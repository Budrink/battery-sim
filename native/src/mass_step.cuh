// Optional leading fracture, then two conservative transfer operators in one
// transaction. Thermal kernels/contact laws are unchanged. Pressure transport
// reads the completed thermal/production/LTE snapshot, never partial neighbours.
double CudaSolver::Impl::prepare_split_step(double time,double cap,double safety,const RuntimeParameters& runtime) {
    for(int retry=0;retry<32;++retry) {
        struct RestoreInputs {Impl& owner;~RestoreInputs(){owner.restore_prefix_inputs();}} restore{*this};
        begin_prefix_inputs();
        // A: heat exchange + local material chemistry. Gas production is a
        // conservative transfer from condensed inventory, NOT advection.
        propose_gas_transport(time);
        select_solid_step(time,cap,safety,runtime);
        select_gas_step(time,safety,runtime);
        if(prefix_solid) {
            double before,after;
            check(cudaMemcpy(&before,dt.p,sizeof(double),cudaMemcpyDeviceToHost),"leading shell dt");
            apply_leading_shell(time,runtime);
            // Both transfer geometries see the NEW aperture and the energy
            // already spent on fracture. If that needs a smaller dt, redo the
            // whole proposal, including destruction, from committed inputs.
            propose_gas_transport(time);
            select_solid_step(time,before,safety,runtime);
            select_gas_step(time,safety,runtime);
            check(cudaMemcpy(&after,dt.p,sizeof(double),cudaMemcpyDeviceToHost),"post-fracture budget");
            if(after<before*(1-1e-12)) {cap=std::min(cap,.95*after);continue;}
        }
        propose_edges(runtime,time);
        propose_local_chemistry(time,runtime);
        gather_solid_candidate(time,runtime);
        if(gas_a.n)gas_heat_stage_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,gas_a.p,gas_stage->p,
            gas_patches_rows->view(),gas_flux.p,gas_sources_rows->view(),source_transfer.p,scenario.gas_model,dt.p);
        equilibrate(state_b.p,gas_stage->p,runtime.runaway_power_scale);

        if(mc_gas) {
            if(channels.n)gas_channel_kernel<<<work_blocks(channels.n),work_threads>>>(channels.n,channels.p,
                channel_state.p,channel_next.p,gas_stage->p,macro_cells.p,scenario.gas_model,time,gas_config.p,
                gas_channels_rows->view(),channel_mass_cap.p,state_b.p,volumes.p,materials.p);
            double duration;
            check(cudaMemcpy(&duration,dt.p,sizeof(double),cudaMemcpyDeviceToHost),"MC gas interval");
            const auto started=std::chrono::steady_clock::now();
            mc_gas->prepare(gas_stage->p,gas_b.p,state_b.p,channel_next.p,duration);
            profiling.cuda_gas_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            equilibrate(state_b.p,gas_b.p,runtime.runaway_power_scale);
            if(gas_a.n)mixture_aging_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,mass_bindings->p,
                volumes.p,materials.p,state_a.p,state_b.p,runtime,time,dt.p);
            return close_and_validate_candidate(time,runtime);
        }

        if(cpu_gas) {
            if(channels.n)gas_channel_kernel<<<work_blocks(channels.n),work_threads>>>(channels.n,channels.p,
                channel_state.p,channel_next.p,gas_stage->p,macro_cells.p,scenario.gas_model,time,gas_config.p,
                gas_channels_rows->view(),channel_mass_cap.p,state_b.p,volumes.p,materials.p);
            double duration;
            check(cudaMemcpy(&duration,dt.p,sizeof(double),cudaMemcpyDeviceToHost),"CPU gas interval");
            // Download is the synchronization boundary: no CUDA kernel reads
            // partial CPU results. CPU only modifies disposable candidates.
            auto stamp=std::chrono::steady_clock::now();
            gas_stage->download(host_gas);state_b.download(host_solid);channel_next.download(host_channels);
            auto now=std::chrono::steady_clock::now();profiling.download_s+=std::chrono::duration<double>(now-stamp).count();stamp=now;
            cpu_gas->advance(host_gas,host_solid,host_channels,duration,&profiling.gas);
            now=std::chrono::steady_clock::now();profiling.cpu_gas_s+=std::chrono::duration<double>(now-stamp).count();stamp=now;
            gas_b.upload(host_gas);state_b.upload(host_solid);channel_next.upload(host_channels);
            profiling.upload_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-stamp).count();
            equilibrate(state_b.p,gas_b.p,runtime.runaway_power_scale);
            if(gas_a.n)mixture_aging_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,mass_bindings->p,
                volumes.p,materials.p,state_a.p,state_b.p,runtime,time,dt.p);
            return close_and_validate_candidate(time,runtime);
        }

        // B: independent pressure geometry. Heat just supplied in A can change
        // pressure substantially, so recheck its budget before moving mass.
        check(cudaMemcpyAsync(mass_dt->p,dt.p,sizeof(double),cudaMemcpyDeviceToDevice),"mass dt proposal");
        check(cudaMemcpyAsync(mass_dt->p+1,dt.p,sizeof(double),cudaMemcpyDeviceToDevice),"thermal dt snapshot");
        if(channels.n)gas_channel_kernel<<<work_blocks(channels.n),work_threads>>>(channels.n,channels.p,
            channel_state.p,channel_next.p,gas_stage->p,macro_cells.p,scenario.gas_model,time,gas_config.p,
            gas_channels_rows->view(),channel_mass_cap.p,state_b.p,volumes.p,materials.p);
        if(shell_patches)shell_feedback_limit_kernel<<<blocks(channels.n),128>>>(channels.n,channels.p,object_offsets.p,
            shell_patches->p,shell_damage->p,shell_rates->p,gas_stage->p,gas_config.p,gas_channels_rows->view(),
            scenario.gas_model,state_b.p,volumes.p,materials.p,mass_dt->p);
        if(gas_a.n)gas_limit_kernel<<<blocks(gas_a.n),128>>>(gas_a.n,gas_stage->p,gas_channels_rows->view(),
            channels.p,channel_next.p,gas_patches_rows->view(),gas_conductance.p,gas_flux.p,gas_sources_rows->view(),
            volumes.p,materials.p,state_b.p,scenario.gas_model,mass_dt->p,runtime.runaway_power_scale,safety,
            gas_config.p,channel_mass_cap.p,shell_work_power?shell_work_power->p:nullptr,true);
        double proposed[2];
        check(cudaMemcpy(proposed,mass_dt->p,sizeof(proposed),cudaMemcpyDeviceToHost),"split budget completion");
        if(!std::isfinite(proposed[0]) || proposed[0]<=0)throw std::runtime_error("Invalid mass substep budget");
        if(proposed[0]<proposed[1]*(1-1e-12)) {
            cap=std::min(cap,.95*proposed[0]);
            continue; // all candidate inventories are discarded, including A
        }
        if(channels.n)gas_transfer_kernel<<<work_blocks(channels.n),work_threads>>>(channels.n,channels.p,
            channel_next.p,channel_state.p,channel_next.p,gas_stage->p,scenario.gas_model,dt.p,channel_mass_cap.p,channel_transfer.p);
        if(gas_a.n) {
            gas_reaction_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,gas_stage->p,gas_config.p,
                gas_channels_rows->view(),channels.p,channel_transfer.p,gas_sources_rows->view(),source_transfer.p,dt.p,gas_reaction.p,true);
            gas_advance_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,gas_stage->p,gas_b.p,gas_config.p,
                gas_channels_rows->view(),channels.p,channel_transfer.p,gas_patches_rows->view(),gas_flux.p,
                gas_sources_rows->view(),source_transfer.p,gas_reaction.p,scenario.gas_model,dt.p,true);
        }
        equilibrate(state_b.p,gas_b.p,runtime.runaway_power_scale);
        if(gas_a.n)mixture_aging_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,mass_bindings->p,
            volumes.p,materials.p,state_a.p,state_b.p,runtime,time,dt.p);
        // New thermal weakening belongs to the next leading fracture step.
        // Legacy modes may still perform trailing mechanical work in closure.
        // Validate all ledgers before anything can be committed.
        return close_and_validate_candidate(time,runtime);
    }
    throw std::runtime_error("Split thermal/mass timestep failed to converge");
}

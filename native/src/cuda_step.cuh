// Included after CudaSolver::Impl. This file owns execution order, not laws.
// Every stage is ordered on the default CUDA stream. Split-step budget checks
// and final validation synchronize on the host; retries discard all proposals.
void CudaSolver::Impl::select_solid_step(double time,double cap,double safety,const RuntimeParameters& runtime) {
    dt.upload({cap});
    if(cached_interface_scale!=runtime.interface_conductance_scale || cached_ambient_scale!=runtime.ambient_conductance_scale || cached_safety!=safety) {
        cfl.upload({1e300});
        if(has_spatial)cfl_kernel<<<blocks(volumes.n),128>>>(volumes.n,volumes.p,links.p,offsets.p,adjacency.p,
            materials.p,runtime,safety,cfl.p,volume_patches->view(),gas_patches.p);
        if(has_isothermal)isothermal_cfl_kernel<<<scenario.cells.size(),128>>>(volumes.p,links.p,object_offsets.p,
            offsets.p,adjacency.p,materials.p,runtime,safety,cfl.p);
        cached_interface_scale=runtime.interface_conductance_scale;
        cached_ambient_scale=runtime.ambient_conductance_scale;cached_safety=safety;
    }
    stability_kernel<<<blocks(volumes.n),128>>>(volumes.n,state_a.p,volumes.p,materials.p,runtime,cfl.p,dt.p,time,
        safety,volume_patches->view(),gas_patches.p,gas_conductance.p);
}
void CudaSolver::Impl::propose_gas_transport(double time) {
    if(channels.n)gas_channel_kernel<<<work_blocks(channels.n),work_threads>>>(channels.n,channels.p,
        channel_state.p,channel_next.p,gas_a.p,macro_cells.p,scenario.gas_model,time,gas_config.p,
        gas_channels_rows->view(),channel_mass_cap.p,state_a.p,volumes.p,materials.p);
    if(gas_patches.n)gas_contact_kernel<<<work_blocks(gas_patches.n),work_threads>>>(gas_patches.n,
        gas_patches.p,gas_a.p,state_a.p,gas_flux.p,gas_conductance.p);
    if(aperture_patch_ids->n)aperture_flux_kernel<<<work_blocks(aperture_patch_ids->n),work_threads>>>(aperture_patch_ids->n,
        aperture_patch_ids->p,gas_patches.p,channels.p,channel_next.p,shell_damage->p,gas_a.p,state_a.p,
        scenario.gas_model,gas_flux.p,gas_conductance.p);
}
void CudaSolver::Impl::select_gas_step(double time,double safety,const RuntimeParameters& runtime) {
    if(prefix_solid && !prefix_applied)leading_shell_limit_kernel<<<blocks(volumes.n),128>>>(volumes.n,
        shell_patches->p,channels.p,shell_damage->p,gas_a.p,gas_config.p,gas_channels_rows->view(),scenario.gas_model,
        state_a.p,volumes.p,materials.p,runtime.damage_rate_scale,dt.p);
    if(shell_patches)shell_damage_proposal_kernel<<<blocks(volumes.n),128>>>(volumes.n,shell_patches->p,channels.p,
        state_a.p,shell_damage->p,gas_a.p,shell_rates->p,runtime.damage_rate_scale,dt.p);
    if(shell_patches)shell_feedback_limit_kernel<<<blocks(channels.n),128>>>(channels.n,channels.p,object_offsets.p,
        shell_patches->p,shell_damage->p,shell_rates->p,gas_a.p,gas_config.p,gas_channels_rows->view(),
        scenario.gas_model,state_a.p,volumes.p,materials.p,dt.p);
    if(shell_work_power)shell_work_power_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,gas_channels_rows->view(),
        channels.p,object_offsets.p,shell_patches->p,shell_damage->p,shell_rates->p,shell_work_power->p);
    if(gas_a.n)gas_limit_kernel<<<blocks(gas_a.n),128>>>(gas_a.n,gas_a.p,gas_channels_rows->view(),
        channels.p,channel_next.p,gas_patches_rows->view(),gas_conductance.p,gas_flux.p,gas_sources_rows->view(),
        volumes.p,materials.p,state_a.p,scenario.gas_model,dt.p,runtime.runaway_power_scale,safety,gas_config.p,channel_mass_cap.p,
        shell_work_power?shell_work_power->p:nullptr,false,time,runtime.max_dt_s);
}
void CudaSolver::Impl::propose_local_chemistry(double time,const RuntimeParameters& runtime) {
    if(conversion_ids->n)source_kernel<true><<<work_blocks(conversion_ids->n),work_threads>>>(conversion_ids->n,
        conversion_ids->p,volumes.p,materials.p,state_a.p,local_state.p,local_heat.p,source_transfer.p,runtime,time,dt.p);
    if(legacy_ids->n)source_kernel<false><<<work_blocks(legacy_ids->n),work_threads>>>(legacy_ids->n,
        legacy_ids->p,volumes.p,materials.p,state_a.p,local_state.p,local_heat.p,source_transfer.p,runtime,time,dt.p);
}
void CudaSolver::Impl::propose_processes(double time,const RuntimeParameters& runtime) {
    propose_edges(runtime,time);
    propose_local_chemistry(time,runtime);
    if(channels.n)gas_transfer_kernel<<<work_blocks(channels.n),work_threads>>>(channels.n,channels.p,
        channel_next.p,channel_state.p,channel_next.p,gas_a.p,scenario.gas_model,dt.p,channel_mass_cap.p,channel_transfer.p);
    if(aperture_patch_ids->n) {
        // Re-evaluate with ACCEPTED, mass-capped flow, not the larger proposal.
        // The conductance can only decrease; previously selected bounds hold.
        aperture_flux_kernel<<<work_blocks(aperture_patch_ids->n),work_threads>>>(aperture_patch_ids->n,
            aperture_patch_ids->p,gas_patches.p,channels.p,channel_next.p,shell_damage->p,gas_a.p,state_a.p,
            scenario.gas_model,gas_flux.p,gas_conductance.p);
        aperture_account_kernel<<<work_blocks(channels.n),work_threads>>>(channels.n,aperture_channel_rows->view(),
            gas_flux.p,channel_state.p,channel_next.p,dt.p);
    }
    if(gas_a.n)gas_reaction_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,gas_a.p,gas_config.p,
        gas_channels_rows->view(),channels.p,channel_transfer.p,gas_sources_rows->view(),source_transfer.p,dt.p,gas_reaction.p);
}
void CudaSolver::Impl::gather_solid_candidate(double time,const RuntimeParameters& runtime) {
    if(has_spatial)volume_kernel<<<work_blocks(volumes.n),work_threads>>>(volumes.n,volumes.p,materials.p,
        links.p,offsets.p,adjacency.p,flux.p,state_a.p,state_b.p,runtime,time,dt.p,volume_patches->view(),
        gas_flux.p,local_state.p,local_heat.p);
    if(has_isothermal)isothermal_volume_kernel<<<scenario.cells.size(),128>>>(volumes.p,materials.p,links.p,
        object_offsets.p,offsets.p,adjacency.p,flux.p,state_a.p,state_b.p,runtime,time,dt.p,local_state.p,local_heat.p);
}
void CudaSolver::Impl::gather_candidate(double time,const RuntimeParameters& runtime) {
    gather_solid_candidate(time,runtime);
    if(gas_a.n)gas_advance_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,gas_a.p,gas_b.p,gas_config.p,
        gas_channels_rows->view(),channels.p,channel_transfer.p,gas_patches_rows->view(),gas_flux.p,
        gas_sources_rows->view(),source_transfer.p,gas_reaction.p,scenario.gas_model,dt.p);
}
double CudaSolver::Impl::close_and_validate_candidate(double time,const RuntimeParameters& runtime) {
    if(channels.n)shell_damage_apply_kernel<<<work_blocks(volumes.n),work_threads>>>(volumes.n,volumes.p,materials.p,
        state_a.p,state_b.p,shell_channels_rows->view(),channels.p,channel_next.p,shell_patches?shell_patches->p:nullptr,
        shell_patches?shell_damage->p:nullptr,shell_patches?shell_next->p:nullptr,shell_patches?shell_rates->p:nullptr,dt.p,runtime.runaway_power_scale);
    if(shell_work_power)shell_work_apply_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,gas_channels_rows->view(),
        channels.p,object_offsets.p,shell_patches->p,shell_damage->p,shell_next->p,volumes.p,materials.p,state_a.p,state_b.p,
        gas_b.p,scenario.gas_model,channel_next.p,runtime.runaway_power_scale);
    if(shell_event_pressure_snapshot) {
        check(cudaMemcpyAsync(shell_event_pressure_snapshot->p,gas_b.p,gas_b.n*sizeof(GasState),cudaMemcpyDeviceToDevice),"fracture event snapshot");
        shell_event_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,gas_channels_rows->view(),channels.p,
            object_offsets.p,shell_patches->p,shell_next->p,volumes.p,materials.p,state_b.p,
            shell_event_pressure_snapshot->p,gas_b.p,scenario.gas_model,channel_next.p,runtime.runaway_power_scale);
    }
    if(shell_patches)shell_aperture_kernel<<<work_blocks(channels.n),work_threads>>>(channels.n,channels.p,object_offsets.p,
        shell_patches->p,shell_next->p,channel_next.p,time,dt.p);
    if(scenario.gas_model.split_transport)equilibrate(state_b.p,gas_b.p,runtime.runaway_power_scale);
    check(cudaMemsetAsync(result.p,0,sizeof(StepResult)),"CUDA candidate status reset");
    if(channels.n)shell_validate_kernel<<<work_blocks(std::max(shell_patches?volumes.n:0,channels.n)),work_threads>>>(
        shell_patches?volumes.n:0,shell_patches?shell_next->p:nullptr,channels.n,channel_next.p,result.p);
    validate_candidate_kernel<<<work_blocks(std::max({volumes.n,gas_a.n,links.n})),work_threads>>>(volumes.n,
        state_b.p,gas_a.n,gas_b.p,links.n,edge_next.p,dt.p,result.p);
    check(cudaGetLastError(),"CUDA candidate launch");
    StepResult status{};
    check(cudaMemcpy(&status,result.p,sizeof(status),cudaMemcpyDeviceToHost),"CUDA candidate completion");
    if(!std::isfinite(status.dt)||status.dt<=0||status.invalid)
        throw std::runtime_error("Invalid proposed substep (node/gas/edge/shell/channel flags="+std::to_string(status.invalid)+")");
    return status.dt;
}

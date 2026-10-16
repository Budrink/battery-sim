// Transaction boundary for the leading fracture operator. Keep rollback out
// of the constitutive kernels and restore inputs on return, retry or exception.
void CudaSolver::Impl::begin_prefix_inputs() {
    if(!prefix_solid)return;
    check(cudaMemcpyAsync(prefix_solid->p,state_a.p,state_a.n*sizeof(ThermalState),cudaMemcpyDeviceToDevice),"prefix solid copy");
    check(cudaMemcpyAsync(prefix_gas->p,gas_a.p,gas_a.n*sizeof(GasState),cudaMemcpyDeviceToDevice),"prefix gas copy");
    check(cudaMemcpyAsync(prefix_channels->p,channel_state.p,channel_state.n*sizeof(GasChannelState),cudaMemcpyDeviceToDevice),"prefix channels copy");
    check(cudaMemcpyAsync(prefix_shell->p,shell_damage->p,shell_damage->n*sizeof(ShellDamageState),cudaMemcpyDeviceToDevice),"prefix shell copy");
    std::swap(prefix_solid->p,state_a.p);std::swap(prefix_gas->p,gas_a.p);
    std::swap(prefix_channels->p,channel_state.p);std::swap(prefix_shell->p,shell_damage->p);
    prefix_inputs_active=true;prefix_applied=false;
}
void CudaSolver::Impl::restore_prefix_inputs() noexcept {
    if(!prefix_inputs_active)return;
    std::swap(prefix_solid->p,state_a.p);std::swap(prefix_gas->p,gas_a.p);
    std::swap(prefix_channels->p,channel_state.p);std::swap(prefix_shell->p,shell_damage->p);
    prefix_inputs_active=false;prefix_applied=false;
}
void CudaSolver::Impl::apply_leading_shell(double time,const RuntimeParameters& runtime) {
    check(cudaMemcpyAsync(gas_stage->p,gas_a.p,gas_a.n*sizeof(GasState),cudaMemcpyDeviceToDevice),"leading pressure snapshot");
    leading_shell_apply_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,gas_channels_rows->view(),channels.p,object_offsets.p,
        shell_patches->p,shell_damage->p,volumes.p,materials.p,state_a.p,gas_stage->p,gas_a.p,scenario.gas_model,
        channel_state.p,dt.p,runtime.damage_rate_scale,runtime.runaway_power_scale);
    shell_aperture_kernel<<<work_blocks(channels.n),work_threads>>>(channels.n,channels.p,object_offsets.p,
        shell_patches->p,shell_damage->p,channel_state.p,time,dt.p,true);
    equilibrate(state_a.p,gas_a.p,runtime.runaway_power_scale);
    prefix_applied=true;
}

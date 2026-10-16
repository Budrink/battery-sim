// Directional connections exposed by a damaged barrier. Static visibility is
// built in aperture_geometry.cpp; only committed local damage/old T are read.
__global__ void aperture_flux_kernel(int count,const int* ids,const GasThermalPatch* patches,
    const DeviceChannel* channels,const GasChannelState* channel_state,const ShellDamageState* damage,
    const GasState* gas,const ThermalState* solid,GasModel model,double* flux,double* conductance) {
    const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
    const int i=ids[k];const auto& p=patches[i];const auto& c=channels[p.channel];
    const auto& state=channel_state[p.channel];const auto d=damage[p.barrier_volume];
    const double opening=(1.-p.seam_fraction)*shell_open_fraction(c.opening,d.bulk)+
                         p.seam_fraction*shell_open_fraction(c.opening,d.seam);
    const double g=aperture_conductance(p.conductance_w_per_k,opening,state.opening_fraction,
        p.aperture_area_fraction,state.mass_flow_kg_s,model.cv_j_per_kg_k+model.specific_r_j_per_kg_k);
    conductance[i]=g;
    flux[i]=g*(gas[c.from].temperature_k-solid[p.volume].temperature_k);
}
// One channel owns its cumulative diagnostic. No floating-point atomics and
// no mutation of committed state during prepare_step.
__global__ void aperture_account_kernel(int count,DeviceRows rows,const double* flux,
    const GasChannelState* old,GasChannelState* next,const double* step) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    double power=0.;for(int k=rows.offsets[i];k<rows.offsets[i+1];++k)power+=flux[rows.ids[k]];
    next[i].surface_energy_j=old[i].surface_energy_j+power**step;
}

// Mechanical work of exposing an aperture. Pressure supplies this work from
// the existing internal gas inventory. Outflow enthalpy is a DIFFERENT ledger:
// it leaves through a channel; fracture work stays in the damaged structure.
__device__ double patch_opening(const ShellOpeningLaw& law,double seam,ShellDamageState d) {
    return (1.-seam)*shell_open_fraction(law,d.bulk)+seam*shell_open_fraction(law,d.seam);
}
__device__ ShellDamageState interpolate_damage(ShellDamageState a,ShellDamageState b,double scale) {
    // Energy limits mechanical growth only; thermal weakening is not undone.
    return {a.bulk+scale*(b.bulk-a.bulk),a.seam+scale*(b.seam-a.seam),b.thermal,b.failure_hazard};
}
__device__ double channel_fracture_work(const DeviceChannel& c,const int* offsets,
    const DeviceShellPatch* patches,const ShellDamageState* old,const ShellDamageState* next,double scale) {
    if(!c.opening.enabled || discrete_shell_events(c.opening) || c.opening.deterministic_aperture || c.opening.fracture_energy_j_per_m2==0)return 0.;
    const int begin=shell_begin(c,offsets),end=shell_end(c,offsets);double delta=0.;
    for(int k=begin;k<end;++k) {
        const double seam=patches[k].seam_fraction;
        delta+=shell_remaining_barrier(old[k].thermal)*fmax(0.,patch_opening(c.opening,seam,interpolate_damage(old[k],next[k],scale))-
                       patch_opening(c.opening,seam,old[k]));
    }
    return c.opening.fracture_energy_j_per_m2*c.area*delta/(end-begin);
}
// Upper bound on pressure-driven fracture power. Thermal weakening alone is
// not mechanical work and cannot open the barrier. Feed the bound
// into the same timestep budget as gas flow, reaction and thermal contacts.
__global__ void shell_work_power_kernel(int count,DeviceRows rows,const DeviceChannel* channels,
    const int* offsets,const DeviceShellPatch* patches,const ShellDamageState* damage,
    const ShellDamageState* rates,double* power) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;double total=0.;
    for(int j=rows.offsets[i];j<rows.offsets[i+1];++j) {
        const auto& c=channels[rows.ids[j]];const auto& law=c.opening;
        if(c.from!=i || !law.enabled || discrete_shell_events(law) || law.deterministic_aperture || law.fracture_energy_j_per_m2==0)continue;
        const int begin=shell_begin(c,offsets),end=shell_end(c,offsets);double rate=0.;
        for(int k=begin;k<end;++k) {
            const double f=patches[k].seam_fraction;
            rate+=shell_remaining_barrier(damage[k].thermal)*((1.-f)*rates[k].bulk*(1.-damage[k].bulk)+f*rates[k].seam*(1.-damage[k].seam));
        }
        total+=law.fracture_energy_j_per_m2*c.area*law.exponent/(1.-law.damage_onset)*rate/(end-begin);
    }
    power[i]=total;
}
// Each gas node owns its outgoing panels. If the requested work exhausts its
// remaining sensible energy, jointly reduce their proposed damage increments.
// A fixed-count monotone solve is deterministic and does not depend on tiling.
__global__ void shell_work_apply_kernel(int count,DeviceRows rows,const DeviceChannel* channels,
    const int* offsets,const DeviceShellPatch* patches,const ShellDamageState* old,ShellDamageState* next,
    const DeviceVolume* volumes,const Material* materials,const ThermalState* solid_old,ThermalState* solid,
    GasState* gas,GasModel model,GasChannelState* state,double speed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    double requested=0.;
    for(int j=rows.offsets[i];j<rows.offsets[i+1];++j) {
        const auto& c=channels[rows.ids[j]];if(c.from==i)requested+=channel_fracture_work(c,offsets,patches,old,next,1.);
    }
    if(requested<=0)return;
    // Numerical positivity floor, not a fitted thermodynamic threshold.
    auto& g=gas[i];const double available=fmax(0.,g.mass_kg*model.cv_j_per_kg_k*(g.temperature_k-1.));
    double scale=1.;
    if(requested>available) {
        double lo=0.,hi=1.;
        for(int n=0;n<52;++n) {
            const double mid=(lo+hi)/2;double work=0.;
            for(int j=rows.offsets[i];j<rows.offsets[i+1];++j) {
                const auto& c=channels[rows.ids[j]];if(c.from==i)work+=channel_fracture_work(c,offsets,patches,old,next,mid);
            }
            if(work<=available)lo=mid;else hi=mid;
        }
        scale=lo;
    }
    double spent=0.;
    for(int j=rows.offsets[i];j<rows.offsets[i+1];++j) {
        const int id=rows.ids[j];const auto& c=channels[id];
        if(c.from!=i || !c.opening.enabled || discrete_shell_events(c.opening) || c.opening.deterministic_aperture || c.opening.fracture_energy_j_per_m2==0)continue;
        const double work=channel_fracture_work(c,offsets,patches,old,next,scale);
        spent+=work;state[id].fracture_work_j+=work;
        if(scale<1.)for(int k=shell_begin(c,offsets);k<shell_end(c,offsets);++k) {
            next[k]=interpolate_damage(old[k],next[k],scale);const double f=patches[k].seam_fraction;
            solid[k].damage=shell_material_damage(next[k],f);
            if(solid[k].damage<1.)solid[k].mode=solid_old[k].mode;
            close_thermal_state(materials[volumes[k].object],volumes[k],solid[k],speed);
        }
    }
    g.energy_j-=spent;
    g.temperature_k=model.reference_temperature_k+g.energy_j/(g.mass_kg*model.cv_j_per_kg_k);
    g.pressure_pa=g.mass_kg*model.specific_r_j_per_kg_k*g.temperature_k/g.free_volume_m3;
}

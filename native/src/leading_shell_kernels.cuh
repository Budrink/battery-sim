// Operator 0: deterministic aperture growth and irreversible fracture work.
// Frozen input pressure/D; each gas node owns all outgoing work claims. No RNG,
// atomics, thermal flux or mass transfer. This operates on rollbackable copies.
__global__ void leading_shell_limit_kernel(int count,const DeviceShellPatch* patches,const DeviceChannel* channels,
    const ShellDamageState* damage,const GasState* gas,const DeviceGas* config,DeviceRows rows,GasModel model,
    const ThermalState* solid,const DeviceVolume* volumes,const Material* materials,double speed,double* step) {
    const int k=blockIdx.x*blockDim.x+threadIdx.x;double limit=1e300;
    if(k<count && patches[k].channel>=0) {
        const auto& p=patches[k];const auto& c=channels[p.channel];const auto d=damage[k];
        if(c.opening.deterministic_aperture && d.bulk<1.) {
            const double dp=gas[c.from].pressure_pa-(c.to<0?c.outside_pressure:gas[c.to].pressure_pa);
            const double strength=c.threshold*shell_remaining_barrier(d.thermal);
            const double factor=d.bulk<p.weak_fraction?p.weak_factor:1.;
            const double rate=speed*aperture_rate(c.opening,dp,strength*factor);
            limit=shell_damage_dt(c.opening,rate,d.bulk);
            if(rate>0) {
                // A mass cap guarantees positivity, NOT temporal resolution of
                // pressure unloading. Couple prospective area to raw flow, or
                // a large dt can open a whole seam before pressure is updated.
                const double area=c.area*d.bulk,growth=c.area*rate*(1.-d.bulk);
                const double pb=c.to<0?c.outside_pressure:gas[c.to].pressure_pa;
                const double flow=gas_orifice_flow(gas[c.from].pressure_pa,pb,gas[c.from].temperature_k,1.,c.cd,model);
                // Resolve the excess ABOVE the active rupture threshold, not
                // the much larger absolute pressure or entire equalization gap.
                const double budget=model.max_fraction_per_step*fmax(0.,1.-strength*factor/dp)*
                    gas_pressure_mass_cap(c,gas,config,rows,model,solid,volumes,materials);
                const double unloading=shell_feedback_dt(area,growth,flow,budget,0.);
                const double slow=shell_slow_aperture_dt(area,growth,model.max_fraction_per_step,c.area);
                limit=fmin(limit,fmax(unloading,slow));
            }
        }
    }
    block_minimum(limit,step);
}
__device__ double leading_increment(const DeviceChannel& c,const DeviceShellPatch& p,ShellDamageState d,
    const GasState* gas,double dt,double speed) {
    const double dp=gas[c.from].pressure_pa-(c.to<0?c.outside_pressure:gas[c.to].pressure_pa);
    return deterministic_fraction(c.opening,d.bulk,dp,c.threshold*shell_remaining_barrier(d.thermal),
        p.weak_fraction,p.weak_factor,dt,speed)-d.bulk;
}
__global__ void leading_shell_apply_kernel(int count,DeviceRows rows,const DeviceChannel* channels,const int* offsets,
    const DeviceShellPatch* patches,ShellDamageState* damage,const DeviceVolume* volumes,const Material* materials,
    ThermalState* solid,const GasState* pressure_snapshot,GasState* gas,GasModel model,GasChannelState* states,
    const double* step,double damage_speed,double chemistry_speed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    double requested=0.;
    for(int j=rows.offsets[i];j<rows.offsets[i+1];++j) {
        const auto& c=channels[rows.ids[j]];if(c.from!=i || !c.opening.deterministic_aperture)continue;
        const int k=offsets[c.shell]+c.shell_thermal_volume;const auto d=damage[k];
        requested+=c.opening.fracture_energy_j_per_m2*patches[k].area_m2*shell_remaining_barrier(d.thermal)*
            leading_increment(c,patches[k],d,pressure_snapshot,*step,damage_speed);
    }
    auto& g=gas[i];const double available=fmax(0.,g.mass_kg*model.cv_j_per_kg_k*(g.temperature_k-1.));
    const double scale=requested>available?available/requested:1.;double spent=0.;
    for(int j=rows.offsets[i];j<rows.offsets[i+1];++j) {
        const int id=rows.ids[j];const auto& c=channels[id];if(c.from!=i || !c.opening.deterministic_aperture)continue;
        const int k=offsets[c.shell]+c.shell_thermal_volume;auto& d=damage[k];
        const double da=scale*leading_increment(c,patches[k],d,pressure_snapshot,*step,damage_speed);
        const double work=c.opening.fracture_energy_j_per_m2*patches[k].area_m2*shell_remaining_barrier(d.thermal)*da;
        d.bulk=d.seam=fmin(1.,d.bulk+da);states[id].fracture_work_j+=work;spent+=work;
        solid[k].damage=shell_material_damage(d,patches[k].seam_fraction);
        if(solid[k].damage>=1.)solid[k].mode=CellMode::Spent;
        close_thermal_state(materials[volumes[k].object],volumes[k],solid[k],chemistry_speed);
    }
    g.energy_j-=spent;
    g.temperature_k=model.reference_temperature_k+g.energy_j/(g.mass_kg*model.cv_j_per_kg_k);
    g.pressure_pa=g.mass_kg*model.specific_r_j_per_kg_k*g.temperature_k/g.free_volume_m3;
}

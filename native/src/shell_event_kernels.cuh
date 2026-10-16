// Discrete fracture events: whole pixels (legacy) or random physical fragments.
// One gas node owns the event batch and its energy debit. All predicates read
// the same pre-fracture pressure snapshot; no atomics or traversal-dependent
// selection of which simultaneous pixel gets the energy first.
__device__ bool shell_event_due(const DeviceChannel& c,const DeviceShellPatch& p,
    ShellDamageState d,double pressure_drop) {
    return d.bulk<1. && (!(c.opening.stochastic_pixels || c.opening.stochastic_area) ||
        pixel_hazard_reached(d.failure_hazard,shell_event_threshold(c.opening,p,d))) &&
        pressure_drop>binary_shell_strength(c.threshold,d.thermal,
        p.seam_fraction,c.opening.seam_strength_factor);
}
__device__ double shell_event_increment(const DeviceChannel& c,const DeviceShellPatch& p,ShellDamageState d) {
    if(c.opening.fixed_area_fraction>0)return fixed_area_event_increment(d.bulk,c.opening.fixed_area_fraction);
    return c.opening.stochastic_area?area_event_increment(d.bulk,p.area_m2,c.opening.fragment_mean_area_m2,p.random_key,d.rupture_events):1.;
}
__global__ void shell_event_kernel(int count,DeviceRows rows,const DeviceChannel* channels,
    const int* offsets,const DeviceShellPatch* patches,ShellDamageState* damage,
    const DeviceVolume* volumes,const Material* materials,ThermalState* solid,
    const GasState* pressure_snapshot,GasState* gas,GasModel model,GasChannelState* states,double speed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    double requested=0.;
    for(int j=rows.offsets[i];j<rows.offsets[i+1];++j) {
        const auto& c=channels[rows.ids[j]];if(c.from!=i || !discrete_shell_events(c.opening))continue;
        const int begin=shell_begin(c,offsets),end=shell_end(c,offsets);
        const double dp=pressure_snapshot[i].pressure_pa-(c.to<0?c.outside_pressure:pressure_snapshot[c.to].pressure_pa);
        const double cost=c.opening.fracture_energy_j_per_m2*c.area/(end-begin);
        for(int k=begin;k<end;++k)if(shell_event_due(c,patches[k],damage[k],dp))
            requested+=cost*shell_remaining_barrier(damage[k].thermal)*shell_event_increment(c,patches[k],damage[k]);
    }
    // A batch is indivisible: insufficient energy defers it, never opens a
    // resized fragment or selects patches by thread/index order. This corner
    // case must be checked when choosing fracture work and gas inventories.
    auto& g=gas[i];
    const double available=fmax(0.,g.mass_kg*model.cv_j_per_kg_k*(g.temperature_k-1.));
    if(requested>available)return;
    for(int j=rows.offsets[i];j<rows.offsets[i+1];++j) {
        const int id=rows.ids[j];const auto& c=channels[id];if(c.from!=i || !discrete_shell_events(c.opening))continue;
        const int begin=shell_begin(c,offsets),end=shell_end(c,offsets);
        const double dp=pressure_snapshot[i].pressure_pa-(c.to<0?c.outside_pressure:pressure_snapshot[c.to].pressure_pa);
        const double cost=c.opening.fracture_energy_j_per_m2*c.area/(end-begin);
        double spent=0.;
        for(int k=begin;k<end;++k)if(shell_event_due(c,patches[k],damage[k],dp)) {
            const double increment=shell_event_increment(c,patches[k],damage[k]);
            spent+=cost*shell_remaining_barrier(damage[k].thermal)*increment;
            damage[k].bulk=damage[k].seam=fmin(1.,damage[k].bulk+increment);
            if(c.opening.stochastic_area) {
                // Event consumed only by an accepted candidate. A rejected
                // transaction restores both its clock and its fragment mark.
                damage[k].failure_hazard=0.;++damage[k].rupture_events;
            }
            solid[k].damage=shell_material_damage(damage[k],patches[k].seam_fraction);
            if(solid[k].damage>=1.)solid[k].mode=CellMode::Spent;
            close_thermal_state(materials[volumes[k].object],volumes[k],solid[k],speed);
        }
        states[id].fracture_work_j+=spent;
    }
    g.energy_j-=requested;
    g.temperature_k=model.reference_temperature_k+g.energy_j/(g.mass_kg*model.cv_j_per_kg_k);
    g.pressure_pa=g.mass_kg*model.specific_r_j_per_kg_k*g.temperature_k/g.free_volume_m3;
}

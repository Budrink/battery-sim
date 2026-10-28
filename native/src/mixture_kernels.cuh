// Local thermodynamic closure, not a heat-conductivity law. One thread owns
// one mobile phase and its unique solid volume; there are no scatter updates.
__global__ void gas_heat_stage_kernel(int count,const GasState* old,GasState* next,
    DeviceRows patches,const double* flux,DeviceRows sources,const GasTransfer* production,
    GasModel model,const double* dt) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;auto g=old[i];
    for(int j=patches.offsets[i];j<patches.offsets[i+1];++j)g.energy_j-=flux[patches.ids[j]]**dt;
    for(int j=sources.offsets[i];j<sources.offsets[i+1];++j) {
        const auto& q=production[sources.ids[j]];
        g.mass_kg+=q.mass;g.energy_j+=q.energy;g.chemical_j+=q.chemical;g.oxidizer_kg+=q.oxidizer;
    }
    g.temperature_k=model.reference_temperature_k+g.energy_j/(g.mass_kg*model.cv_j_per_kg_k);
    g.pressure_pa=g.mass_kg*model.specific_r_j_per_kg_k*g.temperature_k/g.free_volume_m3;
    next[i]=g;
}
__global__ void mixture_equilibrium_kernel(int count,const MassBinding* bindings,const DeviceGas* config,
    const DeviceVolume* volumes,const Material* materials,ThermalState* solid,GasState* gas,GasModel model,double speed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const auto& b=bindings[i];
    if(b.thermal_volume<0)return;
    auto& g=gas[i];auto& s=solid[b.thermal_volume];const auto& v=volumes[b.thermal_volume];const auto& m=materials[v.object];
    const double total=s.stored_energy_j+g.energy_j, cg=g.mass_kg*model.cv_j_per_kg_k,tref=model.reference_temperature_k;
    const double temperature=mixture_temperature(m.capacity_law,m.conversion,s,g,v.capacity_j_per_k,
        v.capacity_j_per_k/m.capacity,model);
    g.energy_j=cg*(temperature-tref);s.stored_energy_j=total-g.energy_j;
    g.temperature_k=s.temperature_k=temperature;s.mobile_capacity_j_per_k=cg;
    if(m.conversion.enabled)refresh_conversion(m.conversion,s,v.capacity_j_per_k/m.capacity,speed);
    else if(m.pulse.profile==RunawayProfile::Kinetic)refresh_reaction(m.kinetics,m.structure,s,m.energy*v.capacity_j_per_k/m.capacity,speed);
    g.free_volume_m3=config[i].volume;
    if(b.reactant_density>0)g.free_volume_m3+=b.initial_reactant_kg/b.reactant_density-
        (s.reactant_kg+s.activated_kg)/b.reactant_density-s.residue_kg/b.residue_density;
    if(b.initial_liquid_kg>0.)g.free_volume_m3+=(b.initial_liquid_kg-s.liquid_kg)/b.liquid_density;
    // Invalid mixture geometry fails validation, never clamps away inventory.
    g.pressure_pa=(g.free_volume_m3>0 && g.free_volume_m3<=b.geometric_volume_m3*(1+1e-10))?
        g.mass_kg*model.specific_r_j_per_kg_k*temperature/g.free_volume_m3:nan("");
}

// Candidate-only phase closure. One writer owns a unique gas/material pair.
__global__ void volatile_phase_kernel(int count,const MassBinding* bindings,
    const DeviceVolume* volumes,const Material* materials,ThermalState* solid,GasState* gas,
    GasModel model,const double* dt) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const int j=bindings[i].thermal_volume;if(j<0)return;
    const auto& v=volumes[j];const auto& m=materials[v.object];
    if(!m.conversion.enabled || !(m.conversion.volatile_phase.initial_mass_kg>0.))return;
    advance_volatile(m.conversion,solid[j],gas[i],v.capacity_j_per_k,v.capacity_j_per_k/m.capacity,model,*dt);
}

// Recompute passive-surface aging from the old state AFTER the coupled gas
// operator supplied its heat. The earlier disposable thermal proposal's aging
// is replaced, not accumulated twice. No source-bearing volume is eligible.
__global__ void finalize_film_aging_kernel(int count,const int* ids,const DeviceVolume* volumes,
    const Material* materials,const ThermalState* old,ThermalState* next,RuntimeParameters runtime,
    double time,const double* step) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const int index=ids[i];auto& s=next[index];const auto& v=volumes[index];const auto& m=materials[v.object];
    s.damage=old[index].damage;s.mode=old[index].mode;s.runaway_started_s=old[index].runaway_started_s;
    if(s.mode==CellMode::Healthy) {
        const double t=.5*(s.temperature_k+old[index].temperature_k);
        s.damage=fmin(1.,s.damage+damage_rate(t,m.onset,m.rate,m.temperature_scale)*m.aging_multiplier*runtime.damage_rate_scale**step);
        if(s.damage>=1.)s.mode=CellMode::Spent;
    }
    close_thermal_state(m,v,s,runtime.runaway_power_scale);
}

// Once per accepted candidate, after both operators have supplied their heat.
// Chemical conversion has its own inventory-based damage; do not age it twice.
__global__ void mixture_aging_kernel(int count,const MassBinding* bindings,
    const DeviceVolume* volumes,const Material* materials,const ThermalState* old,ThermalState* next,
    RuntimeParameters runtime,double time,const double* step) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const int index=bindings[i].thermal_volume;if(index<0)return;
    auto& s=next[index];const auto& v=volumes[index];const auto& m=materials[v.object];
    if(m.conversion.enabled)return;
    if(s.mode==CellMode::Healthy || (m.pulse.profile==RunawayProfile::Kinetic && reaction_live(s))) {
        const double t=.5*(s.temperature_k+old[index].temperature_k);
        s.damage=fmin(1.,s.damage+damage_rate(t,m.onset,m.rate,m.temperature_scale)*m.aging_multiplier*runtime.damage_rate_scale**step);
        const double threshold=m.pulse.profile==RunawayProfile::Kinetic?m.kinetics.activation_damage:1.;
        if(s.mode==CellMode::Healthy && s.damage>=threshold) {
            if(m.is_battery)s.runaway_started_s=time+*step;
            s.mode=m.is_battery && m.energy>0?CellMode::Runaway:CellMode::Spent;
        }
    }
    close_thermal_state(m,v,s,runtime.runaway_power_scale);
}

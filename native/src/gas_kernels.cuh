// Included inside cuda_solver.cu's private namespace. All fluxes use old states;
// every channel/thermal patch has a single flux shared by its two recipients.
struct DeviceGas { double volume, reaction_rate, oxidizer_heat, ignition_temperature; int thermal_volume=-1; };
__device__ double gas_local_capacity(const DeviceGas& config,const GasState& gas,GasModel model,
    const ThermalState* solid,const DeviceVolume* volumes,const Material* materials) {
    double capacity=gas.mass_kg*model.cv_j_per_kg_k;
    if(config.thermal_volume>=0) {
        const int j=config.thermal_volume;const auto& v=volumes[j];const auto& m=materials[v.object];
        capacity+=m.conversion.enabled?conversion_capacity(m.conversion,solid[j],v.capacity_j_per_k,v.capacity_j_per_k/m.capacity):
            v.capacity_j_per_k*capacity_factor(m.capacity_law,solid[j].temperature_k,solid[j].damage);
    }
    return capacity;
}
// Extensive payload for one accepted transfer (not a rate). One record is
// gathered with opposite signs at the two endpoints, and used for diagnostics.
struct GasTransfer { double mass=0.,energy=0.,chemical=0.,oxidizer=0.; };
struct DeviceChannel {
    int from,to,shell;
    double area,cd,threshold,outside_pressure,strength_reference,strength_scale;
    bool rupture;
    ShellOpeningLaw opening;
    int shell_thermal_volume=-1;
};
__device__ int shell_begin(const DeviceChannel& c,const int* offsets) {
    return offsets[c.shell]+(c.shell_thermal_volume<0?0:c.shell_thermal_volume);
}
__device__ int shell_end(const DeviceChannel& c,const int* offsets) {
    return c.shell_thermal_volume<0?offsets[c.shell+1]:shell_begin(c,offsets)+1;
}
// Frozen-state pressure relaxation mass, shared conservatively among incident
// channels. Used both by transport and by prospective aperture feedback.
__device__ double gas_pressure_mass_cap(const DeviceChannel& c,const GasState* gas,
    const DeviceGas* config,DeviceRows rows,GasModel model,const ThermalState* thermal,
    const DeviceVolume* volumes,const Material* materials) {
    const auto& a=gas[c.from];const double pb=c.to<0?c.outside_pressure:gas[c.to].pressure_pa;
    const bool forward=a.pressure_pa>=pb;
    if(!forward && c.to<0)return 0.;
    const int donor=forward?c.from:c.to,receiver=forward?c.to:c.from;
    const int nd=rows.offsets[donor+1]-rows.offsets[donor];
    const int nr=receiver>=0?rows.offsets[receiver+1]-rows.offsets[receiver]:0;
    const double gamma=1+model.specific_r_j_per_kg_k/model.cv_j_per_kg_k;
    double donor_slope=gamma*model.specific_r_j_per_kg_k*gas[donor].temperature_k/gas[donor].free_volume_m3;
    double receiver_slope=receiver>=0?gamma*model.specific_r_j_per_kg_k*gas[donor].temperature_k/gas[receiver].free_volume_m3:0.;
    if(model.split_transport) {
        const auto& d=gas[donor];const double cd=gas_local_capacity(config[donor],d,model,thermal,volumes,materials);
        donor_slope=model.specific_r_j_per_kg_k*d.temperature_k*(1+d.mass_kg*model.specific_r_j_per_kg_k/cd)/d.free_volume_m3;
        if(receiver>=0) {
            const auto& r=gas[receiver];const double cr=gas_local_capacity(config[receiver],r,model,thermal,volumes,materials);
            const double mobile_fraction=r.mass_kg*model.cv_j_per_kg_k/cr;
            receiver_slope=model.specific_r_j_per_kg_k*((1-mobile_fraction)*r.temperature_k+mobile_fraction*gamma*d.temperature_k)/r.free_volume_m3;
        }
    }
    // Allocate each endpoint's pressure response among its own incident edges.
    // max(nd,nr)*(donor_slope+receiver_slope) would spuriously impose a large
    // reservoir's entire degree on a small source pore's much faster response.
    return fabs(a.pressure_pa-pb)/(nd*donor_slope+nr*receiver_slope);
}
__global__ void gas_channel_kernel(int count,const DeviceChannel* channels,const GasChannelState* old,GasChannelState* states,
    const GasState* gas,const CellState* solid,GasModel model,double time,const DeviceGas* config,DeviceRows rows,double* mass_cap,
    const ThermalState* thermal,const DeviceVolume* volumes,const Material* materials) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& c=channels[i];auto& s=states[i];s=old[i];const auto& a=gas[c.from];
    const double pb=c.to<0?c.outside_pressure:gas[c.to].pressure_pa;
    double strength=c.threshold;
    if(c.rupture) {
        const auto& shell=solid[c.shell];
        strength*=fmax(.01,1-shell.max_damage)*exp(fmax(-80.,-fmax(0.,shell.max_temperature_k-c.strength_reference)/c.strength_scale));
    }
    if(!c.opening.enabled) {
        const bool unobstructed=model.split_transport && !c.rupture && c.threshold==0;
        if(!s.open && (unobstructed || a.pressure_pa-pb>=strength)) {s.open=true;s.opened_s=time;}
        s.opening_fraction=s.open?1.:0.;s.effective_area_m2=c.area*s.opening_fraction;
    }
    s.mass_flow_kg_s=0.;
    s.proposed_mass_flow_kg_s=0.;
    mass_cap[i]=0.;
    if(s.open) {
        if(c.to>=0 && pb>a.pressure_pa)
            s.mass_flow_kg_s=-gas_orifice_flow(pb,a.pressure_pa,gas[c.to].temperature_k,s.effective_area_m2,c.cd,model);
        else s.mass_flow_kg_s=gas_orifice_flow(a.pressure_pa,pb,a.temperature_k,s.effective_area_m2,c.cd,model);
        s.proposed_mass_flow_kg_s=s.mass_flow_kg_s;
        // A frozen explicit orifice rate must not exchange more mass than
        // equalizes pressure during this substep. Otherwise sqrt(dp) chatters
        // around equilibrium and a tiny source plenum throttles the whole run.
        // Shared-node degree gives a conservative cap for multiple channels.
        if(s.mass_flow_kg_s!=0)mass_cap[i]=gas_pressure_mass_cap(c,gas,config,rows,model,thermal,volumes,materials);
    }
}
__global__ void gas_transfer_kernel(int count,const DeviceChannel* channels,const GasChannelState* proposed,
    const GasChannelState* old,GasChannelState* next,const GasState* gas,GasModel model,const double* step,const double* mass_cap,GasTransfer* transfer) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    auto s=proposed[i];const double dm=copysign(fmin(fabs(s.mass_flow_kg_s)**step,mass_cap[i]),s.mass_flow_kg_s);
    s.mass_flow_kg_s=dm/ *step;
    const auto& donor=gas[dm>=0?channels[i].from:channels[i].to];
    const GasTransfer q{dm,dm*gas_enthalpy(donor,model),dm*donor.chemical_j/donor.mass_kg,dm*donor.oxidizer_kg/donor.mass_kg};
    transfer[i]=q;
    s.transferred_kg=old[i].transferred_kg+q.mass;s.transferred_energy_j=old[i].transferred_energy_j+q.energy;
    s.transferred_chemical_j=old[i].transferred_chemical_j+q.chemical;s.transferred_oxidizer_kg=old[i].transferred_oxidizer_kg+q.oxidizer;
    next[i]=s;
}
__global__ void gas_contact_kernel(int count,const GasThermalPatch* patches,const GasState* gas,const ThermalState* solid,double* flux,double* conductance) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const auto& p=patches[i];
    if(p.channel>=0)return; // directional pool is handled by aperture_flux_kernel
    // h is an effective gas-to-volume-centre coefficient including the unresolved
    // boundary layer/half-volume, not a separately measured surface-film h.
    flux[i]=p.conductance_w_per_k*(gas[p.compartment].temperature_k-solid[p.volume].temperature_k);
    conductance[i]=p.conductance_w_per_k;
}
struct GasRateBudget {double outgoing_mass=0.,thermal_power=0.,pressure_dt=1e300;};
__device__ GasRateBudget gas_rate_budget(int node,DeviceRows rows,const DeviceChannel* channels,
    const GasChannelState* proposed,const double* caps,const GasState* gas,GasModel model) {
    GasRateBudget budget;
    for(int k=rows.offsets[node];k<rows.offsets[node+1];++k) {
        const int j=rows.ids[k];const auto& c=channels[j];const double flow=proposed[j].mass_flow_kg_s;
        if(flow==0.)continue;
        const int donor=flow>0?c.from:c.to;
        if(donor==node)budget.outgoing_mass+=fabs(flow);
        const double coefficient=donor==node?model.specific_r_j_per_kg_k*gas[node].temperature_k:
            fabs(gas_enthalpy(gas[donor],model)-model.cv_j_per_kg_k*(gas[node].temperature_k-model.reference_temperature_k));
        budget.thermal_power+=fabs(flow)*coefficient;
        const double pb=c.to<0?c.outside_pressure:gas[c.to].pressure_pa;
        budget.pressure_dt=fmin(budget.pressure_dt,gas_pressure_dt(gas[c.from].pressure_pa,pb,flow,caps[j],
            model.pressure_relaxation_fraction,model.pressure_absolute_tolerance_pa,model.pressure_relative_tolerance));
    }
    return budget;
}
__global__ void gas_limit_kernel(int count,const GasState* gas,DeviceRows channel_rows,const DeviceChannel* channels,const GasChannelState* channel_state,
    DeviceRows patch_rows,const double* conductance,const double* flux,DeviceRows source_rows,const DeviceVolume* volumes,const Material* materials,const ThermalState* solid,
    GasModel model,double* step,double speed,double safety,const DeviceGas* config,const double* mass_cap,const double* fracture_power,
    bool mass_stage=false,double time=0.,double horizon=0.) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;double limit=1e300;
    if(i<count) {
        const auto& s=gas[i];double power=fracture_power?fracture_power[i]:0.,sumg=0;
        if(s.temperature_k>=config[i].ignition_temperature)
            power+=config[i].reaction_rate*fmin(s.chemical_j,s.oxidizer_kg*config[i].oxidizer_heat);
        if(!mass_stage)for(int k=patch_rows.offsets[i];k<patch_rows.offsets[i+1];++k) {const int j=patch_rows.ids[k];power+=fabs(flux[j]);sumg+=conductance[j];}
        if(!mass_stage)for(int k=source_rows.offsets[i];k<source_rows.offsets[i+1];++k) {const int j=source_rows.ids[k];
            const auto& m=materials[volumes[j].object];const auto& c=m.conversion;
            const double current=m.electrical.enabled?maximum_charging_current(m.electrical,m.current,time,time+horizon):0.;
            const double voltage=m.voltage.count?sample_drive(m.voltage,time):c.precursor_voltage_ref_v;
            const double ka=degradation_rate(c,solid[j].temperature_k)+overcharge_r1_rate(c,m.electrical,solid[j].temperature_k,current);
            const double phi=precursor_branch(c,solid[j].temperature_k,voltage);
            power+=(completion_rate(c,solid[j].temperature_k)*solid[j].activated_kg+
                phi*ka*solid[j].reactant_kg)*speed*maximum_gas_yield(c)*
                model.cv_j_per_kg_k*fabs(solid[j].temperature_k-s.temperature_k);
        }
        // CPU groups resolve pressure/mass stiffness internally. Keep the
        // thermal/source/reaction/work bounds on the outer CUDA transaction.
        const auto budget=(model.cpu_local_transport || model.cuda_kawasaki_transport)?GasRateBudget{}:
            gas_rate_budget(i,channel_rows,channels,channel_state,mass_cap,gas,model);
        limit=fmin(limit,budget.pressure_dt);
        // Use uncapped rates here: a bounded packet must not silently relax the
        // time-accuracy constraint. Transfer caps remain a final safety net.
        if(budget.outgoing_mass>0)limit=fmin(limit,model.max_fraction_per_step*s.mass_kg/budget.outgoing_mass);
        const double capacity=gas_local_capacity(config[i],s,model,solid,volumes,materials);
        if(power+budget.thermal_power>0)limit=fmin(limit,
            fmin(model.max_temperature_step_k,.05*s.temperature_k)*capacity/(power+budget.thermal_power));
        if(sumg>0)limit=fmin(limit,safety*capacity/sumg);
    }
    block_minimum(limit,step);
}
// Resolve competing claims BEFORE application: transport has the established
// priority; old-state chemistry may consume only the remaining fuel/oxidizer.
// Incoming fuel cannot enlarge the old-state kinetic proposal. This preserves
// the existing explicit discretization without double spending a shared stock.
__global__ void gas_reaction_kernel(int count,const GasState* old,const DeviceGas* config,
    DeviceRows rows,const DeviceChannel* channels,const GasTransfer* transfer,
    DeviceRows source_rows,const GasTransfer* sources,const double* step,GasTransfer* reaction,bool mass_stage=false) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& c=config[i];reaction[i]={};
    if(old[i].temperature_k<c.ignition_temperature || c.reaction_rate<=0)return;
    double chemical=old[i].chemical_j,oxidizer=old[i].oxidizer_kg;
    for(int k=rows.offsets[i];k<rows.offsets[i+1];++k) {
        const int j=rows.ids[k];const double sign=channels[j].from==i?-1.:1.;
        chemical+=sign*transfer[j].chemical;oxidizer+=sign*transfer[j].oxidizer;
    }
    if(!mass_stage)for(int k=source_rows.offsets[i];k<source_rows.offsets[i+1];++k)chemical+=sources[source_rows.ids[k]].chemical;
    const double available=fmin(old[i].chemical_j,old[i].oxidizer_kg*c.oxidizer_heat);
    const double q=fmax(0.,fmin(available*(-expm1(-c.reaction_rate**step)),fmin(chemical,oxidizer*c.oxidizer_heat)));
    reaction[i]={0.,q,-q,-q/c.oxidizer_heat};
}
__global__ void gas_advance_kernel(int count,const GasState* old,GasState* next,const DeviceGas* config,
    DeviceRows channel_rows,const DeviceChannel* channels,const GasTransfer* transfer,DeviceRows patch_rows,const double* flux,
    DeviceRows source_rows,const GasTransfer* sources,const GasTransfer* reaction,GasModel model,const double* step,bool mass_stage=false) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;auto s=old[i];const double dt=*step;
    for(int k=channel_rows.offsets[i];k<channel_rows.offsets[i+1];++k) {const int j=channel_rows.ids[k];
        const auto& c=channels[j];const double sign=c.from==i?-1.:1.;
        const auto& q=transfer[j];
        s.mass_kg+=sign*q.mass;s.energy_j+=sign*q.energy;s.chemical_j+=sign*q.chemical;s.oxidizer_kg+=sign*q.oxidizer;
    }
    // Ordinary contact: heat leaves this gas. Aperture contact: the arriving
    // packet above loses exactly the heat gathered by the receiving solid.
    if(!mass_stage)for(int k=patch_rows.offsets[i];k<patch_rows.offsets[i+1];++k)s.energy_j-=flux[patch_rows.ids[k]]*dt;
    if(!mass_stage)for(int k=source_rows.offsets[i];k<source_rows.offsets[i+1];++k) {const int j=source_rows.ids[k];
        s.mass_kg+=sources[j].mass;s.energy_j+=sources[j].energy;s.chemical_j+=sources[j].chemical;
    }
    const auto& q=reaction[i];
    s.chemical_j+=q.chemical;s.energy_j+=q.energy;s.released_energy_j+=q.energy;
    s.oxidizer_kg+=q.oxidizer;s.consumed_oxidizer_kg-=q.oxidizer;
    s.temperature_k=model.reference_temperature_k+s.energy_j/(s.mass_kg*model.cv_j_per_kg_k);
    s.pressure_pa=s.mass_kg*model.specific_r_j_per_kg_k*s.temperature_k/s.free_volume_m3;next[i]=s;
}

// Shell integrity only. Gas thermodynamics and fluxes stay in gas_kernels.cuh.
// Bulk and weak perimeter have independent irreversible damage at one local T.
struct DeviceShellPatch {
    int channel=-1;
    double seam_fraction=0., failure_hazard=1., area_m2=0.;
    std::uint64_t random_key=0;
    OpeningRectangle rectangle;
    double weak_fraction=0.,weak_factor=1.;
};
__device__ double shell_event_threshold(const ShellOpeningLaw& law,const DeviceShellPatch& p,ShellDamageState d) {
    return law.stochastic_area?area_event_exponential(p.random_key,d.rupture_events,0):p.failure_hazard;
}
__global__ void shell_damage_proposal_kernel(int count,const DeviceShellPatch* patches,
    const DeviceChannel* channels,const ThermalState* solid,const ShellDamageState* damage,
    const GasState* gas,ShellDamageState* rates,double multiplier,double* step) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;double limit=1e300;
    if(i<count) {
        rates[i]={};const auto& p=patches[i];
        if(p.channel>=0 && (channels[p.channel].opening.stochastic_pixels || channels[p.channel].opening.stochastic_area) && damage[i].bulk<1.) {
            const auto& c=channels[p.channel];const auto& law=c.opening;const auto d=damage[i];
            const double dp=gas[c.from].pressure_pa-(c.to<0?c.outside_pressure:gas[c.to].pressure_pa);
            const double critical=binary_shell_strength(c.threshold,d.thermal,p.seam_fraction,law.seam_strength_factor);
            double rate=multiplier*pixel_failure_rate(dp,critical,law.pixel_rate_per_s,law.pixel_pressure_exponent);
            if(law.stochastic_area)rate=law.fixed_area_fraction>0?
                fixed_area_event_rate(rate,law.fixed_area_fraction,d.bulk):area_event_rate(rate,p.area_m2,law.fragment_mean_area_m2,d.bulk);
            rates[i].failure_hazard=rate;
            // Stop at the earliest scheduled pixel event. The next transaction
            // transports through the new hole and re-evaluates every hazard.
            limit=pixel_event_dt(d.failure_hazard,shell_event_threshold(law,p,d),rate);
        }
        if(p.channel>=0 && !discrete_shell_events(channels[p.channel].opening) && !channels[p.channel].opening.deterministic_aperture) {
            const auto& c=channels[p.channel];const auto d=damage[i];const double t=solid[i].temperature_k;
            // This closure models outward opening, not inward shell buckling.
            const double dp=gas[c.from].pressure_pa-(c.to<0?c.outside_pressure:gas[c.to].pressure_pa);
            const double rb=multiplier*shell_pressure_damage_rate(c.opening,dp,shell_strength(c.threshold,t,shell_combined_damage(d.thermal,d.bulk),c.strength_reference,c.strength_scale),d.bulk);
            const double rs=p.seam_fraction>0?multiplier*shell_pressure_damage_rate(c.opening,dp,shell_strength(c.threshold*c.opening.seam_strength_factor,t,shell_combined_damage(d.thermal,d.seam),c.strength_reference,c.strength_scale),d.seam):0.;
            rates[i]={rb,rs};limit=fmin(shell_damage_dt(c.opening,rb,d.bulk),shell_damage_dt(c.opening,rs,d.seam));
        }
    }
    block_minimum(limit,step);
}
__global__ void shell_damage_apply_kernel(int count,const DeviceVolume* volumes,const Material* materials,
    const ThermalState* old,ThermalState* next,DeviceRows rows,const DeviceChannel* channels,
    const GasChannelState* channel_next,const DeviceShellPatch* patches,const ShellDamageState* damage,
    ShellDamageState* updated,const ShellDamageState* rates,const double* step,double speed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const int object=volumes[i].object;const double before=next[i].damage;
    if(patches) {
        updated[i]=damage[i];const auto& p=patches[i];
        if(p.channel>=0) {
            // Recover thermal loss of remaining integrity from the solid step.
            // Keep it separate: only pressure advances mechanical opening.
            const double remaining=1.-old[i].damage;
            const double fraction=remaining>0?fmin(1.,fmax(0.,next[i].damage-old[i].damage)/remaining):0.;
            auto d=damage[i];d.thermal+=(1.-d.thermal)*fraction;
            d.failure_hazard+=rates[i].failure_hazard**step;
            d.bulk=advance_shell_damage(d.bulk,rates[i].bulk,*step);
            d.seam=advance_shell_damage(d.seam,rates[i].seam,*step);updated[i]=d;
            next[i].damage=shell_material_damage(d,p.seam_fraction);
        }
    }
    for(int j=rows.offsets[object];j<rows.offsets[object+1];++j) {
        const int c=rows.ids[j];if(!channels[c].opening.enabled && channel_next[c].open) {
            next[i].damage=1.;break; // Frozen legacy law, opt-out compatibility.
        }
    }
    if(next[i].damage>=1.)next[i].mode=CellMode::Spent;
    // Damage changes constitutive T(U,D), never adds heat to U.
    if(next[i].damage!=before)close_thermal_state(materials[object],volumes[i],next[i],speed);
}
__global__ void shell_aperture_kernel(int count,const DeviceChannel* channels,const int* offsets,
    const DeviceShellPatch* patches,const ShellDamageState* damage,GasChannelState* state,double time,const double* step,bool leading=false) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count || !channels[i].opening.enabled)return;
    const auto& c=channels[i];double fraction=0.,bulk=0.,seam=0.,thermal=0.;
    const int begin=shell_begin(c,offsets),end=shell_end(c,offsets);
    // Uniform tangential grid: equal patch areas. Nonlinear law BEFORE averaging.
    for(int j=begin;j<end;++j) {
        const double f=patches[j].seam_fraction;const auto d=damage[j];
        fraction+=(1-f)*shell_open_fraction(c.opening,d.bulk)+f*shell_open_fraction(c.opening,d.seam);
        bulk=fmax(bulk,d.bulk);if(f>0)seam=fmax(seam,d.seam);
        thermal=fmax(thermal,d.thermal);
    }
    auto& s=state[i];s.opening_fraction=fmin(1.,fraction/(end-begin));
    s.max_bulk_damage=bulk;s.max_seam_damage=seam;s.max_thermal_damage=thermal;
    s.effective_area_m2=c.area*s.opening_fraction;
    if(!s.open && s.opening_fraction>0) {s.open=true;s.opened_s=leading?time:time+*step;}
}
__global__ void shell_validate_kernel(int count,const ShellDamageState* damage,int nc,const GasChannelState* channels,StepResult* status) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && (!isfinite(damage[i].bulk) || !isfinite(damage[i].seam) || !isfinite(damage[i].thermal) || !isfinite(damage[i].failure_hazard) || damage[i].failure_hazard<0 || damage[i].bulk<0 || damage[i].bulk>1 || damage[i].seam<0 || damage[i].seam>1 || damage[i].thermal<0 || damage[i].thermal>1))atomicOr(&status->invalid,8u);
    if(i<nc) {const auto& c=channels[i];
        if(!isfinite(c.opening_fraction) || c.opening_fraction<0 || c.opening_fraction>1 ||
            !isfinite(c.effective_area_m2) || c.effective_area_m2<0 || !isfinite(c.mass_flow_kg_s))atomicOr(&status->invalid,16u);
    }
}

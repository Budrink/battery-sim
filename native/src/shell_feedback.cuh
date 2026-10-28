// Numerical coupling of aperture growth and pressure unloading. No new
// constitutive parameter and no change to the orifice law or thermal graph.
__device__ double prospective_shell_area(const DeviceChannel& c,const int* offsets,
    const DeviceShellPatch* patches,const ShellDamageState* damage,const ShellDamageState* rates,double dt) {
    const int begin=shell_begin(c,offsets),end=shell_end(c,offsets);double fraction=0.;
    for(int k=begin;k<end;++k) {
        const double f=patches[k].seam_fraction;const auto d=damage[k],r=rates[k];
        fraction+=(1-f)*shell_open_fraction(c.opening,advance_shell_damage(d.bulk,r.bulk,dt))+
                  f*shell_open_fraction(c.opening,advance_shell_damage(d.seam,r.seam,dt));
    }
    return c.area*fraction/(end-begin);
}
__global__ void shell_feedback_limit_kernel(int count,const DeviceChannel* channels,
    const int* offsets,const DeviceShellPatch* patches,const ShellDamageState* damage,
    const ShellDamageState* rates,const GasState* gas,const DeviceGas* config,DeviceRows rows,
    GasModel model,const ThermalState* solid,const DeviceVolume* volumes,const Material* materials,
    double* step) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;double limit=1e300;
    if(i<count && channels[i].opening.enabled && !discrete_shell_events(channels[i].opening) && !channels[i].opening.deterministic_aperture) {
        const auto& c=channels[i];const auto& law=c.opening;
        const double pb=c.to<0?c.outside_pressure:gas[c.to].pressure_pa;
        if(gas[c.from].pressure_pa>pb) {
            double area=0.,growth=0.,onset=1e300;
            const int begin=shell_begin(c,offsets),end=shell_end(c,offsets);
            for(int k=begin;k<end;++k) {
                const double f=patches[k].seam_fraction;const auto d=damage[k],r=rates[k];
                area+=(1-f)*shell_open_fraction(law,d.bulk)+f*shell_open_fraction(law,d.seam);
                growth+=(1-f)*r.bulk*(1-d.bulk)+f*r.seam*(1-d.seam);
                if(r.bulk>0 && f<1)onset=fmin(onset,shell_aperture_onset_time(law,d.bulk,r.bulk));
                if(r.seam>0 && f>0)onset=fmin(onset,shell_aperture_onset_time(law,d.seam,r.seam));
            }
            const double weight=c.area/(end-begin);
            // Global Lipschitz bound of f(D), with D'(t)<=rate*(1-Dold).
            const double area_rate=weight*growth*law.exponent/(1-law.damage_onset);
            const double unit_flow=gas_orifice_flow(gas[c.from].pressure_pa,pb,gas[c.from].temperature_k,1.,c.cd,model);
            // Independent mass-geometry budget. Do not multiply it by the
            // thermal CFL factor: refinement should halve, not quarter it.
            const double budget=model.max_fraction_per_step*
                gas_pressure_mass_cap(c,gas,config,rows,model,solid,volumes,materials);
            limit=shell_feedback_dt(area*weight,area_rate,unit_flow,budget,area>0?0.:onset);
            const double slow_limit=shell_slow_aperture_dt(area*weight,area_rate,model.max_fraction_per_step,c.area);
            // Either resolve pressure unloading or keep the aperture effectively
            // resolved to .5% of the WHOLE element area at default mass
            // accuracy .02, not .5% of an arbitrarily tiny hole. The same mass and
            // energy transfer still applies; this does not alter permeability.
            if(slow_limit>limit) {
                limit=slow_limit;
            }
            // If growth adds at most 1% of the old area, the conservative
            // closed-form bound is already tight. Avoid a costly root solve.
            else if(limit<1e299 && area_rate*limit>.01*area*weight) {
                // The global Lipschitz estimate is safe, but too restrictive
                // near f(D)=0 for quadratic apertures. Solve the same monotone
                // budget using the actual frozen-rate opening law. No fit knob.
                double lo=0.,hi=limit;
                bool bracket=false;
                for(int n=0;n<32;++n) {
                    const double transferred=unit_flow*hi*prospective_shell_area(c,offsets,patches,damage,rates,hi);
                    if(transferred>budget) {bracket=true;break;}
                    lo=hi;hi*=2.;
                }
                if(bracket)for(int n=0;n<40;++n) {
                    const double mid=(lo+hi)/2;
                    const double transferred=unit_flow*mid*prospective_shell_area(c,offsets,patches,damage,rates,mid);
                    if(transferred<=budget)lo=mid;else hi=mid;
                }
                limit=lo>0?lo:limit;
            }
        }
    }
    block_minimum(limit,step);
}

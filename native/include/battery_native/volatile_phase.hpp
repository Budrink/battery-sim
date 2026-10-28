#pragma once
#include "battery_native/conversion.hpp"
#ifdef __CUDACC__
#define VP_HD __host__ __device__
#else
#define VP_HD
#endif
namespace battery {
// Constant-latent Clausius-Clapeyron pseudo-fluid closure, not identified
// electrolyte chemistry. Reference boiling point is NOT a trigger.
VP_HD inline double saturation_pressure(const VolatileLaw& p,double t) {
    return p.saturation_reference_pa*exp(fmax(-80.,fmin(80.,p.latent_j_per_kg/p.vapor_specific_r*
        (1/p.boiling_temperature_k-1/t))));
}
struct PhaseTrial {double residual,derivative,temperature;};
VP_HD inline PhaseTrial phase_trial(const VolatileLaw& p,const GasState& g,double sensible,
    double capacity,double cv,double tref,double kd,double dm) {
    const double a=cv-p.liquid_cp,cap=capacity+a*dm;
    const double t=tref+(sensible-p.latent_j_per_kg*dm)/cap;
    const double v=g.free_volume_m3+dm/p.density_kg_m3;
    const double sat=saturation_pressure(p,t)*v/(p.vapor_specific_r*t);
    const double td=(-p.latent_j_per_kg*capacity-a*sensible)/(cap*cap);
    const double sd=sat*((p.latent_j_per_kg/(p.vapor_specific_r*t*t)-1/t)*td+1/(p.density_kg_m3*v));
    return {dm-kd*(sat-g.vapor_kg-dm),1+kd-kd*sd,t};
}
// One local reservoir + its gas pore. The transferred latent energy remains
// mobile and is returned by condensation; it is not destruction heat or fuel.
VP_HD inline double advance_volatile(const ConversionLaw& c,ThermalState& s,GasState& g,
    double cref,double fraction,const GasModel& model,double duration) {
    const auto& p=c.volatile_phase;
    if(!(p.initial_mass_kg>0. && duration>0.))return 0.;
    const double tref=model.reference_temperature_k,cv=model.cv_j_per_kg_k;
    const double cs=conversion_capacity(c,s,cref,fraction),capacity=cs+cv*g.mass_kg;
    const double sensible=s.stored_energy_j+g.energy_j-cref*tref;
    // This model has ONE declared volatile pseudo-species. Condensate may
    // return only up to the original local liquid capacity (no unbounded pool).
    double lo=-fmin(g.vapor_kg,fmax(0.,p.initial_mass_kg*fraction-s.liquid_kg));
    double hi=s.liquid_kg;
    // Prevent an energy-starved evaporation trial from crossing absolute zero.
    const double denominator=p.latent_j_per_kg+(cv-p.liquid_cp)*(1.-tref);
    if(denominator>0)hi=fmin(hi,fmax(0.,(sensible-capacity*(1.-tref))/denominator));
    // Fully implicit relaxation: dm = k dt (m_sat(T(dm),V(dm))-m_vapor(dm)).
    // Monotone in this subcritical pseudo-fluid regime; bounded roots require
    // no extra GLOBAL thermal microsteps. Refinement tests bound splitting error.
    const double kd=p.relaxation_per_s*duration;
    double dm=0.;
    if(phase_trial(p,g,sensible,capacity,cv,tref,kd,hi).residual<=0.)dm=hi;
    else if(phase_trial(p,g,sensible,capacity,cv,tref,kd,lo).residual>=0.)dm=lo;
    else for(int n=0;n<32;++n) {
        const auto trial=phase_trial(p,g,sensible,capacity,cv,tref,kd,dm);
        if(fabs(trial.residual)<=1e-14*fmax(p.initial_mass_kg*fraction,g.vapor_kg))break;
        if(trial.residual>0)hi=dm;else lo=dm;
        const double next=dm-trial.residual/trial.derivative;
        dm=(next>lo && next<hi)?next:.5*(lo+hi);
    }
    const double cap=capacity+dm*(cv-p.liquid_cp);
    const double t=tref+(sensible-p.latent_j_per_kg*dm)/cap;
    s.liquid_kg-=dm;s.evaporated_kg+=fmax(0.,dm);s.condensed_kg+=fmax(0.,-dm);
    g.mass_kg+=dm;g.vapor_kg+=dm;g.latent_j+=dm*p.latent_j_per_kg;
    g.chemical_j+=dm*p.chemical_j_per_kg;
    g.free_volume_m3+=dm/p.density_kg_m3;
    s.temperature_k=g.temperature_k=t;
    s.stored_energy_j=cref*tref+(cs-dm*p.liquid_cp)*(t-tref);
    g.energy_j=g.mass_kg*cv*(t-tref);s.mobile_capacity_j_per_k=g.mass_kg*cv;
    g.pressure_pa=g.mass_kg*model.specific_r_j_per_kg_k*t/g.free_volume_m3;
    return dm;
}
}
#undef VP_HD

#pragma once
#include "battery_native/cuda_solver.hpp"
#include <cmath>
#ifdef __CUDACC__
#define BN_HD __host__ __device__
#else
#define BN_HD
#endif
namespace battery {
BN_HD inline double maximum_gas_yield(const ConversionLaw& p) {
    if(p.gas_yield==0.)return 0.;
    if(p.gas_yield_temperature_per_k==0. && p.gas_yield_density_exponent==0.)return p.gas_yield;
    return p.gas_yield_max;
}
BN_HD inline double effective_gas_yield(const ConversionLaw& p,const ThermalState& s,double fraction) {
    // Ratio of bulk condensed densities in the same fixed control volume.
    const double ratio=fmax(1e-12,(s.reactant_kg+s.activated_kg+s.residue_kg)/(p.mass_kg*fraction));
    const double exponent=p.gas_yield_temperature_per_k*(s.temperature_k-p.gas_yield_reference_k)+
        p.gas_yield_density_exponent*log(ratio);
    return fmin(p.gas_yield_max,p.gas_yield*exp(fmin(80.,fmax(-80.,exponent))));
}
BN_HD inline double arrhenius_rate(double rate,double activation,double reference,double temperature) {
    return rate*exp(fmin(80.,fmax(-80.,activation/8.31446261815324*(1/reference-1/temperature))));
}
BN_HD inline double conversion_capacity(const ConversionLaw& p,const ThermalState& s,double cref,double fraction) {
    return cref-p.mass_kg*fraction*p.reactant_cp+s.reactant_kg*p.reactant_cp+
        s.activated_kg*p.activated_cp+s.residue_kg*p.residue_cp;
}
BN_HD inline double conversion_min_capacity(const ConversionLaw& p,double cref,double fraction) {
    return cref-p.mass_kg*fraction*p.reactant_cp+
        p.mass_kg*fraction*fmin(p.reactant_cp,fmin(p.activated_cp,(1-maximum_gas_yield(p))*p.residue_cp));
}
BN_HD inline double degradation_rate(const ConversionLaw& p,double t) {
    return arrhenius_rate(p.degradation_rate_per_s,p.degradation_activation_j_per_mol,p.degradation_reference_k,t);
}
BN_HD inline double completion_rate(const ConversionLaw& p,double t) {
    return arrhenius_rate(p.reaction_rate_per_s,p.reaction_activation_j_per_mol,p.reaction_reference_k,t);
}
BN_HD inline double conversion_chemical(const ConversionLaw& p,const ThermalState& s) {
    return p.net_heat_j_per_kg*(s.reactant_kg+s.activated_kg);
}
BN_HD inline void refresh_conversion(const ConversionLaw& p,ThermalState& s,double fraction,double speed) {
    const double m0=p.mass_kg*fraction;
    s.damage=fmin(1.,fmax(0.,1-s.reactant_kg/m0));
    s.chemical_energy_remaining_j=conversion_chemical(p,s);
    const double power=completion_rate(p,s.temperature_k)*s.activated_kg*
        p.net_heat_j_per_kg*(1-p.gas_chemical_fraction)*speed;
    s.chemical_power_w=power;
    const double progress=fmin(1.,fmax(0.,1-(s.reactant_kg+s.activated_kg)/m0));
    for(int k=0;k<3;++k)s.conductivity_factor[k]=exp(progress*log(p.residue_conductivity[k]));
    if(s.runaway_started_s>=0) {
        if(s.reactant_kg+s.activated_kg<=m0*1e-12)s.mode=CellMode::Spent;
        else {
            const double rate=completion_rate(p,s.temperature_k)*s.activated_kg*speed;
            s.mode=rate>m0*p.runaway_reporting_rate_per_s?CellMode::Runaway:CellMode::Quenched;
        }
    }
}
// Exact coupled two-stage chain at frozen T. Stable also for kb >> ka: the
// short-lived intermediate need not impose its lifetime on the transport step.
BN_HD inline double advance_conversion(const ConversionLaw& p,const GasModel& g,ThermalState& s,
    double fraction,double dt,double speed,double time,double additional_r1_rate=0.) {
    s.generated_gas_kg=s.generated_gas_energy_j=s.generated_gas_chemical_j=0.;
    const double ka=(degradation_rate(p,s.temperature_k)+additional_r1_rate)*speed;
    const double kb=completion_rate(p,s.temperature_k)*speed;
    const double da=s.reactant_kg*(-expm1(-ka*dt));
    const double difference=fabs(kb-ka);
    const double feed=s.reactant_kg*ka*exp(-fmin(ka,kb)*dt)*
        (difference>0?-expm1(-difference*dt)/difference:dt);
    const double db=s.activated_kg*(-expm1(-kb*dt))+fmax(0.,da-fmin(da,feed));
    const double yield=effective_gas_yield(p,s,fraction);
    const bool reporting_onset=kb*s.activated_kg>p.mass_kg*fraction*p.runaway_reporting_rate_per_s;
    if(reporting_onset && s.runaway_started_s<0)s.runaway_started_s=time;
    s.reactant_kg-=da;s.activated_kg+=da-db;s.residue_kg+=(1-yield)*db;
    const double fuel=p.net_heat_j_per_kg*p.gas_chemical_fraction*db;
    const double generated=p.net_heat_j_per_kg*db-fuel;
    s.gross_reaction_energy_j+=generated;s.released_energy_j+=generated;
    s.generated_gas_kg=yield*db;
    s.generated_gas_energy_j=s.generated_gas_kg*g.cv_j_per_kg_k*(s.temperature_k-g.reference_temperature_k);
    s.generated_gas_chemical_j=fuel;
    return generated-s.generated_gas_energy_j;
}
BN_HD inline double gas_orifice_flow(double pup,double pdown,double temperature,double area,double cd,const GasModel& g) {
    if(pup<=pdown || pup<=0 || area<=0)return 0.;
    const double gamma=1+g.specific_r_j_per_kg_k/g.cv_j_per_kg_k;
    const double ratio=fmax(0.,pdown/pup),critical=pow(2/(gamma+1),gamma/(gamma-1));
    const double f=ratio<=critical?sqrt(gamma)*pow(2/(gamma+1),(gamma+1)/(2*(gamma-1))):
        sqrt(fmax(0.,2*gamma/(gamma-1)*(pow(ratio,2/gamma)-pow(ratio,(gamma+1)/gamma))));
    return cd*area*pup/sqrt(g.specific_r_j_per_kg_k*temperature)*f;
}
BN_HD inline double gas_enthalpy(const GasState& s,const GasModel& g) {
    return g.cv_j_per_kg_k*(s.temperature_k-g.reference_temperature_k)+g.specific_r_j_per_kg_k*s.temperature_k;
}
} // namespace battery
#undef BN_HD

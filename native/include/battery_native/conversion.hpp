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
        s.activated_kg*p.activated_cp+s.residue_kg*p.residue_cp-
        (p.volatile_phase.initial_mass_kg*fraction-s.liquid_kg)*p.volatile_phase.liquid_cp;
}
BN_HD inline double conversion_min_capacity(const ConversionLaw& p,double cref,double fraction) {
    return cref-p.mass_kg*fraction*p.reactant_cp-p.volatile_phase.initial_mass_kg*fraction*p.volatile_phase.liquid_cp+
        p.mass_kg*fraction*fmin(p.reactant_cp,fmin(p.activated_cp,(1-maximum_gas_yield(p))*p.residue_cp));
}
BN_HD inline double degradation_rate(const ConversionLaw& p,double t) {
    return arrhenius_rate(p.degradation_rate_per_s,p.degradation_activation_j_per_mol,p.degradation_reference_k,t);
}
BN_HD inline double completion_rate(const ConversionLaw& p,double t) {
    return arrhenius_rate(p.reaction_rate_per_s,p.reaction_activation_j_per_mol,p.reaction_reference_k,t);
}
BN_HD inline double conversion_chemical(const ConversionLaw& p,const ThermalState& s) {
    return p.net_heat_j_per_kg*(s.reactant_kg+s.activated_kg)+s.precursor_charge_energy_j+
        s.liquid_kg*p.volatile_phase.chemical_j_per_kg;
}
BN_HD inline double healthy_mass(const ThermalState& s) {
    return s.reactant_kg+s.healthy_charged_kg;
}
BN_HD inline double charge_capacity_fraction(const ConversionLaw& p,const ThermalState& s,double fraction) {
    return fmin(1.,fmax(0.,healthy_mass(s)/(p.mass_kg*fraction)));
}
// Frozen-on-substep R1 byproduct fraction. Charging never uses this branch.
BN_HD inline double precursor_branch(const ConversionLaw& p,double temperature) {
    if(p.precursor_yield>=1. && p.precursor_yield_temperature_per_k==0.)
        return 1.;
    const double exponent=p.precursor_yield_temperature_per_k*(temperature-p.precursor_temperature_ref_k);
    return fmin(1.,fmax(0.,p.precursor_yield*exp(fmin(80.,fmax(-80.,exponent)))));
}
BN_HD inline void refresh_conversion(const ConversionLaw& p,ThermalState& s,double fraction,double speed) {
    const double m0=p.mass_kg*fraction;
    s.damage=fmin(1.,fmax(0.,s.degraded_mass_kg/m0));
    s.chemical_energy_remaining_j=conversion_chemical(p,s);
    const double power=completion_rate(p,s.temperature_k)*
        (s.activated_kg*p.net_heat_j_per_kg+s.precursor_charge_energy_j)*(1-p.gas_chemical_fraction)*speed;
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
// Stable convolution of two decays, including coincident/zero rates.
BN_HD inline double decay_feed(double a,double b,double dt) {
    const double difference=fabs(a-b);
    return exp(-fmin(a,b)*dt)*(difference>0?-expm1(-difference*dt)/difference:dt);
}
// All charge-induced B costs electrical work, even if it reacts in this step.
// q_charge=q_intrinsic is the minimal explicit energy-scale PRIOR, not measured
// chemistry. The pre-existing q*A is never counted as that electrical payment.
BN_HD inline double charging_rate_within_budget(const ConversionLaw& p,double unconstrained,double reactant_kg,
    double dt,double speed,double room_j) {
    if(!(unconstrained>0.))return 0.;
    const double price=p.net_heat_j_per_kg;
    if(!(price>0.) || !(reactant_kg>0.))return unconstrained;
    if(!(room_j>0.) || !(dt>0.) || !(speed>0.))return 0.;
    return fmin(unconstrained,room_j/(price*reactant_kg*dt*speed));
}
struct MaterialTransfers {
    double degraded_kg, charged_kg, reacted_kg, charge_work_j, sterilized_j, released_j;
};
// Exact frozen-T triangular system, not split instantaneous reactions:
// A --charge--> Bc; A,Bc --R1--> phi*Bd+(1-phi)*S;
// Bc,Bd --R2--> condensed products + gas. Bc is healthy charged material.
// All rates have units 1/s. Equal rates require no singular divisions.
BN_HD inline MaterialTransfers advance_material_inventory(const ConversionLaw& p,ThermalState& s,
    double dt,double speed,double charge_rate,double phi) {
    const double r=degradation_rate(p,s.temperature_k)*speed,c=charge_rate*speed;
    const double b=completion_rate(p,s.temperature_k)*speed,a=r+c,h=r+b;
    const double before_a=s.reactant_kg,before_b=s.activated_kg,before_c=s.healthy_charged_kg;
    const double decay=exp(-b*dt),da=before_a*(-expm1(-a*dt));
    const double r1_a=a>0?da*r/a:0.,charged=a>0?da*c/a:0.;
    const double new_c=before_c*exp(-h*dt)+c*before_a*decay_feed(a,h,dt);
    const double c_loss=fmax(0.,before_c+charged-new_c);
    const double r1_c=h>0?c_loss*r/h:0.;
    const double degraded=r1_a+r1_c,inert=(1-phi)*degraded;
    const double new_bd=fmax(0.,(before_b-before_c)*decay+
        phi*fmax(0.,a*before_a*decay_feed(a,b,dt)+before_c*decay-new_c));
    const double new_b=new_c+new_bd;
    const double reacted=b>0?fmax(0.,before_b+charged+phi*r1_a-(1-phi)*r1_c-new_b):0.;
    const double price=p.net_heat_j_per_kg,charge_work=price*charged;
    // Every healthy Bc owns exactly q of paid charge energy per kg. R1 may
    // retain it in Bd or make it inaccessible in S, but never release it as heat.
    const double sterile_extra=price*(1-phi)*r1_c;
    const double new_extra=b>0?fmax(0.,
        (s.precursor_charge_energy_j-price*(1-phi)*before_c)*decay+
        price*phi*c*before_a*decay_feed(a,b,dt)+price*(1-phi)*new_c):
        fmax(0.,s.precursor_charge_energy_j+charge_work-sterile_extra);
    const double extra_released=b>0?fmax(0.,s.precursor_charge_energy_j+charge_work-sterile_extra-new_extra):0.;
    const double sterile=price*inert+sterile_extra,released=price*reacted+extra_released;
    s.reactant_kg=before_a-da;s.healthy_charged_kg=new_c;s.activated_kg=new_b;
    s.degraded_mass_kg+=degraded;s.precursor_charge_energy_j=new_extra;
    s.precursor_from_charge_kg+=charged;s.precursor_from_degradation_kg+=phi*degraded;
    s.electrical_chemical_input_j+=charge_work;s.sterilized_chemical_j+=sterile;
    s.residue_kg+=inert;
    return {degraded,charged,reacted,charge_work,sterile,released};
}
// R2 energy/gas closure. Transfer geometries are deliberately outside this law.
BN_HD inline double advance_conversion(const ConversionLaw& p,const GasModel& g,ThermalState& s,
    double fraction,double dt,double speed,double time,double charge_rate=0.) {
    s.generated_gas_kg=s.generated_gas_energy_j=s.generated_gas_chemical_j=0.;
    const double kb=completion_rate(p,s.temperature_k)*speed;
    const double phi=precursor_branch(p,s.temperature_k);
    const double yield=effective_gas_yield(p,s,fraction);
    const bool reporting_onset=kb*s.activated_kg>p.mass_kg*fraction*p.runaway_reporting_rate_per_s;
    if(reporting_onset && s.runaway_started_s<0)s.runaway_started_s=time;
    const auto flow=advance_material_inventory(p,s,dt,speed,charge_rate,phi);
    s.residue_kg+=(1-yield)*flow.reacted_kg;
    const double released=flow.released_j;
    const double fuel=p.gas_chemical_fraction*released;
    const double generated=released-fuel;
    s.gross_reaction_energy_j+=generated;s.released_energy_j+=generated;
    s.generated_gas_kg=yield*flow.reacted_kg;
    s.r2_generated_gas_kg+=s.generated_gas_kg;s.r2_reacted_kg+=flow.reacted_kg;
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

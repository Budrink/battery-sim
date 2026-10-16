#pragma once
#include "battery_native/cuda_solver.hpp"
#include <cmath>
#ifdef __CUDACC__
#define BATTERY_REACTION_HD __host__ __device__
#else
#define BATTERY_REACTION_HD
#endif
namespace battery {
BATTERY_REACTION_HD inline double reaction_rate(const ReactionKinetics& p,double t,double d) {
    // Numerically bounded Arrhenius ratio. Clipping is only an overflow guard.
    const double exponent=p.activation_energy_j_per_mol/8.31446261815324*(1/p.reference_temperature_k-1/t)
        +p.damage_sensitivity*(d-p.activation_damage);
    return p.rate_at_reference_per_s*exp(fmin(80.,fmax(-80.,exponent)));
}
BATTERY_REACTION_HD inline double reaction_conversion(const ThermalState& s,double budget) {
    return budget>0?fmin(1.,fmax(0.,1-s.chemical_energy_remaining_j/budget)):0.;
}
BATTERY_REACTION_HD inline double smooth_conversion(double q,double a,double b) {
    const double x=fmin(1.,fmax(0.,(q-a)/(b-a)));
    return x*x*(3-2*x);
}
BATTERY_REACTION_HD inline double outlet_opening(const EnergyOutletLaw& p,double q) {
    if(!p.enabled)return 0.;
    const double opened=smooth_conversion(q,p.opening_start_conversion,p.opening_end_conversion);
    return p.closure_start_conversion>=1?opened:
        opened*(1-smooth_conversion(q,p.closure_start_conversion,p.closure_end_conversion));
}
BATTERY_REACTION_HD inline bool reaction_live(const ThermalState& s) {
    return s.mode==CellMode::Runaway || s.mode==CellMode::Quenched;
}
BATTERY_REACTION_HD inline void refresh_reaction(const ReactionKinetics& p,const ReactionStructure& structure,
    ThermalState& s,double budget,double speed) {
    const double k=reaction_rate(p,s.temperature_k,s.damage)*speed;
    s.chemical_power_w=reaction_live(s)?k*s.chemical_energy_remaining_j:0.;
    if(reaction_live(s))s.mode=k<p.quench_rate_per_s?CellMode::Quenched:CellMode::Runaway;
    if(structure.enabled && s.runaway_started_s>=0) {
        const double q=reaction_conversion(s,budget);
        const double activity=budget>0?fmin(1.,s.chemical_power_w/(budget*structure.activity_rate_per_s)):0.;
        for(int axis=0;axis<3;++axis)
            s.conductivity_factor[axis]=exp((1-q)*activity*log(structure.active_factor[axis])+q*log(structure.residue_factor[axis]));
    }
}
// Exact finite-reserve integral with rates frozen at the OLD thermal state.
// A CUDA step limiter resolves the subsequent temperature/damage feedback.
BATTERY_REACTION_HD inline double advance_reaction(const ReactionKinetics& p,const EnergyOutletLaw& outlet,
    ThermalState& s,double budget,double dt,double speed) {
    if(!reaction_live(s) || budget<=0)return 0.;
    const double k=reaction_rate(p,s.temperature_k,s.damage)*speed;
    const double opened=outlet_opening(outlet,reaction_conversion(s,budget));
    const double ratio=opened*outlet.ejected_energy_per_released_energy;
    const double consumed=s.chemical_energy_remaining_j*(-expm1(-k*(1+ratio)*dt));
    const double generated=consumed/(1+ratio),ejected=consumed-generated;
    const double vented=generated*opened*outlet.heat_fraction;
    s.chemical_energy_remaining_j=fmax(0.,s.chemical_energy_remaining_j-consumed);
    s.released_energy_j+=generated;s.ejected_chemical_energy_j+=ejected;s.vent_heat_out_j+=vented;
    s.damage=fmin(1.,s.damage+p.damage_per_conversion*generated/budget);
    // No truncation/release of the residual: Spent is an exhaustion tolerance;
    // the tiny remaining reserve stays visible in the chemical ledger.
    if(s.chemical_energy_remaining_j<=budget*1e-12) {s.mode=CellMode::Spent;s.damage=1.;}
    return generated-vented;
}
// Source-owned directed structural metric; never average S_ij and S_ji.
BATTERY_REACTION_HD inline double reaction_bulk_factor(const ThermalState& a,const ThermalState& b,int axis,int direction=0) {
    return (direction>0 || (direction==0 && a.temperature_k>=b.temperature_k))?
        a.conductivity_factor[axis]:b.conductivity_factor[axis];
}
inline double maximum_reaction_factor(const CellConfig& c,int axis) {
    return c.reaction_structure.enabled?fmax(1.,fmax(c.reaction_structure.active_factor[axis],c.reaction_structure.residue_factor[axis])):1.;
}
} // namespace battery
#undef BATTERY_REACTION_HD

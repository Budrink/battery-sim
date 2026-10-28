#pragma once
#include "battery_native/model.hpp"
#include "battery_native/pixel_rupture.hpp"
#include "battery_native/area_rupture.hpp"
#include <cmath>
#ifdef __CUDACC__
#define BATTERY_SHELL_HD __host__ __device__
#else
#define BATTERY_SHELL_HD
#endif
namespace battery {
// Independent histories: thermal weakening never creates an aperture by itself.
struct ShellDamageState {
    double bulk = 0., seam = 0., thermal = 0., failure_hazard = 0.;
    std::uint64_t rupture_events = 0;
};
BATTERY_SHELL_HD inline double shell_combined_damage(double thermal,double mechanical) {
    return thermal+(1.-thermal)*mechanical;
}
BATTERY_SHELL_HD inline double shell_material_damage(ShellDamageState d,double seam_fraction) {
    return shell_combined_damage(d.thermal,(1.-seam_fraction)*d.bulk+seam_fraction*d.seam);
}
inline ShellDamageState initial_shell_state(const Scenario& scenario,int object,int local_volume=-1) {
    ShellDamageState d;d.thermal=scenario.cells[object].initial_damage;
    for(const auto& c:scenario.gas_channels)if(c.shell_object==object && c.opening.enabled &&
        (c.shell_thermal_volume<0 || c.shell_thermal_volume==local_volume))
        d.bulk=d.seam=c.opening.initial_mechanical_damage;
    return d;
}
BATTERY_SHELL_HD inline double shell_remaining_barrier(double damage) {
    // A degraded but still present barrier retains a residual resistance.
    return fmax(.01,1.-damage);
}
// Fraction of a rectangular patch occupied by the panel's perimeter band.
inline double shell_seam_fraction(double u,double v,double du,double dv,double width,double height,double band) {
    if(band<=0)return 0.;
    const double inner_u=fmax(0.,fmin(u+du/2,width/2-band)-fmax(u-du/2,-width/2+band));
    const double inner_v=fmax(0.,fmin(v+dv/2,height/2-band)-fmax(v-dv/2,-height/2+band));
    return fmin(1.,fmax(0.,1-inner_u*inner_v/(du*dv)));
}
// Constitutive closure only: no solver state, allocation, flow or CUDA launch.
BATTERY_SHELL_HD inline double shell_open_fraction(const ShellOpeningLaw& law,double damage) {
    if(law.stochastic_area || law.deterministic_aperture)return fmin(1.,fmax(0.,damage));
    if(law.binary_pixels)return damage>=1.?1.:0.;
    const double x=fmin(1.,fmax(0.,(damage-law.damage_onset)/(1.-law.damage_onset)));
    return pow(x,law.exponent);
}
// Binary pixels use damage alone to weaken the pressure threshold. The seam
// overlap modifies the pixel's effective strength, never its open fraction.
BATTERY_SHELL_HD inline double binary_shell_strength(double reference,double damage,double seam_fraction,double seam_factor) {
    return reference*shell_remaining_barrier(damage)*(1.-seam_fraction*(1.-seam_factor));
}
BATTERY_SHELL_HD inline double shell_strength(double reference,double t,double damage,double tref,double scale) {
    return reference*shell_remaining_barrier(damage)*exp(fmax(-80.,-fmax(0.,t-tref)/scale));
}
BATTERY_SHELL_HD inline double shell_pressure_damage_rate(const ShellOpeningLaw& law,double dp,double strength,double damage) {
    if(damage>=1. || law.pressure_damage_rate_per_s==0.)return 0.;
    return law.pressure_damage_rate_per_s*fmax(0.,dp/strength-1.);
}
BATTERY_SHELL_HD inline double advance_shell_damage(double damage,double rate,double dt) {
    const double next=fmin(1.,damage+(1.-damage)*(-expm1(-rate*dt)));
    // Exact terminal state for legacy connectivity thresholds at D=1. Without
    // this numerical closure, an O(1e-12) residue can retain intact contacts.
    return next>=1.-1e-10?1.:next;
}
BATTERY_SHELL_HD inline double shell_damage_dt(const ShellOpeningLaw& law,double rate,double damage=0.) {
    // Bound absolute increment of D. Once less than the allowed increment is
    // left, even complete destruction cannot exceed the budget. A fractional
    // remainder bound needlessly throttles saturated patches at roundoff.
    const double remaining=1.-damage;
    return rate>0 && remaining>law.max_damage_step?
        -log1p(-law.max_damage_step/remaining)/rate:1e300;
}
BATTERY_SHELL_HD inline double shell_aperture_onset_time(const ShellOpeningLaw& law,double damage,double rate) {
    if(damage>=law.damage_onset)return 0.;
    return rate>0?log((1.-damage)/(1.-law.damage_onset))/rate:1e300;
}
// A(t) <= Aold + area_rate*t. Resolve the pressure unload through the
// prospective aperture, not just the (possibly closed) old aperture.
BATTERY_SHELL_HD inline double shell_feedback_dt(double area,double area_rate,double unit_flow,
    double mass_budget,double onset_time) {
    if(area_rate<=0 || unit_flow<=0 || mass_budget<=0)return 1e300;
    const double linear=unit_flow*area,quadratic=unit_flow*area_rate;
    const double root=2*mass_budget/(linear+sqrt(linear*linear+4*quadratic*mass_budget));
    // Before the first opening no flow exists. Landing exactly on its onset
    // avoids throttling a closed barrier with a hypothetical full-face orifice.
    return fmax(onset_time,root);
}
// Alternative accuracy budget when pressure equilibration is much faster than
// aperture evolution. Budget growth against the WHOLE shell element, not the
// already open pinhole. Default mass accuracy .02 allows .5% of the element
// area per interval; refinement halves this absolute geometric tolerance.
// Preserve the first-opening onset/pressure-work bound when no hole exists.
// This changes numerical resolution only, never the physical aperture law.
BATTERY_SHELL_HD inline double shell_slow_aperture_dt(double area,double area_rate,
    double mass_fraction,double element_area) {
    return area_rate>0 ? (area>0 ? .25*mass_fraction*element_area/area_rate : 0.) : 1e300;
}
} // namespace battery
#undef BATTERY_SHELL_HD

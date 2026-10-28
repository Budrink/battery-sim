#pragma once
#include "battery_native/model.hpp"
#include "battery_native/constitutive.hpp"
#include "battery_native/radiation.hpp"
#include <cmath>

#ifdef __CUDACC__
#define BATTERY_GEOMETRY_HD __host__ __device__
#else
#define BATTERY_GEOMETRY_HD
#endif

namespace battery {
// Shared reference normalization for bulk, boundaries and gas surfaces.
// S=0 gives Gref. An unspecified destruction law does not invent a conductivity
// collapse: retain the same reference range until a material law supplies one.
BATTERY_GEOMETRY_HD inline ConnectivityLaw reference_directed_connectivity(double reference_g) {
    ConnectivityLaw law;law.kind=TransitionKind::DirectedGeometry;
    law.background_conductance_w_per_k=.01*reference_g;
    law.active_conductance_w_per_k=1.98*reference_g;
    law.destroyed_background_conductance_w_per_k=law.background_conductance_w_per_k;
    law.destroyed_active_conductance_w_per_k=law.active_conductance_w_per_k;
    law.endpoint_damage_weight=1.;return law;
}
BATTERY_GEOMETRY_HD inline double geometry_component(Vec3 v,int axis) {
    return axis==0?v.x:axis==1?v.y:v.z;
}
// n^T S_i(T_i,D_i) n for the aligned planar mesh is the corresponding
// principal eigenvalue. Compute its logarithm, never store a dense S field.
BATTERY_GEOMETRY_HD inline double directional_log_metric(const SurfaceConnectivity& r,int axis,double t,double d) {
    const double theta=(t-r.reference_temperature_k)/r.temperature_scale_k;
    return (r.temperature_log_response.count?table_value(r.temperature_log_response,t):
           geometry_component(r.linear,axis)*theta+geometry_component(r.quadratic,axis)*theta*theta)
          +geometry_component(r.damage,axis)*d;
}
// Legacy v5 symmetric closure, retained to reproduce old scenarios/comparisons.
BATTERY_GEOMETRY_HD inline InterfaceState evaluate_state_geometry(const ConnectivityLaw& l,InterfaceState s,
    int axis,double ta,double tb,double da,double db,double dt,double weight,double series_r,double scale=1.) {
    const double t=.5*(ta+tb);
    s.damage=fmax(s.damage,l.endpoint_damage_weight*fmax(da,db));
    if(t>=l.damage_onset_k && l.damage_rate_at_onset_per_s>0 && dt>0)
        s.damage+=dt*l.damage_rate_at_onset_per_s*exp(fmin(80.,(t-l.damage_onset_k)/l.damage_temperature_scale_k));
    s.damage=fmin(1.,s.damage);
    if(s.damage>=l.destruction_damage)s.phase=EdgePhase::Destroyed;
    const bool destroyed=s.phase==EdgePhase::Destroyed;
    s.l0_m=destroyed?l.destroyed_l0_m:l.l0_m;
    const double base=destroyed?l.destroyed_l_m-l.destroyed_l0_m:1.;
    double z=destroyed?(base>0?log(base/l.activation_width_m):-80.):l.geometry_reference_logit;
    z+=.5*(directional_log_metric(l.endpoint_a,axis,ta,da)+directional_log_metric(l.endpoint_b,axis,tb,db));
    z=fmax(-40.,fmin(40.,z));
    const double opening=destroyed && base<=0?0.:exp(z);
    s.l_m=s.l0_m+l.activation_width_m*opening;
    const double fraction=opening/(1.+opening);
    const double floor=destroyed?l.destroyed_background_conductance_w_per_k:l.background_conductance_w_per_k;
    const double span=destroyed?l.destroyed_active_conductance_w_per_k:l.active_conductance_w_per_k;
    const double contact_g=(floor+span*fraction)*weight*scale;
    s.conductance_w_per_k=contact_g>0?1./(1./contact_g+series_r):0.;
    s.destroyed_fraction=destroyed?1.:0.;
    return s;
}

// Ordered source -> receiver response. No averaging of endpoint responses or
// forward/reverse conductances. The recipient participates through state contrast;
// material-pair reference geometry is supplied by the contact law.
BATTERY_GEOMETRY_HD inline double directed_log_response(const SurfaceConnectivity& source,
    int axis,double ts,double tr,double ds,double dr) {
    return directional_log_metric(source,axis,ts,ds)
        +geometry_component(source.temperature_contrast,axis)*(ts-tr)/source.temperature_scale_k
        +geometry_component(source.damage_contrast,axis)*(ds-dr);
}
// direction=0 selects the hotter donor; +/-1 evaluates a->b / b->a for diagnostics.
// Each call advances the ONE physical contact damage history at most once. Pass
// dt=0 for observations. One selected flux is debited/credited by the GPU gather.
BATTERY_GEOMETRY_HD inline InterfaceState evaluate_directed_geometry(const ConnectivityLaw& l,InterfaceState s,
    int axis,double ta,double tb,double da,double db,double dt,double weight,double series_r,double scale=1.,int direction=0,double destroyed_multiplier=1.,double radiative_resistance_a=-1.) {
    if(l.radiation_coefficient_w_per_k4>0.) {
        // Boundary endpoint a is the finite solid, b the environment. This is
        // not an extra heat step or a temperature clamp: GPU gather still
        // debits the single edge flux and credits the boundary energy ledger.
        const double b=l.radiation_coefficient_w_per_k4*weight*scale;
        s.conductance_w_per_k=radiative_resistance_a>=0.?
            radiative_pair_conductance(b,ta,tb,radiative_resistance_a,
                fmax(0.,series_r-radiative_resistance_a),l.radiation_temperature_limit_k):
            radiative_conductance(b,ta,tb,series_r,l.radiation_temperature_limit_k);
        return s;
    }
    s.damage=fmax(s.damage,l.endpoint_damage_weight*fmax(da,db));
    const double t=.5*(ta+tb); // physical interface aging closure, NOT a transport average
    if(t>=l.damage_onset_k && l.damage_rate_at_onset_per_s>0 && dt>0)
        s.damage+=dt*l.damage_rate_at_onset_per_s*exp(fmin(80.,(t-l.damage_onset_k)/l.damage_temperature_scale_k));
    s.damage=fmin(1.,s.damage);
    if(s.damage>=l.destruction_damage)s.phase=EdgePhase::Destroyed;
    const bool destroyed=s.phase==EdgePhase::Destroyed;
    const bool forward=direction>0 || (direction==0 && ta>=tb);
    const auto& source=forward?l.endpoint_a:l.endpoint_b;
    s.l0_m=0.;
    double z=destroyed?l.destroyed_geometry_reference_logit:l.geometry_reference_logit;
    z+=l.interface_damage_log_response*s.damage;
    z+=directed_log_response(source,axis,forward?ta:tb,forward?tb:ta,forward?da:db,forward?db:da);
    z=fmax(-40.,fmin(40.,z));
    const double opening=exp(z);
    s.l_m=0.;
    const double fraction=opening/(1.+opening);
    const double floor=destroyed?l.destroyed_background_conductance_w_per_k:l.background_conductance_w_per_k;
    const double span=destroyed?l.destroyed_active_conductance_w_per_k:l.active_conductance_w_per_k;
    const double g=(floor+span*fraction)*weight*scale*(destroyed?destroyed_multiplier:1.);
    s.conductance_w_per_k=g>0?1./(1./g+series_r):0.;
    s.destroyed_fraction=destroyed?1.:0.;
    return s;
}
} // namespace battery
#undef BATTERY_GEOMETRY_HD

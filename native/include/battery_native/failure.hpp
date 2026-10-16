#pragma once
#include "battery_native/model.hpp"
#include <cmath>
#ifdef __CUDACC__
#define BATTERY_FAILURE_HD __host__ __device__
#else
#define BATTERY_FAILURE_HD
#endif
namespace battery {
// Each exponential is normalized and stops after six time constants. The
// cumulative law is exactly zero/one at its endpoints: no infinite tail budget.
BATTERY_FAILURE_HD inline double pulse_component(double age,double tau) {
    return age<=0?0.:age>=6*tau?1.:-expm1(-age/tau)/(-expm1(-6.));
}
BATTERY_FAILURE_HD inline double pulse_fraction(const RunawayPulse& p,double fast_time,double age) {
    if(p.profile==RunawayProfile::ConstantPower)return fmin(1.,fmax(0.,age/fast_time));
    return p.fast_fraction*pulse_component(age,fast_time)+(1-p.fast_fraction)*pulse_component(age,p.tail_time_s);
}
BATTERY_FAILURE_HD inline double pulse_end(const RunawayPulse& p,double fast_time) {
    if(p.profile==RunawayProfile::ConstantPower)return fast_time;
    return 6*fmax(p.fast_fraction>0?fast_time:0.,p.fast_fraction<1?p.tail_time_s:0.);
}
struct BulkBoundaryPhase {double factor=1.,threshold=1.,peak_factor=1.,relaxation_time_s=0.;};
inline BulkBoundaryPhase boundary_phase(const CellConfig& c) {
    return c.bulk_destruction_on_boundary?BulkBoundaryPhase{c.bulk_connectivity.destroyed_background_conductance_w_per_k,
        c.bulk_connectivity.destruction_damage,c.bulk_failure_transient.peak_factor,
        c.bulk_failure_transient.relaxation_time_s}:BulkBoundaryPhase{};
}
BATTERY_FAILURE_HD inline double phase_factor(BulkBoundaryPhase p,double damage,double time=0.,double start=-1.) {
    if(damage<p.threshold)return 1.;
    if(p.relaxation_time_s<=0 || start<0)return p.factor;
    const double remaining=1.-pulse_component(fmax(0.,time-start),p.relaxation_time_s);
    return p.factor+(p.peak_factor-p.factor)*remaining;
}
BATTERY_FAILURE_HD inline double phase_resistance(double total,double a,double da,double db,BulkBoundaryPhase pa,BulkBoundaryPhase pb,
    double time=0.,double start_a=-1.,double start_b=-1.,double reaction_a=1.,double reaction_b=1.) {
    const double fa=phase_factor(pa,da,time,start_a)*reaction_a,fb=phase_factor(pb,db,time,start_b)*reaction_b;
    // Preserve legacy arithmetic exactly while neither medium has changed.
    return fa==1. && fb==1.?total:a/fa+(total-a)/fb;
}
BATTERY_FAILURE_HD inline double minimum_phase_resistance(double total,double a,BulkBoundaryPhase pa,BulkBoundaryPhase pb,
    double reaction_a=1.,double reaction_b=1.) {
    pa.factor=fmax(1.,fmax(pa.factor,pa.relaxation_time_s>0?pa.peak_factor:1.));
    pb.factor=fmax(1.,fmax(pb.factor,pb.relaxation_time_s>0?pb.peak_factor:1.));
    return phase_resistance(total,a,1.,1.,pa,pb,0.,-1.,-1.,reaction_a,reaction_b);
}
// Ordered internal branch: use the failed donor's structural age; if only the
// recipient has failed, it controls the rupture. Never average two directions.
BATTERY_FAILURE_HD inline double bulk_failure_multiplier(BulkBoundaryPhase p,double ta,double tb,double da,double db,
    double start_a,double start_b,double time,int direction=0) {
    if(p.relaxation_time_s<=0 || (da<p.threshold && db<p.threshold))return 1.;
    const bool forward=direction>0 || (direction==0 && ta>=tb);
    const bool use_a=da>=p.threshold && (db<p.threshold || forward);
    return phase_factor(p,1.,time,use_a?start_a:start_b)/p.factor;
}
} // namespace battery
#undef BATTERY_FAILURE_HD

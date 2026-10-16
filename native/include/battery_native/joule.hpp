#pragma once
#include "battery_native/model.hpp"
#include <cmath>
#ifdef __CUDACC__
#define JOULE_HD __host__ __device__
#else
#define JOULE_HD
#endif
namespace battery {
struct CurrentPoint {double time_s,current_a;};
struct CurrentView {const CurrentPoint* points=nullptr;int count=0;};
JOULE_HD inline double electrical_resistance(const ElectricalLaw& p,double temperature,double damage) {
    const double exponent=p.temperature_log_slope_per_k*(temperature-p.reference_temperature_k)+
        p.damage_log_slope*fmin(1.,fmax(0.,damage));
    return fmin(p.max_resistance_ohm,fmax(p.min_resistance_ohm,p.resistance_ohm*exp(fmin(80.,fmax(-80.,exponent)))));
}
JOULE_HD inline int current_segment(CurrentView v,double time) {
    int lo=0,hi=v.count-1;
    while(hi-lo>1){const int mid=(lo+hi)/2;if(v.points[mid].time_s<=time)lo=mid;else hi=mid;}
    return lo;
}
JOULE_HD inline double current_at(const ElectricalLaw& p,CurrentView v,double time) {
    if(!v.count)return p.current_a;
    if(time<=v.points[0].time_s)return v.points[0].current_a;
    if(time>=v.points[v.count-1].time_s)return v.points[v.count-1].current_a;
    const int i=current_segment(v,time);const auto a=v.points[i],b=v.points[i+1];
    return a.current_a+(b.current_a-a.current_a)*(time-a.time_s)/(b.time_s-a.time_s);
}
// Exact integral of I(t)^2 for piecewise-linear current, including zero/reverse
// current and intervals spanning multiple knots. R is frozen on the thermal step.
JOULE_HD inline double squared_current_integral(const ElectricalLaw& p,CurrentView v,double time,double duration) {
    if(!v.count)return p.current_a*p.current_a*duration;
    const double end=time+duration;double total=0.;int i=current_segment(v,time);
    while(time<end) {
        double next=end;
        if(time<v.points[0].time_s)next=fmin(end,v.points[0].time_s);
        else if(i+1<v.count && v.points[i+1].time_s>time)next=fmin(end,v.points[i+1].time_s);
        const double a=current_at(p,v,time),b=current_at(p,v,next);
        total+=(next-time)*(a*a+a*b+b*b)/3.;time=next;
        if(i+1<v.count && time>=v.points[i+1].time_s)++i;
    }
    return total;
}
JOULE_HD inline double maximum_current(const ElectricalLaw& p,CurrentView v,double begin,double end) {
    double value=fmax(fabs(current_at(p,v,begin)),fabs(current_at(p,v,end)));
    if(v.count)for(int i=current_segment(v,begin);i<v.count && v.points[i].time_s<=end;++i)
        if(v.points[i].time_s>=begin)value=fmax(value,fabs(v.points[i].current_a));
    return value;
}
JOULE_HD inline double positive_current_integral(const ElectricalLaw& p,CurrentView v,double time,double duration) {
    if(!v.count)return fmax(0.,p.current_a)*duration;
    const double end=time+duration;double total=0.;int i=current_segment(v,time);
    while(time<end) {
        double next=end;
        if(time<v.points[0].time_s)next=fmin(end,v.points[0].time_s);
        else if(i+1<v.count && v.points[i+1].time_s>time)next=fmin(end,v.points[i+1].time_s);
        const double a=current_at(p,v,time),b=current_at(p,v,next),dt=next-time;
        if(a>=0. && b>=0.)total+=.5*(a+b)*dt;
        else if(a>0. || b>0.){const double positive=fmax(a,b);total+=.5*dt*positive*positive/fabs(b-a);}
        time=next;if(i+1<v.count && time>=v.points[i+1].time_s)++i;
    }
    return total;
}
JOULE_HD inline double maximum_charging_current(const ElectricalLaw& p,CurrentView v,double begin,double end) {
    double value=fmax(0.,fmax(current_at(p,v,begin),current_at(p,v,end)));
    if(v.count)for(int i=current_segment(v,begin);i<v.count && v.points[i].time_s<=end;++i)
        if(v.points[i].time_s>=begin)value=fmax(value,v.points[i].current_a);
    return value;
}
// Same Arrhenius as ordinary R1, with an accumulated-charge-driven prefactor.
// Overcharge is explicitly supplied by the experiment/SOC controller; normal
// charging and discharging never silently become abuse.
JOULE_HD inline double overcharge_r1_rate(const ConversionLaw& c,const ElectricalLaw& e,double temperature,double charging_current) {
    if(!e.enabled || e.overcharge_fraction<=0. || c.overcharge_conversion_per_charge<=0.)return 0.;
    const double exponent=c.degradation_activation_j_per_mol/8.31446261815324*
        (1./e.reference_temperature_k-1./temperature);
    return c.overcharge_conversion_per_charge*e.overcharge_fraction*fmax(0.,charging_current)/e.nominal_capacity_coulomb*
        exp(fmin(80.,fmax(-80.,exponent)));
}
} // namespace battery
#undef JOULE_HD

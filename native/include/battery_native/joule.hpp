#pragma once
#include "battery_native/constitutive.hpp"
#include <cmath>
#ifdef __CUDACC__
#define JOULE_HD __host__ __device__
#else
#define JOULE_HD
#endif
namespace battery {
struct CurrentPoint {double time_s,current_a;};
struct CurrentView {const CurrentPoint* points=nullptr;int count=0;};
// Additional charging polarization is a state of matter, not a clock driven
// by all positive current. Only accepted, unconsumed charging precursor counts;
// R1-derived precursor belongs to the reduced-conductivity mixture instead.
JOULE_HD inline double electrical_charge_polarization_fraction(double reactant,double healthy_charged) {
    const double healthy=reactant+healthy_charged;
    return healthy>0.?fmin(1.,fmax(0.,healthy_charged/healthy)):0.;
}
JOULE_HD inline double electrical_resistance(const ElectricalLaw& p,double temperature,double polarization=0.) {
    const double thermal=p.temperature_log_response.count?table_value(p.temperature_log_response,temperature):
        p.temperature_log_slope_per_k*(temperature-p.reference_temperature_k);
    const double exponent=thermal+p.charge_log_slope*polarization;
    return fmin(p.max_resistance_ohm,fmax(p.min_resistance_ohm,p.resistance_ohm*exp(fmin(80.,fmax(-80.,exponent)))));
}
// One effective coefficient for R1-derived B and ALL condensed products.
// A and healthy charged Bc retain intact conductivity; gas/void has none.
JOULE_HD inline double electrical_mixture_factor(const ElectricalLaw& e,const ConversionLaw& c,
    double reactant,double activated,double residue,double reference_mass,double healthy_charged=0.) {
    if(!c.enabled)return 1.;
    return fmax(0.,(reactant+healthy_charged+e.product_conductivity_ratio*(activated-healthy_charged+residue))/reference_mass);
}
// Healthy material and condensed products conduct in parallel within one
// control volume. Only the healthy phase owns charging polarization. Applying
// its exp(b*Bc/(A+Bc)) to the products leaves an artificial charged-residue
// resistor after R2 has consumed almost all healthy matter.
JOULE_HD inline double electrical_mixture_conductivity(const ElectricalLaw& e,const ConversionLaw& c,
    double temperature,double reactant,double activated,double residue,double reference_mass,double healthy_charged=0.) {
    const double baseline=electrical_resistance(e,temperature,0.);
    if(!c.enabled)return 1./baseline;
    const double healthy=fmax(0.,reactant+healthy_charged);
    const double products=fmax(0.,activated-healthy_charged+residue);
    const double polarization=electrical_charge_polarization_fraction(reactant,healthy_charged);
    return (healthy/electrical_resistance(e,temperature,polarization)+
        e.product_conductivity_ratio*products/baseline)/reference_mass;
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
// Charging formation and thermal degradation are distinct sources of the SAME
// precursor. Charging follows charge throughput, not R1's thermal Arrhenius.
// Charging acceptance is explicit; positive current is not itself damage.
JOULE_HD inline double sample_drive(CurrentView v,double time) {
    if(!v.count)return 0.;
    if(time<=v.points[0].time_s)return v.points[0].current_a;
    if(time>=v.points[v.count-1].time_s)return v.points[v.count-1].current_a;
    const int i=current_segment(v,time);const auto a=v.points[i],b=v.points[i+1];
    return a.current_a+(b.current_a-a.current_a)*(time-a.time_s)/(b.time_s-a.time_s);
}
JOULE_HD inline double drive_integral(CurrentView v,double time,double duration) {
    if(!v.count || duration<=0.)return 0.;
    const double end=time+duration;double total=0.;
    while(time<end) {
        double next=end;
        if(time<v.points[0].time_s)next=fmin(end,v.points[0].time_s);
        else {
            const int i=current_segment(v,time);
            if(i+1<v.count && v.points[i+1].time_s>time)next=fmin(end,v.points[i+1].time_s);
        }
        const double a=sample_drive(v,time),b=sample_drive(v,next);
        total+=.5*(a+b)*(next-time);time=next;
    }
    return total;
}
JOULE_HD inline double charging_precursor_rate(const ConversionLaw& c,const ElectricalLaw& e,double charging_current,double capacity_fraction=1.) {
    if(!e.enabled || e.charge_acceptance_fraction<=0. || c.charge_conversion_per_nominal_charge<=0.)return 0.;
    if(!(capacity_fraction>1e-12))return 0.;
    return c.charge_conversion_per_nominal_charge*e.charge_acceptance_fraction*fmax(0.,charging_current)/(e.nominal_capacity_coulomb*capacity_fraction);
}
JOULE_HD inline double minimum_current(const ElectricalLaw& p,CurrentView v,double begin,double end) {
    double result=fmin(current_at(p,v,begin),current_at(p,v,end));
    for(int i=v.count?current_segment(v,begin)+1:0;i<v.count && v.points[i].time_s<end;++i)
        result=fmin(result,v.points[i].current_a);
    return result;
}
JOULE_HD inline double next_drive_knot(CurrentView view,double time,double end) {
    if(!view.count)return end;
    // Drive sampling holds the last value after the recording ends. Returning
    // that already-passed knot would move integration backwards, then leave
    // time unchanged forever. This also matters when I is already zero.
    if(time>=view.points[view.count-1].time_s)return end;
    if(time<view.points[0].time_s)return fmin(end,view.points[0].time_s);
    const int i=current_segment(view,time);
    return i+1<view.count?fmin(end,view.points[i+1].time_s):end;
}
// Exact terminal electrical work. This is a bound, NEVER an extra heat source.
JOULE_HD inline double electrical_work_integral(const ElectricalLaw& p,CurrentView current,CurrentView voltage,double time,double duration) {
    const double end=time+duration;double work=0.;
    while(time<end) {
        const double next=fmin(next_drive_knot(current,time,end),next_drive_knot(voltage,time,end));
        const double ia=current_at(p,current,time),ib=current_at(p,current,next);
        const double ua=sample_drive(voltage,time),ub=sample_drive(voltage,next);
        work+=(next-time)*(2.*ia*ua+ia*ub+ib*ua+2.*ib*ub)/6.;time=next;
    }
    return work;
}
// Dissipative envelope: measured electrical work minus the declared reversible
// floor. ONLY I²R heat is bounded here. Charging storage pays from FULL U·I
// after Joule heat, not from this remainder; U·I is never an extra heat source.
JOULE_HD inline double electrical_dissipative_budget(const ElectricalLaw& p,CurrentView current,CurrentView voltage,double time,double duration) {
    return electrical_work_integral(p,current,voltage,time,duration)-
        p.reversible_voltage_floor_v*positive_current_integral(p,current,time,duration);
}
struct ElectricalInterval {
    double charge_c=0.,i2_a2_s=0.,work_j=0.,limited_time_s=0.;
};
JOULE_HD inline double compliant_current(const ElectricalLaw& p,double command,double voltage,double resistance) {
    return fmin(fmax(0.,command),resistance>0.?fmax(0.,voltage-p.reversible_voltage_floor_v)/resistance:0.);
}
// Exact integration of min(I_set, max(0,U-E_floor)/R) for frozen Kirchhoff R.
// Split at both profile knots AND branch intersections. No heat clipping,
// microsteps, per-node quadrature or independently inconsistent charge budget.
JOULE_HD inline ElectricalInterval compliant_interval(const ElectricalLaw& p,CurrentView current,CurrentView voltage,
    double resistance,double time,double duration) {
    ElectricalInterval out;const double end=time+duration;
    while(time<end) {
        double next=fmin(next_drive_knot(current,time,end),next_drive_knot(voltage,time,end));
        double ua=sample_drive(voltage,time),ub=sample_drive(voltage,next);
        const double va=ua-p.reversible_voltage_floor_v,vb=ub-p.reversible_voltage_floor_v;
        if((va<0. && vb>0.) || (va>0. && vb<0.)) {
            const double cross=time+(next-time)*va/(va-vb);
            if(cross>time && cross<next)next=cross;
        }
        ub=sample_drive(voltage,next);
        const double ca=current_at(p,current,time),cb=current_at(p,current,next);
        const double da=ca-(resistance>0.?fmax(0.,ua-p.reversible_voltage_floor_v)/resistance:0.);
        const double db=cb-(resistance>0.?fmax(0.,ub-p.reversible_voltage_floor_v)/resistance:0.);
        if((da<0. && db>0.) || (da>0. && db<0.)) {
            const double cross=time+(next-time)*da/(da-db);
            if(cross>time && cross<next)next=cross;
        }
        ub=sample_drive(voltage,next);
        const double ia=compliant_current(p,ca,ua,resistance);
        const double ib=compliant_current(p,current_at(p,current,next),ub,resistance),dt=next-time;
        out.charge_c+=dt*(ia+ib)/2.;
        out.i2_a2_s+=dt*(ia*ia+ia*ib+ib*ib)/3.;
        out.work_j+=dt*(2.*ia*ua+ia*ub+ib*ua+2.*ib*ub)/6.;
        const double mid=(time+next)/2.;
        if(compliant_current(p,current_at(p,current,mid),sample_drive(voltage,mid),resistance)<current_at(p,current,mid))
            out.limited_time_s+=dt;
        time=next;
    }
    return out;
}
} // namespace battery
#undef JOULE_HD

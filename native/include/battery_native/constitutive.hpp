#pragma once
#include "battery_native/model.hpp"
#include <cmath>
#ifdef __CUDACC__
#define BATTERY_CONSTITUTIVE_HD __host__ __device__
#else
#define BATTERY_CONSTITUTIVE_HD
#endif
namespace battery {
BATTERY_CONSTITUTIVE_HD inline double table_value(const ScalarTable& t,double x) {
    if(!t.count)return 0.;
    if(x<=t.temperature_k[0])return t.value[0];
    for(int i=1;i<t.count;++i)if(x<t.temperature_k[i]) {
        const double f=(x-t.temperature_k[i-1])/(t.temperature_k[i]-t.temperature_k[i-1]);
        return t.value[i-1]+f*(t.value[i]-t.value[i-1]);
    }
    return t.value[t.count-1];
}
// Primitive measured from the first knot, including constant endpoint tails.
BATTERY_CONSTITUTIVE_HD inline double table_integral(const ScalarTable& t,double x) {
    if(x<=t.temperature_k[0])return (x-t.temperature_k[0])*t.value[0];
    double sum=0.;
    for(int i=1;i<t.count;++i) {
        const double width=t.temperature_k[i]-t.temperature_k[i-1];
        const double h=fmin(x-t.temperature_k[i-1],width);
        sum+=h*(t.value[i-1]+.5*h*(t.value[i]-t.value[i-1])/width);
        if(x<=t.temperature_k[i])return sum;
    }
    return sum+(x-t.temperature_k[t.count-1])*t.value[t.count-1];
}
BATTERY_CONSTITUTIVE_HD inline double table_integral_inverse(const ScalarTable& t,double u) {
    if(u<=0.)return t.temperature_k[0]+u/t.value[0];
    for(int i=1;i<t.count;++i) {
        const double h=t.temperature_k[i]-t.temperature_k[i-1];
        const double segment=.5*h*(t.value[i-1]+t.value[i]);
        if(u<=segment) {
            const double slope=(t.value[i]-t.value[i-1])/h;
            // Stable positive root, including the zero-slope limit.
            return t.temperature_k[i-1]+2.*u/(t.value[i-1]+sqrt(fmax(0.,t.value[i-1]*t.value[i-1]+2.*slope*u)));
        }
        u-=segment;
    }
    return t.temperature_k[t.count-1]+u/t.value[t.count-1];
}
BATTERY_CONSTITUTIVE_HD inline double defect_capacity_factor(const HeatCapacityLaw& l,double d) {
    return 1.+(l.damage_factor_at_one-1.)*fmax(0.,fmin(1.,d));
}
BATTERY_CONSTITUTIVE_HD inline double capacity_factor(const HeatCapacityLaw& l,double t,double d) {
    return l.factor.count?table_value(l.factor,t)*defect_capacity_factor(l,d):1.;
}
BATTERY_CONSTITUTIVE_HD inline double minimum_capacity_factor(const HeatCapacityLaw& l) {
    if(!l.factor.count)return 1.;
    double v=l.factor.value[0];for(int i=1;i<l.factor.count;++i)v=fmin(v,l.factor.value[i]);
    return v*fmin(1.,l.damage_factor_at_one);
}
// U(T_ref,D)=C_ref*T_ref for every D. Defect/reaction energy sources are NOT
// hidden here. This is a constitutive choice, not a calibrated defect model.
BATTERY_CONSTITUTIVE_HD inline double thermal_energy(const HeatCapacityLaw& l,double cref,double t,double d) {
    if(!l.factor.count)return cref*t;
    return cref*(l.reference_temperature_k+defect_capacity_factor(l,d)*
        (table_integral(l.factor,t)-table_integral(l.factor,l.reference_temperature_k)));
}
BATTERY_CONSTITUTIVE_HD inline double thermal_temperature(const HeatCapacityLaw& l,double cref,double u,double d) {
    if(!l.factor.count)return u/cref;
    return table_integral_inverse(l.factor,(u/cref-l.reference_temperature_k)/defect_capacity_factor(l,d)
        +table_integral(l.factor,l.reference_temperature_k));
}
} // namespace battery
#undef BATTERY_CONSTITUTIVE_HD

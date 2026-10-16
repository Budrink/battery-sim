#pragma once
#include "battery_native/conversion.hpp"
#include "battery_native/constitutive.hpp"
#ifdef __CUDACC__
#define MIXTURE_HD __host__ __device__
#else
#define MIXTURE_HD
#endif
namespace battery {
// Shared CPU/CUDA inversion of the SAME extensive energy equation. Chemical
// inventories and free volume are unchanged by this local equilibrium closure.
MIXTURE_HD inline double mixture_temperature(const HeatCapacityLaw& law,const ConversionLaw& conversion,
    const ThermalState& s,const GasState& g,double capacity,double fraction,GasModel model) {
    const double total=s.stored_energy_j+g.energy_j,cg=g.mass_kg*model.cv_j_per_kg_k,tref=model.reference_temperature_k;
    if(!law.factor.count || conversion.enabled) {
        const double cs=conversion.enabled?conversion_capacity(conversion,s,capacity,fraction):capacity;
        const double offset=conversion.enabled?(capacity-cs)*tref:0.;
        return (total-offset+cg*tref)/(cs+cg);
    }
    double lo=0.,hi=fmax(1.,fmax(s.temperature_k,g.temperature_k));
    for(int n=0;n<64 && thermal_energy(law,capacity,hi,s.damage)+cg*(hi-tref)<total;++n)hi*=2.;
    for(int n=0;n<64;++n) {
        const double mid=(lo+hi)/2;
        if(thermal_energy(law,capacity,mid,s.damage)+cg*(mid-tref)<total)lo=mid;else hi=mid;
    }
    return (lo+hi)/2;
}
}
#undef MIXTURE_HD

#pragma once
#include "battery_native/cuda_solver.hpp"
#include "battery_native/gas_local_reactor.hpp"
#include "battery_native/gas_thermal_contact.hpp"

#ifdef __CUDACC__
#define BATTERY_EXCHANGE_HD __host__ __device__
#else
#define BATTERY_EXCHANGE_HD
#endif
namespace battery {
// Freeze the selected directed conductances for ONE local integration stage.
// Parallel contact reduction is not a symmetrization of Sij and Sji.
BATTERY_EXCHANGE_HD inline GasFilmState evaluate_gas_surface_contacts(int gas,
    double temperature,const GasFilmView& v) {
    GasFilmState f;
    for(int j=v.offsets[gas];j<v.offsets[gas+1];++j) {
        const int id=v.ids[j];const auto& p=v.patches[id];const auto& s=v.solid[p.volume];
        const double g=gas_contact_conductance(p,temperature,s.temperature_k,s.damage);
        v.conductance[id]=g;f.conductance+=g;
        f.wall_temperature+=g*s.temperature_k;
        f.receiving_capacity+=g*v.capacity[p.volume];
    }
    if(f.conductance>0.) {
        f.wall_temperature/=f.conductance;f.receiving_capacity/=f.conductance;
    }
    return f;
}
// Credit precisely the heat removed by the local reactor, using the SAME
// directed conductances. Each gas owns its patch records; gather has one writer
// per solid. The final nonzero patch closes floating-point roundoff.
BATTERY_EXCHANGE_HD inline void record_gas_surface_heat(int gas,const GasFilmView& v,
    const GasFilmState& f,double duration) {
    if(!(f.conductance>0.))return;
    int last=-1;
    for(int j=v.offsets[gas];j<v.offsets[gas+1];++j)
        if(v.conductance[v.ids[j]]>0.)last=j;
    double sum=0.;
    for(int j=v.offsets[gas];j<v.offsets[gas+1];++j) {
        const int id=v.ids[j];const double g=v.conductance[id];
        if(!(g>0.))continue;
        const auto& p=v.patches[id];
        const double q=j==last?f.heat_to_solid-sum:
            g*(f.heat_to_solid/f.conductance+
               (f.wall_temperature-v.solid[p.volume].temperature_k)*duration);
        v.heat[id]+=q;sum+=q;
    }
}
}
#undef BATTERY_EXCHANGE_HD

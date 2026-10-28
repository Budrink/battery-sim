#pragma once
#include "battery_native/gas_combustion.hpp"
#include "battery_native/gas_mixing.hpp"

#ifdef __CUDACC__
#define RTL_HD __host__ __device__
#else
#define RTL_HD
#endif
namespace battery {
// Accuracy bounds only. No reaction/flow is applied or suppressed here.
RTL_HD inline double combustion_available(const kawasaki::Node& n,const GasCombustionLaw& law) {
    return fmax(0.,fmin(n.chemical,n.oxidizer*law.oxidizer_heat_j_per_kg));
}
RTL_HD inline double resident_reaction_step(const kawasaki::Node& n,const GasCombustionLaw& law,
    const kawasaki::Settings& s,double fraction,double temperature_step) {
    const double capacity=n.solid_capacity+s.cv*n.mass;
    // Even instantaneous consumption of this stock cannot violate the thermal
    // accuracy budget. burn_gas STILL integrates it and keeps both ledgers.
    if(combustion_available(n,law)<=capacity*fmin(temperature_step,.05*kawasaki::temperature(n,s)))return 1e300;
    return combustion_step_limit(n,law,s,fraction,temperature_step);
}
RTL_HD inline double reaction_rise_bound(const kawasaki::Node& n,const GasCombustionLaw& law,
    const kawasaki::Settings& s,double horizon,double temperature_step) {
    const double capacity=n.solid_capacity+s.cv*n.mass,t=kawasaki::temperature(n,s);
    const double budget=fmin(temperature_step,.05*t),available=combustion_available(n,law);
    if(!(available>0.))return 0.;
    return fmin(budget,fmin(available,combustion_rate(law,t+budget)*available*horizon)/capacity);
}
struct ReactionArrivals {
    double energy=0.,chemical=0.,oxygen_heat=0.;
    double max_temperature=0.,max_specific_fuel=0.,max_specific_oxygen_heat=0.;
    double fuel_rate=0.,oxygen_heat_rate=0.; // reagent supply, J/s
};
RTL_HD inline void bound_reaction_arrival(ReactionArrivals& out,const kawasaki::Node& donor,
    const kawasaki::Node& receiver,const GasCombustionLaw& donor_law,const GasCombustionLaw& receiver_law,
    kawasaki::Edge edge,const kawasaki::Settings& s,double horizon,double temperature_step,double fraction) {
    horizon=fmin(horizon,resident_reaction_step(donor,donor_law,s,fraction,temperature_step));
    auto up=donor;
    const double td=kawasaki::temperature(donor,s)+reaction_rise_bound(donor,donor_law,s,horizon,temperature_step);
    up.energy=td*(donor.solid_capacity+s.cv*donor.mass);
    const double pa=kawasaki::pressure(up,s),pb=kawasaki::pressure(receiver,s);
    if(pa-pb<=s.equilibrium_atol_pa+s.equilibrium_rtol*pa)return;
    // The actual gather retains at least half the donor mass. Each edge is
    // additionally capped by pressure equalization. A one-sided receiver
    // pressure bound is looser than that cap, but cheap and conservative.
    const double donor_floor=kawasaki::temperature(donor,s)*pow(.5,s.R/s.cv);
    const double tmin=fmin(kawasaki::temperature(receiver,s),(1+s.R/s.cv)*donor_floor);
    const double pressure_cap=(pa-pb)*receiver.volume/(s.R*tmin);
    const double flow=kawasaki::flow(up,receiver,edge,s);
    const double dm=fmin(.5*donor.mass,fmin(pressure_cap,flow*horizon));
    out.energy+=dm*(s.cv+s.R)*td; // packet enthalpy <= inlet enthalpy
    out.chemical+=dm/donor.mass*donor.chemical;
    out.oxygen_heat+=dm/donor.mass*donor.oxidizer*receiver_law.oxidizer_heat_j_per_kg;
    out.fuel_rate+=flow/donor.mass*donor.chemical;
    out.oxygen_heat_rate+=flow/donor.mass*donor.oxidizer*receiver_law.oxidizer_heat_j_per_kg;
    if(dm>0.) {
        out.max_temperature=fmax(out.max_temperature,(1+s.R/s.cv)*td);
        out.max_specific_fuel=fmax(out.max_specific_fuel,donor.chemical/donor.mass);
        out.max_specific_oxygen_heat=fmax(out.max_specific_oxygen_heat,donor.oxidizer/donor.mass*receiver_law.oxidizer_heat_j_per_kg);
    }
}
// A pressure-equilibrated neighbour can still supply the missing reagent.
// Bound the gross incoming half of counterexchange; ignore its outgoing half.
RTL_HD inline void bound_mixing_arrival(ReactionArrivals& out,const kawasaki::Node& donor,
    const kawasaki::Node& receiver,const GasCombustionLaw& donor_law,const GasCombustionLaw& receiver_law,
    const kawasaki::Edge& edge,const kawasaki::Settings& s,const GasModel& model,double horizon) {
    if(!(model.mixing_diffusivity_m2_s>0. || model.mixing_flow_fraction>0.))return;
    const double flow=kawasaki::flow(donor,receiver,edge,s)+kawasaki::flow(receiver,donor,edge,s);
    const double rate=maximum_mixing_conductance(donor,receiver,edge,s,model.mixing_diffusivity_m2_s,model.mixing_flow_fraction,flow);
    const double dm=fmin(.5*donor.mass,rate*horizon);
    if(!(dm>0.))return;
    const double td=kawasaki::temperature(donor,s)+reaction_rise_bound(donor,donor_law,s,horizon,model.max_temperature_step_k);
    out.energy+=dm*s.cv*td;out.chemical+=dm/donor.mass*donor.chemical;
    out.oxygen_heat+=dm/donor.mass*donor.oxidizer*receiver_law.oxidizer_heat_j_per_kg;
    out.fuel_rate+=rate/donor.mass*donor.chemical;
    out.oxygen_heat_rate+=rate/donor.mass*donor.oxidizer*receiver_law.oxidizer_heat_j_per_kg;
    out.max_temperature=fmax(out.max_temperature,td);
    out.max_specific_fuel=fmax(out.max_specific_fuel,donor.chemical/donor.mass);
    out.max_specific_oxygen_heat=fmax(out.max_specific_oxygen_heat,donor.oxidizer/donor.mass*receiver_law.oxidizer_heat_j_per_kg);
}
RTL_HD inline double arrival_reaction_step(const kawasaki::Node& n,const GasCombustionLaw& law,
    const kawasaki::Settings& s,ReactionArrivals incoming,double horizon,double fraction,double temperature_step,
    double integrated_resident_limit=-1.) {
    const double resident=integrated_resident_limit>=0.?integrated_resident_limit:resident_reaction_step(n,law,s,fraction,temperature_step);
    if(incoming.energy==0. && incoming.chemical==0. && incoming.oxygen_heat==0.)return resident;
    // Simultaneous outgoing packets remove at most half the resident mass.
    // Ignoring their cooling and all added heat capacity gives an upper T bound.
    const double capacity_min=n.solid_capacity+.5*s.cv*n.mass;
    const double available=fmin(n.chemical+incoming.chemical,
        n.oxidizer*law.oxidizer_heat_j_per_kg+incoming.oxygen_heat);
    const double t=kawasaki::temperature(n,s),rise=fmin(temperature_step,.05*t);
    // Adding mass also adds heat capacity. The mixture cannot exceed its
    // hottest inlet enthalpy temperature or its largest specific reactant stock,
    // even when a nearly empty receiver fills with a much heavier packet.
    const double fuel_specific=fmax(n.chemical/n.mass,incoming.max_specific_fuel);
    const double oxygen_specific=fmax(n.oxidizer*law.oxidizer_heat_j_per_kg/n.mass,incoming.max_specific_oxygen_heat);
    const double heat_per_capacity=fmin(available/capacity_min,fmin(fuel_specific,oxygen_specific)/s.cv);
    if(heat_per_capacity<=rise)return resident;
    const double resident_upper=t+reaction_rise_bound(n,law,s,fmin(horizon,resident),temperature_step);
    const double upper=fmin(resident_upper+incoming.energy/capacity_min,
        fmax(resident_upper,incoming.max_temperature));
    const double rate=combustion_rate(law,upper+(law.activation_j_per_mol>0.?rise:0.));
    if(!(rate>0.))return resident;
    const double kinetic_limit=fmin(resident,fmin(combustion_interval(rate,fraction),rise/(rate*heat_per_capacity)));
    // Alternative safe bound for a fast, continuously fed flame: extra heat
    // cannot exceed incoming fuel/O2 supply, even at infinite reaction speed.
    // Do not impose the lifetime of FUTURE reactants on the entire gas graph.
    // Split the thermal budget between existing stock and arriving reactants.
    const double old_heat=combustion_available(n,law),half_budget=.5*rise*capacity_min;
    // This is a THERMAL coupling budget, independently safe even when the
    // small resident reagent stock reacts completely. Its chemical lifetime
    // must not reintroduce global subcycling: burn_gas resolves that locally.
    const double stock_limit=integrated_resident_limit>=0.?resident:
        (old_heat<=half_budget?1e300:half_budget/(rate*old_heat));
    const double extra=fmax(0.,available-old_heat);
    // min(F+rf*t,O+ro*t)-min(F,O) <= r_limiting*t. Excess fuel must
    // never throttle an oxygen-starved flame (and conversely for fuel).
    const double old_oxygen=n.oxidizer*law.oxidizer_heat_j_per_kg;
    const double supply=n.chemical<old_oxygen?incoming.fuel_rate:
        n.chemical>old_oxygen?incoming.oxygen_heat_rate:fmin(incoming.fuel_rate,incoming.oxygen_heat_rate);
    const double supply_limit=extra<=half_budget || !(supply>0.)?1e300:half_budget/supply;
    // Each criterion independently bounds the same temperature increment.
    // Either can admit the interval; local chemistry is STILL integrated by
    // burn_gas before/after transport, conserving fuel, oxygen and energy.
    return fmax(kinetic_limit,fmin(stock_limit,supply_limit));
}
} // namespace battery
#undef RTL_HD

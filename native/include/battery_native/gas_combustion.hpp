#pragma once
#include "battery_native/kawasaki_gas_math.hpp"
#include "battery_native/model.hpp"
#include <cmath>
#include <cfloat>

#ifdef __CUDACC__
#define BATTERY_COMBUSTION_HD __host__ __device__
#else
#define BATTERY_COMBUSTION_HD
#endif
namespace battery {
struct GasFilmState;
struct GasThermalPatch;
struct ThermalState;
struct GasFilmView {
    const GasThermalPatch* patches=nullptr;
    const int* offsets=nullptr;
    const int* ids=nullptr;
    const ThermalState* solid=nullptr;
    const double* capacity=nullptr;
    double* heat=nullptr;
    double* conductance=nullptr;
    const int* solid_offsets=nullptr;
    const int* solid_ids=nullptr;
    int solid_count=0;
};
// Local effective chemistry. Laws belong to volumes; fuel/oxidizer move with gas.
struct GasCombustionLaw {
    double rate_per_s=0., ignition_temperature_k=0., oxidizer_heat_j_per_kg=13.1e6;
    double reference_temperature_k=600., activation_j_per_mol=0.;
};
struct GasCombustionView {
    const GasCombustionLaw* laws=nullptr;
    bool enabled=false;
    GasFilmState* films=nullptr;
    GasFilmView exchange;
};
inline GasCombustionLaw gas_combustion_law(const GasCompartment& g) {
    return {g.reaction_rate_per_s,g.ignition_temperature_k,g.oxidizer_heat_j_per_kg,
            g.reaction_reference_k,g.reaction_activation_j_per_mol};
}
// Normalized Arrhenius. Chemical reserve is NOT part of sensible temperature.
// Ea=0 is the exact constant-rate analytical limit, not a separate mechanism.
BATTERY_COMBUSTION_HD inline double combustion_rate(const GasCombustionLaw& law,double temperature) {
    if(!(law.rate_per_s>0.) || temperature<law.ignition_temperature_k)return 0.;
    const double exponent=law.activation_j_per_mol/8.31446261815324*
        (1./law.reference_temperature_k-1./temperature);
    return law.rate_per_s*exp(fmin(80.,fmax(-80.,exponent)));
}
BATTERY_COMBUSTION_HD inline double combustion_interval(double rate,double fraction) {
    return rate>0.?-log1p(-fraction)/rate:1e300;
}
BATTERY_COMBUSTION_HD inline bool resolved_combustion_heat(const kawasaki::Node& node,const GasCombustionLaw& law) {
    return fmin(node.chemical,node.oxidizer*law.oxidizer_heat_j_per_kg)>
        32*DBL_EPSILON*fmax(1.,node.energy);
}
BATTERY_COMBUSTION_HD inline double combustion_step_limit(const kawasaki::Node& node,
    const GasCombustionLaw& law,const kawasaki::Settings& settings,double fraction,double temperature_step) {
    const double available=fmax(0.,fmin(node.chemical,node.oxidizer*law.oxidizer_heat_j_per_kg));
    const double temperature=kawasaki::temperature(node,settings);
    if(!(available>0.) || combustion_rate(law,temperature)==0.)return 1e300;
    const double rise=fmin(temperature_step,.05*temperature);
    const double rate=combustion_rate(law,temperature+(law.activation_j_per_mol>0.?rise:0.));
    double limit=combustion_interval(rate,fraction);
    if(law.activation_j_per_mol>0.)
        limit=fmin(limit,rise*(node.solid_capacity+settings.cv*node.mass)/(rate*available));
    return limit;
}
// Exact frozen-rate reaction of a locally equilibrated mixture. Chemical energy
// becomes sensible energy, oxygen is a tracer within the conserved total mass.
// Diagnostics stay in the reacting volume and are never advected.
BATTERY_COMBUSTION_HD inline bool frozen_gas_combustion(kawasaki::Node& node,const GasCombustionLaw& law,
    const kawasaki::Settings& settings,double duration) {
    const double rate=combustion_rate(law,kawasaki::temperature(node,settings));
    const double available=fmax(0.,fmin(node.chemical,node.oxidizer*law.oxidizer_heat_j_per_kg));
    const double heat=available*(-expm1(-rate*duration));
    if(settings.vapor_fuel_heat_j_kg>0. && node.chemical>0.)node.vapor*=1-heat/node.chemical;
    node.chemical=fmax(0.,node.chemical-heat);
    node.oxidizer=fmax(0.,node.oxidizer-heat/law.oxidizer_heat_j_per_kg);
    node.energy+=heat;node.reaction_released_j+=heat;
    node.reaction_oxidizer_kg+=heat/law.oxidizer_heat_j_per_kg;
    return std::isfinite(node.energy) && node.energy>0.;
}
BATTERY_COMBUSTION_HD inline bool burn_gas(kawasaki::Node& node,const GasCombustionLaw& law,
    const kawasaki::Settings& settings,double duration,double fraction=.1,double temperature_step=1.) {
    double elapsed=0.;unsigned iterations=0;
    while(elapsed<duration) {
        const double rate=combustion_rate(law,kawasaki::temperature(node,settings));
        const double available=fmax(0.,fmin(node.chemical,node.oxidizer*law.oxidizer_heat_j_per_kg));
        if(!(rate>0. && available>0.))return true;
        // Cold mixtures remain cheap; self-heating re-evaluates k(T) locally.
        const double step=law.activation_j_per_mol>0. && resolved_combustion_heat(node,law)?
            fmin(duration-elapsed,combustion_step_limit(node,law,settings,fraction,temperature_step)):duration-elapsed;
        if(!(step>0.) || elapsed+step==elapsed || ++iterations>200000)return false;
        const double heat=available*(-expm1(-rate*step));
        // Well-mixed fuels react in proportion to their chemical inventories.
        // Oxidized vapor is no longer condensable; its mass stays in products.
        // Its phase-reference energy stays mobile (no unaccounted heat pulse).
        if(settings.vapor_fuel_heat_j_kg>0.)node.vapor*=1-heat/node.chemical;
        node.chemical-=heat;
        node.oxidizer=fmax(0.,node.oxidizer-heat/law.oxidizer_heat_j_per_kg);
        node.energy+=heat;
        node.reaction_released_j+=heat;
        node.reaction_oxidizer_kg+=heat/law.oxidizer_heat_j_per_kg;
        elapsed+=step;
    }
    return true;
}
BATTERY_COMBUSTION_HD inline bool resolved_gas_fuel(const kawasaki::Node& node) {
    // Only suppress extra transport subcycling for a roundoff-sized heat stock.
    // burn_gas still consumes that stock; there is no physical reaction cutoff.
    return node.chemical>32*DBL_EPSILON*fmax(1.,node.energy);
}
} // namespace battery
#undef BATTERY_COMBUSTION_HD

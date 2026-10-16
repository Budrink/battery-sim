#pragma once
#include "battery_native/kawasaki_gas_math.hpp"
#include <cmath>
#include <cfloat>

#ifdef __CUDACC__
#define BATTERY_COMBUSTION_HD __host__ __device__
#else
#define BATTERY_COMBUSTION_HD
#endif
namespace battery {
// Local effective chemistry. Laws belong to volumes; fuel/oxidizer move with gas.
struct GasCombustionLaw {
    double rate_per_s=0., ignition_temperature_k=0., oxidizer_heat_j_per_kg=13.1e6;
};
struct GasCombustionView {
    const GasCombustionLaw* laws=nullptr;
    double maximum_rate_per_s=0.;
};
// Exact frozen-rate reaction of a locally equilibrated mixture. Chemical energy
// becomes sensible energy, oxygen is a tracer within the conserved total mass.
// Diagnostics stay in the reacting volume and are never advected.
BATTERY_COMBUSTION_HD inline void burn_gas(kawasaki::Node& node,const GasCombustionLaw& law,
    const kawasaki::Settings& settings,double duration) {
    if(!(law.rate_per_s>0. && duration>0.) ||
       kawasaki::temperature(node,settings)<law.ignition_temperature_k)return;
    const double available=fmax(0.,fmin(node.chemical,node.oxidizer*law.oxidizer_heat_j_per_kg));
    if(!(available>0.))return;
    const double heat=available*(-expm1(-law.rate_per_s*duration));
    node.chemical-=heat;
    node.oxidizer=fmax(0.,node.oxidizer-heat/law.oxidizer_heat_j_per_kg);
    node.energy+=heat;
    node.reaction_released_j+=heat;
    node.reaction_oxidizer_kg+=heat/law.oxidizer_heat_j_per_kg;
}
// At most the existing mass-step fraction reacts in one coupled gas interval.
// The exponential update itself is exact; this bound resolves its feedback on
// pressure, enthalpy flow and mixing. It is a numerical control, not a fit law.
BATTERY_COMBUSTION_HD inline double combustion_interval(double maximum_rate,double fraction) {
    return maximum_rate>0.?-log1p(-fraction)/maximum_rate:1e300;
}
BATTERY_COMBUSTION_HD inline bool resolved_gas_fuel(const kawasaki::Node& node) {
    // Only suppress extra transport subcycling for a roundoff-sized heat stock.
    // burn_gas still consumes that stock; there is no physical reaction cutoff.
    return node.chemical>32*DBL_EPSILON*fmax(1.,node.energy);
}
} // namespace battery
#undef BATTERY_COMBUSTION_HD

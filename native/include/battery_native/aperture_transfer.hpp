#pragma once
#include <cmath>
#ifdef __CUDACC__
#define BATTERY_APERTURE_HD __host__ __device__
#else
#define BATTERY_APERTURE_HD
#endif
namespace battery {
// Existing contact conductance, gated by aperture and bounded by the heat
// capacity rate of the gas actually crossing it. No pulse, timer or heat source.
BATTERY_APERTURE_HD inline double aperture_conductance(double full_g,double local_open,
    double mean_open,double area_fraction,double outward_mass_rate,double gas_cp) {
    if(mean_open<=0 || outward_mass_rate<=0)return 0.;
    const double flow_share=area_fraction*local_open/mean_open;
    return fmin(full_g*local_open,outward_mass_rate*gas_cp*flow_share);
}
} // namespace battery
#undef BATTERY_APERTURE_HD

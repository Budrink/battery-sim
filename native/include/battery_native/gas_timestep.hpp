#pragma once
#include <cmath>
#ifdef __CUDACC__
#define BATTERY_GAS_DT_HD __host__ __device__
#else
#define BATTERY_GAS_DT_HD
#endif
namespace battery {
// Accuracy bound, separate from the conservative transfer cap. equalization_mass
// is |dp| / (degree_d * dP_d/dm + degree_r * dP_r/dm). It includes both endpoint
// volumes and their gas/solid heat capacities. Near equilibrium the pressure
// tolerance prevents chasing vanishing sqrt(dp) relaxation times indefinitely.
// This is a local timescale criterion, NOT an embedded error estimate.
BATTERY_GAS_DT_HD inline double gas_pressure_dt(double pa,double pb,double raw_flow,
    double equalization_mass,double fraction,double absolute_tolerance,double relative_tolerance) {
    const double dp=fabs(pa-pb),rate=fabs(raw_flow);
    if(rate==0. || dp==0.)return 1e300;
    const double tolerance=absolute_tolerance+relative_tolerance*fmax(fabs(pa),fabs(pb));
    return fraction*(equalization_mass/rate)*fmax(1.,tolerance/dp);
}
} // namespace battery
#undef BATTERY_GAS_DT_HD

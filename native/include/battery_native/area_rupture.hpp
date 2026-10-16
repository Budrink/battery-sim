#pragma once
#include "battery_native/pixel_rupture.hpp"
#include <cmath>
#ifdef __CUDACC__
#define BATTERY_AREA_HD __host__ __device__
#else
#define BATTERY_AREA_HD
#endif
namespace battery {
// Counter-based independent marks: stream 0 is the waiting hazard, stream 1
// is the fragment area. Retries never consume RNG state. Physical fragment
// scale is independent of the numerical surface mesh.
BATTERY_AREA_HD inline double area_event_exponential(std::uint64_t key,std::uint64_t event,std::uint64_t stream) {
    const auto bits=pixel_mix(key^pixel_mix(event)^pixel_mix(stream+0x41f8d9a3ULL));
    const double u=(double(bits>>12)+.5)*0x1p-52;
    return -log(u);
}
BATTERY_AREA_HD inline double area_event_rate(double pressure_rate,double area,double mean_fragment,double opened) {
    return pressure_rate*(area/mean_fragment)*fmax(0.,1.-opened);
}
BATTERY_AREA_HD inline double area_event_increment(double opened,double area,double mean_fragment,
    std::uint64_t key,std::uint64_t event) {
    return fmin(fmax(0.,1.-opened),mean_fragment*area_event_exponential(key,event,1)/area);
}
// A fixed fraction of the ORIGINAL patch, not of its shrinking remainder.
// Normalize the event clock by that fraction so changing resolution does not
// silently change the expected opening speed at a frozen pressure/state.
BATTERY_AREA_HD inline double fixed_area_event_rate(double pressure_rate,double fraction,double opened) {
    return pressure_rate/fraction*fmax(0.,1.-opened);
}
BATTERY_AREA_HD inline double fixed_area_event_increment(double opened,double fraction) {
    return fmin(fmax(0.,1.-opened),fraction);
}
} // namespace battery
#undef BATTERY_AREA_HD

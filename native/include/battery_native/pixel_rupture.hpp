#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#ifdef __CUDACC__
#define BATTERY_PIXEL_HD __host__ __device__
#else
#define BATTERY_PIXEL_HD
#endif
namespace battery {
// Stateless, seeded per physical channel identifier and local pixel index.
// Draw once during mesh upload: retries and CUDA tiling never consume a stream.
inline std::uint64_t pixel_channel_key(const std::string& id) {
    std::uint64_t h=14695981039346656037ULL;
    for(unsigned char c:id) {h^=c;h*=1099511628211ULL;}
    return h;
}
BATTERY_PIXEL_HD inline std::uint64_t pixel_mix(std::uint64_t z) {
    z+=0x9e3779b97f4a7c15ULL;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}
inline double pixel_failure_hazard(std::uint64_t seed,std::uint64_t channel,std::uint64_t pixel) {
    const auto bits=pixel_mix(seed^pixel_mix(channel)^pixel_mix(pixel));
    const double u=(double(bits>>12)+.5)*0x1p-52; // strictly inside (0,1)
    return -std::log(u);
}
BATTERY_PIXEL_HD inline double pixel_failure_rate(double dp,double critical,double rate,double exponent) {
    return dp>critical ? rate*pow(dp/critical-1.,exponent) : 0.;
}
BATTERY_PIXEL_HD inline bool pixel_hazard_reached(double accumulated,double threshold) {
    return accumulated>=threshold-1e-12*fmax(1.,threshold);
}
BATTERY_PIXEL_HD inline double pixel_event_dt(double accumulated,double threshold,double rate) {
    return rate>0 && !pixel_hazard_reached(accumulated,threshold)?(threshold-accumulated)/rate:1e300;
}
} // namespace battery
#undef BATTERY_PIXEL_HD

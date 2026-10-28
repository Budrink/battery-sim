#pragma once
#include <cmath>
#ifdef __CUDACC__
#define TSP_HD __host__ __device__
#else
#define TSP_HD
#endif
namespace battery {
// Accuracy estimates may be relaxed, explicit conduction stability may NOT.
// A shorter externally requested interval is an event/output/end alignment,
// not a stiff-process adaptive retry. The caller records these separately.
TSP_HD inline double bounded_time_step(double requested,double hard_limit,double cap,double floor) {
    const double lower=fmin(cap,floor);
    if(hard_limit<lower*(1.-1e-12))return -1.;
    return fmin(cap,fmax(requested,lower));
}
// lower_bound belongs to the ORIGINAL external interval, never a retry cap.
// Otherwise a .95 safety margin can masquerade as a short external alignment.
TSP_HD inline double retry_time_cap(double cap,double proposed,double lower_bound) {
    return fmin(cap,fmax(lower_bound,.95*proposed));
}
// Equal-sized remaining intervals avoid a tiny final transport remainder.
TSP_HD inline double balanced_transport_step(double remaining,double desired,double floor) {
    if(!(floor>0.))return fmin(remaining,desired);
    const double maximum_count=fmax(1.,::floor(remaining/floor+1e-12));
    const double count=fmin(maximum_count,fmax(1.,ceil(remaining/fmax(desired,floor))));
    return remaining/count;
}
}
#undef TSP_HD

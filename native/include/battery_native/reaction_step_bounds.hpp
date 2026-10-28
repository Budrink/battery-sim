#pragma once
#include <cmath>
#ifdef __CUDACC__
#define RSB_HD __host__ __device__
#else
#define RSB_HD
#endif
namespace battery {
// Accuracy bound for A -> B -> products with frozen per-step rates. Newly
// formed B cannot complete faster than an equal stock present at interval start.
// For any accepted h <= horizon:
// reacted_fresh <= feed_rate*h*(1-exp(-k*h)) <= feed_rate*h*bound_fraction.
// This changes ONLY timestep selection, never reaction heat or inventories.
RSB_HD inline double fresh_completion_bound(double rate,double horizon) {
    return -expm1(-rate*fmax(0.,horizon));
}
RSB_HD inline double completion_rate_bound(double rate,double stock,double feed,double horizon) {
    return rate*stock+feed*fresh_completion_bound(rate,horizon);
}
}
#undef RSB_HD

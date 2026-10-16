#pragma once
#include <cmath>
#ifdef __CUDACC__
#define BATTERY_BUDGET_HD __host__ __device__
#else
#define BATTERY_BUDGET_HD
#endif
namespace battery {
struct CappedDemand { double rate,cap; };
// Largest admissible t for linear*t + sum(min(rate_i*t, cap_i)) <= budget.
// Demands and budget are nonnegative. Start with the raw-rate lower bound,
// remove saturated demands, then solve the remaining affine segment. Each
// increase can saturate another demand; at most count+1 passes are necessary.
// This avoids both a dt-dependent preliminary flow and unnecessarily tiny
// raw-orifice timesteps near pressure equilibrium. No iterative state mutation.
template<class Demands>
BATTERY_BUDGET_HD double capped_budget_dt(double budget,double linear,int count,const Demands& demands) {
    double rate=linear;
    for(int j=0;j<count;++j)rate+=demands(j).rate;
    if(rate<=0)return 1e300;
    double dt=budget/rate;
    for(int pass=0;pass<=count;++pass) {
        double spent=0;rate=linear;
        for(int j=0;j<count;++j) {
            const auto d=demands(j);
            if(d.rate>0 && dt>=d.cap/d.rate)spent+=d.cap;
            else rate+=d.rate;
        }
        if(rate<=0)return 1e300;
        const double next=fmax(0.,budget-spent)/rate;
        if(next<=dt)return dt;
        dt=next;
    }
    return dt;
}
} // namespace battery
#undef BATTERY_BUDGET_HD

#pragma once
#include <cmath>
#ifdef __CUDACC__
#define BATTERY_RADIATION_HD __host__ __device__
#else
#define BATTERY_RADIATION_HD
#endif

namespace battery {
inline constexpr double stefan_boltzmann_w_m2_k4=5.670374419e-8;

// One ordinary graph edge from a finite solid node to a radiative reservoir.
// B = sigma * emissivity * visible area. The material half-cell is in series,
// not hidden in a fitted emissivity: Ts + R*B*(Ts^4-Tr^4) = Tnode.
// Newton starts above the unique root of this monotone convex equation. A
// fixed iteration count keeps this constitutive evaluation SIMD-friendly.
BATTERY_RADIATION_HD inline double radiative_conductance(double b,double node_k,
    double reservoir_k,double material_r,double temperature_limit_k=4000.) {
    if(b<=0.)return 0.;
    if(node_k<=0. || reservoir_k<=0. || material_r<0. ||
       node_k>temperature_limit_k || reservoir_k>temperature_limit_k)return NAN;
    if(material_r==0.)return b*(node_k+reservoir_k)*(node_k*node_k+reservoir_k*reservoir_k);
    double surface=fmax(node_k,reservoir_k);
    for(int i=0;i<20;++i) {
        const double secant=b*(surface+reservoir_k)*(surface*surface+reservoir_k*reservoir_k);
        const double residual=surface-node_k+material_r*secant*(surface-reservoir_k);
        surface-=residual/(1.+4.*material_r*b*surface*surface*surface);
    }
    const double g=b*(surface+reservoir_k)*(surface*surface+reservoir_k*reservoir_k);
    return g/(1.+material_r*g);
}

// Reciprocal finite-to-finite radiative edge. Both material half cells remain
// in series: Tsa=Ta-Ra*q, Tsb=Tb+Rb*q, q=B*(Tsa^4-Tsb^4). No reservoir and no
// new energy source. Safeguarded Newton operates only within passive bounds.
BATTERY_RADIATION_HD inline double radiative_pair_conductance(double b,double ta,
    double tb,double ra,double rb,double temperature_limit_k=4000.) {
    if(b<=0.)return 0.;
    if(ta<=0. || tb<=0. || ra<0. || rb<0. ||
       ta>temperature_limit_k || tb>temperature_limit_k)return NAN;
    if(ra==0.)return radiative_conductance(b,tb,ta,rb,temperature_limit_k);
    if(rb==0.)return radiative_conductance(b,ta,tb,ra,temperature_limit_k);
    const double delta=ta-tb,r=ra+rb;
    if(delta==0.) {const double g=4.*b*ta*ta*ta;return g/(1.+r*g);}
    double lo=fmin(0.,delta/r),hi=fmax(0.,delta/r),q=0.;
    for(int i=0;i<24;++i) {
        const double a=ta-ra*q,c=tb+rb*q;
        const double g=b*(a+c)*(a*a+c*c);
        const double f=q-g*(a-c);
        if(f==0.)break;
        if(f>0.)hi=q;else lo=q;
        const double next=q-f/(1.+4.*b*(ra*a*a*a+rb*c*c*c));
        q=(next>=lo && next<=hi)?next:.5*(lo+hi);
    }
    // Evaluating the secant avoids dividing two small numbers near equality.
    const double a=ta-ra*q,c=tb+rb*q,g=b*(a+c)*(a*a+c*c);
    return g/(1.+r*g);
}
} // namespace battery
#undef BATTERY_RADIATION_HD

#pragma once
#include "battery_native/kawasaki_gas.hpp"
#include <cmath>
#ifdef __CUDACC__
#define KG_HD __host__ __device__
#else
#define KG_HD
#endif
namespace battery::kawasaki {
KG_HD inline double temperature(const Node& n,const Settings& s) {return n.energy/(n.solid_capacity+s.cv*n.mass);}
KG_HD inline double pressure(const Node& n,const Settings& s) {return n.mass*s.R*temperature(n,s)/n.volume;}
KG_HD inline double entropy(const Node& n,const Settings& s) {
    // Reference constants cancel because total mass/material is unchanged.
    return (n.solid_capacity+s.cv*n.mass)*log(temperature(n,s))+n.mass*s.R*log(n.volume/n.mass);
}
KG_HD inline std::uint64_t mix(std::uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
KG_HD inline double uniform(std::uint64_t seed,int edge,std::uint64_t event,int draw) {
    auto key=mix(seed^0x62d4157b39c81a05ULL);
    key=mix(key^static_cast<std::uint64_t>(edge));key=mix(key^event);
    const auto bits=mix(key^static_cast<std::uint64_t>(draw))>>12;
    return (static_cast<double>(bits)+.5)*0x1p-52; // strictly (0,1), no mutable thread RNG
}
KG_HD inline double packet_energy(const Node& donor,double dm,const Settings& s) {
    const double c=donor.solid_capacity+s.cv*donor.mass;
    return -donor.energy*expm1((1+s.R/s.cv)*log1p(-s.cv*dm/c));
}
KG_HD inline double cap(const Node& donor,const Node& receiver,const Settings& s) {
    const double pa=pressure(donor,s),pb=pressure(receiver,s);
    if(pa<=pb)return 0;
    if(donor.solid_capacity==0 && receiver.solid_capacity==0) {
        const double relative=(pb-pa)/pa*receiver.volume/(donor.volume+receiver.volume);
        return -donor.mass*expm1(log1p(relative)/(1+s.R/s.cv));
    }
    // Constant-capacity LTE case: safeguarded Newton on the same monotone root.
    // The bracket remains the fallback; this does not approximate material cp.
    double lo=0,hi=donor.mass*(1-1e-12),dm=0;
    for(int k=0;k<64;++k) {
        const double q=packet_energy(donor,dm,s);
        auto a=donor,b=receiver;a.mass-=dm;a.energy-=q;b.mass+=dm;b.energy+=q;
        const double ta=temperature(a,s),tb=temperature(b,s);
        const double ca=a.solid_capacity+s.cv*a.mass,cb=b.solid_capacity+s.cv*b.mass;
        const double gap=pressure(a,s)-pressure(b,s);
        const double slope=s.R*ta/a.volume*(1+a.mass*s.R/ca)+
            s.R/b.volume*(tb+b.mass*((s.cv+s.R)*ta-s.cv*tb)/cb);
        if(gap>0)lo=dm;else hi=dm;
        if(dm>0 && fabs(gap)<=2e-14*fmax(pa,pb)) {
            // Round toward the donor side. Avoid returning a stale zero lower
            // bracket after convergence from above.
            return fmax(0.,dm-fabs(gap)/slope-8e-15*dm);
        }
        const double proposal=dm+gap/slope;
        const double next=(proposal>lo && proposal<hi)?proposal:(lo+hi)/2;
        if(next==dm || next==lo || next==hi)return lo;
        dm=next;
    }
    return lo;
}
KG_HD inline double flow(const Node& a,const Node& b,const Edge& e,const Settings& s) {
    const double pa=pressure(a,s),pb=pressure(b,s);
    if(pa<=pb || e.area==0 || e.cd==0)return 0;
    const double gamma=1+s.R/s.cv,ratio=pb/pa;
    const double critical=pow(2/(gamma+1),gamma/(gamma-1));
    const double lr=log1p((pb-pa)/pa);
    const double f2=ratio<=critical?gamma*pow(2/(gamma+1),(gamma+1)/(gamma-1)):
        2*gamma/(gamma-1)*exp(2/gamma*lr)*(-expm1((gamma-1)/gamma*lr));
    return e.cd*e.area*pa*sqrt(f2/(s.R*temperature(a,s)));
}
KG_HD inline bool transfer(Node& donor,Node& receiver,double dm,const Settings& s) {
    if(!(dm>0 && dm<donor.mass))return false;
    const double q=packet_energy(donor,dm,s),f=dm/donor.mass;
    const double chemical=f*donor.chemical,oxidizer=f*donor.oxidizer;
    const double before=entropy(donor,s)+entropy(receiver,s);
    auto a=donor,b=receiver;
    a.mass-=dm;a.energy-=q;a.chemical-=chemical;a.oxidizer-=oxidizer;
    b.mass+=dm;b.energy+=q;b.chemical+=chemical;b.oxidizer+=oxidizer;
    const double after=entropy(a,s)+entropy(b,s);
    if(!(a.mass>0 && a.energy>0 && b.mass>0 && b.energy>0) ||
       !(after>=before-1e-12*fmax(1.,fabs(before))))return false;
    donor=a;receiver=b;return true;
}
// Exact random-time-change clock for THIS edge while other edges are frozen.
// Network color splitting still has to converge with duration -> 0.
#ifdef __CUDACC__
// Near equilibrium, F ~ sqrt(dp), dp ~ remaining equalizing mass.
// Integrate dM/dt = -F0 sqrt(M/M0), then use the SAME conservative energy packet.
// This removes stochastic variance only in the explicitly bounded tail region.
__device__ inline int relax_tail(Node& d,Node& r,const Edge& edge,Clock& clock,
    const Settings& s,double equalizing,double duration) {
    const double rate=flow(d,r,edge,s);
    const double fraction=fmin(1.,rate*duration/(2*equalizing));
    const double dm=equalizing*fraction*(2-fraction);
    if(!(dm>0))return 0;
    if(!transfer(d,r,dm,s))return 1;
    ++clock.tails;clock.tail_mass+=dm;
    return 0;
}
__device__ inline int evolve(Node& a,Node& b,const Edge& edge,Clock& clock,const Settings& s,
    double duration,std::uint64_t seed,int id) {
    if(edge.area==0 || edge.cd==0)return 0;
    if(!clock.initialized){clock.hazard=-log(uniform(seed,id,clock.events,0));clock.initialized=true;}
    double left=duration;
    for(int attempt=0;attempt<200000;++attempt) {
        const bool forward=pressure(a,s)>=pressure(b,s);
        Node& d=forward?a:b;Node& r=forward?b:a;
        const double pa=pressure(d,s),pb=pressure(r,s),gap=pa-pb;
        if(gap<=0)return 0;
        const double equalizing=cap(d,r,s);
        if(!(equalizing>0))return 0;
        if(s.tail_atol_pa>0 && gap<=fmin(s.tail_atol_pa,s.tail_rtol*pa))
            return relax_tail(d,r,edge,clock,s,equalizing,left);
        // Without this explicitly bounded closure, sqrt(dp) and ever smaller
        // packets produce infinitely many events at finite equalization time.
        // It is a numerical tolerance, NOT additional physical conductivity.
        if(gap<=s.equilibrium_atol_pa+s.equilibrium_rtol*pa) {
            if(!transfer(d,r,equalizing,s))return 1;
            ++clock.snaps;clock.snapped_mass+=equalizing;return 0;
        }
        const double maximum=s.packet_fraction*equalizing;
        const double rate=2*flow(d,r,edge,s)/maximum;
        if(rate==0)return 0;
        if(!(rate>0 && rate<1e300))return 2;
        const double wait=clock.hazard/rate;
        if(wait>left) {clock.hazard-=rate*left;return 0;}
        if(!(left-wait<left) && wait>0)return 3;
        left-=wait;
        const double dm=maximum*uniform(seed,id,clock.events,1);
        if(!transfer(d,r,dm,s))return 1;
        ++clock.events;clock.hazard=-log(uniform(seed,id,clock.events,0));
        if(left<=0)return 0;
    }
    return 4; // never silently truncate a saturated event loop
}
#endif
} // namespace battery::kawasaki
#undef KG_HD

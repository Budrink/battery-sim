#pragma once
#include <cmath>

#ifdef __CUDACC__
#define BRT_HD __host__ __device__
#else
#define BRT_HD
#endif
namespace battery::reactive_transit {
// Exact frozen-coefficient stirred-volume transit. This is numerical
// integration of the existing R3 law, NOT a new chemical reaction or time fit.
// In dimensionless time x=t/dt:
//   dF/dx=Fin-h*F-s*min(F,O), dO/dx=Oin-h*O-s*min(F,O).
// O is oxygen expressed in its chemical-energy equivalent (kg * J/kg).
// h is the carrier turnover exposure, s=k(T)*dt. There is at most ONE switch
// of the limiting constituent, so two analytical segments suffice.
BRT_HD inline double phi(double x) {
    if(x<1e-5)return 1.-x/2+x*x/6-x*x*x/24;
    return -expm1(-x)/x;
}
BRT_HD inline double psi(double x) {
    if(x<1e-4)return .5-x/6+x*x/24-x*x*x/120;
    return (1.-phi(x))/x;
}
struct Pair {
    double fuel=0., oxygen=0., fuel_out=0., oxygen_out=0., heat=0.;
};
BRT_HD inline void segment(Pair& p,double fin,double oin,double h,double s,double length) {
    const double z=p.fuel-p.oxygen,zin=fin-oin;
    const bool fuel_limited=z<0. || (z==0. && zin<=0.);
    const double stock=fmin(p.fuel,p.oxygen),source=fuel_limited?fin:oin;
    const double a=h+s;
    const double li=stock*length*phi(a*length)+source*length*length*psi(a*length);
    const double zi=z*length*phi(h*length)+zin*length*length*psi(h*length);
    const double next_l=stock*exp(-a*length)+source*length*phi(a*length);
    const double next_z=z*exp(-h*length)+zin*length*phi(h*length);
    p.heat+=s*li;
    p.fuel_out+=h*(li+(fuel_limited?0.:zi));
    p.oxygen_out+=h*(li+(fuel_limited?-zi:0.));
    p.fuel=next_l+fmax(0.,next_z);p.oxygen=next_l+fmax(0.,-next_z);
}
BRT_HD inline Pair integrate(double fuel,double oxygen,double fin,double oin,double h,double s) {
    Pair p{fuel,oxygen};
    const double z=fuel-oxygen,zin=fin-oin;
    const double zend=z*exp(-h)+zin*phi(h);
    double split=1.;
    if(z*zend<0.)split=h>1e-12?log1p(-h*z/zin)/h:-z/zin;
    if(split>0. && split<1.) {
        segment(p,fin,oin,h,s,split);
        // The difference is exactly zero at the analytically located switch.
        // Its next sign is selected by the incoming constituent imbalance.
        const double equal=.5*(p.fuel+p.oxygen);p.fuel=p.oxygen=equal;
        segment(p,fin,oin,h,s,1.-split);
    }else segment(p,fin,oin,h,s,1.);
    return p;
}
// The transported carrier is fixed by the accepted pressure-network mass
// budget. Solve its analytic turnover to match THAT mass, not an arbitrary
// residence time. Never change dt or invent a smaller physical substep.
BRT_HD inline double carrier_out(double initial,double incoming,double h) {
    return initial*(-expm1(-h))+incoming*(1.-phi(h));
}
BRT_HD inline bool exposure(double initial,double incoming,double outgoing,double& h) {
    if(outgoing==0.){h=0.;return true;}
    const double final_mass=initial+incoming-outgoing;
    if(!(initial>0. && incoming>=0. && outgoing>0. && final_mass>0.))return false;
    if(incoming==outgoing){h=outgoing/initial;return true;}
    if(incoming==0.){h=-log1p(-outgoing/initial);return true;}
    double lower=0.,upper=2*outgoing/fmin(initial,final_mass);
    h=2*outgoing/(initial+final_mass);
    for(int iteration=0;iteration<20;++iteration) {
        const double residual=carrier_out(initial,incoming,h)-outgoing;
        if(fabs(residual)<=2e-13*(initial+incoming))return true;
        if(residual>0.)upper=h;else lower=h;
        const double derivative=initial*exp(-h)+incoming*(phi(h)-exp(-h))/h;
        const double proposed=h-residual/derivative;
        h=proposed>lower && proposed<upper?proposed:.5*(lower+upper);
    }
    return fabs(carrier_out(initial,incoming,h)-outgoing)<=1e-11*(initial+incoming);
}
} // namespace battery::reactive_transit
#undef BRT_HD

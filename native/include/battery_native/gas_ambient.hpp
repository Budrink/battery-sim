#pragma once
#include "battery_native/kawasaki_gas_math.hpp"
#include "battery_native/model.hpp"
#include <cfloat>

#ifdef __CUDACC__
#define AMBIENT_HD __host__ __device__
#else
#define AMBIENT_HD
#endif
namespace battery::gas_ambient {
namespace kg=kawasaki;
struct Transfer {
    double mass=0., energy=0., chemical=0., oxidizer=0.; // net outward
    double gross_in=0., gross_out=0., entropy=0.;
};
AMBIENT_HD inline kg::Node reservoir(const GasAmbientBoundary& b,const kg::Settings& s){
    const double mass=b.pressure_pa/(s.R*b.temperature_k);
    return {mass,mass*s.cv*b.temperature_k,1.,0.,0.,mass*b.oxidizer_mass_fraction};
}
// Implicit compressible orifice, bracketed by EXACT pressure equalization.
// Bisection is algebraic work, not hidden physical timesteps. Adiabatic outflow
// carries cp*T enthalpy; inflow carries the ambient donor's cp*T.
AMBIENT_HD inline void pressure_exchange(kg::Node& n,const GasAmbientBoundary& b,
    const kg::Settings& s,double dt,Transfer& transfer){
    if(!(b.area_m2>0. && b.discharge_coefficient>0. && dt>0.))return;
    const double p=kg::pressure(n,s);if(p==b.pressure_pa)return;
    const bool outward=p>b.pressure_pa;
    const double gamma=1.+s.R/s.cv;
    const double cap=outward?-n.mass*expm1(log(b.pressure_pa/p)/gamma):
        (b.pressure_pa*n.volume/s.R-n.energy/s.cv)/(gamma*b.temperature_k);
    if(!(cap>0.))return;
    kg::Edge edge;edge.area=b.area_m2;edge.cd=b.discharge_coefficient;
    const auto bath=reservoir(b,s);double low=0.,high=cap;
    for(int k=0;k<48;++k){const double dm=(low+high)/2.;auto probe=n;
        if(outward){probe.mass-=dm;probe.energy-=kg::packet_energy(n,dm,s);}
        else {probe.mass+=dm;probe.energy+=(s.cv+s.R)*b.temperature_k*dm;}
        const double flow=outward?kg::flow(probe,bath,edge,s):kg::flow(bath,probe,edge,s);
        if(dm<dt*flow)low=dm;else high=dm;
    }
    const double dm=(low+high)/2.;
    if(outward){const double remaining=1.-dm/n.mass;
        n.energy-=kg::packet_energy(n,dm,s);n.mass-=dm;
        n.chemical*=remaining;n.oxidizer*=remaining;n.vapor*=remaining;n.latent*=remaining;
        transfer.gross_out+=dm;
    }else{
        n.mass+=dm;n.energy+=(s.cv+s.R)*b.temperature_k*dm;
        n.oxidizer+=b.oxidizer_mass_fraction*dm;transfer.gross_in+=dm;
    }
}
// Throughflow/entrainment at an OPEN external surface, not a heat-only sink.
// Equal mass enters/leaves; composition relaxes at mdot/m and sensible energy
// at gamma*mdot/m because exhaust/inlet transport ENTHALPY, not cv*T packets.
AMBIENT_HD inline void renew(kg::Node& n,const GasAmbientBoundary& b,
    const kg::Settings& s,double dt,Transfer& transfer){
    const double mdot=b.renewal_flow_m3_s*b.pressure_pa/(s.R*b.temperature_k);
    if(!(mdot>0. && dt>0.))return;
    const double turnover=mdot*dt/n.mass;
    const double fraction=-expm1(-turnover),thermal=-expm1(-(1.+s.R/s.cv)*turnover);
    n.energy+=(n.mass*s.cv*b.temperature_k-n.energy)*thermal;
    n.chemical*=1.-fraction;n.oxidizer+=(n.mass*b.oxidizer_mass_fraction-n.oxidizer)*fraction;
    n.vapor*=1.-fraction;n.latent*=1.-fraction;
    transfer.gross_in+=mdot*dt;transfer.gross_out+=mdot*dt;
}
AMBIENT_HD inline bool apply(kg::Node& n,const GasAmbientBoundary& b,
    const kg::Settings& s,double reference_k,double dt,Transfer& out){
    out={};if(!b.enabled || !(dt>0.) || (b.area_m2==0. && b.renewal_flow_m3_s==0.))return true;
    if(n.solid_capacity!=0.)return false;
    const auto before=n;
    pressure_exchange(n,b,s,dt/2.,out);renew(n,b,s,dt,out);pressure_exchange(n,b,s,dt/2.,out);
    out.mass=before.mass-n.mass;out.chemical=before.chemical-n.chemical;out.oxidizer=before.oxidizer-n.oxidizer;
    // Same reference convention as native export; latent travels with vapor.
    out.energy=before.mass*s.cv*(kg::temperature(before,s)-reference_k)+before.latent-
        (n.mass*s.cv*(kg::temperature(n,s)-reference_k)+n.latent);
    const double absolute_out=before.energy-n.energy;
    const double sa=s.cv*log(b.temperature_k)+s.R*log(s.R*b.temperature_k/b.pressure_pa);
    out.entropy=kg::entropy(n,s)-kg::entropy(before,s)+absolute_out/b.temperature_k-
        (s.cv+s.R-sa)*out.mass;
    return std::isfinite(n.energy) && n.energy>0. && n.mass>0. && n.chemical>=0. && n.oxidizer>=0. &&
        n.vapor>=0. && n.oxidizer+n.vapor<=n.mass+1e-14 && n.latent>=0. &&
        out.entropy>=-1e-10*fmax(1.,fabs(kg::entropy(before,s)));
}
} // namespace battery::gas_ambient
#undef AMBIENT_HD

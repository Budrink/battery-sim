#pragma once
#include "battery_native/kawasaki_gas_math.hpp"
#ifdef __CUDACC__
#define GM_HD __host__ __device__
#else
#define GM_HD
#endif
namespace battery {
struct MixingTransfer {
    double mass_each_way=0.,energy=0.,chemical=0.,oxidizer=0.,vapor=0.,latent=0.;
};
GM_HD inline double gas_effective_diffusivity(const kawasaki::Node& a,const kawasaki::Node& b,
    const kawasaki::Edge& edge,const kawasaki::Settings& settings,double molecular) {
    if(edge.molecular_mixing || settings.thermal_stirring_coefficient==0.)return molecular;
    const double ta=kawasaki::temperature(a,settings),tb=kawasaki::temperature(b,settings);
    // Physical eddy size is NOT the numerical centre-to-centre distance.
    // Evaluate the resolved gradient over that fixed physical length. For a
    // linear T field, h-refinement then preserves D rather than making it ~h^2.
    const double length=fmin(settings.thermal_stirring_length_m,settings.mixing_length_cap_m);
    const double contrast=fmin(1.,length*fabs(ta-tb)/(edge.mixing_length*fmax(.5*(ta+tb),1.)));
    const double velocity=sqrt(9.81*length*contrast);
    return molecular+settings.thermal_stirring_coefficient*length*velocity;
}
// A solid-coupled endpoint does not make an opened exterior aperture a pore.
// Only a passage wholly inside the same condensed object retains its internal
// molecular reference. The existing open-area gate still blocks closed shells.
GM_HD inline bool internal_porous_mixing(bool free_gas_only,bool rupture,int object_a,int object_b) {
    return free_gas_only && !rupture && object_a>=0 && object_a==object_b;
}
GM_HD inline double reference_mixing_rate(const kawasaki::Node& a,const kawasaki::Node& b,
    const kawasaki::Edge& edge,double diffusivity,double entrainment,double pressure_mass_flow) {
    if(!(edge.area>0. && edge.cd>0.))return 0.;
    const double ra=a.mass/a.volume,rb=b.mass/b.volume;
    return diffusivity*edge.area*(2*ra*rb/(ra+rb))/edge.mixing_length+entrainment*fabs(pressure_mass_flow);
}
// One conservative counterexchange packet carries sensible heat AND composition.
// Select the hotter gas as the ordered S source; never average Sij and Sji.
// Density and aperture determine the thermal reference; current gas temperatures
// determine S even at dP=0. Pressure-flow entrainment is a separate additive
// contribution: do not multiply its already specified coefficient by S again.
GM_HD inline double mixing_conductance(const kawasaki::Node& a,const kawasaki::Node& b,
    const kawasaki::Edge& edge,const kawasaki::Settings& settings,
    double diffusivity,double entrainment,double pressure_mass_flow) {
    if(!(edge.area>0. && edge.cd>0.))return 0.;
    const double mechanical=entrainment*fabs(pressure_mass_flow);
    const double rate=reference_mixing_rate(a,b,edge,gas_effective_diffusivity(a,b,edge,settings,diffusivity),0.,0.);
    if(!(rate>0.))return mechanical;
    // G(S)/Gunit is dimensionless. Keeping the reference evaluation separate
    // also makes a neutral profile EXACTLY the original rate (no cv roundtrip).
    const auto state=edge.molecular_mixing?
        evaluate_directed_geometry(reference_directed_connectivity(1.),{},edge.geometry_axis,
            kawasaki::temperature(a,settings),kawasaki::temperature(b,settings),0.,0.,0.,1.,0.):
        evaluate_directed_geometry(settings.mixing_connectivity,{},edge.geometry_axis,
            kawasaki::temperature(a,settings),kawasaki::temperature(b,settings),0.,0.,0.,1.,0.);
    return mechanical+rate*state.conductance_w_per_k;
}
// Safe envelope for kinetic arrival accuracy controls, NOT an applied flux.
GM_HD inline double maximum_mixing_conductance(const kawasaki::Node& a,const kawasaki::Node& b,
    const kawasaki::Edge& edge,const kawasaki::Settings& settings,
    double diffusivity,double entrainment,double pressure_mass_flow) {
    const auto& law=settings.mixing_connectivity;
    if(!(edge.area>0. && edge.cd>0.))return 0.;
    return reference_mixing_rate(a,b,edge,gas_effective_diffusivity(a,b,edge,settings,diffusivity),0.,0.)*
        (edge.molecular_mixing?1.99:law.background_conductance_w_per_k+law.active_conductance_w_per_k)+entrainment*fabs(pressure_mass_flow);
}
GM_HD inline double mixing_request(const kawasaki::Node& a,const kawasaki::Node& b,
    const kawasaki::Edge& edge,const kawasaki::Settings& settings,double diffusivity,
    double entrainment,double pressure_mass_flow,double dt) {
    if(!(edge.area>0. && edge.cd>0. && dt>0.))return 0.;
    const double conductance=mixing_conductance(a,b,edge,settings,diffusivity,entrainment,pressure_mass_flow);
    const double reduced=a.mass*b.mass/(a.mass+b.mass);
    // Exact frozen-coefficient pairwise mixing; approaches equilibrium, cannot
    // exchange an arbitrary fraction of the gas just because dt is large.
    return reduced*(-expm1(-conductance*dt/reduced));
}
GM_HD inline MixingTransfer mixing_transfer(const kawasaki::Node& a,const kawasaki::Node& b,
    const kawasaki::Settings& settings,double mass) {
    return {mass,mass*settings.cv*(kawasaki::temperature(a,settings)-kawasaki::temperature(b,settings)),
        mass*(a.chemical/a.mass-b.chemical/b.mass),mass*(a.oxidizer/a.mass-b.oxidizer/b.mass),
        mass*(a.vapor/a.mass-b.vapor/b.mass),mass*(a.latent/a.mass-b.latent/b.mass)};
}
}
#undef GM_HD

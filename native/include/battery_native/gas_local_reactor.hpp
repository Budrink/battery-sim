#pragma once
#include "battery_native/gas_combustion.hpp"
#ifdef __CUDACC__
#define GLR_HD __host__ __device__
#else
#define GLR_HD
#endif
namespace battery {
// Numerical scratch only. No material heat capacity or chemical law is changed.
struct GasFilmState {
    double conductance=0., wall_temperature=0., receiving_capacity=0.;
    double integrated_temperature=0., heat_to_solid=0.;
};
// Exact frozen-k solution of C dT/dt = k L exp(-kt)-G(T-Twall).
// It includes cooling DURING combustion, not an after-the-fact temperature cap.
GLR_HD inline void frozen_gas_reactor(kawasaki::Node& n,const GasCombustionLaw& law,
    const kawasaki::Settings& settings,double dt,GasFilmState& film) {
    const double capacity=n.solid_capacity+settings.cv*n.mass;
    const double t=kawasaki::temperature(n,settings),a=film.conductance/capacity;
    const double k=combustion_rate(law,t);
    const double stock=fmax(0.,fmin(n.chemical,n.oxidizer*law.oxidizer_heat_j_per_kg));
    const double heat=stock*(-expm1(-k*dt));
    const double delta=fabs(a-k);
    const double convolution=exp(-fmin(a,k)*dt)*(delta>0.?-expm1(-delta*dt)/delta:dt);
    const double next=film.wall_temperature+(t-film.wall_temperature)*exp(-a*dt)+k*stock/capacity*convolution;
    const double loss=capacity*(t-next)+heat;
    if(settings.vapor_fuel_heat_j_kg>0. && n.chemical>0.)n.vapor*=1-heat/n.chemical;
    n.chemical=fmax(0.,n.chemical-heat);
    n.oxidizer=fmax(0.,n.oxidizer-heat/law.oxidizer_heat_j_per_kg);
    n.energy=capacity*next;n.reaction_released_j+=heat;
    n.reaction_oxidizer_kg+=heat/law.oxidizer_heat_j_per_kg;
    film.heat_to_solid+=loss;
    film.integrated_temperature+=film.wall_temperature*dt+loss/film.conductance;
}
GLR_HD inline bool advance_gas_reactor(kawasaki::Node& n,const GasCombustionLaw& law,
    const kawasaki::Settings& settings,double duration,GasFilmState& film,
    double temperature_tolerance=.05,double fraction=.02,unsigned* iteration_count=nullptr) {
    if(iteration_count)*iteration_count=0;
    if(!(film.conductance>0.))return burn_gas(n,law,settings,duration,fraction,temperature_tolerance);
    if(!(fmin(n.chemical,n.oxidizer*law.oxidizer_heat_j_per_kg)>0.) ||
        !(law.rate_per_s>0.) || law.activation_j_per_mol==0.) {
        frozen_gas_reactor(n,law,settings,duration,film);return n.energy>0.;
    }
    double elapsed=0.,h=duration;unsigned iterations=0;
    while(elapsed<duration) {
        h=fmin(h,duration-elapsed);
        if(!(h>0.) || elapsed+h==elapsed || ++iterations>200000)return false;
        if(iteration_count)*iteration_count=iterations;
        auto full=n,half=n;auto f1=film,f2=film;
        frozen_gas_reactor(full,law,settings,h,f1);
        frozen_gas_reactor(half,law,settings,h/2,f2);
        frozen_gas_reactor(half,law,settings,h/2,f2);
        const double stock=fmax(0.,fmin(n.chemical,n.oxidizer*law.oxidizer_heat_j_per_kg));
        const double error=fmax(fabs(kawasaki::temperature(full,settings)-kawasaki::temperature(half,settings))/temperature_tolerance,
            fabs(full.chemical-half.chemical)/(1e-10+.001*stock));
        if(!isfinite(error) || !(half.energy>0.))return false;
        if(error>1.) {h*=fmax(.1,.8/sqrt(error));continue;}
        n=half;film=f2;elapsed+=h;
        h*=error>0.?fmin(2.,.9/sqrt(error)):2.;
    }
    return true;
}
// Pressure/transport splitting accuracy is distinct from local ODE accuracy.
// A gas packet has tiny capacity; bound its coupling to the receiving solids
// and at most 2% of gas temperature, while the ODE above stays much tighter.
GLR_HD inline double gas_coupling_rise(const kawasaki::Node& n,const kawasaki::Settings& settings,
    const GasFilmState& film,double base) {
    const double capacity=n.solid_capacity+settings.cv*n.mass;
    return base*fmax(1.,fmin(.02*kawasaki::temperature(n,settings),.1*film.receiving_capacity/capacity));
}
} // namespace battery
#undef GLR_HD

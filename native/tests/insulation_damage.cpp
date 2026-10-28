#include "battery_native/state_geometry.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
ConnectivityLaw law() {
    auto l=reference_directed_connectivity(500.);
    l.endpoint_damage_weight=0.;
    l.geometry_reference_logit=std::log(70./920.);
    l.interface_damage_log_response=std::log(945./45.)-l.geometry_reference_logit;
    l.damage_onset_k=423.15;l.damage_rate_at_onset_per_s=1e-6;l.damage_temperature_scale_k=5.;
    l.destroyed_background_conductance_w_per_k=950.;l.destroyed_active_conductance_w_per_k=0.;
    return l;
}
double ramp(double dt) {
    auto l=law();InterfaceState s;
    for(int i=0;i<static_cast<int>(800./dt);++i) {
        const double t=423.15+.1*i*dt;
        s=evaluate_directed_geometry(l,s,0,t,t,0,0,dt,1.,0.);
        if(s.damage>=1.)return t-273.15;
    }
    throw std::runtime_error("Ramp never destroyed insulation");
}
int main(){try {
    auto l=law();InterfaceState s;
    s=evaluate_directed_geometry(l,s,0,298.15,298.15,0,0,100.,1.,0.);
    require(s.damage==0. && std::abs(s.conductance_w_per_k-75.)<1e-10,"Cold insulation changed");
    double previous=75.;
    for(int i=0;i<1000;++i) {
        s=evaluate_directed_geometry(l,s,0,473.15,473.15,0,0,.01,1.,0.);
        require(s.conductance_w_per_k>=previous,"Damage decreases conductance");previous=s.conductance_w_per_k;
    }
    const auto partial=s;
    s=evaluate_directed_geometry(l,s,0,298.15,298.15,0,0,10000.,1.,0.);
    require(s.damage==partial.damage && s.conductance_w_per_k==partial.conductance_w_per_k,"Partial insulation healed");
    for(int i=0;i<5000;++i)s=evaluate_directed_geometry(l,s,0,473.15,473.15,0,0,.01,1.,0.);
    require(s.damage==1. && s.phase==EdgePhase::Destroyed,"Hot hold did not destroy insulation");
    s=evaluate_directed_geometry(l,s,0,298.15,298.15,0,0,10000.,1.,0.);
    require(s.damage==1. && std::abs(s.conductance_w_per_k-950.)<1e-10,"Destroyed insulation healed");
    const double coarse=ramp(.1),fine=ramp(.01);
    require(std::abs(coarse-fine)<.03 && std::abs(fine-199.52)<.1,"Ramp threshold or timestep convergence failed");
    std::cout<<"irreversible partial/full damage, monotone conductance, ramp convergence pass: "<<coarse<<" / "<<fine<<" C\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

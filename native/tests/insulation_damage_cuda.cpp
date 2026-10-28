#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
int main(){try {
    Scenario s;s.settings.duration_s=60.;s.settings.max_dt_s=.05;
    s.settings.output_interval_s=1.;s.settings.stability_factor=.2;
    CellConfig c;c.id="B";c.dimensions_m={.01,.03,.05};
    c.heat_capacity_j_per_k=100.;c.initial_temperature_k=473.15;
    c.thermal_divisions={1,1,1};c.conductivity_w_per_m_k={1,1,1};s.cells={c};
    ShellConfig sh;sh.contents_id="B";sh.material_id="test-shell";sh.thickness_m=.001;
    sh.density_kg_m3=2000.;sh.specific_heat_j_kg_k=800.;sh.conductivity_w_per_m_k={2,3,4};
    sh.initial_temperature_k=473.15;sh.contents_contact_h=500.;sh.seam_contact_h=500.;
    add_physical_shell(s,sh);
    for(auto& e:s.interfaces)if(e.cell_a==0) {
        const double area=e.law.background_conductance_w_per_k/5.;
        e.has_endpoint_response=true;
        auto& l=e.law;l.geometry_reference_logit=std::log(70./920.);
        l.interface_damage_log_response=std::log(945./45.)-l.geometry_reference_logit;
        l.damage_onset_k=423.15;l.damage_rate_at_onset_per_s=1e-6;l.damage_temperature_scale_k=5.;
        l.destroyed_background_conductance_w_per_k=950.*area;l.destroyed_active_conductance_w_per_k=0.;
    }
    Simulation sim(s);sim.run_to_end();
    int destroyed=0;const auto states=sim.thermal_link_states();const auto& mesh=sim.thermal_mesh();
    for(std::size_t i=0;i<mesh.links.size();++i) {
        const auto& e=mesh.links[i];if(e.interface_index<0)continue;
        const auto& config=s.interfaces[e.interface_index];const auto& state=states[i];
        if(config.cell_a==0) {
            if(state.damage!=1. || state.phase!=EdgePhase::Destroyed)
                throw std::runtime_error("CUDA contact did not retain destruction");
            const double g=config.law.destroyed_background_conductance_w_per_k*e.weight;
            const double expected=1./(1./g+e.series_resistance_k_per_w);
            if(std::abs(state.conductance_w_per_k-expected)>1e-10)
                throw std::runtime_error("CUDA ignores destroyed insulation conductance");
            ++destroyed;
        }else if(state.damage!=0.)throw std::runtime_error("Insulation changed seam damage");
    }
    if(destroyed!=6 || std::abs(sim.energy_balance_residual_j())>1e-8)
        throw std::runtime_error("CUDA insulation count/energy balance failed");
    std::cout<<"CUDA: six insulation contacts destroyed, seams unchanged, energy conserved\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

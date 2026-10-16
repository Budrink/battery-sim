// Read-only depth diagnostics. Integration uses the unchanged native CUDA core.
#include "battery_native/simulation.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using J=nlohmann::json;

int main(int argc,char** argv) {try {
    if(argc!=2)throw std::runtime_error("usage: battery_sim_profile_probe SCENARIO");
    const auto scenario=battery::load_scenario(argv[1]);
    for(const auto& c:scenario.cells)if(c.is_battery) {
        if(c.has_bulk_connectivity)throw std::runtime_error("Depth diagnostic requires constant battery bulk conductivity");
        if(c.thermal_divisions[0]%2)throw std::runtime_error("Even x divisions required for an exact midplane");
    }
    const auto started=std::chrono::steady_clock::now();battery::Simulation sim(scenario);
    const auto& mesh=sim.thermal_mesh();
    J result={{"time_s",J::array()},{"objects",J::object()},
        {"power_convention","Actual instantaneous Fourier flux across local x=0, positive -x to +x. Null for an isothermal constraint, NOT zero transport."}};
    for(std::size_t object=0;object<scenario.cells.size();++object) {
        const auto& c=scenario.cells[object];if(!c.is_battery)continue;
        const int nx=c.thermal_divisions[0];std::vector<double> centers(nx);
        for(int i=0;i<nx;++i)centers[i]=((i+.5)/nx-.5)*c.dimensions_m[0];
        result["objects"][c.id]={{"divisions",c.thermal_divisions},{"isothermal",c.isothermal},
            {"heat_capacity_j_per_k",c.heat_capacity_j_per_k},{"conductivity_w_per_m_k",c.conductivity_w_per_m_k},
            {"slab_center_x_m",centers},{"slab_temperature_k",J::array()},
            {"left_half_temperature_k",J::array()},{"right_half_temperature_k",J::array()},
            {"mean_temperature_k",J::array()},{"midplane_left_to_right_power_w",J::array()}};
    }
    auto sample=[&] {
        const auto& field=sim.thermal_states();result["time_s"].push_back(sim.time_s());
        for(std::size_t object=0;object<scenario.cells.size();++object) {
            const auto& c=scenario.cells[object];if(!c.is_battery)continue;
            const int nx=c.thermal_divisions[0];std::vector<double> temperature(nx,0),capacity(nx,0);
            double left_energy=0,right_energy=0,left_capacity=0,right_capacity=0;
            for(int j=mesh.object_offsets[object];j<mesh.object_offsets[object+1];++j) {
                const auto& v=mesh.volumes[j];
                const int slab=std::clamp(static_cast<int>(std::floor((v.local_center_m.x/c.dimensions_m[0]+.5)*nx)),0,nx-1);
                const double energy=v.capacity_j_per_k*field[j].temperature_k;
                temperature[slab]+=energy;capacity[slab]+=v.capacity_j_per_k;
                if(v.local_center_m.x<0) {left_energy+=energy;left_capacity+=v.capacity_j_per_k;}
                else {right_energy+=energy;right_capacity+=v.capacity_j_per_k;}
            }
            for(int i=0;i<nx;++i)temperature[i]/=capacity[i];
            auto& track=result["objects"][c.id];track["slab_temperature_k"].push_back(temperature);
            track["left_half_temperature_k"].push_back(left_energy/left_capacity);
            track["right_half_temperature_k"].push_back(right_energy/right_capacity);
            track["mean_temperature_k"].push_back((left_energy+right_energy)/(left_capacity+right_capacity));
            if(c.isothermal) {track["midplane_left_to_right_power_w"].push_back(nullptr);continue;}
            double power=0;int crossings=0;
            for(const auto& edge:mesh.links) {
                if(edge.b<0 || mesh.volumes[edge.a].object!=static_cast<int>(object) ||
                   mesh.volumes[edge.b].object!=static_cast<int>(object))continue;
                const double xa=mesh.volumes[edge.a].local_center_m.x,xb=mesh.volumes[edge.b].local_center_m.x;
                if(xa*xb>=0)continue;
                if(edge.law.kind!=battery::TransitionKind::Constant || edge.law.active_conductance_w_per_k!=0)
                    throw std::runtime_error("Expected constant bulk face");
                const double raw=edge.law.background_conductance_w_per_k*edge.weight;
                const double g=raw>0?1./(1./raw+edge.series_resistance_k_per_w):0.;
                power+=(xa<xb?1.:-1.)*g*(field[edge.a].temperature_k-field[edge.b].temperature_k);++crossings;
            }
            if(crossings!=c.thermal_divisions[1]*c.thermal_divisions[2])throw std::runtime_error("Incomplete midplane cut");
            track["midplane_left_to_right_power_w"].push_back(power);
        }
    };
    sample();double next=scenario.settings.output_interval_s;
    while(!sim.finished()) {
        sim.step(std::max(1e-12,std::min(next,scenario.settings.duration_s)-sim.time_s()));
        if(sim.time_s()>=next-1e-9 || sim.finished()) {sample();next+=scenario.settings.output_interval_s;}
    }
    result["energy_residual_j"]=sim.energy_balance_residual_j();
    result["volumes"]=mesh.volumes.size();result["steps"]=sim.step_count();
    result["wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::cout<<result.dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

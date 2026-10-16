// Diagnostic runner for a CLOSED, fixed-geometry, gas-only network.
// Uses production CUDA transport; never fits material parameters.
#include "battery_native/simulation.hpp"
#include "battery_native/conversion.hpp"
#include "battery_native/gas_timestep.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

using J=nlohmann::json;
using namespace battery;

// Host mirror for diagnostics ONLY (this runner forbids LTE and external sinks).
// Reports what sets the timestep; production decisions remain entirely on CUDA.
J pressure_bottleneck(const Scenario& s,const std::vector<GasState>& gas,const std::vector<GasChannelState>& states) {
    std::vector<int> degree(gas.size());
    for(const auto& c:s.gas_channels){++degree[c.from];++degree[c.to];}
    double best=1e300;J out;const auto& model=s.gas_model;
    for(std::size_t i=0;i<s.gas_channels.size();++i) {
        const auto& c=s.gas_channels[i];const auto& a=gas[c.from];const auto& b=gas[c.to];
        const auto& donor=a.pressure_pa>=b.pressure_pa?a:b;
        const double flow=gas_orifice_flow(std::max(a.pressure_pa,b.pressure_pa),std::min(a.pressure_pa,b.pressure_pa),
            donor.temperature_k,states[i].effective_area_m2,c.discharge_coefficient,model);
        const double slope=(1+model.specific_r_j_per_kg_k/model.cv_j_per_kg_k)*model.specific_r_j_per_kg_k*donor.temperature_k*
            (degree[c.from]/a.free_volume_m3+degree[c.to]/b.free_volume_m3);
        const double cap=std::abs(a.pressure_pa-b.pressure_pa)/slope;
        const double dt=gas_pressure_dt(a.pressure_pa,b.pressure_pa,flow,cap,model.pressure_relaxation_fraction,
            model.pressure_absolute_tolerance_pa,model.pressure_relative_tolerance);
        if(dt<best) {best=dt;out={{"path",c.id},{"dt_s",dt},{"dp_pa",std::abs(a.pressure_pa-b.pressure_pa)},
            {"raw_flow_kg_s",flow},{"from_volume_m3",a.free_volume_m3},{"to_volume_m3",b.free_volume_m3}};}
    }
    return out;
}

void validate_isolated(const Scenario& s) {
    if(!s.gas_model.split_transport || !s.interfaces.empty() || !s.boundaries.empty() ||
       !s.gas_contacts.empty() || !s.primary_failures.empty())
        throw std::runtime_error("Audit requires isolated gas transport");
    for(const auto& c:s.cells)if(c.is_battery || c.conversion.enabled || c.runaway_energy_j!=0 || c.damage_rate_at_onset_per_s!=0)
        throw std::runtime_error("Audit must not contain active solids");
    for(const auto& g:s.gas_compartments)if(g.owner_cell>=0 || g.thermal_cell>=0 || g.reaction_rate_per_s!=0 || g.initial_chemical_j!=0)
        throw std::runtime_error("Audit must not contain LTE/chemistry");
    for(const auto& c:s.gas_channels)if(!c.passage || c.to<0 || c.opening.enabled || c.rupture)
        throw std::runtime_error("Audit requires fixed internal passages");
}

J run(Scenario s,double cap,double mass_budget,double duration,double sample_interval) {
    for(double v:{cap,mass_budget,duration,sample_interval})if(!std::isfinite(v)||v<=0)
        throw std::runtime_error("Positive finite audit controls required");
    s.gas_model.max_fraction_per_step=mass_budget;
    validate_isolated(s);
    auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh);RuntimeParameters runtime;
    const auto started=std::chrono::steady_clock::now();
    J out={{"max_dt_s",cap},{"mass_budget",mass_budget},{"duration_s",duration},
        {"pressure_relaxation_fraction",s.gas_model.pressure_relaxation_fraction},
        {"pressure_absolute_tolerance_pa",s.gas_model.pressure_absolute_tolerance_pa},
        {"pressure_relative_tolerance",s.gas_model.pressure_relative_tolerance},
        {"time_s",J::array()},{"gas",J::object()},{"channels",J::object()}};
    struct Totals {double proposed=0,accepted=0,clipped_time=0;std::size_t clipped_steps=0;};
    std::vector<Totals> totals(s.gas_channels.size());
    std::vector<GasState> gas;std::vector<GasChannelState> channels;
    double t=0,min_dt=1e300,max_dt=0,max_mass_error=0,max_energy_error=0;
    std::size_t steps=0,steps_with_clipping=0;
    auto sample=[&] {
        const auto d=solver.process_diagnostics();out["time_s"].push_back(t);
        max_mass_error=std::max(max_mass_error,std::abs(d.mass_residual_kg));
        max_energy_error=std::max(max_energy_error,std::abs(d.thermal_residual_j));
        for(std::size_t k=0;k<d.gas.size();++k) {
            const auto& g=d.gas[k];auto& row=out["gas"][s.gas_compartments[k].id];
            row["pressure_pa"].push_back(g.pressure_pa);row["mass_kg"].push_back(g.mass_kg);
            row["temperature_k"].push_back(g.temperature_k);
            row["energy_j"].push_back(g.energy_j);
        }
        for(std::size_t k=0;k<d.channels.size();++k)
            out["channels"][s.gas_channels[k].id]["transferred_mass_kg"].push_back(d.channels[k].transferred_kg);
    };
    sample();std::size_t tick=1;
    while(t<duration) {
        const double target=std::min(duration,tick*sample_interval);
        while(target-t>1e-13*duration) {
            const double dt=solver.step(t,std::min(cap,target-t),s.settings.stability_factor,runtime);
            if(t+dt==t || ++steps>2000000)throw std::runtime_error("Audit timestep/count limit");
            solver.download_gas_states(gas,channels);bool clipped=false;
            for(std::size_t k=0;k<channels.size();++k) {
                const auto& c=channels[k];auto& a=totals[k];
                if(std::abs(c.effective_area_m2-s.gas_channels[k].area_m2)>1e-12*s.gas_channels[k].area_m2)
                    throw std::runtime_error("Fixed aperture changed");
                const double proposed=std::abs(c.proposed_mass_flow_kg_s)*dt,accepted=std::abs(c.mass_flow_kg_s)*dt;
                a.proposed+=proposed;a.accepted+=accepted;
                // Ignore roundoff-scale packets when counting clipped steps.
                if(proposed-accepted>std::max(1e-18,proposed*1e-8)) {
                    a.clipped_time+=dt;++a.clipped_steps;clipped=true;
                }
            }
            steps_with_clipping+=clipped;t+=dt;min_dt=std::min(min_dt,dt);max_dt=std::max(max_dt,dt);
            if(std::getenv("BATTERY_SIM_PROGRESS") && steps%10000==0)
                std::cerr<<"gas audit t="<<t<<" / "<<duration<<", steps="<<steps
                         <<", pressure bound "<<pressure_bottleneck(s,gas,channels).dump()<<'\n';
        }
        sample();++tick;
        if(duration-t<=1e-13*duration)break;
    }
    for(std::size_t k=0;k<totals.size();++k) {
        const auto& a=totals[k];out["channels"][s.gas_channels[k].id].update({
            {"proposed_abs_mass_kg",a.proposed},{"accepted_abs_mass_kg",a.accepted},
            {"clipped_time_s",a.clipped_time},{"clipped_steps",a.clipped_steps}});
    }
    out.update({{"steps",steps},{"steps_with_clipping",steps_with_clipping},{"min_actual_dt_s",min_dt},
        {"max_actual_dt_s",max_dt},{"max_mass_residual_kg",max_mass_error},{"max_thermal_residual_j",max_energy_error},
        {"wall_s",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}});
    return out;
}

int main(int argc,char** argv) {try {
    if(argc!=6)throw std::runtime_error("Usage: gas_transport_audit SCENE MAX_DT MASS_BUDGET DURATION SAMPLE_INTERVAL");
    const auto s=load_scenario(argv[1]);
    std::cout<<run(s,std::stod(argv[2]),std::stod(argv[3]),std::stod(argv[4]),std::stod(argv[5])).dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

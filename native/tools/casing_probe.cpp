// General opt-in casing calibration replay. No optimizer or thermal evolution on CPU.
#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include "battery_native/failure.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
using J=nlohmann::json;
struct Probe {std::string id;int edge,object,axis;double area;};
J profile_json(const battery::SolverProfile& p) {
    const auto& g=p.gas;
    return {{"prepare_s",p.prepare_s},{"cpu_gas_s",p.cpu_gas_s},{"download_s",p.download_s},{"upload_s",p.upload_s},
        {"cuda_gas_s",p.cuda_gas_s},{"deterministic_pressure_gap_pa",p.deterministic_pressure_gap_pa},
        {"kawasaki_events",p.kawasaki_events},{"kawasaki_tail_updates",p.kawasaki_tail_updates},
        {"gas_batches",p.gas_batches},{"gas_limited_batches",p.gas_limited_batches},
        {"windows",g.windows},{"groups",g.groups},{"explicit_substeps",g.explicit_substeps},
        {"implicit_steps",g.implicit_steps},{"rejections",g.rejections},{"newton_iterations",g.newton_iterations},
        {"residual_evaluations",g.residual_evaluations}};
}
J replay(const std::string& path,const J& request) {
    auto s=battery::load_scenario(path);J root;std::ifstream(path)>>root;
    s.settings.max_dt_s=request.value("max_dt_s",s.settings.max_dt_s);
    const double refinement=request.value("refinement",1.);
    if(!std::isfinite(refinement)||refinement<=0 || refinement>1)throw std::runtime_error("Invalid refinement");
    s.settings.max_dt_s*=refinement;s.gas_model.max_fraction_per_step*=refinement;s.gas_model.max_temperature_step_k*=refinement;
    s.gas_model.local_exchange_interval_s*=refinement;
    s.gas_model.pressure_relaxation_fraction*=refinement;
    s.gas_model.pressure_absolute_tolerance_pa*=refinement;
    s.gas_model.pressure_relative_tolerance*=refinement;
    s.gas_model.kawasaki_tail_atol_pa*=refinement;
    s.gas_model.kawasaki_tail_rtol*=refinement;
    s.gas_model.batch_pressure_fraction*=refinement;
    s.gas_model.implicit_temperature_tolerance_k*=refinement;
    s.gas_model.implicit_tracer_relative_tolerance*=refinement;
    for(auto& c:s.cells) {c.conversion.max_fraction_per_step*=refinement;c.conversion.max_temperature_step_k*=refinement;}
    for(auto& c:s.gas_channels)if(c.opening.enabled)c.opening.max_damage_step*=refinement;
    std::vector<Probe> probes;
    for(const auto& p:root.at("measurement_probes")) {
        auto e=std::find_if(s.interfaces.begin(),s.interfaces.end(),[&](auto& x){return x.id==p.at("interface_id");});
        auto c=std::find_if(s.cells.begin(),s.cells.end(),[&](auto& x){return x.id==p.at("cell_id");});
        if(e==s.interfaces.end() || c==s.cells.end())throw std::runtime_error("Unknown physical surface probe");
        int object=static_cast<int>(c-s.cells.begin());
        if(object!=static_cast<int>(e->cell_a) && object!=static_cast<int>(e->cell_b))throw std::runtime_error("Probe object not on contact");
        auto q=std::find_if(s.ports.begin(),s.ports.end(),[&](auto& x){return x.id==(object==static_cast<int>(e->cell_a)?e->port_a:e->port_b);});
        int axis=q->surface[1]-'x';
        if(std::abs(q->area_m2-c->dimensions_m[(axis+1)%3]*c->dimensions_m[(axis+2)%3])>1e-12)throw std::runtime_error("Full panel face required");
        probes.push_back({p.at("id"),static_cast<int>(e-s.interfaces.begin()),object,axis,q->area_m2});
    }
    const auto started=std::chrono::steady_clock::now();battery::Simulation sim(s);const auto& m=sim.thermal_mesh();
    J out={{"time_s",J::array()},{"temperature_k",J::object()},{"applied_objects",J::array()},
        {"max_thermal_residual_j",0.},{"max_chemical_residual_j",0.},{"max_mass_residual_kg",0.}};
    for(auto& c:s.cells)out["applied_objects"].push_back({{"id",c.id},{"capacity_j_k",c.heat_capacity_j_per_k},
        {"dimensions_m",c.dimensions_m},{"conductivity_w_m_k",c.conductivity_w_per_m_k},{"shell_contents_id",c.shell_contents_id}});
    auto sample=[&] {
        const auto& field=sim.thermal_states();const auto edges=sim.thermal_link_states();out["time_s"].push_back(sim.time_s());
        for(auto& p:probes) {
            const auto& c=s.cells[p.object];double sum=0,weight=0;
            for(int j=m.interface_offsets[p.edge];j<m.interface_offsets[p.edge+1];++j) {
                int index=m.interface_links[j];const auto& e=m.links[index];
                const auto& a=field[e.a];const auto& b=field[e.b];
                double r=battery::phase_resistance(e.series_resistance_k_per_w,e.series_resistance_a_k_per_w,a.damage,b.damage,
                    battery::boundary_phase(s.cells[m.volumes[e.a].object]),battery::boundary_phase(s.cells[m.volumes[e.b].object]),
                    sim.time_s(),a.runaway_started_s,b.runaway_started_s,a.conductivity_factor[e.geometry_axis],b.conductivity_factor[e.geometry_axis]);
                if(e.law.kind!=battery::TransitionKind::DirectedGeometry)throw std::runtime_error("External sensor contact must be directed");
                auto now=battery::evaluate_directed_geometry(e.law,edges[index],e.geometry_axis,a.temperature_k,b.temperature_k,a.damage,b.damage,0,e.weight,r);
                int own=m.volumes[e.a].object==p.object?e.a:e.b,other=own==e.a?e.b:e.a;
                double half=c.dimensions_m[p.axis]/c.thermal_divisions[p.axis]/(2*c.conductivity_w_per_m_k[p.axis]*p.area*e.weight);
                // Match the solver: panel bulk destruction affects internal edges,
                // while boundary halves use boundary_phase and structural factors.
                const double f=battery::phase_factor(battery::boundary_phase(c),field[own].damage,sim.time_s(),field[own].runaway_started_s)*field[own].conductivity_factor[p.axis];
                sum+=e.weight*(field[own].temperature_k+now.conductance_w_per_k*(field[other].temperature_k-field[own].temperature_k)*half/f);weight+=e.weight;
            }
            out["temperature_k"][p.id].push_back(sum/weight);
        }
        for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].is_battery) {
            const auto& c=sim.cells()[i];out["cell_mean_temperature_k"][s.cells[i].id].push_back(c.temperature_k);
            out["joule_energy_j"][s.cells[i].id].push_back(c.joule_energy_j);
            if(s.cells[i].conversion.enabled) {
                double virgin=0.,precursor=0.,residue=0.;
                for(int j=m.object_offsets[i];j<m.object_offsets[i+1];++j) {
                    virgin+=field[j].reactant_kg;precursor+=field[j].activated_kg;residue+=field[j].residue_kg;
                }
                auto& chemistry=out["chemistry"][s.cells[i].id];
                chemistry["virgin_kg"].push_back(virgin);
                chemistry["r2_precursor_kg"].push_back(precursor);
                chemistry["residue_kg"].push_back(residue);
                chemistry["r2_power_w"].push_back(c.chemical_power_w);
                out["released_energy_j_history"].push_back(c.released_energy_j);out["remaining_energy_j"].push_back(c.chemical_energy_remaining_j);
                out["source_damage"].push_back(c.damage);out["source_max_temperature_k"].push_back(c.max_temperature_k);
            }
        }
        if(!s.gas_compartments.empty()) {
            auto d=sim.process_diagnostics();
            for(std::size_t i=0;i<d.channels.size();++i) {
                auto& channel=out["channels"][s.gas_channels[i].id];const auto& v=d.channels[i];
                channel["opening_fraction"].push_back(v.opening_fraction);
                channel["effective_area_m2"].push_back(v.effective_area_m2);
                if(s.gas_channels[i].opening.deterministic_aperture) {
                    channel["aperture_radius_m"].push_back(v.aperture_radius_m);
                    channel["aperture_strips_m"].push_back(v.aperture_strips_m);
                }
                channel["max_bulk_damage"].push_back(v.max_bulk_damage);
                channel["max_seam_damage"].push_back(v.max_seam_damage);
                channel["max_thermal_damage"].push_back(v.max_thermal_damage);
                channel["mass_flow_kg_s"].push_back(v.mass_flow_kg_s);
                channel["proposed_mass_flow_kg_s"].push_back(v.proposed_mass_flow_kg_s);
                channel["transferred_mass_kg"].push_back(v.transferred_kg);
                channel["transferred_chemical_j"].push_back(v.transferred_chemical_j);
                channel["transferred_oxidizer_kg"].push_back(v.transferred_oxidizer_kg);
                channel["surface_energy_j"].push_back(v.surface_energy_j);
                channel["fracture_work_j"].push_back(v.fracture_work_j);
                channel["delivered_gas_energy_j"].push_back(v.transferred_energy_j-v.surface_energy_j);
            }
            for(std::size_t i=0;i<d.gas.size();++i) {
                auto& g=out["gas"][s.gas_compartments[i].id];g["pressure_pa"].push_back(d.gas[i].pressure_pa);
                g["temperature_k"].push_back(d.gas[i].temperature_k);g["mass_kg"].push_back(d.gas[i].mass_kg);
                g["free_volume_m3"].push_back(d.gas[i].free_volume_m3);
                g["heat_capacity_j_per_k"].push_back(d.gas[i].mass_kg*s.gas_model.cv_j_per_kg_k);
                g["chemical_j"].push_back(d.gas[i].chemical_j);g["released_energy_j"].push_back(d.gas[i].released_energy_j);
                g["oxidizer_kg"].push_back(d.gas[i].oxidizer_kg);g["consumed_oxidizer_kg"].push_back(d.gas[i].consumed_oxidizer_kg);
            }
            out["max_mass_residual_kg"]=std::max(out["max_mass_residual_kg"].get<double>(),std::abs(d.mass_residual_kg));
            out["max_oxidizer_residual_kg"]=std::max(out.value("max_oxidizer_residual_kg",0.),std::abs(d.oxidizer_residual_kg));
        }
        out["max_thermal_residual_j"]=std::max(out["max_thermal_residual_j"].get<double>(),std::abs(sim.energy_balance_residual_j()));
        out["max_chemical_residual_j"]=std::max(out["max_chemical_residual_j"].get<double>(),std::abs(sim.chemical_balance_residual_j()));
    };
    std::vector<double> times=root.value("failure_sample_times_s",std::vector<double>{});
    if(times.empty()) {times.push_back(0);for(double t=s.settings.output_interval_s;t<s.settings.duration_s-1e-9;t+=s.settings.output_interval_s)times.push_back(t);times.push_back(s.settings.duration_s);}
    if(times.size()<2 || times.front()!=0 || times.back()!=s.settings.duration_s)throw std::runtime_error("Invalid observation interval");
    for(std::size_t i=1;i<times.size();++i)if(!std::isfinite(times[i])||times[i]<=times[i-1])throw std::runtime_error("Unordered observation times");
    const bool progress=std::getenv("BATTERY_SIM_PROGRESS")!=nullptr;auto last_progress=started;
    const std::string checkpoint=request.value("checkpoint_path",std::string{});
    auto last_checkpoint=started;
    auto save_checkpoint=[&] {
        const auto now=std::chrono::steady_clock::now();
        if(checkpoint.empty() || std::chrono::duration<double>(now-last_checkpoint).count()<=20)return;
        // Retain only completed observations; a slow interval must not erase
        // all earlier evidence when its replay budget expires.
        out["steps"]=sim.step_count();out["wall_s"]=std::chrono::duration<double>(now-started).count();
        out["partial"]=true;out["advanced_to_s"]=sim.time_s();out["solver_profile"]=profile_json(sim.profile());
        {std::ofstream file(checkpoint+".tmp");file<<out.dump();if(!file)throw std::runtime_error("Checkpoint write failed");}
        std::filesystem::rename(checkpoint+".tmp",checkpoint);last_checkpoint=now;
    };
    sample();for(std::size_t i=1;i<times.size();++i) {
        while(sim.time_s()<times[i]-1e-10) {
            sim.step(times[i]-sim.time_s());
            save_checkpoint();
            if(progress) {
                const auto now=std::chrono::steady_clock::now();
                if(std::chrono::duration<double>(now-last_progress).count()>10) {
                    std::cerr<<"Replay: t="<<sim.time_s()<<" / "<<s.settings.duration_s
                             <<" s, steps="<<sim.step_count()<<", dt="<<sim.last_dt_s()<<'\n';last_progress=now;
                }
            }
        }
        sample();
        save_checkpoint();
    }
    const auto& field=sim.thermal_states();for(std::size_t i=0;i<s.cells.size();++i)
        for(int j=m.object_offsets[i];j<m.object_offsets[i+1];++j)out["final_fields_k"][s.cells[i].id].push_back(field[j].temperature_k);
    out["events"]=J::array();for(auto& e:sim.events())out["events"].push_back({{"time_s",e.time_s},{"kind",e.kind},{"id",e.entity_id}});
    out["partial"]=false;
    out["solver_profile"]=profile_json(sim.profile());
    out["energy_residual_j"]=sim.energy_balance_residual_j();out["steps"]=sim.step_count();out["volumes"]=m.volumes.size();
    out["wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();return out;
}
int main(int argc,char** argv) {if(argc!=2)return 2;std::string line;while(std::getline(std::cin,line))try {
    const auto request=J::parse(line);for(auto& item:request.items())if(item.key()!="scenario_path" && item.key()!="max_dt_s" && item.key()!="refinement" && item.key()!="checkpoint_path")throw std::runtime_error("Unknown replay control");
    std::cout<<replay(request.value("scenario_path",std::string(argv[1])),request).dump()<<std::endl;
}catch(const std::exception& e){std::cout<<J({{"error",e.what()}}).dump()<<std::endl;}}

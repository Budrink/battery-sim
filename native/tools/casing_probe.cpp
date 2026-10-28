// General opt-in casing calibration replay. No optimizer or thermal evolution on CPU.
#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include "battery_native/failure.hpp"
#include "battery_native/conversion.hpp"
#include <nlohmann/json.hpp>
#include "pad_ablation.hpp"
#include "gas_film_trace.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
using J=nlohmann::json;
#include "process_history.hpp"
#include "gas_field_observer.hpp"
#include "visual_recording_observer.hpp"
struct Probe {std::string id;int edge,object,axis;double area;};
J restart_identity(J root,const std::string& core,double refinement,double dt) {
    for(const auto* key:{"metadata","name","source","failure_sample_times_s","measurement_probes","observations","thermal_path_trace","gas_film_trace","surface_probe_trace","process_history_selection"})root.erase(key);
    root["simulation"].erase("duration_s");root["restart_core"]=core;
    root["restart_refinement"]=refinement;root["restart_max_dt_s"]=dt;return root;
}
J unchanged_rebase_physics(J root) {
    root=restart_identity(std::move(root),"",1.,0.);
    for(const auto* key:{"compartments","channels","thermal_contacts"})root["gas_network"].erase(key);
    auto& ports=root["ports"];
    ports.erase(std::remove_if(ports.begin(),ports.end(),[](const J& p){
        const auto id=p.at("id").get<std::string>();
        return id.rfind("chamber-wall:gas-partition:",0)==0 || id.rfind("ablation-contact:",0)==0;
    }),ports.end());return root;
}
#include "gas_response_restart.hpp"
#include "transport_suffix_restart.hpp"
double radiative_material_half(const battery::Scenario& s,const battery::ThermalMesh& mesh,
    const battery::ThermalLink& e,const battery::ThermalState& a,double time) {
    if(e.b<0 || e.law.radiation_coefficient_w_per_k4<=0.)return -1.;
    return battery::phase_resistance(e.series_resistance_a_k_per_w,e.series_resistance_a_k_per_w,
        a.damage,0.,battery::boundary_phase(s.cells[mesh.volumes[e.a].object]),{},
        time,a.runaway_started_s,-1.,a.conductivity_factor[e.geometry_axis],1.);
}
// Read-only snapshots. Fluxes are instantaneous, not integrated ledger entries.
// Re-evaluate G(T,D) at dt=0, exactly as surface thermocouple observations do.
void sample_thermal_paths(J& out,const battery::Scenario& s,const battery::ThermalMesh& mesh,
    const std::vector<battery::ThermalState>& field,const std::vector<battery::InterfaceState>& edges,double time) {
    for(std::size_t i=0;i<s.cells.size();++i) {
        double weighted=0.,capacity=0.,minimum=1e300,maximum=-1e300;
        for(int j=mesh.object_offsets[i];j<mesh.object_offsets[i+1];++j) {
            const double t=field[j].temperature_k,c=mesh.volumes[j].capacity_j_per_k;
            weighted+=c*t;capacity+=c;minimum=std::min(minimum,t);maximum=std::max(maximum,t);
        }
        auto& v=out["thermal_paths"]["objects"][s.cells[i].id];
        v["reference_capacity_weighted_temperature_k"].push_back(weighted/capacity);
        v["minimum_temperature_k"].push_back(minimum);v["maximum_temperature_k"].push_back(maximum);
    }
    for(std::size_t i=0;i<s.interfaces.size();++i) {
        double flux=0.,conductance=0.,ta=0.,tb=0.,weight=0.;
        for(int j=mesh.interface_offsets[i];j<mesh.interface_offsets[i+1];++j) {
            const int index=mesh.interface_links[j];const auto& e=mesh.links[index];
            const auto& a=field[e.a];const auto& b=field[e.b];
            const double r=battery::phase_resistance(e.series_resistance_k_per_w,e.series_resistance_a_k_per_w,a.damage,b.damage,
                battery::boundary_phase(s.cells[mesh.volumes[e.a].object]),battery::boundary_phase(s.cells[mesh.volumes[e.b].object]),
                time,a.runaway_started_s,b.runaway_started_s,a.conductivity_factor[e.geometry_axis],b.conductivity_factor[e.geometry_axis]);
            if(e.law.kind!=battery::TransitionKind::DirectedGeometry)throw std::runtime_error("Path trace requires directed contacts");
            const auto now=battery::evaluate_directed_geometry(e.law,edges[index],e.geometry_axis,a.temperature_k,b.temperature_k,
                a.damage,b.damage,0,e.weight,r,1.,0,1.,radiative_material_half(s,mesh,e,a,time));
            flux+=now.conductance_w_per_k*(a.temperature_k-b.temperature_k);
            conductance+=now.conductance_w_per_k;ta+=e.weight*a.temperature_k;tb+=e.weight*b.temperature_k;weight+=e.weight;
        }
        const auto& c=s.interfaces[i];auto& v=out["thermal_paths"]["interfaces"][c.id];
        v["from"]=s.cells[c.cell_a].id;v["to"]=s.cells[c.cell_b].id;
        v["a_to_b_power_w"].push_back(flux);v["conductance_w_per_k"].push_back(conductance);
        v["a_adjacent_temperature_k"].push_back(ta/weight);v["b_adjacent_temperature_k"].push_back(tb/weight);
    }
}
J profile_json(const battery::SolverProfile& p) {
    const auto& g=p.gas;
    return {{"prepare_s",p.prepare_s},{"cpu_gas_s",p.cpu_gas_s},{"download_s",p.download_s},{"upload_s",p.upload_s},
        {"coarse_floor_activations",p.coarse_floor_activations},{"aligned_short_steps",p.aligned_short_steps},
        {"minimum_regular_step_s",p.minimum_regular_step_s},
        {"split_step_retries",p.split_step_retries},{"floor_bounded_retries",p.floor_bounded_retries},
        {"cuda_gas_s",p.cuda_gas_s},{"deterministic_pressure_gap_pa",p.deterministic_pressure_gap_pa},
        {"kawasaki_events",p.kawasaki_events},{"kawasaki_tail_updates",p.kawasaki_tail_updates},
        {"gas_batches",p.gas_batches},{"gas_limited_batches",p.gas_limited_batches},
        {"gas_last_reaction_intervals",p.gas_last_reaction_intervals},
        {"gas_max_local_reactor_iterations",p.gas_max_local_reactor_iterations},
        {"gas_last_reaction_step_s",p.gas_last_reaction_step_s},
        {"gas_last_pressure_iterations",p.gas_last_pressure_iterations},
        {"gas_pressure_pattern_attempts",p.gas_pressure_pattern_attempts},
        {"gas_pressure_pattern_accepted",p.gas_pressure_pattern_accepted},
        {"gas_last_mixing_iterations",p.gas_last_mixing_iterations},
        {"gas_last_mixing_residual",p.gas_last_mixing_residual},
        {"gas_last_mixing_raw_residual",p.gas_last_mixing_raw_residual},
        {"gas_averaged_min_scale",p.gas_averaged_min_scale},{"gas_averaged_windows",p.gas_averaged_windows},
        {"gas_averaged_limited_windows",p.gas_averaged_limited_windows},
        {"gas_averaged_entropy_backtracks",p.gas_averaged_entropy_backtracks},
        {"windows",g.windows},{"groups",g.groups},{"explicit_substeps",g.explicit_substeps},
        {"implicit_steps",g.implicit_steps},{"rejections",g.rejections},{"newton_iterations",g.newton_iterations},
        {"residual_evaluations",g.residual_evaluations}};
}
J replay(const std::string& path,const J& request) {
    auto s=battery::load_scenario(path);J root;std::ifstream(path)>>root;
    const bool zone_diagnostics=root.value("metadata",J::object()).value("zone_diagnostics",false);
    const ProcessHistorySelection history(root.value("process_history_selection",J::object()));
    s.settings.max_dt_s=request.value("max_dt_s",s.settings.max_dt_s);
    const double refinement=request.value("refinement",1.);
    if(!std::isfinite(refinement)||refinement<=0 || refinement>1)throw std::runtime_error("Invalid refinement");
    s.settings.max_dt_s*=refinement;s.gas_model.max_fraction_per_step*=refinement;s.gas_model.max_temperature_step_k*=refinement;
    s.gas_model.local_exchange_interval_s*=refinement;
    s.gas_model.coupled_transport_interval_s*=refinement;
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
    const std::string restart_input=request.value("restart_from",std::string{});
    const std::string sealed_path=request.value("save_unvented_restart",std::string{});
    const std::string committed_path=request.value("save_committed_restart",std::string{});
    const double committed_at=request.value("committed_restart_at_s",-1.);
    if(!committed_path.empty() && (committed_at<0. || committed_at>s.settings.duration_s || !sealed_path.empty()))
        throw std::runtime_error("Invalid/exclusive committed restart capture");
    // Observation-only checkpoint limit. Do not add time boundaries or change
    // integration: retain the latest already sampled sealed state <= this time.
    const double sealed_until=request.value("unvented_restart_until_s",s.settings.duration_s);
    if(!std::isfinite(sealed_until)||sealed_until<0.||sealed_until>s.settings.duration_s)
        throw std::runtime_error("Invalid unvented checkpoint time limit");
    const bool stop_at_rupture=request.value("stop_after_first_rupture",true);
    J identity=restart_identity(root,request.value("restart_contract",std::string{}),refinement,s.settings.max_dt_s);
    const std::string restart_contract=identity.dump();
    if((!restart_input.empty() || !sealed_path.empty() || !committed_path.empty()) && identity["restart_core"].get<std::string>().empty())
        throw std::runtime_error("Explicit core fingerprint required for restart");
    if(!restart_input.empty()) {
        std::ifstream file(restart_input,std::ios::binary);if(!file)throw std::runtime_error("Cannot open restart input");
        const std::string bytes((std::istreambuf_iterator<char>(file)),{});
        if(request.contains("restart_transport_scenario")) {
            if(refinement!=1.)throw std::runtime_error("Transport suffix requires unchanged numerical controls");
            J original;std::ifstream(request.at("restart_transport_scenario").get<std::string>())>>original;
            check_transport_suffix_restart(original,root,request.value("diagnostic_reactive_transit",false),
                request.value("diagnostic_closed_exhausts",false));
            const auto source_contract=restart_identity(original,request.at("restart_contract"),1.,s.settings.max_dt_s).dump();
            sim.restore_checkpoint(bytes,source_contract);
            if(sim.save_checkpoint(source_contract)!=bytes)
                throw std::runtime_error("Conditioned transport suffix changed its initial physical state");
        }else if(request.contains("restart_gas_response_scenario")) {
            if(refinement!=1.)throw std::runtime_error("Gas-response suffix requires unrefined controls");
            J original;std::ifstream(request.at("restart_gas_response_scenario").get<std::string>())>>original;
            check_gas_response_restart(original,root,request.value("diagnostic_transport_probe",false));
            sim.restore_checkpoint(bytes,restart_identity(original,request.at("restart_contract"),1.,s.settings.max_dt_s).dump());
        }else if(request.contains("restart_source_scenario")) {
            if(refinement!=1.)throw std::runtime_error("Graph rebase requires unrefined controls");
            const auto source_path=request.at("restart_source_scenario").get<std::string>();
            J original;std::ifstream(source_path)>>original;
            if(unchanged_rebase_physics(root)!=unchanged_rebase_physics(expected_pad_ablation(original,root,request)))
                throw std::runtime_error("Gas rebase changes solid physics or transport laws");
            auto source_scenario=battery::load_scenario(source_path);source_scenario.settings.max_dt_s=s.settings.max_dt_s;
            // Receiver EOS/reaction laws must remain identical to every donor.
            auto law=[](J c){for(const auto* k:{"id","volume_m3","initial_mass_kg","local_equilibrium","contents_id"})c.erase(k);return c;};
            std::vector<battery::GasRemap> map;
            for(const auto& e:request.at("unvented_gas_remap")) {
                const auto a=e.at("from").get<std::size_t>(),b=e.at("to").get<std::size_t>();
                if(law(original.at("gas_network").at("compartments").at(a))!=law(root.at("gas_network").at("compartments").at(b)))
                    throw std::runtime_error("Gas rebase changes reaction/EOS/initial state");
                map.push_back({a,b,e.at("fraction").get<double>()});
            }
            battery::Simulation source(source_scenario);
            source.restore_checkpoint(bytes,restart_identity(original,request.at("restart_contract"),1.,s.settings.max_dt_s).dump());
            sim.rebase_unvented_from(source,map);
        }else sim.restore_checkpoint(bytes,restart_contract);
    }
    const double started_at=sim.time_s();const auto steps_at_start=sim.step_count();
    VisualRecordingObserver visual(request.value("visual_recording_path",std::string{}),path,s);
    bool rupture_seen=false;double sealed_time=-1.;
    J out={{"time_s",J::array()},{"temperature_k",J::object()},{"applied_objects",J::array()},
        {"max_thermal_residual_j",0.},{"max_chemical_residual_j",0.},{"max_mass_residual_kg",0.}};
    out["started_at_s"]=started_at;out["steps_at_start"]=steps_at_start;
    GasFilmTrace film_trace(root.value("gas_film_trace",false),s,m);
    GasFieldObserver gas_fields(root,request,s);
    if(root.value("metadata",J::object()).contains("pad_ablation"))out["pad_ablation"]=root["metadata"]["pad_ablation"];
    if(!restart_input.empty())out["restart_from"]=restart_input;
    if(request.contains("restart_transport_scenario")) {
        out["diagnostic_conditioned_transport_suffix"]=true;
        out["suffix_initial_state_bitwise_equal"]=true;
        out["full_warmup_validation"]=false;
        out["diagnostic_reactive_transit"]=request.value("diagnostic_reactive_transit",false);
        out["diagnostic_closed_exhausts"]=request.value("diagnostic_closed_exhausts",false);
    }
    if(request.contains("restart_gas_response_scenario"))out["diagnostic_gas_response_suffix"]=request.at("restart_gas_response_scenario");
    if(request.value("diagnostic_transport_probe",false))out["diagnostic_transport_probe"]=true;
    if(request.contains("restart_source_scenario")) {
        out["rebase_source_scenario"]=request.at("restart_source_scenario");
        const auto d=sim.process_diagnostics();
        out["rebase_initial_residuals"]={{"thermal_j",d.thermal_residual_j},{"chemical_j",d.chemical_residual_j},
            {"mass_kg",d.mass_residual_kg},{"oxidizer_kg",d.oxidizer_residual_kg}};
        if(std::abs(d.thermal_residual_j)>1e-4 || std::abs(d.chemical_residual_j)>1e-4 ||
           std::abs(d.mass_residual_kg)>1e-10 || std::abs(d.oxidizer_residual_kg)>1e-10)
            throw std::runtime_error("Unvented graph rebase violates conservation");
    }
    const auto save_sealed=[&] {
        if(sealed_path.empty() || rupture_seen || sim.time_s()>sealed_until+1e-9)return;
        const auto d=sim.process_diagnostics();
        for(std::size_t i=0;i<s.gas_channels.size();++i)if(s.gas_channels[i].rupture &&
            (d.channels[i].open || d.channels[i].opening_fraction>0. || d.channels[i].transferred_kg!=0.))
            throw std::runtime_error("Unvented checkpoint requested after a rupture");
        const auto bytes=sim.save_checkpoint(restart_contract);
        {std::ofstream file(sealed_path+".tmp",std::ios::binary);file.write(bytes.data(),bytes.size());
            if(!file)throw std::runtime_error("Cannot write restart state");}
        std::filesystem::rename(sealed_path+".tmp",sealed_path);sealed_time=sim.time_s();
        out["unvented_restart"]={{"path",sealed_path},{"time_s",sealed_time}};
    };
    for(auto& c:s.cells)out["applied_objects"].push_back({{"id",c.id},{"capacity_j_k",c.heat_capacity_j_per_k},
        {"dimensions_m",c.dimensions_m},{"conductivity_w_m_k",c.conductivity_w_per_m_k},{"shell_contents_id",c.shell_contents_id}});
    std::vector<std::pair<std::string,std::vector<std::size_t>>> pressure_observers;
    const J observer_settings=root.value("metadata",J::object()).value("gas_pressure_observers",J::object());
    for(const auto& item:observer_settings.items()) {
        if(item.value().at("mode")!="volume_mean")throw std::runtime_error("Unknown gas pressure observer");
        std::vector<std::size_t> indices;
        for(const auto& id:item.value().at("compartments")) {
            auto it=std::find_if(s.gas_compartments.begin(),s.gas_compartments.end(),[&](const auto& c){return c.id==id;});
            if(it==s.gas_compartments.end())throw std::runtime_error("Unknown gas observer compartment");
            const auto i=static_cast<std::size_t>(it-s.gas_compartments.begin());
            if(std::find(indices.begin(),indices.end(),i)!=indices.end())throw std::runtime_error("Duplicate gas observer compartment");
            indices.push_back(i);
        }
        if(indices.empty())throw std::runtime_error("Empty gas pressure observer");
        pressure_observers.emplace_back(item.key(),std::move(indices));
    }
    auto sample=[&] {
        const auto& field=sim.thermal_states();const auto edges=sim.thermal_link_states();out["time_s"].push_back(sim.time_s());
        if(root.value("thermal_path_trace",false))sample_thermal_paths(out,s,m,field,edges,sim.time_s());
        const auto electrical=sim.electrical_states();
        for(std::size_t i=0;i<s.heating_protocols.size();++i) {
            const auto& p=s.heating_protocols[i];const auto& state=sim.heating_protocol_states()[i];
            auto& v=out["heating_protocols"][p.id];
            v["sensor_temperature_k"].push_back(sim.cells()[p.sensor_object].temperature_k);
            v["target_temperature_k"].push_back(p.initial_temperature_k+p.ramp_k_per_s*sim.time_s());
            v["commanded_power_w"].push_back(state.commanded_power_w);
            v["cutoff_time_s"].push_back(state.cutoff_time_s);
            v["joule_energy_j"].push_back(sim.cells()[p.heater_object].joule_energy_j);
        }
        for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].electrical.enabled) {
            const auto& e=electrical[i];auto& v=out["electrical"][s.cells[i].id];
            v["snapshot_time_s"].push_back(e.time_s);v["resistance_ohm"].push_back(e.resistance_ohm);
            v["current_a"].push_back(e.current_a);v["outlet_current_a"].push_back(e.outlet_current_a);
            v["ohmic_drop_v"].push_back(e.ohmic_drop_v);v["power_w"].push_back(e.power_w);
            v["kcl_relative_residual"].push_back(e.kcl_relative_residual);
            v["power_relative_residual"].push_back(e.power_relative_residual);v["iterations"].push_back(e.iterations);
            if(s.cells[i].electrical.voltage_limited_current) {
                v["actual_positive_charge_c"].push_back(e.positive_charge_c);
                v["actual_terminal_work_j"].push_back(e.terminal_work_j);
                v["voltage_limited_time_s"].push_back(e.voltage_limited_time_s);
            }
        }
        for(auto& p:probes) {
            const auto& c=s.cells[p.object];double sum=0,weight=0,minimum=1e300,maximum=-1e300;
            for(int j=m.interface_offsets[p.edge];j<m.interface_offsets[p.edge+1];++j) {
                int index=m.interface_links[j];const auto& e=m.links[index];
                const auto& a=field[e.a];const auto& b=field[e.b];
                double r=battery::phase_resistance(e.series_resistance_k_per_w,e.series_resistance_a_k_per_w,a.damage,b.damage,
                    battery::boundary_phase(s.cells[m.volumes[e.a].object]),battery::boundary_phase(s.cells[m.volumes[e.b].object]),
                    sim.time_s(),a.runaway_started_s,b.runaway_started_s,a.conductivity_factor[e.geometry_axis],b.conductivity_factor[e.geometry_axis]);
                if(e.law.kind!=battery::TransitionKind::DirectedGeometry)throw std::runtime_error("External sensor contact must be directed");
                auto now=battery::evaluate_directed_geometry(e.law,edges[index],e.geometry_axis,a.temperature_k,b.temperature_k,a.damage,b.damage,0,e.weight,r,
                    1.,0,1.,radiative_material_half(s,m,e,a,sim.time_s()));
                int own=m.volumes[e.a].object==p.object?e.a:e.b,other=own==e.a?e.b:e.a;
                // Use the ACTUAL overlap area, which can be smaller than the
                // casing face when a pad has open perimeter bands.
                double half=own==e.a?e.series_resistance_a_k_per_w:
                    e.series_resistance_k_per_w-e.series_resistance_a_k_per_w;
                // Match the solver: panel bulk destruction affects internal edges,
                // while boundary halves use boundary_phase and structural factors.
                const double f=battery::phase_factor(battery::boundary_phase(c),field[own].damage,sim.time_s(),field[own].runaway_started_s)*field[own].conductivity_factor[p.axis];
                const double surface=field[own].temperature_k+now.conductance_w_per_k*(field[other].temperature_k-field[own].temperature_k)*half/f;
                sum+=e.weight*surface;weight+=e.weight;
                minimum=std::min(minimum,surface);maximum=std::max(maximum,surface);
            }
            out["temperature_k"][p.id].push_back(sum/weight);
            // A point thermocouple is not a face average. Report the spatial
            // envelope without choosing a fitted hot spot or changing physics.
            if(root.value("surface_probe_trace",false)) {
                auto& trace=out["surface_probe_trace"][p.id];
                trace["minimum_temperature_k"].push_back(minimum);
                trace["maximum_temperature_k"].push_back(maximum);
            }
        }
        for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].is_battery) {
            const auto& c=sim.cells()[i];out["cell_mean_temperature_k"][s.cells[i].id].push_back(c.temperature_k);
            out["joule_energy_j"][s.cells[i].id].push_back(c.joule_energy_j);
            if(s.cells[i].conversion.enabled) {
                double virgin=0.,precursor=0.,residue=0.,charge_energy=0.,electrical_input=0.,from_charge=0.,from_degradation=0.;
                double healthy_charged=0.,degraded=0.,liquid=0.,evaporated=0.,condensed=0.,r2_gas=0.;
                for(int j=m.object_offsets[i];j<m.object_offsets[i+1];++j) {
                    virgin+=field[j].reactant_kg;precursor+=field[j].activated_kg;residue+=field[j].residue_kg;
                    charge_energy+=field[j].precursor_charge_energy_j;electrical_input+=field[j].electrical_chemical_input_j;
                    from_charge+=field[j].precursor_from_charge_kg;from_degradation+=field[j].precursor_from_degradation_kg;
                    healthy_charged+=field[j].healthy_charged_kg;degraded+=field[j].degraded_mass_kg;
                    liquid+=field[j].liquid_kg;evaporated+=field[j].evaporated_kg;condensed+=field[j].condensed_kg;
                    r2_gas+=field[j].r2_generated_gas_kg;
                    if(zone_diagnostics) {
                        auto& z=out["chemistry_zones"][s.cells[i].id][std::to_string(j-m.object_offsets[i])];
                        const auto& f=field[j];const auto& law=s.cells[i].conversion;
                        z["temperature_k"].push_back(f.temperature_k);
                        z["precursor_kg"].push_back(f.activated_kg);z["virgin_kg"].push_back(f.reactant_kg);
                        z["r2_rate_per_s"].push_back(battery::completion_rate(law,f.temperature_k));
                        z["r2_power_w"].push_back(f.chemical_power_w);
                        z["r2_generated_gas_kg"].push_back(f.r2_generated_gas_kg);
                        z["r2_reacted_kg"].push_back(f.r2_reacted_kg);
                        z["liquid_kg"].push_back(f.liquid_kg);
                    }
                }
                auto& chemistry=out["chemistry"][s.cells[i].id];
                chemistry["virgin_kg"].push_back(virgin);
                chemistry["r2_precursor_kg"].push_back(precursor);
                chemistry["residue_kg"].push_back(residue);
                chemistry["healthy_charged_kg"].push_back(healthy_charged);
                chemistry["degraded_mass_kg"].push_back(degraded);
                chemistry["charge_capacity_fraction"].push_back((virgin+healthy_charged)/s.cells[i].conversion.mass_kg);
                chemistry["r2_power_w"].push_back(c.chemical_power_w);
                chemistry["liquid_kg"].push_back(liquid);chemistry["evaporated_kg"].push_back(evaporated);
                chemistry["condensed_kg"].push_back(condensed);chemistry["r2_generated_gas_kg"].push_back(r2_gas);
                chemistry["precursor_charge_energy_j"].push_back(charge_energy);
                chemistry["electrical_chemical_input_j"].push_back(electrical_input);
                chemistry["precursor_from_charge_kg"].push_back(from_charge);
                chemistry["precursor_from_degradation_kg"].push_back(from_degradation);
                out["released_energy_j_history"].push_back(c.released_energy_j);out["remaining_energy_j"].push_back(c.chemical_energy_remaining_j);
                out["source_damage"].push_back(c.damage);out["source_max_temperature_k"].push_back(c.max_temperature_k);
            }
        }
        if(!s.gas_compartments.empty()) {
            auto d=sim.process_diagnostics();
            gas_fields.sample(out,d.gas,sim.time_s());gas_fields.describe(out,s);
            film_trace.sample(out,field,d.gas);
            double gas_mass=0.,gas_energy=0.,gas_latent=0.,gas_chemical=0.,gas_released=0.,oxidizer=0.;
            for(const auto& g:d.gas){gas_mass+=g.mass_kg;gas_energy+=g.energy_j;gas_latent+=g.latent_j;
                gas_chemical+=g.chemical_j;gas_released+=g.released_energy_j;oxidizer+=g.oxidizer_kg;}
            auto& totals=out["process_totals"];
            totals["thermal_boundary_energy_out_j"].push_back(sim.boundary_energy_out_j());
            totals["gas_mass_kg"].push_back(gas_mass);totals["gas_sensible_energy_j"].push_back(gas_energy);
            totals["gas_latent_energy_j"].push_back(gas_latent);totals["gas_chemical_j"].push_back(gas_chemical);
            totals["gas_released_energy_j"].push_back(gas_released);totals["gas_oxidizer_kg"].push_back(oxidizer);
            totals["escaped_mass_kg"].push_back(d.escaped_mass_kg);totals["escaped_energy_j"].push_back(d.escaped_energy_j);
            totals["escaped_chemical_j"].push_back(d.escaped_chemical_j);totals["escaped_oxidizer_kg"].push_back(d.escaped_oxidizer_kg);
            double gross_in=0.,gross_out=0.,entropy=0.;for(const auto& g:d.gas){gross_in+=g.ambient_gross_in_kg;gross_out+=g.ambient_gross_out_kg;entropy+=g.ambient_entropy_j_k;}
            totals["ambient_gross_in_kg"].push_back(gross_in);totals["ambient_gross_out_kg"].push_back(gross_out);
            totals["ambient_entropy_j_k"].push_back(entropy);
            for(std::size_t i=0;i<d.channels.size();++i) {
                if(!history.includes("channel_prefixes",s.gas_channels[i].id))continue;
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
                channel["mixed_mass_kg"].push_back(v.mixed_mass_kg);
                channel["transferred_vapor_kg"].push_back(v.transferred_vapor_kg);
                channel["transferred_latent_j"].push_back(v.transferred_latent_j);
                channel["transferred_chemical_j"].push_back(v.transferred_chemical_j);
                channel["transferred_oxidizer_kg"].push_back(v.transferred_oxidizer_kg);
                channel["surface_energy_j"].push_back(v.surface_energy_j);
                channel["fracture_work_j"].push_back(v.fracture_work_j);
                channel["delivered_gas_energy_j"].push_back(v.transferred_energy_j-v.surface_energy_j);
            }
            for(std::size_t i=0;i<d.gas.size();++i) {
                if(!history.includes("gas_prefixes",s.gas_compartments[i].id))continue;
                auto& g=out["gas"][s.gas_compartments[i].id];g["pressure_pa"].push_back(d.gas[i].pressure_pa);
                g["temperature_k"].push_back(d.gas[i].temperature_k);g["mass_kg"].push_back(d.gas[i].mass_kg);
                g["vapor_kg"].push_back(d.gas[i].vapor_kg);g["latent_j"].push_back(d.gas[i].latent_j);
                g["free_volume_m3"].push_back(d.gas[i].free_volume_m3);
                g["heat_capacity_j_per_k"].push_back(d.gas[i].mass_kg*s.gas_model.cv_j_per_kg_k);
                g["chemical_j"].push_back(d.gas[i].chemical_j);g["released_energy_j"].push_back(d.gas[i].released_energy_j);
                g["oxidizer_kg"].push_back(d.gas[i].oxidizer_kg);g["consumed_oxidizer_kg"].push_back(d.gas[i].consumed_oxidizer_kg);
            }
            for(const auto& observer:pressure_observers) {
                double pv=0.,volume=0.,minimum=1e300,maximum=0.,mass=0.,vapor=0.,energy=0.;
                for(auto i:observer.second) {
                    const auto& g=d.gas[i];pv+=g.pressure_pa*g.free_volume_m3;volume+=g.free_volume_m3;
                    mass+=g.mass_kg;vapor+=g.vapor_kg;energy+=g.energy_j;
                    minimum=std::min(minimum,g.pressure_pa);maximum=std::max(maximum,g.pressure_pa);
                }
                auto& o=out["gas_observers"][observer.first];o["pressure_pa"].push_back(pv/volume);
                o["min_pressure_pa"].push_back(minimum);o["max_pressure_pa"].push_back(maximum);
                o["volume_m3"].push_back(volume);
                o["mass_kg"].push_back(mass);o["vapor_kg"].push_back(vapor);o["sensible_energy_j"].push_back(energy);
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
    if(!committed_path.empty() && std::none_of(times.begin(),times.end(),[&](double t){return std::abs(t-committed_at)<1e-8;}))
        throw std::runtime_error("Committed capture must use an existing observation, not alter the integration grid");
    bool committed_saved=false;
    const auto save_committed=[&] {
        if(committed_path.empty() || committed_saved || std::abs(sim.time_s()-committed_at)>1e-8)return;
        const auto bytes=sim.save_checkpoint(restart_contract);
        {std::ofstream file(committed_path+".tmp",std::ios::binary);file.write(bytes.data(),bytes.size());
            if(!file)throw std::runtime_error("Cannot write committed restart");}
        std::filesystem::rename(committed_path+".tmp",committed_path);committed_saved=true;
        out["committed_restart"]={{"path",committed_path},{"time_s",sim.time_s()}};
    };
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
        // Separate timestamp: these are LIVE values, not a fabricated extra
        // thermocouple observation at the last completed sample time.
        out["live_state"]={{"time_s",sim.time_s()}};
        for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].conversion.enabled)
            out["live_state"]["objects"][s.cells[i].id]={{"max_temperature_k",sim.cells()[i].max_temperature_k},
                {"r2_power_w",sim.cells()[i].chemical_power_w}};
        {std::ofstream file(checkpoint+".tmp");file<<out.dump();if(!file)throw std::runtime_error("Checkpoint write failed");}
        std::filesystem::rename(checkpoint+".tmp",checkpoint);last_checkpoint=now;
    };
    sample();visual.observe(sim,true);save_sealed();save_committed();for(std::size_t i=1;i<times.size();++i) {
        if(committed_saved && request.value("stop_after_committed_restart",false))break;
        if(times[i]<=started_at+1e-10)continue;
        while(sim.time_s()<times[i]-1e-10) {
            sim.step(times[i]-sim.time_s());
            visual.observe(sim);
            film_trace.collect(sim);
            if(!sealed_path.empty() && !rupture_seen) {
                for(const auto& event:sim.events())if(event.kind=="shell_rupture") {rupture_seen=true;break;}
                if(rupture_seen){out["first_rupture_s"]=sim.time_s();if(stop_at_rupture)break;}
            }
            save_checkpoint();
            if(progress) {
                const auto now=std::chrono::steady_clock::now();
                if(std::chrono::duration<double>(now-last_progress).count()>10) {
                    double power=0.,remaining=0.,precursor=0.;
                    for(const auto& node:sim.thermal_states()) {
                        power=std::max(power,node.chemical_power_w);remaining+=node.reactant_kg;precursor+=node.activated_kg;
                    }
                    std::cerr<<std::setprecision(17)<<"Replay: t="<<sim.time_s()<<" / "<<s.settings.duration_s
                             <<" s, wall_s="<<std::chrono::duration<double>(now-started).count()
                             <<", steps="<<sim.step_count()<<", dt="<<sim.last_dt_s()
                             <<", Tmax_K="<<sim.max_temperature_k()<<", max_R2_W="<<power
                             <<", A_kg="<<remaining<<", B_kg="<<precursor<<'\n';last_progress=now;
                }
            }
        }
        sample();
        visual.observe(sim,true);
        save_committed();
        if(rupture_seen && stop_at_rupture)break;
        save_sealed();
        save_checkpoint();
    }
    const auto& field=sim.thermal_states();for(std::size_t i=0;i<s.cells.size();++i)
        for(int j=m.object_offsets[i];j<m.object_offsets[i+1];++j)out["final_fields_k"][s.cells[i].id].push_back(field[j].temperature_k);
    out["events"]=J::array();for(auto& e:sim.events())out["events"].push_back({{"time_s",e.time_s},{"kind",e.kind},{"id",e.entity_id}});
    out["partial"]=(rupture_seen && stop_at_rupture) || (committed_saved && request.value("stop_after_committed_restart",false));
    out["simulated_duration_s"]=sim.time_s()-started_at;
    out["advanced_to_s"]=sim.time_s();out.erase("live_state");
    visual.observe(sim,true,sim.time_s()>=s.settings.duration_s-1e-8);
    out["solver_profile"]=profile_json(sim.profile());
    out["energy_residual_j"]=sim.energy_balance_residual_j();out["steps"]=sim.step_count();out["volumes"]=m.volumes.size();
    out["wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();return out;
}
int main(int argc,char** argv) {if(argc!=2)return 2;std::string line;while(std::getline(std::cin,line))try {
    const auto request=J::parse(line);
    static constexpr std::array controls={"scenario_path","max_dt_s","refinement","checkpoint_path",
        "restart_from","save_unvented_restart","unvented_restart_until_s","restart_contract","stop_after_first_rupture",
        "restart_source_scenario","restart_gas_response_scenario","unvented_gas_remap",
        "diagnostic_pad_ablation","diagnostic_transport_probe","restart_transport_scenario",
        "save_committed_restart","committed_restart_at_s","stop_after_committed_restart","diagnostic_reactive_transit",
        "diagnostic_closed_exhausts","gas_frames_path","visual_recording_path"};
    for(const auto& item:request.items())if(std::none_of(controls.begin(),controls.end(),
        [&](const char* key){return item.key()==key;}))throw std::runtime_error("Unknown replay control: "+item.key());
    std::cout<<replay(request.value("scenario_path",std::string(argv[1])),request).dump()<<std::endl;
}catch(const std::exception& e){std::cout<<J({{"error",e.what()}}).dump()<<std::endl;}}

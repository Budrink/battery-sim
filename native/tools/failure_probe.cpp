// Post-event source/contact identification. Physics remains in the CUDA core.
// This probe never fits heat capacity, intact links, initial fields or sensors.
#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include "battery_native/failure.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using J=nlohmann::json;
struct FaceProbe {std::string id;int edge,object,axis;double area;};

int main(int argc,char** argv) {try {
    if(argc!=2)throw std::runtime_error("usage: battery_sim_failure_probe SCENARIO; JSON requests on stdin");
    const auto base=battery::load_scenario(argv[1]);
    J root;std::ifstream(argv[1])>>root;
    const bool post=root.value("metadata",J::object()).value("failure_calibration",false);
    const auto model=root.value("metadata",J::object()).value("failure_model",std::string{});
    const bool transient=model=="medium_transition_v3";
    const bool transition=transient || model=="medium_transition_v2";
    const bool kinetic=model=="kinetic_process_v4";
    if((transition || kinetic) && !post)throw std::runtime_error("Medium transition requires failure protocol");
    if(post && !kinetic)for(const auto& c:base.cells)if(c.runaway_pulse.profile==battery::RunawayProfile::Kinetic)
        throw std::runtime_error("Legacy pulse calibration cannot fit kinetic reactions; use the reaction probe and a dedicated identification protocol");
    if(kinetic)for(const auto& c:base.cells) {
        if(c.id=="C2") {
            if(c.runaway_pulse.profile!=battery::RunawayProfile::Kinetic || !c.reaction_structure.enabled || c.energy_outlet.enabled ||
               c.bulk_destruction_on_boundary || c.bulk_failure_transient.relaxation_time_s!=0 || c.bulk_connectivity.endpoint_damage_weight!=0 ||
               c.damage_rate_at_onset_per_s!=0 || c.reaction_kinetics.reference_temperature_k!=400. || c.reaction_kinetics.activation_damage!=.2 ||
               c.reaction_kinetics.damage_sensitivity!=2. || c.reaction_kinetics.damage_per_conversion!=1.)
                throw std::runtime_error("Kinetic protocol requires frozen 400K reaction prior, independent structure, no gas outlet or thermal ignition fit");
        } else if(c.runaway_energy_j!=0 || c.runaway_pulse.profile!=battery::RunawayProfile::ConstantPower || c.reaction_structure.enabled || c.energy_outlet.enabled)
            throw std::runtime_error("Kinetic calibration must not introduce witness chemistry");
    }
    for(const auto& c:base.cells)if(c.heat_capacity_law.factor.count)
        throw std::runtime_error("Failure trial requires all capacities fixed");
    std::vector<FaceProbe> probes;
    for(const auto& p:root.value("measurement_probes",J::array())) {
        auto e=std::find_if(base.interfaces.begin(),base.interfaces.end(),[&](auto& x){return x.id==p.at("interface_id");});
        auto c=std::find_if(base.cells.begin(),base.cells.end(),[&](auto& x){return x.id==p.at("cell_id");});
        if(e==base.interfaces.end() || c==base.cells.end())throw std::runtime_error("Unknown face probe");
        int object=static_cast<int>(c-base.cells.begin());
        if(object!=static_cast<int>(e->cell_a) && object!=static_cast<int>(e->cell_b))throw std::runtime_error("Wrong probe cell");
        auto port=std::find_if(base.ports.begin(),base.ports.end(),[&](auto& q){return q.id==(object==static_cast<int>(e->cell_a)?e->port_a:e->port_b);});
        int axis=port->surface[1]-'x';
        if(c->isothermal || std::abs(port->area_m2-c->dimensions_m[(axis+1)%3]*c->dimensions_m[(axis+2)%3])>1e-12)
            throw std::runtime_error("Finite-volume full-face probe required");
        probes.push_back({p.at("id"),static_cast<int>(e-base.interfaces.begin()),object,axis,port->area_m2});
    }
    std::string line;
    while(std::getline(std::cin,line)) {try {
        J p=J::parse(line);auto s=base;
        for(auto& item:p.items()) {
            const auto& k=item.key();const double v=item.value();
            if(!std::isfinite(v) || (k=="fast_fraction"?v<0:v<=0))throw std::runtime_error("Positive finite parameters / nonnegative fraction required");
            bool valid=k=="max_dt_s";
            if(post)valid=valid || k=="energy_j" || (!kinetic && k=="duration_s") || k=="destroyed_iss_h" || k=="destroyed_tim_h";
            if(transition)valid=valid || k=="fast_fraction" || k=="tail_time_s" || k=="destroyed_bulk_factor";
            if(transient)valid=valid || k=="bulk_peak_factor" || k=="bulk_relaxation_time_s";
            if(kinetic)valid=valid || k=="rate_400_per_s" || k=="residue_through" || k=="residue_inplane" ||
                k=="max_fraction_per_step" || k=="max_temperature_rise_per_step_k";
            if(!valid)throw std::runtime_error("Frozen or unsupported parameter: "+k);
            if((k=="energy_j" && (v<1000 || v>1000000)) || (k=="duration_s" && (v<(transition?.1:.5) || v>(transition?5.:120.))) ||
               (k=="fast_fraction" && v>1) || (k=="tail_time_s" && (v<5 || v>120)) ||
               (k=="destroyed_bulk_factor" && (v<.005 || v>30)) ||
               (k=="bulk_peak_factor" && (v<.1 || v>100)) || (k=="bulk_relaxation_time_s" && (v<.05 || v>120)) ||
               (k=="rate_400_per_s" && (v<.0001 || v>10.)) ||
               ((k=="residue_through" || k=="residue_inplane") && (v<.005 || v>10.)) ||
               (k=="max_fraction_per_step" && v>.02) || (k=="max_temperature_rise_per_step_k" && v>2.) ||
               ((k=="destroyed_iss_h" || k=="destroyed_tim_h") && (v<(kinetic?.01:1.) || v>10000)))
                throw std::runtime_error("Failure parameter outside exploratory bounds");
        }
        if(post) {
            auto c=std::find_if(s.cells.begin(),s.cells.end(),[](auto& x){return x.id=="C2";});
            if(c==s.cells.end() || !c->is_battery)throw std::runtime_error("Solved source C2 required");
            c->runaway_energy_j=p.value("energy_j",c->runaway_energy_j);
            c->runaway_duration_s=p.value("duration_s",c->runaway_duration_s);
            if(kinetic) {
                auto& k=c->reaction_kinetics;k.rate_at_reference_per_s=p.value("rate_400_per_s",k.rate_at_reference_per_s);
                k.max_fraction_per_step=p.value("max_fraction_per_step",k.max_fraction_per_step);
                k.max_temperature_rise_per_step_k=p.value("max_temperature_rise_per_step_k",k.max_temperature_rise_per_step_k);
                auto& f=c->reaction_structure.residue_factor;
                f[0]=p.value("residue_through",f[0]);f[1]=f[2]=p.value("residue_inplane",f[1]);
            }
            if(transition) {
                if(c->runaway_pulse.profile!=battery::RunawayProfile::BurstTail || !c->bulk_destruction_on_boundary)
                    throw std::runtime_error("Burst-tail source and bulk boundary switch required");
                c->runaway_pulse.fast_fraction=p.value("fast_fraction",c->runaway_pulse.fast_fraction);
                c->runaway_pulse.tail_time_s=p.value("tail_time_s",c->runaway_pulse.tail_time_s);
                c->bulk_connectivity.destroyed_background_conductance_w_per_k=p.value("destroyed_bulk_factor",c->bulk_connectivity.destroyed_background_conductance_w_per_k);
                if(transient) {
                    if(c->bulk_failure_transient.relaxation_time_s<=0)throw std::runtime_error("Transient bulk structure required");
                    c->bulk_failure_transient.peak_factor=p.value("bulk_peak_factor",c->bulk_failure_transient.peak_factor);
                    c->bulk_failure_transient.relaxation_time_s=p.value("bulk_relaxation_time_s",c->bulk_failure_transient.relaxation_time_s);
                }
            }
            for(auto& e:s.interfaces)if(e.id=="failure-C2-ISS" || e.id=="failure-C2-TIM") {
                const auto key=e.id=="failure-C2-ISS"?"destroyed_iss_h":"destroyed_tim_h";
                auto port=std::find_if(s.ports.begin(),s.ports.end(),[&](auto& q){return q.id==e.port_a;});
                e.law.destroyed_background_conductance_w_per_k=p.value(key,e.law.destroyed_background_conductance_w_per_k/port->area_m2)*port->area_m2;
                if(e.law.kind!=battery::TransitionKind::DirectedGeometry || e.law.endpoint_damage_weight!=1 || e.law.destruction_damage!=1 || e.law.destroyed_active_conductance_w_per_k!=0)
                    throw std::runtime_error("Expected irreversible source contact switch");
            }
        }
        s.settings.max_dt_s=p.value("max_dt_s",s.settings.max_dt_s);
        auto started=std::chrono::steady_clock::now();battery::Simulation sim(s);
        const auto& m=sim.thermal_mesh();
        J out={{"time_s",J::array()},{"temperature_k",J::object()},{"parameters",p},{"applied_objects",J::array()},{"applied_source_contacts",J::array()}};
        for(auto& c:s.cells)out["applied_objects"].push_back({{"id",c.id},{"capacity_j_k",c.heat_capacity_j_per_k},
            {"conductivity_w_m_k",c.conductivity_w_per_m_k},{"divisions",c.thermal_divisions},
            {"energy_j",c.runaway_energy_j},{"duration_s",c.runaway_duration_s}});
        if(transition)for(std::size_t i=0;i<s.cells.size();++i) {
            const auto& c=s.cells[i];out["applied_objects"][i]["failure_state"]={
                {"profile",c.runaway_pulse.profile==battery::RunawayProfile::BurstTail?"burst_tail":"constant_power"},
                {"fast_fraction",c.runaway_pulse.fast_fraction},{"tail_time_s",c.runaway_pulse.tail_time_s},
                {"destroyed_bulk_factor",c.bulk_connectivity.destroyed_background_conductance_w_per_k},
                {"bulk_endpoint_damage_weight",c.bulk_connectivity.endpoint_damage_weight},
                {"bulk_destruction_on_boundary",c.bulk_destruction_on_boundary}};
            if(transient)out["applied_objects"][i]["failure_state"].update({
                {"bulk_peak_factor",c.bulk_failure_transient.peak_factor},
                {"bulk_relaxation_time_s",c.bulk_failure_transient.relaxation_time_s}});
        }
        if(kinetic)for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].id=="C2") {
            const auto& c=s.cells[i];const auto& k=c.reaction_kinetics;const auto& f=c.reaction_structure;
            out["applied_objects"][i]["kinetic_state"]={{"reference_temperature_k",k.reference_temperature_k},
                {"rate_400_per_s",k.rate_at_reference_per_s},{"activation_energy_j_per_mol",k.activation_energy_j_per_mol},
                {"activation_damage",k.activation_damage},{"damage_sensitivity",k.damage_sensitivity},{"damage_per_conversion",k.damage_per_conversion},
                {"active_factor",{f.active_factor[0],f.active_factor[1],f.active_factor[2]}},
                {"residue_factor",{f.residue_factor[0],f.residue_factor[1],f.residue_factor[2]}},
                {"activity_rate_per_s",f.activity_rate_per_s},{"outlet_enabled",c.energy_outlet.enabled},
                {"max_fraction_per_step",k.max_fraction_per_step},{"max_temperature_rise_per_step_k",k.max_temperature_rise_per_step_k}};
        }
        for(auto& e:s.interfaces)if(e.id.starts_with("failure-C2-"))out["applied_source_contacts"].push_back({{"id",e.id},
            {"intact_reference_logit",e.law.geometry_reference_logit},{"destroyed_g_w_k",e.law.destroyed_background_conductance_w_per_k}});
        auto sample=[&] {
            const auto& field=sim.thermal_states();auto history=sim.thermal_link_states();
            out["time_s"].push_back(sim.time_s());
            for(const auto& p:probes) {
                const auto& c=s.cells[p.object];double temperature=0,weight=0;
                const double half_r=c.dimensions_m[p.axis]/c.thermal_divisions[p.axis]/(2*c.conductivity_w_per_m_k[p.axis]);
                for(int j=m.interface_offsets[p.edge];j<m.interface_offsets[p.edge+1];++j) {
                    const int index=m.interface_links[j];const auto& e=m.links[index];
                    const auto pa=battery::boundary_phase(s.cells[m.volumes[e.a].object]);
                    const auto pb=battery::boundary_phase(s.cells[m.volumes[e.b].object]);
                    const double series_r=battery::phase_resistance(e.series_resistance_k_per_w,e.series_resistance_a_k_per_w,
                        field[e.a].damage,field[e.b].damage,pa,pb,sim.time_s(),field[e.a].runaway_started_s,field[e.b].runaway_started_s,
                        field[e.a].conductivity_factor[e.geometry_axis],field[e.b].conductivity_factor[e.geometry_axis]);
                    if(e.law.kind!=battery::TransitionKind::DirectedGeometry)throw std::runtime_error("Directed contacts required");
                    auto now=battery::evaluate_directed_geometry(e.law,history[index],e.geometry_axis,
                        field[e.a].temperature_k,field[e.b].temperature_k,field[e.a].damage,field[e.b].damage,
                        0,e.weight,series_r);
                    const int own=m.volumes[e.a].object==p.object?e.a:e.b,other=own==e.a?e.b:e.a;
                    const double flux=now.conductance_w_per_k*(field[other].temperature_k-field[own].temperature_k);
                    const double factor=battery::phase_factor(battery::boundary_phase(c),field[own].damage,sim.time_s(),field[own].runaway_started_s)*field[own].conductivity_factor[p.axis];
                    temperature+=e.weight*(field[own].temperature_k+flux*half_r/(factor*p.area*e.weight));weight+=e.weight;
                }
                auto& a=out["temperature_k"][p.id];if(a.is_null())a=J::array();a.push_back(temperature/weight);
            }
            for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].is_battery) {
                auto& a=out["cell_mean_temperature_k"][s.cells[i].id];if(a.is_null())a=J::array();a.push_back(sim.cells()[i].temperature_k);
            }
            if(kinetic) {
                for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].id=="C2") {
                    const auto& c=sim.cells()[i];
                    for(auto [key,value]:std::initializer_list<std::pair<const char*,double>>{
                        {"released_energy_j_history",c.released_energy_j},{"remaining_energy_j",c.chemical_energy_remaining_j},
                        {"chemical_power_w",c.chemical_power_w},{"source_damage",c.damage},
                        {"source_min_temperature_k",c.min_temperature_k},{"source_max_temperature_k",c.max_temperature_k}}) {
                        auto& a=out[key];if(a.is_null())a=J::array();a.push_back(value);
                    }
                    double lo[3]{INFINITY,INFINITY,INFINITY},hi[3]{0.,0.,0.};
                    for(int j=m.object_offsets[i];j<m.object_offsets[i+1];++j)for(int axis=0;axis<3;++axis) {
                        lo[axis]=std::min(lo[axis],field[j].conductivity_factor[axis]);hi[axis]=std::max(hi[axis],field[j].conductivity_factor[axis]);
                    }
                    out["structure_min_xyz"].push_back({lo[0],lo[1],lo[2]});out["structure_max_xyz"].push_back({hi[0],hi[1],hi[2]});
                }
                out["max_thermal_residual_j"]=std::max(out.value("max_thermal_residual_j",0.),std::abs(sim.energy_balance_residual_j()));
                out["max_chemical_residual_j"]=std::max(out.value("max_chemical_residual_j",0.),std::abs(sim.chemical_balance_residual_j()));
            }
            if(transition) {
                int total=0,destroyed=0;double lo=INFINITY,hi=0;
                for(std::size_t k=0;k<m.links.size();++k) {
                    const auto& e=m.links[k];if(e.b<0 || m.volumes[e.a].object!=m.volumes[e.b].object || s.cells[m.volumes[e.a].object].id!="C2")continue;
                    const double multiplier=battery::bulk_failure_multiplier(battery::boundary_phase(s.cells[m.volumes[e.a].object]),
                        field[e.a].temperature_k,field[e.b].temperature_k,field[e.a].damage,field[e.b].damage,
                        field[e.a].runaway_started_s,field[e.b].runaway_started_s,sim.time_s());
                    auto now=battery::evaluate_directed_geometry(e.law,history[k],e.geometry_axis,field[e.a].temperature_k,field[e.b].temperature_k,
                        field[e.a].damage,field[e.b].damage,0,e.weight,e.series_resistance_k_per_w,1.,0,multiplier);
                    ++total;destroyed+=now.phase==battery::EdgePhase::Destroyed;lo=std::min(lo,now.conductance_w_per_k);hi=std::max(hi,now.conductance_w_per_k);
                }
                if(transient) {
                    double lo=INFINITY,hi=0;
                    for(std::size_t j=0;j<field.size();++j)if(s.cells[m.volumes[j].object].id=="C2") {
                        const double f=battery::phase_factor(battery::boundary_phase(s.cells[m.volumes[j].object]),field[j].damage,sim.time_s(),field[j].runaway_started_s);
                        lo=std::min(lo,f);hi=std::max(hi,f);
                    }
                    for(auto [key,value]:std::initializer_list<std::pair<const char*,double>>{{"bulk_factor_min",lo},{"bulk_factor_max",hi}}) {
                        auto& a=out[key];if(a.is_null())a=J::array();a.push_back(value);
                    }
                }
                for(auto [key,value]:std::initializer_list<std::pair<const char*,double>>{{"released_energy_j_history",sim.released_energy_j()},
                    {"destroyed_internal_fraction",total?double(destroyed)/total:0.},{"internal_g_min",lo},{"internal_g_max",hi}}) {
                    auto& a=out[key];if(a.is_null())a=J::array();a.push_back(value);
                }
            }
        };
        std::vector<double> times;
        if(root.contains("failure_sample_times_s"))times=root.at("failure_sample_times_s").get<std::vector<double>>();
        else {
            times.push_back(0.);
            for(double t=s.settings.output_interval_s;t<s.settings.duration_s-1e-9;t+=s.settings.output_interval_s)times.push_back(t);
            times.push_back(s.settings.duration_s);
        }
        if(times.size()<2 || times.front()!=0. || times.back()!=s.settings.duration_s)throw std::runtime_error("Sampling must cover full simulation interval");
        for(std::size_t i=1;i<times.size();++i)if(!std::isfinite(times[i]) || times[i]<=times[i-1])throw std::runtime_error("Strictly ordered finite sampling times required");
        sample();
        for(std::size_t i=1;i<times.size();++i) {
            while(sim.time_s()<times[i]-1e-10)sim.step(times[i]-sim.time_s());
            if(std::abs(sim.time_s()-times[i])>1e-9)throw std::runtime_error("Sampling clock mismatch");
            sample();
        }
        const auto& field=sim.thermal_states();
        for(std::size_t i=0;i<s.cells.size();++i) {
            auto& a=out["final_fields_k"][s.cells[i].id];a=J::array();
            for(int j=m.object_offsets[i];j<m.object_offsets[i+1];++j)a.push_back(field[j].temperature_k);
        }
        out["events"]=J::array();for(auto& e:sim.events())out["events"].push_back({{"time_s",e.time_s},{"kind",e.kind},{"id",e.entity_id}});
        out["energy_residual_j"]=sim.energy_balance_residual_j();out["released_energy_j"]=sim.released_energy_j();
        out["boundary_energy_out_j"]=sim.boundary_energy_out_j();out["steps"]=sim.step_count();out["volumes"]=m.volumes.size();
        if(kinetic) {out["vent_heat_out_j"]=sim.vent_heat_out_j();out["ejected_chemical_energy_j"]=sim.ejected_chemical_energy_j();}
        out["wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::cout<<out.dump()<<std::endl;
    }catch(const std::exception& e){std::cout<<J({{"error",e.what()}}).dump()<<std::endl;}}
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

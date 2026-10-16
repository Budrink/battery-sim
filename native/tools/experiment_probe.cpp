// Dataset-agnostic conditional heat-transfer probe. Integration remains CUDA.
// Sensor faces are reconstructed from the FV center and the local contact flux.
#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include "battery_native/failure.hpp"
#include "battery_native/reaction.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using J=nlohmann::json;

struct Probe {std::string id;int edge,object,axis;double area;};

std::string contact_group(std::string a,std::string b) {
    if(a>b)std::swap(a,b);
    if(a=="ISS" && b=="battery")return "battery_iss_scale";
    if(a=="TIM" && b=="battery")return "battery_tim_scale";
    if(a=="ISS" && b=="aluminium")return "iss_aluminium_scale";
    if(a=="ISS" && b=="steel")return "iss_steel_scale";
    if(a=="TIM" && b=="aluminium")return "tim_aluminium_scale";
    throw std::runtime_error("Unclassified calibration contact: "+a+" / "+b);
}

int main(int argc,char** argv) {try {
    if(argc!=2)throw std::runtime_error("usage: battery_sim_experiment_probe SCENARIO; JSON parameter requests on stdin");
    auto base=battery::load_scenario(argv[1]);
    J root;std::ifstream(argv[1])>>root;
    const std::string family=root.value("metadata",J::object()).value("connectivity_model",std::string{});
    const bool directed=family=="directed_geometry";
    const bool dynamic=directed || family=="state_geometry";
    const bool internal_fit=root.value("metadata",J::object()).value("internal_calibration",false);
    const std::string stage=root.value("metadata",J::object()).value("staged_calibration",std::string{});
    const bool internal_contrast_fit=root.value("metadata",J::object()).value("internal_contrast_calibration",false);
    if(internal_contrast_fit && stage!="links")throw std::runtime_error("Internal contrast fit requires links-only protocol");
    if(!stage.empty() && stage!="conductivity" && stage!="capacity" && stage!="links")throw std::runtime_error("Unknown calibration stage");
    if(!stage.empty() && internal_fit)throw std::runtime_error("Conflicting calibration protocols");
    if(stage=="links")for(const auto& c:base.cells)
        if(c.heat_capacity_law.factor.count)throw std::runtime_error("Links-only calibration requires ALL capacities constant");
    const auto expected_kind=directed?battery::TransitionKind::DirectedGeometry:battery::TransitionKind::StateGeometry;
    std::vector<Probe> probes;
    for(const auto& p:root.at("measurement_probes")) {
        auto edge=std::find_if(base.interfaces.begin(),base.interfaces.end(),[&](auto& e){return e.id==p.at("interface_id");});
        auto cell=std::find_if(base.cells.begin(),base.cells.end(),[&](auto& c){return c.id==p.at("cell_id");});
        if(edge==base.interfaces.end()||cell==base.cells.end())throw std::runtime_error("Unknown probe target");
        const int object=static_cast<int>(cell-base.cells.begin());
        if(object!=static_cast<int>(edge->cell_a) && object!=static_cast<int>(edge->cell_b))throw std::runtime_error("Probe cell not on contact");
        const auto port=std::find_if(base.ports.begin(),base.ports.end(),[&](auto& q){return q.id==(object==static_cast<int>(edge->cell_a)?edge->port_a:edge->port_b);});
        const int axis=port->surface[1]-'x';
        // This probe supports full, coextensive rectangular contact faces.
        if(std::abs(port->area_m2-cell->dimensions_m[(axis+1)%3]*cell->dimensions_m[(axis+2)%3])>1e-12)
            throw std::runtime_error("Surface probe requires a full-face contact");
        probes.push_back({p.at("id"),static_cast<int>(edge-base.interfaces.begin()),object,axis,port->area_m2});
    }
    std::string line;
    while(std::getline(std::cin,line)) {try {
        auto request=J::parse(line);auto s=base;
        // Calibration must never compensate contact/fixture errors by changing
        // measured battery properties. Reject obsolete keys, even scale=1.
        const std::vector<std::string> groups={"battery_iss_scale","battery_tim_scale","iss_aluminium_scale","iss_steel_scale","tim_aluminium_scale"};
        const std::vector<std::string> allowed={"contact_scale","ambient_scale","max_dt_s","mesh_scale",
            "battery_iss_scale","battery_tim_scale","iss_aluminium_scale","iss_steel_scale","tim_aluminium_scale",
            "iss_cp_scale","tim_cp_scale","surface_linear","surface_quadratic",
            "battery_temperature_contrast","iss_temperature_contrast","tim_temperature_contrast",
            "aluminium_temperature_contrast","steel_temperature_contrast","internal_linear","internal_quadratic",
            "internal_log_35","internal_log_50","internal_log_70","capacity_45","capacity_70","internal_temperature_contrast"};
        for(const auto& item:request.items()) {
            const auto& key=item.key();
            if(std::find(allowed.begin(),allowed.end(),key)==allowed.end())
                throw std::runtime_error("Unsupported calibration parameter: "+key+"; battery properties are fixed");
            const double value=item.value().get<double>();
            const bool tabulated=key=="internal_log_35" || key=="internal_log_50" || key=="internal_log_70";
            const bool capacity=key=="capacity_45" || key=="capacity_70";
            const bool numeric=key=="max_dt_s" || key=="mesh_scale";
            const bool contact_fit=key=="battery_iss_scale" || key=="battery_tim_scale";
            const bool selected_link=key=="battery_iss_scale" || key=="iss_aluminium_scale";
            const bool internal_contrast=key=="internal_temperature_contrast";
            if(internal_contrast && !internal_contrast_fit)throw std::runtime_error("Internal contrast requires explicit links-only opt-in");
            if(tabulated && stage!="conductivity" && stage!="links")throw std::runtime_error("Table fit requires conductivity-stage opt-in");
            if(capacity && stage!="capacity")throw std::runtime_error("Capacity fit requires capacity-stage opt-in");
            if(!stage.empty() && !numeric && !(stage=="conductivity"?(tabulated || contact_fit):stage=="links"?(tabulated || selected_link || internal_contrast):capacity))
                throw std::runtime_error("Parameter belongs to the frozen calibration block: "+key);
            if(stage=="links" && ((tabulated && std::abs(value)>6.) ||
                (selected_link && (value<.02 || value>99.99))))throw std::runtime_error("Links-only parameter outside protocol bounds");
            if(capacity && (!std::isfinite(value) || value<.8 || value>1.2))throw std::runtime_error("Capacity trial factor outside [0.8,1.2]");
            const bool internal=key=="internal_linear" || key=="internal_quadratic";
            if(internal && !internal_fit)throw std::runtime_error("Internal calibration requires an explicit opt-in scene");
            if(internal_fit && !internal && key!="max_dt_s" && key!="mesh_scale")
                throw std::runtime_error("External contacts and material properties are frozen during internal calibration");
            const bool contrast=key.ends_with("_temperature_contrast");
            const bool signed_coefficient=key=="surface_linear" || key=="surface_quadratic" || contrast || internal || tabulated;
            if(!std::isfinite(value) || (!signed_coefficient && value<=0))throw std::runtime_error("Positive finite scales / finite signed coefficients required");
            if(signed_coefficient && (!dynamic || std::abs(value)>12))throw std::runtime_error("Surface coefficient requires state_geometry and absolute value <=12");
            if(contrast && !directed)throw std::runtime_error("Temperature contrast requires directed_geometry");
            if(dynamic && (key=="iss_cp_scale" || key=="tim_cp_scale"))throw std::runtime_error("Pad heat capacities are fixed in state_geometry calibration");
        }
        double contact=request.value("contact_scale",1.),ambient=request.value("ambient_scale",1.);
        if(request.contains("contact_scale"))for(const auto& key:groups)
            if(request.contains(key))throw std::runtime_error("Cannot combine shared and material-pair contact scales");
        auto scale=[&](const std::string& group){return request.value(group,contact);};
        auto set_response=[&](battery::SurfaceConnectivity& r,std::string material) {
            if(request.contains("surface_linear")) {double a=request.at("surface_linear");r.linear={a,a,a};}
            if(request.contains("surface_quadratic")) {double b=request.at("surface_quadratic");r.quadratic={b,b,b};}
            if(material=="ISS")material="iss";if(material=="TIM")material="tim";
            const auto key=material+"_temperature_contrast";
            if(request.contains(key)) {double c=request.at(key);r.temperature_contrast={c,c,c};}
        };
        auto set_reference=[&](battery::ConnectivityLaw& law,double factor) {
            const double h=100.*factor;
            if(h<=1 || h>=10000)throw std::runtime_error("State geometry reference contact h must be strictly between 1 and 10000 W/m2/K");
            law.geometry_reference_logit=std::log((h-1)/(10000-h));
        };
        J audit={{"objects",J::array()},{"contacts",J::array()},{"boundaries",J::array()}};
        for(auto& c:s.cells) {
            if(!stage.empty() && c.is_battery) {
                auto& t=c.bulk_response.temperature_log_response;
                if(!c.has_bulk_response || t.count!=4 || c.bulk_connectivity.kind!=battery::TransitionKind::DirectedGeometry)
                    throw std::runtime_error("Staged calibration requires four-knot internal response");
                const char* keys[]={"internal_log_35","internal_log_50","internal_log_70"};
                for(int i=0;i<3;++i)if(request.contains(keys[i]))t.value[i+1]=request.at(keys[i]);
                if(request.contains("internal_temperature_contrast")) {
                    const double v=request.at("internal_temperature_contrast");c.bulk_response.temperature_contrast={v,v,v};
                }
                auto& cp=c.heat_capacity_law.factor;
                if(stage=="capacity") {
                    if(cp.count!=3 || c.heat_capacity_law.damage_factor_at_one!=1.)throw std::runtime_error("Independent capacity table required; defect fit disabled");
                    if(request.contains("capacity_45"))cp.value[1]=request.at("capacity_45");
                    if(request.contains("capacity_70"))cp.value[2]=request.at("capacity_70");
                } else if(cp.count)throw std::runtime_error("Conductivity stage requires constant heat capacity");
            }
            if(internal_fit && c.is_battery) {
                if(!c.has_bulk_connectivity || !c.has_bulk_response || c.bulk_connectivity.kind!=battery::TransitionKind::DirectedGeometry)
                    throw std::runtime_error("Internal fit requires an independent directed bulk response");
                if(request.contains("internal_linear")) {const double x=request.at("internal_linear");c.bulk_response.linear={x,x,x};}
                if(request.contains("internal_quadratic")) {const double x=request.at("internal_quadratic");c.bulk_response.quadratic={x,x,x};}
            }
            if(dynamic)set_response(c.surface_connectivity,c.material_id);
            if(!c.is_battery && c.material_id=="ISS")c.heat_capacity_j_per_k*=request.value("iss_cp_scale",1.);
            if(!c.is_battery && c.material_id=="TIM")c.heat_capacity_j_per_k*=request.value("tim_cp_scale",1.);
            audit["objects"].push_back({{"id",c.id},{"material",c.material_id},
                {"heat_capacity_j_per_k",c.heat_capacity_j_per_k},{"conductivity_w_per_m_k",c.conductivity_w_per_m_k},
                {"surface_linear",c.surface_connectivity.linear.x},{"surface_quadratic",c.surface_connectivity.quadratic.x},
                {"temperature_contrast",c.surface_connectivity.temperature_contrast.x},
                {"has_bulk_response",c.has_bulk_response},
                {"internal_linear",c.bulk_response.linear.x},{"internal_quadratic",c.bulk_response.quadratic.x}});
            auto table_json=[](const battery::ScalarTable& t) {
                J out=J::array();for(int i=0;i<t.count;++i)out.push_back({t.temperature_k[i],t.value[i]});return out;
            };
            if(c.bulk_response.temperature_log_response.count)
                audit["objects"].back()["internal_table"]=table_json(c.bulk_response.temperature_log_response);
            if(stage=="links" && c.is_battery)
                audit["objects"].back()["internal_temperature_contrast"]=c.bulk_response.temperature_contrast.x;
            if(c.heat_capacity_law.factor.count) {
                audit["objects"].back()["capacity_table"]=table_json(c.heat_capacity_law.factor);
                audit["objects"].back()["capacity_reference_temperature_k"]=c.heat_capacity_law.reference_temperature_k;
                audit["objects"].back()["capacity_damage_factor_at_one"]=c.heat_capacity_law.damage_factor_at_one;
            }
        }
        for(auto& e:s.interfaces) {
            if(!dynamic && (e.law.kind!=battery::TransitionKind::Constant || e.law.active_conductance_w_per_k!=0 ||
               e.law.damage_rate_at_onset_per_s!=0 || e.law.endpoint_damage_weight!=0 || e.initial_damage!=0)
               )
                throw std::runtime_error("Conditional surface probe currently requires constant contacts");
            const auto group=contact_group(s.cells[e.cell_a].material_id,s.cells[e.cell_b].material_id);
            if(dynamic) {
                if(e.law.kind!=expected_kind)throw std::runtime_error("Mixed contact models in dynamic calibration");
                if(request.contains(group) || request.contains("contact_scale"))set_reference(e.law,scale(group));
            } else {
                e.law.background_conductance_w_per_k*=scale(group);
                e.law.destroyed_background_conductance_w_per_k*=scale(group);
            }
            audit["contacts"].push_back({{"id",e.id},{"group",group},{"scale",scale(group)},
                {"law",battery::name(e.law.kind)}, {"cell_a",s.cells[e.cell_a].id},{"cell_b",s.cells[e.cell_b].id},
                {"reference_h_w_m2_k",dynamic?1.+9999./(1.+std::exp(-e.law.geometry_reference_logit)):100.*scale(group)},
                {"conductance_w_per_k",dynamic?e.law.background_conductance_w_per_k+e.law.active_conductance_w_per_k/(1.+std::exp(-e.law.geometry_reference_logit)):e.law.background_conductance_w_per_k}});
        }
        for(auto& b:s.boundaries) {
            std::string group="fixed";
            if(b.id.starts_with("source-")) {
                group=contact_group("battery",s.cells[b.cell_index].material_id);
                // In links-only scenes measured forcing has a dynamic contact.
                // Its legacy constant-G fallback is frozen, not multiplied again.
                if(stage!="links")b.conductance_w_per_k*=scale(group);
                if(dynamic) {
                    if(!b.has_connectivity || b.connectivity.kind!=expected_kind)throw std::runtime_error("Dynamic source boundary required");
                    if(request.contains(group) || request.contains("contact_scale"))set_reference(b.connectivity,scale(group));
                    set_response(b.remote_surface_connectivity,"battery");
                }
            }
            if(b.id.starts_with("ambient-"))b.conductance_w_per_k*=ambient;
            if(b.id.starts_with("ambient-"))group="ambient_scale";
            audit["boundaries"].push_back({{"id",b.id},{"group",group},{"conductance_w_per_k",b.conductance_w_per_k},
                {"law",b.has_connectivity?battery::name(b.connectivity.kind):"constant"},
                {"remote_temperature_contrast",b.remote_surface_connectivity.temperature_contrast.x}});
            if(stage=="links" && b.has_connectivity)
                audit["boundaries"].back()["reference_h_w_m2_k"]=1.+9999./(1.+std::exp(-b.connectivity.geometry_reference_logit));
        }
        s.settings.max_dt_s=request.value("max_dt_s",s.settings.max_dt_s);
        battery::refine_environment_mesh(s,request.value("mesh_scale",1));
        const auto started=std::chrono::steady_clock::now();battery::Simulation sim(s);
        J result={{"time_s",J::array()},{"temperature_k",J::object()},{"parameters",request},{"applied_model",audit},
                  {"interface_snapshot",J::object()}};
        for(const auto& p:probes)result["temperature_k"][p.id]=J::array();
        // These are numerical subregions of the SAME battery material, not a
        // second casing material or an isothermal merging of shell nodes.
        for(std::size_t object=0;object<s.cells.size();++object)if(s.cells[object].is_battery) {
            const auto& mesh=sim.thermal_mesh();
            for(const bool core:{true,false}) {
                int count=0;double capacity=0;
                for(int j=mesh.object_offsets[object];j<mesh.object_offsets[object+1];++j) {
                    const auto& v=mesh.volumes[j];
                    if((v.region==battery::ThermalRegion::Core)!=core)continue;
                    ++count;capacity+=v.capacity_j_per_k;
                }
                result["battery_regions"][s.cells[object].id][core?"core":"shell"]={
                    {"volume_count",count},{"heat_capacity_j_per_k",capacity},
                    {"mean_temperature_k",J::array()},{"min_temperature_k",J::array()},
                    {"max_temperature_k",J::array()}};
            }
        }
        auto sample=[&] {
            const auto& m=sim.thermal_mesh();const auto& field=sim.thermal_states();
            const auto edge_states=dynamic?sim.thermal_link_states():std::vector<battery::InterfaceState>{};
            auto snapshot=[&](int index,int direction=0) {
                const auto& link=m.links[index];
                const double series_r=battery::phase_resistance(link.series_resistance_k_per_w,link.series_resistance_a_k_per_w,
                    field[link.a].damage,field[link.b].damage,
                    battery::boundary_phase(s.cells[m.volumes[link.a].object]),
                    battery::boundary_phase(s.cells[m.volumes[link.b].object]),sim.time_s(),
                    field[link.a].runaway_started_s,field[link.b].runaway_started_s,
                    field[link.a].conductivity_factor[link.geometry_axis],field[link.b].conductivity_factor[link.geometry_axis]);
                const double reaction_factor=m.volumes[link.a].object==m.volumes[link.b].object?
                    battery::reaction_bulk_factor(field[link.a],field[link.b],link.geometry_axis,direction):1.;
                const double multiplier=m.volumes[link.a].object==m.volumes[link.b].object?
                    battery::bulk_failure_multiplier(battery::boundary_phase(s.cells[m.volumes[link.a].object]),
                        field[link.a].temperature_k,field[link.b].temperature_k,field[link.a].damage,field[link.b].damage,
                        field[link.a].runaway_started_s,field[link.b].runaway_started_s,sim.time_s(),direction):1.;
                if(directed && link.law.kind==battery::TransitionKind::DirectedGeometry)
                    return battery::evaluate_directed_geometry(link.law,edge_states[index],link.geometry_axis,
                        field[link.a].temperature_k,field[link.b].temperature_k,
                        field[link.a].damage,field[link.b].damage,0,link.weight,series_r,reaction_factor,direction,multiplier);
                if(dynamic && link.law.kind==battery::TransitionKind::StateGeometry) {
                    return battery::evaluate_state_geometry(link.law,edge_states[index],link.geometry_axis,
                        field[link.a].temperature_k,field[link.b].temperature_k,
                        field[link.a].damage,field[link.b].damage,0,link.weight,series_r,reaction_factor);
                }
                battery::InterfaceState state;
                const double g0=link.law.background_conductance_w_per_k*link.weight*reaction_factor;
                state.conductance_w_per_k=g0>0?1/(1/g0+series_r):0;
                return state;
            };
            result["time_s"].push_back(sim.time_s());
            for(std::size_t object=0;object<s.cells.size();++object)if(s.cells[object].is_battery) {
                for(const bool core:{true,false}) {
                    double energy=0,capacity=0,lo=INFINITY,hi=-INFINITY;
                    for(int j=m.object_offsets[object];j<m.object_offsets[object+1];++j) {
                        const auto& v=m.volumes[j];if((v.region==battery::ThermalRegion::Core)!=core)continue;
                        const double t=field[j].temperature_k;
                        const double cp=v.capacity_j_per_k*battery::capacity_factor(s.cells[object].heat_capacity_law,t,field[j].damage);
                        energy+=cp*t;capacity+=cp;
                        lo=std::min(lo,t);hi=std::max(hi,t);
                    }
                    auto& region=result["battery_regions"][s.cells[object].id][core?"core":"shell"];
                    region["mean_temperature_k"].push_back(capacity>0?J(energy/capacity):J(nullptr));
                    region["min_temperature_k"].push_back(capacity>0?J(lo):J(nullptr));
                    region["max_temperature_k"].push_back(capacity>0?J(hi):J(nullptr));
                }
            }
            if(dynamic)for(std::size_t i=0;i<s.interfaces.size();++i) {
                double g=0,l=0,gab=0,gba=0,lab=0,lba=0,power=0;
                for(int j=m.interface_offsets[i];j<m.interface_offsets[i+1];++j) {
                    const int index=m.interface_links[j];const auto state=snapshot(index);
                    g+=state.conductance_w_per_k;l+=m.links[index].weight*state.l_m;
                    if(directed) {
                        const auto ab=snapshot(index,1),ba=snapshot(index,-1);const auto& link=m.links[index];
                        gab+=ab.conductance_w_per_k;gba+=ba.conductance_w_per_k;
                        lab+=link.weight*ab.l_m;lba+=link.weight*ba.l_m;
                        power+=state.conductance_w_per_k*(field[link.a].temperature_k-field[link.b].temperature_k);
                    }
                }
                auto& track=result["interface_snapshot"][s.interfaces[i].id];
                if(track.is_null())track={{"conductance_w_per_k",J::array()},{"l_m",J::array()}};
                track["conductance_w_per_k"].push_back(g);track["l_m"].push_back(l);
                if(directed) {
                    for(auto [key,value]:std::initializer_list<std::pair<const char*,double>>{
                        {"a_to_b_conductance_w_per_k",gab},{"b_to_a_conductance_w_per_k",gba},
                        {"a_to_b_l_m",lab},{"b_to_a_l_m",lba},{"net_a_to_b_power_w",power}}) {
                        if(track[key].is_null())track[key]=J::array();track[key].push_back(value);
                    }
                }
            }
            for(const auto& p:probes) {
                double temperature=0,weight=0;
                const auto& cell=s.cells[p.object];
                const double half_r=cell.isothermal?0:cell.dimensions_m[p.axis]/cell.thermal_divisions[p.axis]/(2*cell.conductivity_w_per_m_k[p.axis]);
                for(int j=m.interface_offsets[p.edge];j<m.interface_offsets[p.edge+1];++j) {
                    const auto& link=m.links[m.interface_links[j]];
                    const bool side_a=m.volumes[link.a].object==p.object;
                    const int own=side_a?link.a:link.b,other=side_a?link.b:link.a;
                    const double g=snapshot(m.interface_links[j]).conductance_w_per_k;
                    const double flux=g*(field[other].temperature_k-field[own].temperature_k);
                    const double factor=battery::phase_factor(battery::boundary_phase(cell),field[own].damage,sim.time_s(),field[own].runaway_started_s)*field[own].conductivity_factor[p.axis];
                    temperature+=link.weight*(field[own].temperature_k+flux*half_r/(factor*p.area*link.weight));
                    weight+=link.weight;
                }
                if(weight<=0)throw std::runtime_error("Empty sensor contact");
                result["temperature_k"][p.id].push_back(temperature/weight);
            }
            if(internal_fit || !stage.empty())for(std::size_t object=0;object<s.cells.size();++object)if(s.cells[object].is_battery) {
                double lo=INFINITY,hi=-INFINITY,power=0;
                for(std::size_t index=0;index<m.links.size();++index) {
                    const auto& e=m.links[index];
                    if(e.b<0 || m.volumes[e.a].object!=static_cast<int>(object) || m.volumes[e.b].object!=static_cast<int>(object))continue;
                    const auto v=m.volumes[e.a].size_m;
                    const double h[]={v.x,v.y,v.z};const int axis=e.geometry_axis;
                    const double base_g=s.cells[object].conductivity_w_per_m_k[axis]*h[(axis+1)%3]*h[(axis+2)%3]/h[axis];
                    const double g=snapshot(index).conductance_w_per_k;
                    lo=std::min(lo,g/base_g);hi=std::max(hi,g/base_g);
                    if(e.region==battery::ThermalLinkRegion::CoreShell)
                        power+=(m.volumes[e.a].region==battery::ThermalRegion::Core?1.:-1.)*g*(field[e.a].temperature_k-field[e.b].temperature_k);
                }
                auto& track=result["internal_snapshot"][s.cells[object].id];
                for(auto [key,value]:std::initializer_list<std::pair<const char*,double>>{
                    {"minimum_factor",lo},{"maximum_factor",hi},{"net_core_to_shell_power_w",power}}) {
                    if(track[key].is_null())track[key]=J::array();track[key].push_back(value);
                }
            }
        };
        sample();double next=s.settings.output_interval_s;
        while(!sim.finished()) {
            sim.step(std::max(1e-12,std::min(next,s.settings.duration_s)-sim.time_s()));
            if(sim.time_s()>=next-1e-9 || sim.finished()) {sample();next+=s.settings.output_interval_s;}
        }
        result["energy_residual_j"]=sim.energy_balance_residual_j();
        result["boundary_energy_out_j"]=sim.boundary_energy_out_j();
        result["volumes"]=sim.thermal_mesh().volumes.size();result["steps"]=sim.step_count();
        result["wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::cout<<result.dump()<<std::endl;
    }catch(const std::exception& e){std::cout<<J({{"error",e.what()}}).dump()<<std::endl;}}
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

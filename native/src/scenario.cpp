#include "battery_native/model.hpp"
#include "battery_native/constitutive.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace battery {
namespace {

using json = nlohmann::json;

Vec3 vec3(const json& value, const std::string& where) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(where + " must contain three numbers");
    }
    return {value.at(0).get<double>(), value.at(1).get<double>(), value.at(2).get<double>()};
}

ShapeKind shape_kind(const std::string& value) {
    if (value == "cylinder") return ShapeKind::Cylinder;
    if (value == "prismatic") return ShapeKind::Prismatic;
    if (value == "pouch") return ShapeKind::Pouch;
    throw std::runtime_error("unknown cell shape: " + value);
}

TransitionKind transition_kind(const std::string& value) {
    if (value == "constant") return TransitionKind::Constant;
    if (value == "smooth") return TransitionKind::Smooth;
    if (value == "first_order") return TransitionKind::FirstOrder;
    if (value == "second_order") return TransitionKind::SecondOrder;
    if (value == "state_geometry") return TransitionKind::StateGeometry;
    if (value == "directed_geometry") return TransitionKind::DirectedGeometry;
    throw std::runtime_error("unknown transition kind: " + value);
}

EdgePhase edge_phase(const std::string& value) {
    if (value == "low") return EdgePhase::Low;
    if (value == "high") return EdgePhase::High;
    if (value == "destroyed") return EdgePhase::Destroyed;
    throw std::runtime_error("unknown edge phase: " + value);
}

ScalarTable parse_table(const json& value,bool positive) {
    if(!value.is_array() || value.size()<2 || value.size()>8)
        throw std::runtime_error("Constitutive table requires 2..8 [temperature_k,value] knots");
    ScalarTable t;t.count=static_cast<int>(value.size());
    for(int i=0;i<t.count;++i) {
        if(!value[i].is_array() || value[i].size()!=2)throw std::runtime_error("Invalid constitutive knot");
        t.temperature_k[i]=value[i][0].get<double>();t.value[i]=value[i][1].get<double>();
        if(!std::isfinite(t.temperature_k[i]) || t.temperature_k[i]<=0 || !std::isfinite(t.value[i]) ||
           (positive && t.value[i]<=0) || (i && t.temperature_k[i]<=t.temperature_k[i-1]))
            throw std::runtime_error("Constitutive knots must be finite, ordered, with positive capacities");
    }
    return t;
}
SurfaceConnectivity parse_surface(const json& value) {
    SurfaceConnectivity r;
    r.reference_temperature_k=value.value("reference_temperature_k",298.15);
    r.temperature_scale_k=value.value("temperature_scale_k",100.);
    r.linear=vec3(value.value("linear",json::array({0.,0.,0.})),"surface_connectivity.linear");
    r.quadratic=vec3(value.value("quadratic",json::array({0.,0.,0.})),"surface_connectivity.quadratic");
    r.damage=vec3(value.value("damage",json::array({0.,0.,0.})),"surface_connectivity.damage");
    r.temperature_contrast=vec3(value.value("temperature_contrast",json::array({0.,0.,0.})),"surface_connectivity.temperature_contrast");
    r.damage_contrast=vec3(value.value("damage_contrast",json::array({0.,0.,0.})),"surface_connectivity.damage_contrast");
    if(value.contains("temperature_log_response")) {
        r.temperature_log_response=parse_table(value.at("temperature_log_response"),false);
        for(double v:{r.linear.x,r.linear.y,r.linear.z,r.quadratic.x,r.quadratic.y,r.quadratic.z})
            if(v!=0.)throw std::runtime_error("Tabulated response cannot be combined with a temperature polynomial");
    }
    return r;
}

ConnectivityLaw parse_law(const json& value) {
    ConnectivityLaw law;
    law.kind = transition_kind(value.value("kind", "constant"));
    law.destroyed_geometry_reference_logit=value.value("destroyed_geometry_reference_logit",0.);
    if(law.kind==TransitionKind::DirectedGeometry &&
       (value.contains("l0_m") || value.contains("destroyed_l_m") || value.contains("tensor_only")))
        throw std::runtime_error("Directed contacts use only tensor geometry; obsolete length/compatibility fields are not accepted");
    law.l0_m = law.kind==TransitionKind::DirectedGeometry?0.:value.value("l0_m",0.);
    law.l_low_m = value.value("l_low_m", law.l0_m);
    law.l_high_m = value.value("l_high_m", law.l_low_m);
    law.transition_temperature_k = value.value("transition_temperature_k", 400.0);
    law.transition_width_k = value.value("transition_width_k", 10.0);
    law.hysteresis_k = value.value("hysteresis_k", 0.0);
    law.relaxation_time_s = value.value("relaxation_time_s", 0.0);
    law.background_conductance_w_per_k = value.value("background_conductance_w_per_k", 0.0);
    law.active_conductance_w_per_k = value.value("active_conductance_w_per_k", 0.0);
    law.activation_width_m = value.value("activation_width_m", 1.0e-5);
    law.geometry_reference_logit=value.value("geometry_reference_logit",0.);
    law.damage_onset_k = value.value("damage_onset_k", 450.0);
    law.damage_rate_at_onset_per_s = value.value("damage_rate_at_onset_per_s", 0.0);
    law.damage_temperature_scale_k = value.value("damage_temperature_scale_k", 25.0);
    law.damage_l_shift_m = value.value("damage_l_shift_m", 0.0);
    law.damage_conductance_factor_at_one = value.value("damage_conductance_factor_at_one", 1.0);
    law.endpoint_damage_weight = value.value("endpoint_damage_weight", 0.0);
    law.destruction_damage = value.value("destruction_damage", 1.0);
    law.destroyed_l0_m = value.value("destroyed_l0_m", law.l0_m);
    law.destroyed_l_m = value.value("destroyed_l_m", law.l_low_m);
    law.destroyed_background_conductance_w_per_k =
        value.value("destroyed_background_conductance_w_per_k", law.background_conductance_w_per_k);
    law.destroyed_active_conductance_w_per_k =
        value.value("destroyed_active_conductance_w_per_k", law.active_conductance_w_per_k);
    if (law.activation_width_m <= 0.0 || law.transition_width_k <= 0.0 ||
        law.damage_temperature_scale_k <= 0.0) {
        throw std::runtime_error("interface law scales must be positive");
    }
    return law;
}

}  // namespace

const char* name(ShapeKind value) {
    switch (value) {
        case ShapeKind::Cylinder: return "cylinder";
        case ShapeKind::Prismatic: return "prismatic";
        case ShapeKind::Pouch: return "pouch";
    }
    return "unknown";
}

const char* name(TransitionKind value) {
    switch (value) {
        case TransitionKind::Constant: return "constant";
        case TransitionKind::Smooth: return "smooth";
        case TransitionKind::FirstOrder: return "first_order";
        case TransitionKind::SecondOrder: return "second_order";
        case TransitionKind::StateGeometry: return "state_geometry";
        case TransitionKind::DirectedGeometry: return "directed_geometry";
    }
    return "unknown";
}

const char* name(EdgePhase value) {
    switch (value) {
        case EdgePhase::Low: return "low";
        case EdgePhase::High: return "high";
        case EdgePhase::Destroyed: return "destroyed";
    }
    return "unknown";
}

const char* name(CellMode value) {
    switch (value) {
        case CellMode::Healthy: return "healthy";
        case CellMode::Runaway: return "runaway";
        case CellMode::Spent: return "spent";
        case CellMode::PartiallySpent: return "partially-spent";
        case CellMode::Quenched: return "quenched";
    }
    return "unknown";
}

void refine_environment_mesh(Scenario& scenario, int scale) {
    if (scale < 1 || scale > 8)
        throw std::runtime_error("mesh-scale must be 1..8");
    const bool passive=std::any_of(scenario.cells.begin(),scenario.cells.end(),[](const auto& c){return !c.is_battery;});
    for(auto& c:scenario.cells)if(!passive || !c.is_battery) {
        if(!c.initial_temperature_field_k.empty() && scale!=1)
            throw std::runtime_error("Cannot refine an initial field without an interpolation policy");
        for(auto& n:c.thermal_divisions)n*=scale;
    }
}

Scenario load_scenario(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open scenario: " + path);
    json root;
    stream >> root;

    Scenario scenario;
    scenario.name = root.at("name").get<std::string>();
    scenario.source = root.value("source", "unspecified");
    const auto metadata = root.value("metadata", json::object());
    scenario.calibration_status = metadata.value("calibration_status", "unspecified");
    scenario.geometry_status = metadata.value("geometry_status", "");
    const auto& simulation = root.at("simulation");
    scenario.settings.duration_s = simulation.at("duration_s").get<double>();
    scenario.settings.max_dt_s = simulation.at("max_dt_s").get<double>();
    scenario.settings.output_interval_s = simulation.at("output_interval_s").get<double>();
    scenario.settings.stability_factor = simulation.value("stability_factor", 0.2);
    scenario.settings.step_interval_s = simulation.value("step_interval_s", 0.0);

    std::unordered_map<std::string, std::size_t> cell_index;
    for (const auto& value : root.at("cells")) {
        CellConfig cell;
        cell.id = value.at("id").get<std::string>();
        const auto role=value.value("role", std::string("battery"));
        if(role!="battery" && role!="passive")throw std::runtime_error("Unknown object role: "+role);
        cell.is_battery=role=="battery";
        cell.material_id=value.value("material_id",std::string("unspecified"));
        if (!cell_index.emplace(cell.id, scenario.cells.size()).second) {
            throw std::runtime_error("duplicate cell id: " + cell.id);
        }
        const auto& geometry = value.at("geometry");
        cell.shape = shape_kind(geometry.at("kind").get<std::string>());
        cell.dimensions_m = geometry.at("dimensions_m").get<std::vector<double>>();
        const std::size_t expected = cell.shape == ShapeKind::Cylinder ? 2 : 3;
        if (cell.dimensions_m.size() != expected) {
            throw std::runtime_error("wrong dimension count for cell " + cell.id);
        }
        const auto pose = value.value("pose", json::object());
        cell.position_m = vec3(pose.value("position_m", json::array({0.0, 0.0, 0.0})),
                               "pose.position_m");
        const auto orientation = pose.value("orientation_xyzw", json::array({0.0, 0.0, 0.0, 1.0}));
        if (!orientation.is_array() || orientation.size() != 4) {
            throw std::runtime_error("orientation_xyzw must contain four numbers");
        }
        for (std::size_t index = 0; index < 4; ++index) {
            cell.orientation_xyzw[index] = orientation.at(index).get<double>();
        }
        const auto& thermal = value.at("thermal");
        cell.heat_capacity_j_per_k = thermal.at("heat_capacity_j_per_k").get<double>();
        cell.initial_temperature_k = thermal.value("initial_temperature_k", 298.15);
        cell.thermal_divisions = thermal.value("divisions", std::array<int,3>{4,4,4});
        cell.conductivity_w_per_m_k = thermal.value("conductivity_w_per_m_k", std::array<double,3>{1,1,1});
        cell.initial_temperature_field_k = thermal.value("initial_temperature_field_k", std::vector<double>{});
        cell.surface_connectivity=parse_surface(thermal.value("surface_connectivity",json::object()));
        cell.bulk_destruction_on_boundary=thermal.value("bulk_destruction_on_boundary",false);
        if(thermal.contains("bulk_failure_transient")) {
            const auto& f=thermal.at("bulk_failure_transient");
            cell.bulk_failure_transient={f.at("peak_factor"),f.at("relaxation_time_s")};
            if(cell.bulk_failure_transient.relaxation_time_s<=0)
                throw std::runtime_error("Explicit bulk transient requires positive relaxation time");
        }
        if(thermal.contains("reaction_structure")) {
            const auto& r=thermal.at("reaction_structure");auto& p=cell.reaction_structure;p.enabled=true;
            const auto a=r.at("active_factor").get<std::array<double,3>>();
            const auto b=r.at("residue_factor").get<std::array<double,3>>();
            for(int axis=0;axis<3;++axis) {p.active_factor[axis]=a[axis];p.residue_factor[axis]=b[axis];}
            p.activity_rate_per_s=r.at("activity_rate_per_s");
        }
        const auto transfer = thermal.value("internal_heat_transfer", std::string("finite_volume"));
        if (transfer != "finite_volume" && transfer != "isothermal")
            throw std::runtime_error("Unknown internal_heat_transfer for cell " + cell.id);
        cell.isothermal = transfer == "isothermal";
        if(thermal.contains("heat_capacity_law")) {
            const auto& law=thermal.at("heat_capacity_law");auto& l=cell.heat_capacity_law;
            l.factor=parse_table(law.at("factor"),true);
            l.reference_temperature_k=law.value("reference_temperature_k",298.15);
            l.damage_factor_at_one=law.value("damage_factor_at_one",1.);
            if(!std::isfinite(l.reference_temperature_k) || l.reference_temperature_k<=0 ||
               !std::isfinite(l.damage_factor_at_one) || l.damage_factor_at_one<=0 ||
               std::abs(table_value(l.factor,l.reference_temperature_k)-1.)>1e-12)
                throw std::runtime_error("Capacity law must be positive and anchored at C_ref");
            if(cell.isothermal)throw std::runtime_error("Variable capacity currently requires finite_volume transfer");
        }
        if (thermal.contains("bulk_connectivity")) {
            auto bulk = thermal.at("bulk_connectivity");
            bulk["background_conductance_w_per_k"] = bulk.value("background_factor", 1.0);
            bulk["active_conductance_w_per_k"] = bulk.value("active_factor", 0.0);
            bulk["destroyed_background_conductance_w_per_k"] = bulk.value("destroyed_background_factor", bulk.value("background_factor", 1.0));
            bulk["destroyed_active_conductance_w_per_k"] = bulk.value("destroyed_active_factor", bulk.value("active_factor", 0.0));
            cell.bulk_connectivity = parse_law(bulk);
            cell.has_bulk_connectivity = true;
        }
        if(thermal.contains("bulk_response")) {
            if(!cell.has_bulk_connectivity)throw std::runtime_error("bulk_response requires bulk_connectivity: "+cell.id);
            cell.bulk_response=parse_surface(thermal.at("bulk_response"));
            cell.has_bulk_response=true;
        }
        const auto degradation = value.value("degradation", json::object());
        cell.aging_fraction = degradation.value("aging_fraction", 0.0);
        cell.initial_damage = degradation.value("initial_damage", 0.0);
        cell.damage_onset_k = degradation.value("onset_k", 420.0);
        cell.damage_rate_at_onset_per_s = degradation.value("rate_at_onset_per_s", 0.0);
        cell.damage_temperature_scale_k = degradation.value("temperature_scale_k", 25.0);
        cell.aging_damage_multiplier = degradation.value("aging_damage_multiplier", 0.0);
        const auto& runaway = value.at("runaway");
        cell.runaway_energy_j = runaway.at("energy_j").get<double>();
        const auto profile=runaway.value("profile",std::string("constant_power"));
        if(profile!="constant_power" && profile!="burst_tail" && profile!="kinetic")throw std::runtime_error("Unknown runaway profile");
        cell.runaway_pulse.profile=profile=="kinetic"?RunawayProfile::Kinetic:
            profile=="burst_tail"?RunawayProfile::BurstTail:RunawayProfile::ConstantPower;
        if(profile=="kinetic") {
            if(runaway.contains("duration_s"))throw std::runtime_error("Kinetic reaction has no prescribed duration");
            const auto& r=runaway.at("kinetics");auto& p=cell.reaction_kinetics;
            p.reference_temperature_k=r.at("reference_temperature_k");
            p.rate_at_reference_per_s=r.at("rate_at_reference_per_s");
            p.activation_energy_j_per_mol=r.at("activation_energy_j_per_mol");
            p.activation_damage=r.at("activation_damage");
            p.damage_sensitivity=r.value("damage_sensitivity",2.);
            p.damage_per_conversion=r.value("damage_per_conversion",1.);
            p.quench_rate_per_s=r.value("quench_rate_per_s",1e-5);
            p.max_fraction_per_step=r.value("max_fraction_per_step",.02);
            p.max_temperature_rise_per_step_k=r.value("max_temperature_rise_per_step_k",2.);
            if(runaway.contains("energy_outlet")) {
                const auto& v=runaway.at("energy_outlet");auto& o=cell.energy_outlet;o.enabled=true;
                if(v.at("sink")!="outside_domain")throw std::runtime_error("Energy outlet currently supports only outside_domain, not a neighboring cell");
                cell.energy_outlet_port=v.at("port_id");
                o.opening_start_conversion=v.at("opening_start_conversion");o.opening_end_conversion=v.at("opening_end_conversion");
                o.closure_start_conversion=v.value("closure_start_conversion",1.);o.closure_end_conversion=v.value("closure_end_conversion",1.);
                o.heat_fraction=v.at("heat_fraction");
                o.ejected_energy_per_released_energy=v.value("ejected_energy_per_released_energy",0.);
            }
        } else {
            cell.runaway_duration_s=runaway.at("duration_s");
            if(runaway.contains("kinetics") || runaway.contains("energy_outlet") || cell.reaction_structure.enabled)
                throw std::runtime_error("Reaction kinetics, structure and energy outlet require kinetic profile");
        }
        cell.runaway_pulse.fast_fraction=runaway.value("fast_fraction",.8);
        cell.runaway_pulse.tail_time_s=runaway.value("tail_time_s",30.);
        if(value.contains("electrical")) {
            const auto& e=value.at("electrical");auto& p=cell.electrical;p.enabled=true;
            p.current_a=e.value("current_a",0.);p.resistance_ohm=e.at("resistance_ohm");
            p.reference_temperature_k=e.value("reference_temperature_k",298.15);
            p.temperature_log_slope_per_k=e.value("temperature_log_slope_per_k",0.);
            p.damage_log_slope=e.value("damage_log_slope",0.);
            p.min_resistance_ohm=e.value("min_resistance_ohm",1e-6);
            p.max_resistance_ohm=e.value("max_resistance_ohm",1.);
            p.max_temperature_step_k=e.value("max_temperature_step_k",.5);
            p.overcharge_fraction=e.value("overcharge_fraction",0.);
            p.nominal_capacity_coulomb=e.value("nominal_capacity_coulomb",0.);
            cell.current_profile_a=e.value("current_profile_a",std::vector<std::array<double,2>>{});
            if(!e.contains("current_a") && cell.current_profile_a.empty())throw std::runtime_error("Electrical heating requires a current drive");
            if(e.contains("current_a") && !cell.current_profile_a.empty())throw std::runtime_error("Specify constant current OR its profile, not both");
        }
        if(profile!="burst_tail" && (runaway.contains("fast_fraction") || runaway.contains("tail_time_s")))
            throw std::runtime_error("Pulse shape parameters require burst_tail");
        if (cell.heat_capacity_j_per_k <= 0.0 || cell.runaway_duration_s <= 0.0) {
            throw std::runtime_error("invalid thermal parameters for cell " + cell.id);
        }
        scenario.cells.push_back(std::move(cell));
    }

    for(const auto& value:root.value("shells",json::array())) {
        ShellConfig sh;sh.contents_id=value.at("contents_id");sh.material_id=value.at("material_id");
        sh.thickness_m=value.at("thickness_m");sh.density_kg_m3=value.at("density_kg_m3");sh.specific_heat_j_kg_k=value.at("specific_heat_j_kg_k");
        sh.conductivity_w_per_m_k=value.at("conductivity_w_per_m_k").get<std::array<double,3>>();
        sh.contents_contact_h=value.at("contents_contact_h");sh.seam_contact_h=value.at("seam_contact_h");
        sh.damage_onset_k=value.value("damage_onset_k",500.);sh.damage_rate_per_s=value.value("damage_rate_per_s",0.);
        sh.damage_scale_k=value.value("damage_scale_k",25.);sh.destroyed_conductance_factor=value.value("destroyed_conductance_factor",.01);
        sh.initial_temperature_k=value.value("initial_temperature_k",-1.);
        if(value.contains("surface_divisions")) {
            sh.surface_divisions=value.at("surface_divisions").get<std::array<int,3>>();
            for(int n:sh.surface_divisions)if(n<=0)throw std::runtime_error("Shell surface_divisions must be positive");
        }
        if(value.contains("initial_temperature_k") && sh.initial_temperature_k<=0)throw std::runtime_error("Shell temperature must be positive");
        add_physical_shell(scenario,sh);
        const auto frame=value.value("conductivity_frame",std::string("global"));
        if(frame!="global" && frame!="normal_tangential")throw std::runtime_error("Unknown casing conductivity frame");
        for(auto& panel:scenario.cells)if(panel.shell_contents_id==sh.contents_id) {
            const auto face=panel.id.substr(panel.id.size()-2);const int axis=face[1]-'x';
            if(frame=="normal_tangential")for(int k=0;k<3;++k)panel.conductivity_w_per_m_k[(axis+k)%3]=sh.conductivity_w_per_m_k[k];
            if(value.contains("surface_connectivity"))panel.surface_connectivity=parse_surface(value.at("surface_connectivity"));
            if(value.contains("initial_temperature_fields_k") && value.at("initial_temperature_fields_k").contains(face))
                panel.initial_temperature_field_k=value.at("initial_temperature_fields_k").at(face).get<std::vector<double>>();
        }
    }
    for(std::size_t i=0;i<scenario.cells.size();++i)cell_index[scenario.cells[i].id]=i;
    std::unordered_map<std::string, std::size_t> port_index;
    for(std::size_t i=0;i<scenario.ports.size();++i)port_index[scenario.ports[i].id]=i;
    for (const auto& value : root.at("ports")) {
        PortConfig port;
        port.id = value.at("id").get<std::string>();
        port.cell_id = value.at("cell_id").get<std::string>();
        const auto cell = cell_index.find(port.cell_id);
        if (cell == cell_index.end()) throw std::runtime_error("unknown cell for port " + port.id);
        port.cell_index = cell->second;
        port.kind = value.value("kind", "contact");
        port.surface = value.value("surface", "unspecified");
        port.area_m2 = value.at("area_m2").get<double>();
        port.rectangle_m=value.value("rectangle_m",std::array<double,2>{0.,0.});
        port.centroid_m = vec3(value.at("centroid_m"), "port.centroid_m");
        port.normal = vec3(value.at("normal"), "port.normal");
        if (!port_index.emplace(port.id, scenario.ports.size()).second) {
            throw std::runtime_error("duplicate port id: " + port.id);
        }
        scenario.ports.push_back(std::move(port));
    }

    for (const auto& value : root.value("interfaces", json::array())) {
        InterfaceConfig edge;
        edge.id = value.at("id").get<std::string>();
        edge.port_a = value.at("port_a").get<std::string>();
        edge.port_b = value.at("port_b").get<std::string>();
        if (!port_index.contains(edge.port_a) || !port_index.contains(edge.port_b)) {
            throw std::runtime_error("unknown interface port for " + edge.id);
        }
        const auto& port_a = scenario.ports[port_index.at(edge.port_a)];
        const auto& port_b = scenario.ports[port_index.at(edge.port_b)];
        edge.cell_a = port_a.cell_index;
        edge.cell_b = port_b.cell_index;
        edge.point_a_m = port_a.centroid_m;
        edge.point_b_m = port_b.centroid_m;
        if (edge.cell_a == edge.cell_b) throw std::runtime_error("self interface: " + edge.id);
        edge.material_a = value.value("material_a", "unspecified");
        edge.material_b = value.value("material_b", "unspecified");
        edge.interface_material = value.value("interface_material", "unspecified");
        edge.law = parse_law(value.at("law"));
        edge.initial_l_m = value.value("initial_l_m", edge.law.l_low_m);
        edge.initial_damage = value.value("initial_damage", 0.0);
        edge.initial_phase = edge_phase(value.value("initial_phase", "low"));
        scenario.interfaces.push_back(std::move(edge));
    }

    for (const auto& value : root.value("boundaries", json::array())) {
        AmbientBoundary boundary;
        boundary.id = value.at("id").get<std::string>();
        boundary.port_id = value.at("port_id").get<std::string>();
        if (!port_index.contains(boundary.port_id)) {
            throw std::runtime_error("unknown boundary port: " + boundary.port_id);
        }
        boundary.cell_index = scenario.ports[port_index.at(boundary.port_id)].cell_index;
        boundary.temperature_k = value.at("temperature_k").get<double>();
        boundary.conductance_w_per_k = value.at("conductance_w_per_k").get<double>();
        boundary.temperature_profile = value.value("temperature_profile", std::vector<std::array<double,2>>{});
        if(value.contains("connectivity")) {
            boundary.has_connectivity=true;
            boundary.connectivity=parse_law(value.at("connectivity"));
            boundary.remote_surface_connectivity=parse_surface(value.value("remote_surface_connectivity",json::object()));
        }
        scenario.boundaries.push_back(std::move(boundary));
    }

    if(root.contains("gas_network")) {
        const auto& net=root.at("gas_network");auto& gm=scenario.gas_model;
        const std::string transport=net.value("transport",std::string("legacy_coupled"));
        if(transport!="legacy_coupled" && transport!="split_local_equilibrium")throw std::runtime_error("Unknown gas transport scheme");
        gm.split_transport=transport=="split_local_equilibrium";
        const auto backend=net.value("mass_backend",std::string("cuda_global"));
        if(backend!="cuda_global" && backend!="cpu_local" && backend!="cuda_kawasaki" && backend!="cuda_batched" && backend!="cuda_deterministic" && backend!="cuda_network")throw std::runtime_error("Unknown mass backend");
        gm.cpu_local_transport=backend=="cpu_local";
        gm.cuda_deterministic_transport=backend=="cuda_deterministic";
        gm.cuda_network_transport=backend=="cuda_network";
        gm.cuda_batched_transport=backend=="cuda_batched" || gm.cuda_deterministic_transport || gm.cuda_network_transport;
        gm.cuda_kawasaki_transport=backend=="cuda_kawasaki" || gm.cuda_batched_transport;
        gm.deterministic_sweeps_per_step=net.value("deterministic_sweeps_per_step",gm.deterministic_sweeps_per_step);
        gm.deterministic_pressure_fraction=net.value("deterministic_pressure_fraction",gm.deterministic_pressure_fraction);
        gm.network_pressure_fraction=net.value("network_pressure_fraction",gm.network_pressure_fraction);
        gm.batch_pressure_fraction=net.value("batch_pressure_fraction",gm.batch_pressure_fraction);
        gm.batch_mean_threshold=net.value("batch_mean_threshold",gm.batch_mean_threshold);
        gm.kawasaki_packet_fraction=net.value("kawasaki_packet_fraction",gm.kawasaki_packet_fraction);
        gm.kawasaki_tail_atol_pa=net.value("kawasaki_tail_atol_pa",gm.kawasaki_tail_atol_pa);
        gm.kawasaki_tail_rtol=net.value("kawasaki_tail_rtol",gm.kawasaki_tail_rtol);
        gm.kawasaki_seed=net.value("kawasaki_seed",gm.kawasaki_seed);
        const auto integrator=net.value("local_integrator",std::string("explicit"));
        if(integrator!="explicit" && integrator!="implicit")throw std::runtime_error("Unknown local gas integrator");
        gm.implicit_local_transport=integrator=="implicit";
        gm.implicit_temperature_tolerance_k=net.value("implicit_temperature_tolerance_k",gm.implicit_temperature_tolerance_k);
        gm.implicit_tracer_relative_tolerance=net.value("implicit_tracer_relative_tolerance",gm.implicit_tracer_relative_tolerance);
        gm.local_exchange_interval_s=net.value("local_exchange_interval_s",gm.local_exchange_interval_s);
        gm.reference_temperature_k=net.at("reference_temperature_k");gm.cv_j_per_kg_k=net.at("cv_j_per_kg_k");
        gm.specific_r_j_per_kg_k=net.at("specific_r_j_per_kg_k");
        gm.max_fraction_per_step=net.value("max_fraction_per_step",.01);gm.max_temperature_step_k=net.value("max_temperature_step_k",.5);
        gm.pressure_relaxation_fraction=net.value("pressure_relaxation_fraction",gm.pressure_relaxation_fraction);
        gm.pressure_absolute_tolerance_pa=net.value("pressure_absolute_tolerance_pa",gm.pressure_absolute_tolerance_pa);
        gm.pressure_relative_tolerance=net.value("pressure_relative_tolerance",gm.pressure_relative_tolerance);
        std::unordered_map<std::string,int> gas_ids;
        for(const auto& v:net.at("compartments")) {
            GasCompartment g;g.id=v.at("id");g.volume_m3=v.at("volume_m3");g.initial_mass_kg=v.at("initial_mass_kg");g.initial_temperature_k=v.at("initial_temperature_k");
            g.initial_chemical_j=v.value("initial_chemical_j",0.);g.initial_oxidizer_mass_fraction=v.value("initial_oxidizer_mass_fraction",0.);
            g.reaction_rate_per_s=v.value("reaction_rate_per_s",0.);g.oxidizer_heat_j_per_kg=v.value("oxidizer_heat_j_per_kg",13.1e6);
            g.ignition_temperature_k=v.value("ignition_temperature_k",0.);
            if(v.contains("local_equilibrium")) {
                const auto& binding=v.at("local_equilibrium");
                for(const auto& item:binding.items())if(item.key()!="cell_id" && item.key()!="thermal_volume" &&
                    item.key()!="reactant_density_kg_m3" && item.key()!="residue_density_kg_m3")
                    throw std::runtime_error("Unknown local_equilibrium property: "+item.key());
                g.thermal_cell=static_cast<int>(cell_index.at(binding.at("cell_id").get<std::string>()));
                g.thermal_volume=binding.at("thermal_volume");
                g.reactant_density_kg_m3=binding.value("reactant_density_kg_m3",0.);
                g.residue_density_kg_m3=binding.value("residue_density_kg_m3",0.);
            }
            if(v.contains("contents_id"))g.owner_cell=static_cast<int>(cell_index.at(v.at("contents_id").get<std::string>()));
            if(!gas_ids.emplace(g.id,static_cast<int>(scenario.gas_compartments.size())).second)throw std::runtime_error("Duplicate gas compartment");
            scenario.gas_compartments.push_back(g);
        }
        for(const auto& v:net.value("channels",json::array())) {
            GasChannel ch;ch.id=v.at("id");ch.from=gas_ids.at(v.at("from").get<std::string>());
            const std::string to=v.at("to");ch.to=to=="outside_domain"?-1:gas_ids.at(to);
            const std::string kind=v.at("kind");if(kind!="vent" && kind!="rupture" && kind!="internal" && kind!="passage")throw std::runtime_error("Unknown gas channel kind");
            ch.internal=kind=="internal";
            ch.passage=kind=="passage";
            ch.shell_thermal_volume=v.value("shell_thermal_volume",-1);
            if(ch.internal || ch.passage) {
                if(v.contains("port_id"))throw std::runtime_error("Internal pore path must not masquerade as an exterior port");
                ch.shell_object=scenario.gas_compartments[ch.from].thermal_cell;
            } else {
                ch.port_id=v.at("port_id");
                const auto& p=scenario.ports.at(port_index.at(ch.port_id));ch.shell_object=static_cast<int>(p.cell_index);
            }
            ch.area_m2=v.value("area_m2",0.);ch.discharge_coefficient=v.value("discharge_coefficient",.7);
            ch.opening_pressure_pa=v.at("opening_pressure_pa");ch.outside_pressure_pa=v.value("outside_pressure_pa",101325.);
            ch.rupture=kind=="rupture";ch.strength_reference_k=v.value("strength_reference_k",298.15);
            ch.strength_temperature_scale_k=v.value("strength_temperature_scale_k",1e6);
            if(v.contains("surface_link")) {
                const auto& link=v.at("surface_link");
                if(!link.is_object() || link.size()!=1 || !link.contains("h_w_per_m2_k"))
                    throw std::runtime_error("surface_link requires only h_w_per_m2_k");
                ch.surface_link_h_w_per_m2_k=link.at("h_w_per_m2_k");
                if(!std::isfinite(ch.surface_link_h_w_per_m2_k) || ch.surface_link_h_w_per_m2_k<=0)
                    throw std::runtime_error("surface_link h must be positive");
            }
            if(v.contains("damage_opening")) {
                const auto& o=v.at("damage_opening");
                if(!o.is_object())throw std::runtime_error("damage_opening must be an object");
                for(const auto& item:o.items())if(item.key()!="damage_onset" && item.key()!="exponent" &&
                    item.key()!="pressure_damage_rate_per_s" && item.key()!="max_damage_step" &&
                    item.key()!="seam_width_m" && item.key()!="seam_strength_factor" && item.key()!="fracture_energy_j_per_m2" &&
                    item.key()!="initial_mechanical_damage" && item.key()!="mode" &&
                    item.key()!="pixel_rate_per_s" && item.key()!="pixel_pressure_exponent" && item.key()!="seed" && item.key()!="fragment_mean_area_m2" && item.key()!="fixed_area_fraction" &&
                    item.key()!="area_rate_per_s" && item.key()!="pressure_exponent" && item.key()!="valve")
                    throw std::runtime_error("Unknown damage_opening property: "+item.key());
                ch.opening.enabled=true;
                const auto mode=o.value("mode",std::string("progressive"));
                if(mode!="progressive" && mode!="binary_pixels" && mode!="stochastic_pixels" && mode!="stochastic_area" && mode!="stochastic_fraction" && mode!="deterministic_aperture")throw std::runtime_error("Unknown shell opening mode");
                ch.opening.deterministic_aperture=mode=="deterministic_aperture";
                if(!ch.opening.deterministic_aperture && (o.contains("area_rate_per_s") || o.contains("pressure_exponent") || o.contains("valve")))throw std::runtime_error("Aperture parameters require deterministic_aperture");
                if(ch.opening.deterministic_aperture) {
                    for(const auto* key:{"seed","pixel_rate_per_s","pixel_pressure_exponent","fragment_mean_area_m2","fixed_area_fraction","damage_onset","exponent","pressure_damage_rate_per_s"})
                        if(o.contains(key))throw std::runtime_error("Inapplicable deterministic aperture property: "+std::string(key));
                    ch.opening.area_rate_per_s=o.value("area_rate_per_s",1.);
                    ch.opening.pressure_exponent=o.value("pressure_exponent",1.);
                    if(o.contains("valve")) {
                        const auto& valve=o.at("valve");
                        if(!valve.is_object())throw std::runtime_error("Valve must be an object");
                        for(const auto& item:valve.items())if(item.key()!="radius_m" && item.key()!="center_m" && item.key()!="pressure_factor")throw std::runtime_error("Unknown valve property");
                        const auto xy=valve.at("center_m").get<std::array<double,2>>();
                        ch.opening.valve_u_m=xy[0];ch.opening.valve_v_m=xy[1];
                        ch.opening.valve_radius_m=valve.at("radius_m");
                        ch.opening.valve_pressure_factor=valve.at("pressure_factor");
                        if(ch.opening.valve_radius_m<=0)throw std::runtime_error("Valve radius must be positive");
                    }
                }
                ch.opening.binary_pixels=mode=="binary_pixels" || mode=="stochastic_pixels";
                ch.opening.stochastic_pixels=mode=="stochastic_pixels";
                ch.opening.stochastic_area=mode=="stochastic_area" || mode=="stochastic_fraction";
                if(o.contains("fixed_area_fraction") && mode!="stochastic_fraction")throw std::runtime_error("fixed_area_fraction requires stochastic_fraction mode");
                if(mode=="stochastic_fraction" && o.contains("fragment_mean_area_m2"))throw std::runtime_error("Fixed-fraction mode has no random fragment area");
                ch.opening.fixed_area_fraction=mode=="stochastic_fraction"?o.value("fixed_area_fraction",.02):0.;
                if(mode=="stochastic_fraction" && (!std::isfinite(ch.opening.fixed_area_fraction) || ch.opening.fixed_area_fraction<=0 || ch.opening.fixed_area_fraction>1))throw std::runtime_error("Invalid fixed area fraction");
                ch.opening.fragment_mean_area_m2=o.value("fragment_mean_area_m2",1e-5);
                ch.opening.pixel_rate_per_s=o.value("pixel_rate_per_s",1.);
                ch.opening.pixel_pressure_exponent=o.value("pixel_pressure_exponent",1.);
                if(o.contains("seed") && !o.at("seed").is_number_unsigned())throw std::runtime_error("Shell seed must be an unsigned integer");
                ch.opening.rupture_seed=o.value("seed",std::uint64_t{1});
                ch.opening.initial_mechanical_damage=o.value("initial_mechanical_damage",0.);
                ch.opening.damage_onset=o.value("damage_onset",.2);ch.opening.exponent=o.value("exponent",2.);
                ch.opening.pressure_damage_rate_per_s=o.value("pressure_damage_rate_per_s",1.);
                ch.opening.max_damage_step=o.value("max_damage_step",.02);
                ch.opening.seam_width_m=o.value("seam_width_m",0.);
                ch.opening.seam_strength_factor=o.value("seam_strength_factor",1.);
                ch.opening.fracture_energy_j_per_m2=o.value("fracture_energy_j_per_m2",0.);
            }
            scenario.gas_channels.push_back(ch);
        }
        for(const auto& v:net.value("thermal_contacts",json::array())) {
            GasContact c;c.compartment=gas_ids.at(v.at("compartment").get<std::string>());c.port_id=v.at("port_id");c.h_w_per_m2_k=v.at("h_w_per_m2_k");c.thermal_volume=v.value("thermal_volume",-1);scenario.gas_contacts.push_back(c);
        }
        for(const auto& v:root.at("cells"))if(v.contains("conversion")) {
            auto& c=scenario.cells[cell_index.at(v.at("id").get<std::string>())];auto& p=c.conversion;const auto& x=v.at("conversion");p.enabled=true;
            if(x.at("chemistry")!="r1-r2-r3")throw std::runtime_error("Only R1/R2/R3 chemistry is supported");
            for(const char* old:{"l0_m","lcrit_m","energy_scale_j_per_kg","activation_rate_per_s","absorption_j_per_kg"})
                if(x.contains(old))throw std::runtime_error(std::string("Obsolete chemistry field: ")+old);
            p.mass_kg=x.at("mass_kg");p.reactant_cp=x.at("reactant_cp");p.activated_cp=x.at("activated_cp");p.residue_cp=x.at("residue_cp");
            p.net_heat_j_per_kg=x.at("net_heat_j_per_kg");
            p.degradation_rate_per_s=x.at("degradation_rate_per_s");
            p.reaction_rate_per_s=x.at("reaction_rate_per_s");
            p.reaction_reference_k=x.at("reaction_reference_k");p.reaction_activation_j_per_mol=x.at("reaction_activation_j_per_mol");
            p.degradation_reference_k=x.at("degradation_reference_k");
            p.degradation_activation_j_per_mol=x.at("degradation_activation_j_per_mol");
            p.initial_precursor_fraction=x.value("initial_precursor_fraction",0.);
            p.overcharge_conversion_per_charge=x.value("overcharge_conversion_per_charge",0.);
            p.runaway_reporting_rate_per_s=x.value("runaway_reporting_rate_per_s",1e-5);
            p.gas_yield_reference_k=x.value("gas_yield_reference_k",p.reaction_reference_k);
            p.gas_yield_temperature_per_k=x.value("gas_yield_temperature_per_k",0.);
            p.gas_yield_density_exponent=x.value("gas_yield_density_exponent",0.);
            p.gas_yield_max=x.value("gas_yield_max",.95);
            p.gas_yield=x.at("gas_yield");p.gas_chemical_fraction=x.at("gas_chemical_fraction");
            p.max_fraction_per_step=x.value("max_fraction_per_step",.01);p.max_temperature_step_k=x.value("max_temperature_step_k",.5);
            const auto f=x.at("residue_conductivity").get<std::array<double,3>>();for(int k=0;k<3;++k)p.residue_conductivity[k]=f[k];
            p.gas_compartment=gas_ids.at(x.at("gas_compartment").get<std::string>());
        }
    } else for(const auto& v:root.at("cells"))if(v.contains("conversion"))throw std::runtime_error("Conversion requires an explicit gas network");

    for (const auto& value : root.value("primary_failures", json::array())) {
        FailureEvent event;
        event.cell_id = value.at("cell_id").get<std::string>();
        if (!cell_index.contains(event.cell_id)) {
            throw std::runtime_error("unknown primary failure cell: " + event.cell_id);
        }
        event.cell_index = cell_index.at(event.cell_id);
        event.time_s = value.at("time_s").get<double>();
        event.thermal_volume = value.value("thermal_volume", -1);
        scenario.primary_failures.push_back(std::move(event));
    }
    std::sort(scenario.primary_failures.begin(), scenario.primary_failures.end(),
              [](const FailureEvent& a, const FailureEvent& b) { return a.time_s < b.time_s; });

    if (root.contains("observations")) {
        const auto& observations = root.at("observations");
        if (observations.contains("propagated") && observations.at("propagated").is_boolean()) {
            scenario.has_expected_propagation = true;
            scenario.expected_propagation = observations.at("propagated").get<bool>();
        }
        scenario.expected_runaway_order =
            observations.value("runaway_order", std::vector<std::string>{});
    }
    return scenario;
}

}  // namespace battery

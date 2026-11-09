#include "scenario_internal.hpp"
#include "battery_native/thermal/constitutive.hpp"
#include "battery_native/thermal/state_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace battery::scenario_detail {
CellIndex parse_thermal_links(const json& root, Scenario& scenario, const CellIndex& cell_index) {
    std::unordered_set<int> driven_heaters;
    for (const auto& value : root.value("heating_protocols", json::array())) {
        HeatingProtocol p;
        p.id = value.at("id");
        p.heater_id = value.at("heater_id");
        p.sensor_id = value.at("sensor_id");
        if (!cell_index.contains(p.heater_id) || !cell_index.contains(p.sensor_id))
            throw std::runtime_error("Unknown heating protocol heater/sensor");
        p.heater_object = cell_index.at(p.heater_id);
        p.sensor_object = cell_index.at(p.sensor_id);
        if (!driven_heaters.insert(p.heater_object).second)
            throw std::runtime_error("Duplicate heater control");
        p.initial_temperature_k =
            value.value("initial_temperature_k", scenario.cells[p.sensor_object].initial_temperature_k);
        p.ramp_k_per_s = value.at("ramp_k_per_s");
        p.feedforward_w = value.at("feedforward_w");
        p.proportional_w_per_k = value.at("proportional_w_per_k");
        p.maximum_power_w = value.at("maximum_power_w");
        p.cutoff_temperature_k = value.at("cutoff_temperature_k");
        p.cutoff_rate_k_per_s = value.at("cutoff_rate_k_per_s");
        validate_heating_protocol(p);
        scenario.heating_protocols.push_back(p);
    }
    std::unordered_map<std::string, std::size_t> port_index;
    for (std::size_t i = 0; i < scenario.ports.size(); ++i)
        port_index[scenario.ports[i].id] = i;
    for (const auto& value : root.at("ports")) {
        PortConfig port;
        port.id = value.at("id").get<std::string>();
        port.cell_id = value.at("cell_id").get<std::string>();
        const auto cell = cell_index.find(port.cell_id);
        if (cell == cell_index.end())
            throw std::runtime_error("unknown cell for port " + port.id);
        port.cell_index = cell->second;
        port.kind = value.value("kind", "contact");
        port.surface = value.value("surface", "unspecified");
        port.area_m2 = value.at("area_m2").get<double>();
        port.rectangle_m = value.value("rectangle_m", std::array<double, 2>{0., 0.});
        port.centroid_m = vec3(value.at("centroid_m"), "port.centroid_m");
        port.normal = vec3(value.at("normal"), "port.normal");
        if (!port_index.emplace(port.id, scenario.ports.size()).second) {
            throw std::runtime_error("duplicate port id: " + port.id);
        }
        scenario.ports.push_back(std::move(port));
    }

    for (const auto& value : root.value("interfaces", json::array())) {
        InterfaceConfig edge;
        edge.lumped_heater = value.value("lumped_heater", false);
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
        if (edge.cell_a == edge.cell_b)
            throw std::runtime_error("self interface: " + edge.id);
        edge.material_a = value.value("material_a", "unspecified");
        edge.material_b = value.value("material_b", "unspecified");
        edge.interface_material = value.value("interface_material", "unspecified");
        if (value.contains("radiation")) {
            if (value.contains("law"))
                throw std::runtime_error("Radiative interface cannot also specify a conductive law");
            const auto& r = value.at("radiation");
            edge.radiative = true;
            edge.radiation_exchange_area_m2 = r.at("exchange_area_m2");
            edge.effective_emissivity = r.at("effective_emissivity");
            edge.radiation_temperature_limit_k = r.value("temperature_limit_k", 4000.);
            edge.law = reference_directed_connectivity(0.);
        } else
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
        boundary.conductance_w_per_k = value.value("conductance_w_per_k", 0.);
        if (value.contains("radiation")) {
            const auto& r = value.at("radiation");
            boundary.radiative = true;
            boundary.effective_emissivity = r.at("effective_emissivity").get<double>();
            boundary.radiating_area_fraction = r.value("area_fraction", 1.);
            boundary.radiation_temperature_limit_k = r.value("temperature_limit_k", 4000.);
            if (value.contains("connectivity") || boundary.conductance_w_per_k != 0.)
                throw std::runtime_error(
                    "Radiation must be a separate geometric edge, not a second copy of convection");
        } else if (!value.contains("conductance_w_per_k"))
            throw std::runtime_error("Ordinary boundary requires conductance_w_per_k");
        boundary.temperature_profile =
            value.value("temperature_profile", std::vector<std::array<double, 2>>{});
        if (value.contains("connectivity")) {
            boundary.has_connectivity = true;
            boundary.connectivity = parse_law(value.at("connectivity"));
            boundary.remote_surface_connectivity =
                parse_surface(value.value("remote_surface_connectivity", json::object()));
        }
        scenario.boundaries.push_back(std::move(boundary));
    }

    return port_index;
}
} // namespace battery::scenario_detail

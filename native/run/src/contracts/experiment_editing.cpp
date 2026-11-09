#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <set>
#include <map>

namespace battery {
using J = nlohmann::json;
J add_external_heater(const J& input, const std::string& bid, const std::string& face, double power, double h,
                      double capacity, bool controlled, double ramp, double cutoff_temperature,
                      double cutoff_rate, bool constant_cutoff) {
    if (face == "wrap") {
        auto e = add_external_heater(input, bid, "-x", power, h, capacity, controlled, ramp,
                                     cutoff_temperature, cutoff_rate, constant_cutoff);
        const auto hid = e.at("assembly").at("passive_cells").back().at("id").get<std::string>();
        return set_heater_mount(e, hid, bid, "wrap");
    }
    auto e = upgrade_experiment(input);
    if (face.size() != 2 || (face[0] != '+' && face[0] != '-') ||
        std::string("xyz").find(face[1]) == std::string::npos || !std::isfinite(power) || power < 0 ||
        !std::isfinite(h) || h <= 0 || !std::isfinite(capacity) || capacity <= 0 || !std::isfinite(ramp) ||
        ramp <= 0 || !std::isfinite(cutoff_temperature) || cutoff_temperature <= 0 ||
        !std::isfinite(cutoff_rate) || cutoff_rate < 0)
        throw std::invalid_argument("Invalid external heater parameters");
    const auto scenario = scenario_from_json(compile_experiment(e));
    auto cell = std::find_if(scenario.cells.begin(), scenario.cells.end(),
                             [&](const auto& c) { return c.id == bid && c.is_battery; });
    if (cell == scenario.cells.end())
        throw std::invalid_argument("Select a battery for the heater");
    const std::string hid = "heater-" + bid + "-" + (face[0] == '+' ? "plus" : "minus") + face[1];
    for (const auto& c : scenario.cells)
        if (c.id == hid)
            throw std::invalid_argument("This heater already exists; edit it in Apparatus");
    if (cell->orientation_xyzw != std::array<double, 4>{0, 0, 0, 1})
        throw std::invalid_argument("Heater wizard currently requires an axis-aligned battery");
    const int axis = int(std::string("xyz").find(face[1])), u = (axis + 1) % 3, v = (axis + 2) % 3;
    const double sign = face[0] == '+' ? 1 : -1, thickness = .001;
    std::array<double, 3> center{cell->position_m.x, cell->position_m.y, cell->position_m.z};
    auto dims = cell->dimensions_m;
    center[axis] += sign * dims[axis] / 2;
    std::string target = hid + "-target";
    const auto shell_port = std::find_if(scenario.ports.begin(), scenario.ports.end(), [&](const auto& p) {
        return p.id == bid + ":case:" + face + ":outer";
    });
    double area = dims[u] * dims[v];
    if (shell_port != scenario.ports.end()) {
        target = shell_port->id;
        center = {shell_port->centroid_m.x, shell_port->centroid_m.y, shell_port->centroid_m.z};
        area = shell_port->area_m2;
        dims[u] = shell_port->rectangle_m[0];
        dims[v] = shell_port->rectangle_m[1];
    } else {
        std::array<double, 3> p{}, normal{};
        p[axis] = sign * dims[axis] / 2;
        normal[axis] = sign;
        e["assembly"]["ports"].push_back({{"id", target},
                                          {"cell_id", bid},
                                          {"frame", "local"},
                                          {"surface", face},
                                          {"centroid_m", p},
                                          {"normal", normal},
                                          {"area_m2", area},
                                          {"rectangle_m", {dims[u], dims[v]}}});
    }
    center[axis] += sign * thickness / 2;
    dims[axis] = thickness;
    const double current = std::sqrt(power);
    J heater{{"id", hid},
             {"role", "passive"},
             {"material_id", "external-resistive-heater"},
             {"runaway", {{"energy_j", 0}, {"duration_s", 1}}},
             {"geometry", {{"kind", "prismatic"}, {"dimensions_m", dims}}},
             {"pose", {{"position_m", center}, {"orientation_xyzw", {0, 0, 0, 1}}}},
             {"thermal",
              {{"heat_capacity_j_per_k", capacity},
               {"initial_temperature_k", cell->initial_temperature_k},
               {"conductivity_w_per_m_k", {20, 20, 20}},
               {"divisions", {1, 1, 1}}}},
             {"electrical",
              {{"resistance_ohm", 1},
               {"min_resistance_ohm", 1},
               {"max_resistance_ohm", 1},
               {"current_a", current},
               {"voltage_profile_v", {{0, current}}}}}};
    e["assembly"]["passive_cells"].push_back(heater);
    std::array<double, 3> hp{}, hn{};
    hp[axis] = -sign * thickness / 2;
    hn[axis] = -sign;
    e["assembly"]["ports"].push_back({{"id", hid + "-contact"},
                                      {"cell_id", hid},
                                      {"frame", "local"},
                                      {"surface", std::string(sign > 0 ? "-" : "+") + face[1]},
                                      {"centroid_m", hp},
                                      {"normal", hn},
                                      {"area_m2", area},
                                      {"rectangle_m", {dims[u], dims[v]}}});
    e["assembly"]["interfaces"].push_back({{"id", hid + "-coupling"},
                                           {"port_a", hid + "-contact"},
                                           {"port_b", target},
                                           {"law",
                                            {{"kind", "directed_geometry"},
                                             {"background_conductance_w_per_k", .01 * h * area},
                                             {"active_conductance_w_per_k", 1.98 * h * area},
                                             {"endpoint_damage_weight", 0}}}});
    if (controlled || constant_cutoff) {
        if (controlled && power <= 0)
            throw std::invalid_argument("Controlled heater requires a positive power limit");
        e["conditions"]["heating_protocols"].push_back(
            {{"id", hid + "-control"},
             {"heater_id", hid},
             {"sensor_id", bid},
             {"initial_temperature_k", cell->initial_temperature_k},
             {"ramp_k_per_s", controlled ? ramp : 0.},
             {"feedforward_w", controlled ? power * .25 : power},
             {"proportional_w_per_k", controlled ? 1. : 0.},
             {"maximum_power_w", controlled ? power : std::max(1., power)},
             {"cutoff_temperature_k", cutoff_temperature},
             {"cutoff_rate_k_per_s", cutoff_rate}});
    }
    e["metadata"]["editor_changes"].push_back(
        {{"kind", "external_heater"},
         {"id", hid},
         {"assumption",
          "Explicit synthetic apparatus: 1 mm plate, 20 W/(m K), user capacity/contact; not calibrated"},
         {"contact_h_w_m2_k", h},
         {"capacity_j_k", capacity},
         {"initial_power_limit_w", power}});
    if (e["provenance"]["status"] != "synthetic")
        e["provenance"]["status"] = "mixed";
    (void)compile_experiment(e);
    if (face == "+z" || face == "-z") {
        for (const auto& b : e.at("assembly").at("batteries"))
            if (b.at("id") == bid)
                for (const auto& m : e.at("battery_models"))
                    if (m.at("id") == b.at("model_id") && m.at("revision") == b.at("model_revision") &&
                        m.at("geometry").at("kind") == "cylinder")
                        return set_heater_mount(e, hid, bid, face);
    }
    return e;
}

J set_heater_cutoff(const J& input, const std::string& hid, double temperature, double rate) {
    if (!std::isfinite(temperature) || temperature <= 0 || !std::isfinite(rate) || rate < 0)
        throw std::invalid_argument("Cutoff requires positive Kelvin temperature and nonnegative rise rate");
    auto e = upgrade_experiment(input);
    auto& protocols = e["conditions"]["heating_protocols"];
    auto controller = std::find_if(protocols.begin(), protocols.end(),
                                   [&](const auto& p) { return p.at("heater_id") == hid; });
    if (controller == protocols.end()) {
        const auto s = scenario_from_json(compile_experiment(e));
        const auto h =
            std::find_if(s.cells.begin(), s.cells.end(), [&](const auto& c) { return c.id == hid; });
        if (h == s.cells.end() ||
            (h->material_id != "external-resistive-heater" && h->material_id != "heater"))
            throw std::invalid_argument("Select an external heater");
        const int index = int(h - s.cells.begin());
        // All sectors of a wrap share one observed battery. Reject genuinely
        // ambiguous multi-battery heaters instead of picking a random neighbour.
        std::set<std::string> targets;
        for (const auto& edge : s.interfaces)
            if (edge.cell_a == index || edge.cell_b == index) {
                const auto& other = s.cells[edge.cell_a == index ? edge.cell_b : edge.cell_a];
                if (other.is_battery || !other.shell_contents_id.empty())
                    targets.insert(other.shell_contents_id.empty() ? other.id : other.shell_contents_id);
            }
        if (targets.size() != 1)
            throw std::invalid_argument("Heater cutoff requires one unambiguous target battery");
        const auto edge = std::find_if(s.interfaces.begin(), s.interfaces.end(),
                                       [&](const auto& x) { return x.cell_a == index || x.cell_b == index; });
        if (edge == s.interfaces.end() || (edge->cell_a != index && edge->cell_b != index))
            throw std::invalid_argument("Heater needs an explicit battery coupling for its cutoff sensor");
        const double power = h->electrical.current_a * h->electrical.current_a * h->electrical.resistance_ohm;
        protocols.push_back({{"id", hid + "-control"},
                             {"heater_id", hid},
                             {"sensor_id", *targets.begin()},
                             {"ramp_k_per_s", 0.},
                             {"feedforward_w", power},
                             {"proportional_w_per_k", 0.},
                             {"maximum_power_w", std::max(1., power)},
                             {"cutoff_temperature_k", temperature},
                             {"cutoff_rate_k_per_s", rate}});
    } else {
        (*controller)["cutoff_temperature_k"] = temperature;
        (*controller)["cutoff_rate_k_per_s"] = rate;
    }
    e["metadata"]["editor_changes"].push_back({{"kind", "heater_cutoff"}, {"id", hid}});
    if (e["provenance"]["status"] != "synthetic")
        e["provenance"]["status"] = "mixed";
    (void)compile_experiment(e);
    return e;
}

J duplicate_battery(const J& input, const std::string& bid) {
    auto e = upgrade_experiment(input);
    auto& a = e["assembly"];
    J source;
    std::set<std::string> ids;
    for (const auto& b : a["batteries"]) {
        ids.insert(b.at("id"));
        if (b.at("id") == bid)
            source = b;
    }
    if (source.is_null())
        throw std::invalid_argument("Select a battery to duplicate");
    for (const auto& p : a.value("passive_cells", J::array()))
        ids.insert(p.at("id"));
    std::string new_id;
    for (int n = 1;; ++n) {
        new_id = bid + "_" + std::to_string(n);
        if (!ids.contains(new_id))
            break;
    }
    const auto old_gas = source.at("gas_compartment").get<std::string>();
    const auto new_gas = "inside-" + new_id;
    const auto compartments = a["gas_network"]["compartments"];
    std::set<std::string> gas_ids;
    for (const auto& g : compartments)
        gas_ids.insert(g.at("id"));
    std::map<std::string, std::string> gas_map;
    for (const auto& g : compartments)
        if (g.at("id") == old_gas || g.value("contents_id", "") == bid ||
            g.value("local_equilibrium", J::object()).value("cell_id", "") == bid) {
            const std::string old = g.at("id");
            const auto next = old == old_gas ? new_gas : old + "_" + new_id;
            if (!gas_ids.insert(next).second)
                throw std::invalid_argument("Duplicate gas ID; rename the source battery first");
            gas_map[old] = next;
            auto copy = g;
            copy["id"] = next;
            if (copy.contains("contents_id"))
                copy["contents_id"] = new_id;
            if (copy.contains("local_equilibrium"))
                copy["local_equilibrium"]["cell_id"] = new_id;
            a["gas_network"]["compartments"].push_back(copy);
        }
    const auto channels = a["gas_network"].value("channels", J::array());
    for (auto channel : channels)
        if (gas_map.contains(channel.at("from")) && gas_map.contains(channel.at("to"))) {
            channel["id"] = channel.at("id").get<std::string>() + "_" + new_id;
            channel["from"] = gas_map.at(channel.at("from"));
            channel["to"] = gas_map.at(channel.at("to"));
            a["gas_network"]["channels"].push_back(channel);
        }
    source["id"] = new_id;
    source["gas_compartment"] = new_gas;
    source["pose"]["position_m"][0] = source["pose"]["position_m"][0].get<double>() + .04;
    a["batteries"].push_back(source);
    const auto routes = a["vent_connections"];
    for (auto route : routes)
        if (route.at("battery_id") == bid) {
            route["id"] = route.at("id").get<std::string>() + "_" + new_id;
            route["battery_id"] = new_id;
            a["vent_connections"].push_back(route);
        }
    e["numerics"]["batteries"][new_id] = e["numerics"]["batteries"][bid];
    (void)compile_experiment(e);
    return e;
}
} // namespace battery

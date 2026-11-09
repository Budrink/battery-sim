#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <stdexcept>

namespace battery {
namespace {
using J = nlohmann::json;
double positive(const J& p, const char* key, bool zero = false) {
    const double v = p.at(key).get<double>();
    if (!std::isfinite(v) || (zero ? v < 0 : v <= 0))
        throw std::invalid_argument(std::string("Invalid case property: ") + key);
    return v;
}
J contact(double g) {
    return {{"kind", "directed_geometry"},
            {"background_conductance_w_per_k", .01 * g},
            {"active_conductance_w_per_k", 1.98 * g}};
}
} // namespace

J set_enclosure(const J& input, const std::string& id, const std::string& gas_id, const J& p) {
    if (id.empty() || gas_id.empty())
        throw std::invalid_argument("Case and gas IDs are required");
    auto e = upgrade_experiment(input);
    auto& a = e["assembly"];
    const auto size = p.at("inside_dimensions_m").get<std::array<double, 3>>();
    const auto center = p.at("position_m").get<std::array<double, 3>>();
    const auto k = p.at("conductivity_w_per_m_k").get<std::array<double, 3>>();
    for (int i = 0; i < 3; ++i)
        if (!std::isfinite(size[i]) || size[i] <= 0 || !std::isfinite(center[i]) || !std::isfinite(k[i]) ||
            k[i] <= 0)
            throw std::invalid_argument("Invalid case dimensions or conductivity");
    const double t = positive(p, "thickness_m"), rho = positive(p, "density_kg_m3"),
                 cp = positive(p, "specific_heat_j_kg_k");
    const double hg = positive(p, "gas_contact_h_w_m2_k"), ha = positive(p, "ambient_h_w_m2_k", true);
    const double initial = positive(p, "initial_temperature_k"),
                 ambient = positive(p, "ambient_temperature_k");
    positive(p, "damage_onset_k");
    positive(p, "damage_rate_per_s", true);
    positive(p, "damage_scale_k");
    const auto material = p.at("material_id").get<std::string>();
    if (material.empty())
        throw std::invalid_argument("Case material ID is required");
    auto& registry = e["metadata"]["editor_enclosures"];
    if (registry.is_null())
        registry = J::object();
    const bool updating = registry.contains(id);
    if (updating && registry.at(id).at("gas_compartment") != gas_id)
        throw std::invalid_argument("Changing the case gas compartment requires explicit gas-route editing");
    std::set<std::string> walls, ports;
    const char* faces[] = {"-x", "+x", "-y", "+y", "-z", "+z"};
    const auto wall_id = [&](const std::string& f) {
        return id + "-" + (f[0] == '+' ? "plus" : "minus") + f[1];
    };
    for (const auto* f : faces) {
        walls.insert(wall_id(f));
        ports.insert(wall_id(f) + ":inner");
        ports.insert(wall_id(f) + ":outer");
    }
    const auto scenario = scenario_from_json(compile_experiment(e));
    double displaced = 0;
    for (const auto& c : scenario.cells) {
        if (walls.contains(c.id)) {
            if (!updating)
                throw std::invalid_argument("Case ID collides with an existing object");
            continue;
        }
        if (c.orientation_xyzw != std::array<double, 4>{0, 0, 0, 1})
            throw std::invalid_argument("Case wizard currently supports axis-aligned contents only");
        const std::array<double, 3> pos{c.position_m.x, c.position_m.y, c.position_m.z};
        for (int axis = 0; axis < 3; ++axis)
            if (std::abs(pos[axis] - center[axis]) + c.dimensions_m[axis] / 2 > size[axis] / 2 + 1e-10)
                throw std::invalid_argument("Case does not contain object: " + c.id);
        // Compiled cylinder contents use an equal-volume prism. Shell cells are disjoint solids.
        displaced += c.dimensions_m[0] * c.dimensions_m[1] * c.dimensions_m[2];
    }
    const double free_volume = size[0] * size[1] * size[2] - displaced;
    if (free_volume <= 0)
        throw std::invalid_argument("Case has no free gas volume");
    auto& compartments = a["gas_network"]["compartments"];
    auto gas = std::find_if(compartments.begin(), compartments.end(),
                            [&](const auto& g) { return g.at("id") == gas_id; });
    if (gas == compartments.end() || gas->contains("contents_id"))
        throw std::invalid_argument("Select a free-space gas compartment, not battery contents");
    const double ratio = free_volume / gas->at("volume_m3").template get<double>();
    for (const char* key : {"initial_mass_kg", "initial_chemical_j"})
        if (gas->contains(key))
            (*gas)[key] = gas->at(key).template get<double>() * ratio;
    (*gas)["volume_m3"] = free_volume;
    auto remove = [&](J& list, auto predicate) {
        if (list.is_null())
            list = J::array();
        list.erase(std::remove_if(list.begin(), list.end(), predicate), list.end());
    };
    remove(a["passive_cells"], [&](const auto& v) { return walls.contains(v.at("id")); });
    remove(a["ports"], [&](const auto& v) { return ports.contains(v.at("id")); });
    remove(a["gas_network"]["thermal_contacts"],
           [&](const auto& v) { return ports.contains(v.value("port_id", std::string{})); });
    remove(e["conditions"]["boundaries"],
           [&](const auto& v) { return ports.contains(v.value("port_id", std::string{})); });
    for (int face = 0; face < 6; ++face) {
        const int axis = face / 2, u = (axis + 1) % 3, v = (axis + 2) % 3;
        const double sign = face % 2 ? 1 : -1;
        auto dims = size, pos = center;
        // Corner strips belong to X, then Y: no intersecting wall volumes.
        for (int d = 0; d < 3; ++d)
            dims[d] = d == axis ? t : size[d] + (d > axis ? 2 * t : 0);
        pos[axis] += sign * (size[axis] + t) / 2;
        const std::string wid = wall_id(faces[face]);
        a["passive_cells"].push_back({{"id", wid},
                                      {"role", "passive"},
                                      {"material_id", material},
                                      {"geometry", {{"kind", "prismatic"}, {"dimensions_m", dims}}},
                                      {"pose", {{"position_m", pos}, {"orientation_xyzw", {0, 0, 0, 1}}}},
                                      {"thermal",
                                       {{"heat_capacity_j_per_k", dims[0] * dims[1] * dims[2] * rho * cp},
                                        {"initial_temperature_k", initial},
                                        {"conductivity_w_per_m_k", k},
                                        {"divisions", {1, 1, 1}}}},
                                      {"degradation",
                                       {{"onset_k", p.at("damage_onset_k")},
                                        {"rate_at_onset_per_s", p.at("damage_rate_per_s")},
                                        {"temperature_scale_k", p.at("damage_scale_k")}}},
                                      {"runaway", {{"energy_j", 0}, {"duration_s", 1}}}});
        for (bool outer : {false, true}) {
            std::array<double, 3> point{}, normal{};
            normal[axis] = (outer ? sign : -sign);
            point[axis] = normal[axis] * t / 2;
            const double area = (outer ? dims[u] * dims[v] : size[u] * size[v]);
            const auto pid = wid + (outer ? ":outer" : ":inner");
            a["ports"].push_back({{"id", pid},
                                  {"cell_id", wid},
                                  {"frame", "local"},
                                  {"surface", std::string(normal[axis] > 0 ? "+" : "-") + "xyz"[axis]},
                                  {"centroid_m", point},
                                  {"normal", normal},
                                  {"area_m2", area},
                                  {"rectangle_m", {outer ? dims[u] : size[u], outer ? dims[v] : size[v]}}});
            if (outer && ha > 0)
                e["conditions"]["boundaries"].push_back({{"id", pid + "-ambient"},
                                                         {"port_id", pid},
                                                         {"temperature_k", ambient},
                                                         {"conductance_w_per_k", ha * area},
                                                         {"connectivity", contact(ha * area)}});
            else if (!outer)
                a["gas_network"]["thermal_contacts"].push_back(
                    {{"compartment", gas_id}, {"port_id", pid}, {"connectivity", contact(hg * area)}});
        }
    }
    registry[id] = {{"properties", p}, {"gas_compartment", gas_id}, {"free_volume_m3", free_volume}};
    const auto previous_mesh = a.value("gas_meshes", J::object()).value(id, J::object());
    a["gas_meshes"][id] = {{"compartment", gas_id},
                           {"inside_dimensions_m", size},
                           {"position_m", center},
                           {"maximum_cell_size_m", p.value("gas_cell_size_m", .02)},
                           {"surface_h_w_m2_k", p.value("solid_contact_h_w_m2_k", 10.)},
                           {"wall_h_w_m2_k", hg},
                           {"discharge_coefficient", p.value("gas_discharge_coefficient", .7)}};
    for (const auto* key : {"openings", "opening_ambient", "battery_surface_connectivity_per_m2"})
        if (previous_mesh.contains(key))
            a["gas_meshes"][id][key] = previous_mesh.at(key);
    if (!updating) {
        const double diameter = std::min(.02, .25 * std::min(size[0], size[1]));
        a["gas_meshes"][id]["openings"] = {{{"id", "top-opening"},
                                            {"face", "+z"},
                                            {"shape", "circle"},
                                            {"center_m", {0., 0.}},
                                            {"size_m", {diameter, diameter}}}};
    }
    e["metadata"]["visual_materials"][material] = "enclosure";
    if (e["provenance"]["status"] != "synthetic")
        e["provenance"]["status"] = "mixed";
    (void)scenario_from_json(compile_experiment(e));
    return e;
}
} // namespace battery

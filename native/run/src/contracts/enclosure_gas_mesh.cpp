#include "contract_internal.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/thermal/object_grid.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace battery::contract_detail {
namespace {
using V = std::array<double, 3>;
constexpr double eps = 1e-10;
struct Box {
    V lo{}, hi{};
};
double volume(const Box& b) {
    return (b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]);
}
V center(const Box& b) {
    return {(b.lo[0] + b.hi[0]) / 2, (b.lo[1] + b.hi[1]) / 2, (b.lo[2] + b.hi[2]) / 2};
}
Box box(const CellConfig& c) {
    Box b;
    const V p{c.position_m.x, c.position_m.y, c.position_m.z};
    for (int d = 0; d < 3; ++d) {
        b.lo[d] = p[d] - c.dimensions_m[d] / 2;
        b.hi[d] = p[d] + c.dimensions_m[d] / 2;
    }
    return b;
}
// Exact axis-aligned CSG. Each subtraction produces disjoint boxes; no gas is
// placed inside a pad, shell or battery, including thin barriers below mesh size.
void subtract(const Box& source, const Box& solid, std::vector<Box>& free) {
    Box intersection;
    for (int d = 0; d < 3; ++d) {
        intersection.lo[d] = std::max(source.lo[d], solid.lo[d]);
        intersection.hi[d] = std::min(source.hi[d], solid.hi[d]);
        if (intersection.hi[d] - intersection.lo[d] <= eps) {
            free.push_back(source);
            return;
        }
    }
    Box rest = source;
    for (int d = 0; d < 3; ++d) {
        if (intersection.lo[d] - rest.lo[d] > eps) {
            auto part = rest;
            part.hi[d] = intersection.lo[d];
            free.push_back(part);
        }
        if (rest.hi[d] - intersection.hi[d] > eps) {
            auto part = rest;
            part.lo[d] = intersection.hi[d];
            free.push_back(part);
        }
        rest.lo[d] = intersection.lo[d];
        rest.hi[d] = intersection.hi[d];
    }
}
// Rectangle of actual shared face, including nonconforming neighbours.
double overlap_face(const Box& a, const Box& b, int axis, double plane, V& middle, V& extent) {
    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
    middle[axis] = plane;
    extent[axis] = 0;
    for (int d : {u, v}) {
        const double lo = std::max(a.lo[d], b.lo[d]), hi = std::min(a.hi[d], b.hi[d]);
        if (hi - lo <= eps)
            return 0;
        middle[d] = (hi + lo) / 2;
        extent[d] = hi - lo;
    }
    return extent[u] * extent[v];
}
J thermal_law(double g) {
    return {{"kind", "directed_geometry"},
            {"background_conductance_w_per_k", .01 * g},
            {"active_conductance_w_per_k", 1.98 * g}};
}
// A contact law per square metre, not a pre-integrated contact from another
// fixture. Only its extensive conductances scale with the exposed face area.
J surface_law(const J& reference, double area) {
    J law = reference;
    for (const auto* key :
         {"background_conductance_w_per_k", "active_conductance_w_per_k",
          "destroyed_background_conductance_w_per_k", "destroyed_active_conductance_w_per_k"})
        if (law.contains(key))
            law[key] = law.at(key).get<double>() * area;
    return law;
}
#include "uniform_gas_grid.hpp"
void route_outlets(J& net, const Scenario& scene, const std::vector<Box>& grid, const std::string& id,
                   const std::string& seed_id) {
    const auto name = [&](std::size_t i) { return id + "-air-" + std::to_string(i); };
    // Each opening discharges locally, not into a global mixing reservoir.
    // Native shell patches remain the aperture resolution; their centres
    // (or valve centres) select the immediately adjacent free gas volume.
    for (auto& ch : net["channels"]) {
        if (ch.at("from") == seed_id)
            fail("gas_mesh", "Outgoing template channels require explicit spatial placement");
        if (ch.at("to") != seed_id)
            continue;
        if (!ch.contains("port_id"))
            fail("gas_mesh", "Cannot place a channel without a physical port");
        const auto p = std::find_if(scene.ports.begin(), scene.ports.end(),
                                    [&](const auto& port) { return port.id == ch.at("port_id"); });
        if (p == scene.ports.end())
            fail("gas_mesh", "Missing discharge port");
        const auto& owner = scene.cells[p->cell_index];
        V point{p->centroid_m.x, p->centroid_m.y, p->centroid_m.z};
        const int axis = p->surface[1] - 'x', u = (axis + 1) % 3, v = (axis + 2) % 3;
        const int local = ch.value("shell_thermal_volume", -1);
        if (local >= 0) {
            const auto n = owner.thermal_divisions;
            const int xyz[] = {local % n[0], (local / n[0]) % n[1], local / (n[0] * n[1])};
            const auto origin = center(box(owner));
            const auto edges = object_grid(scene, owner);
            for (int d : {u, v})
                point[d] = origin[d] + (edges[d][xyz[d]] + edges[d][xyz[d] + 1]) / 2;
            const auto opening = ch.value("damage_opening", J::object());
            if (opening.contains("valve")) {
                point[u] += opening["valve"]["center_m"][0].get<double>();
                point[v] += opening["valve"]["center_m"][1].get<double>();
            }
        }
        const V normal{p->normal.x, p->normal.y, p->normal.z};
        for (int d = 0; d < 3; ++d)
            point[d] += normal[d] * 1e-8;
        int target = -1;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            bool inside = true;
            for (int d = 0; d < 3; ++d)
                inside &= point[d] >= grid[i].lo[d] - eps && point[d] <= grid[i].hi[d] + eps;
            if (inside) {
                target = int(i);
                break;
            }
        }
        if (target < 0)
            fail("gas_mesh",
                 "Opening is blocked by a solid or outside the case: " + ch.at("id").get<std::string>());
        ch["to"] = name(target);
    }
}

int connect_passages(J& net, const std::vector<Box>& grid, const std::string& id, double cd) {
    const auto name = [&](std::size_t i) { return id + "-air-" + std::to_string(i); };
    // Nonconforming adjacency uses intersection area, not the larger face.
    int passages = 0;
    for (std::size_t i = 0; i < grid.size(); ++i)
        for (std::size_t j = i + 1; j < grid.size(); ++j)
            for (int axis = 0; axis < 3; ++axis) {
                const bool forward = std::abs(grid[i].hi[axis] - grid[j].lo[axis]) < eps;
                const bool reverse = std::abs(grid[j].hi[axis] - grid[i].lo[axis]) < eps;
                if (!forward && !reverse)
                    continue;
                V middle{}, extent{};
                const double area = overlap_face(
                    grid[i], grid[j], axis, forward ? grid[i].hi[axis] : grid[j].hi[axis], middle, extent);
                if (area <= 0)
                    continue;
                const double length = std::abs(center(grid[i])[axis] - center(grid[j])[axis]);
                V direction{};
                direction[axis] = forward ? 1 : -1;
                net["channels"].push_back({{"id", id + "-passage-" + std::to_string(passages++)},
                                           {"kind", "passage"},
                                           {"from", name(i)},
                                           {"to", name(j)},
                                           {"area_m2", area},
                                           {"opening_pressure_pa", 0},
                                           {"discharge_coefficient", cd},
                                           {"mixing_axis", axis},
                                           {"mixing_length_m", length},
                                           {"inertial_length_m", length},
                                           {"flow_direction", direction}});
            }
    return passages;
}

std::pair<int, double> connect_solids(J& out, const Scenario& scene, const std::vector<Box>& grid,
                                      const Box& enclosure, const std::string& id, const std::string& seed_id,
                                      double h, double wall_h, const std::vector<EnclosureOpening>& holes,
                                      const J& battery_surface_law) {
    auto& net = out["gas_network"];
    const auto name = [&](std::size_t i) { return id + "-air-" + std::to_string(i); };
    auto& contacts = net["thermal_contacts"];
    if (contacts.is_null())
        contacts = J::array();
    contacts.erase(std::remove_if(contacts.begin(), contacts.end(),
                                  [&](const auto& c) { return c.at("compartment") == seed_id; }),
                   contacts.end());
    int surfaces = 0;
    double area_sum = 0;
    for (std::size_t i = 0; i < grid.size(); ++i)
        for (const auto& cell : scene.cells) {
            const auto solid = box(cell);
            for (int axis = 0; axis < 3; ++axis)
                for (int sign : {-1, 1}) {
                    const double plane = sign > 0 ? solid.hi[axis] : solid.lo[axis];
                    if (std::abs(plane - (sign > 0 ? grid[i].lo[axis] : grid[i].hi[axis])) > eps)
                        continue;
                    V middle{}, extent{};
                    double area = overlap_face(solid, grid[i], axis, plane, middle, extent);
                    const double support_area = area;
                    if (area <= 0)
                        continue;
                    const int u = (axis + 1) % 3, v = (axis + 2) % 3;
                    const auto origin = center(enclosure);
                    for (const auto& hole : holes)
                        if (hole.axis == axis &&
                            std::abs(plane - (hole.sign > 0 ? enclosure.hi[axis] : enclosure.lo[axis])) < eps)
                            area -= hole.overlap({middle[u] - extent[u] / 2 - origin[u],
                                                  middle[v] - extent[v] / 2 - origin[v]},
                                                 {middle[u] + extent[u] / 2 - origin[u],
                                                  middle[v] + extent[v] / 2 - origin[v]});
                    if (area <= 1e-16)
                        continue;
                    V normal{};
                    normal[axis] = sign;
                    const auto pid = id + "-gas-contact-" + std::to_string(surfaces++);
                    out["ports"].push_back({{"id", pid},
                                            {"cell_id", cell.id},
                                            {"surface", std::string(sign > 0 ? "+" : "-") + "xyz"[axis]},
                                            {"normal", normal},
                                            {"centroid_m", middle},
                                            {"area_m2", support_area},
                                            {"rectangle_m", {extent[u], extent[v]}}});
                    const bool wall = solid.lo[axis] >= enclosure.hi[axis] - eps ||
                                      solid.hi[axis] <= enclosure.lo[axis] + eps;
                    const auto law = !cell.shell_contents_id.empty() && !battery_surface_law.is_null()
                                         ? surface_law(battery_surface_law, area)
                                         : thermal_law((wall ? wall_h : h) * area);
                    contacts.push_back({{"compartment", name(i)}, {"port_id", pid}, {"connectivity", law}});
                    area_sum += area;
                }
        }
    return {surfaces, area_sum};
}

void connect_openings(J& out, const Scenario& scene, const std::vector<Box>& grid, const Box& enclosure,
                      const std::string& id, const J& spec, const std::vector<EnclosureOpening>& holes) {
    const auto origin = center(enclosure);
    std::vector<double> areas(grid.size(), 0.);
    std::map<std::string, double> removed;
    for (const auto& hole : holes) {
        const int axis = hole.axis, u = (axis + 1) % 3, v = (axis + 2) % 3;
        const double plane = hole.sign > 0 ? enclosure.hi[axis] : enclosure.lo[axis];
        const std::string wall_id = id + (hole.sign > 0 ? "-plus" : "-minus") + "xyz"[axis];
        const auto wall = std::find_if(scene.cells.begin(), scene.cells.end(),
                                       [&](const auto& c) { return c.id == wall_id; });
        if (wall == scene.cells.end())
            fail("opening", "Missing case wall: " + wall_id);
        double assigned = 0.;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (std::abs((hole.sign > 0 ? grid[i].hi[axis] : grid[i].lo[axis]) - plane) > eps)
                continue;
            const double a = hole.overlap({grid[i].lo[u] - origin[u], grid[i].lo[v] - origin[v]},
                                          {grid[i].hi[u] - origin[u], grid[i].hi[v] - origin[v]});
            assigned += a;
            areas[i] += a;
        }
        if (std::abs(assigned - hole.area()) > std::max(1e-12, hole.area() * 1e-8))
            fail("opening", "Opening blocked by internal solids: " + hole.id);
        removed[wall_id] += hole.area();
        V position = origin;
        position[axis] = plane + hole.sign * wall->dimensions_m[axis];
        position[u] += hole.center[0];
        position[v] += hole.center[1];
        out["metadata"]["case_openings"].push_back({{"id", hole.id},
                                                    {"case_id", id},
                                                    {"wall_id", wall_id},
                                                    {"axis", axis},
                                                    {"sign", hole.sign},
                                                    {"circle", hole.circle},
                                                    {"center_m", position},
                                                    {"size_m", hole.size},
                                                    {"area_m2", assigned}});
    }
    // The existing conservative ambient operator handles both exhaust and
    // fresh-air inflow. No prescribed fan or artificial oxygen replenishment.
    const auto ambient = spec.value("opening_ambient", J::object());
    object(ambient, "opening_ambient", {"pressure_pa", "temperature_k", "oxidizer_mass_fraction"});
    const double pressure = number(ambient.value("pressure_pa", 101325.), "ambient pressure", 1e-12, 1e12);
    const double temperature =
        number(ambient.value("temperature_k", 298.15), "ambient temperature", 1e-12, 1e9);
    const double oxygen = number(ambient.value("oxidizer_mass_fraction", .232), "ambient oxygen", 0, 1);
    for (auto& node : out["gas_network"]["compartments"])
        for (std::size_t i = 0; i < areas.size(); ++i)
            if (areas[i] > 0 && node.at("id") == id + "-air-" + std::to_string(i))
                node["ambient_boundary"] = {{"pressure_pa", pressure},
                                            {"temperature_k", temperature},
                                            {"oxidizer_mass_fraction", oxygen},
                                            {"area_m2", areas[i]},
                                            {"discharge_coefficient", spec.at("discharge_coefficient")},
                                            {"renewal_flow_m3_s", 0.}};
    for (auto& cell : out["cells"]) {
        const std::string cid = cell.at("id");
        if (!removed.contains(cid))
            continue;
        const auto& c =
            *std::find_if(scene.cells.begin(), scene.cells.end(), [&](const auto& c) { return c.id == cid; });
        const int axis = cid.back() - 'x', u = (axis + 1) % 3, v = (axis + 2) % 3;
        const double fraction = 1 - removed[cid] / (c.dimensions_m[u] * c.dimensions_m[v]);
        if (fraction <= 1e-8)
            fail("opening", "Opening removes the entire wall");
        cell["thermal"]["heat_capacity_j_per_k"] =
            cell["thermal"]["heat_capacity_j_per_k"].get<double>() * fraction;
        // Existing exterior cooling is optional. Its area shrinks with the wall,
        // but material coefficients never change.
        for (auto& boundary : out["boundaries"])
            if (boundary.value("port_id", "") == cid + ":outer") {
                if (boundary.contains("conductance_w_per_k"))
                    boundary["conductance_w_per_k"] =
                        boundary["conductance_w_per_k"].get<double>() * fraction;
                if (boundary.contains("connectivity"))
                    for (const char* key : {"background_conductance_w_per_k", "active_conductance_w_per_k"})
                        if (boundary["connectivity"].contains(key))
                            boundary["connectivity"][key] =
                                boundary["connectivity"][key].get<double>() * fraction;
            }
    }
}
} // namespace

void compile_gas_meshes(J& out, const J& meshes) {
    if (!meshes.is_object())
        fail("gas_meshes", "Expected object keyed by enclosure ID");
    for (const auto& [id, spec] : meshes.items()) {
        (void)contract_detail::id(J(id), "gas_mesh/id");
        object(spec, "gas_mesh",
               {"compartment", "inside_dimensions_m", "position_m", "maximum_cell_size_m", "surface_h_w_m2_k",
                "wall_h_w_m2_k", "discharge_coefficient", "openings", "opening_ambient",
                "battery_surface_connectivity_per_m2"},
               {"compartment", "inside_dimensions_m", "position_m", "maximum_cell_size_m", "surface_h_w_m2_k",
                "wall_h_w_m2_k", "discharge_coefficient"});
        const auto seed_id = text(spec.at("compartment"), "gas mesh compartment");
        const auto size = spec.at("inside_dimensions_m").get<V>(), position = spec.at("position_m").get<V>();
        const double step = number(spec.at("maximum_cell_size_m"), "gas mesh size", 1e-5, 10);
        const double h = number(spec.at("surface_h_w_m2_k"), "gas-solid h", 1e-12, 1e9);
        const double wall_h = number(spec.at("wall_h_w_m2_k"), "gas-wall h", 1e-12, 1e9);
        const double cd = number(spec.at("discharge_coefficient"), "gas discharge coefficient", 1e-12, 1);
        const auto battery_law = spec.value("battery_surface_connectivity_per_m2", J());
        if (!battery_law.is_null()) {
            object(battery_law, "battery gas surface law",
                   {"kind", "geometry_reference_logit", "background_conductance_w_per_k",
                    "active_conductance_w_per_k", "destroyed_background_conductance_w_per_k",
                    "destroyed_active_conductance_w_per_k", "endpoint_damage_weight"},
                   {"kind", "geometry_reference_logit", "background_conductance_w_per_k",
                    "active_conductance_w_per_k"});
            if (battery_law.at("kind") != "directed_geometry")
                fail("battery gas surface law", "Expected directed_geometry");
            number(battery_law.at("geometry_reference_logit"), "gas surface logit", -40, 40);
            for (const auto* key :
                 {"background_conductance_w_per_k", "active_conductance_w_per_k",
                  "destroyed_background_conductance_w_per_k", "destroyed_active_conductance_w_per_k"})
                if (battery_law.contains(key))
                    number(battery_law.at(key), "gas surface conductance per area", 0, 1e9);
            if (battery_law.contains("endpoint_damage_weight"))
                number(battery_law.at("endpoint_damage_weight"), "gas surface damage weight", 0, 1);
        }
        const auto holes = parse_enclosure_openings(spec);
        Box enclosure;
        for (int d = 0; d < 3; ++d) {
            if (!std::isfinite(size[d]) || size[d] <= 0 || !std::isfinite(position[d]))
                fail("gas_mesh", "Invalid dimensions");
            enclosure.lo[d] = position[d] - size[d] / 2;
            enclosure.hi[d] = position[d] + size[d] / 2;
        }
        const auto scene = scenario_from_json(out);
        auto& net = out["gas_network"];
        auto& nodes = net["compartments"];
        auto seed_it =
            std::find_if(nodes.begin(), nodes.end(), [&](const auto& n) { return n.at("id") == seed_id; });
        if (seed_it == nodes.end() || seed_it->contains("contents_id") ||
            seed_it->contains("ambient_boundary") || seed_it->contains("thermal_binding"))
            fail("gas_mesh", "Template must be a closed free-space compartment");
        const J seed = *seed_it;
        for (const auto& node : nodes)
            if (node.at("id").get<std::string>().starts_with(id + "-air-"))
                fail("gas_mesh", "Generated compartment ID collision");
        const auto uniform = uniform_gas_grid(enclosure, scene, step);
        const std::vector<Box>& grid = uniform.fragments;
        if (grid.empty())
            fail("gas_mesh", "No accessible gas volume");
        const auto name = [&](std::size_t i) { return id + "-air-" + std::to_string(i); };
        const double original_volume = seed.at("volume_m3");
        nodes.erase(seed_it);
        double total_volume = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            J node = seed;
            const double vol = volume(grid[i]);
            total_volume += vol;
            node["id"] = name(i);
            node["volume_m3"] = vol;
            for (const char* key : {"initial_mass_kg", "initial_chemical_j"})
                if (node.contains(key))
                    node[key] = seed.at(key).get<double>() * vol / original_volume;
            nodes.push_back(node);
            out["metadata"]["gas_volume_geometry"][name(i)] = {{"low_m", grid[i].lo}, {"high_m", grid[i].hi}};
        }
        route_outlets(net, scene, grid, id, seed_id);
        const int passages = connect_passages(net, grid, id, cd);
        const auto [surfaces, area_sum] =
            connect_solids(out, scene, grid, enclosure, id, seed_id, h, wall_h, holes, battery_law);
        connect_openings(out, scene, grid, enclosure, id, spec, holes);
        out["metadata"]["resolved_gas_meshes"][id] = {
            {"cells", grid.size()},
            {"passages", passages},
            {"contacts", surfaces},
            {"free_volume_m3", total_volume},
            {"solid_contact_area_m2", area_sum},
            {"maximum_cell_size_m", step},
            {"geometry", "axis-aligned solids; cylinders retain declared equal-volume prism approximation"}};
        merge_uniform_gas(out, uniform, id);
    }
}
} // namespace battery::contract_detail

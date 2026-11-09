#pragma once
#include "battery_native/contracts/enclosure_openings.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace battery::explicit_case_detail {
using J = nlohmann::json;
using V = std::array<double, 3>;
// Exact port/volume overlap. Contact conductances are extensive quantities;
// scale by remaining area, never by the number of overlapping gas cells.
inline double contact_area(const PortConfig& p, const J& box, const V& origin,
                           const std::vector<EnclosureOpening>& holes, const std::string& face) {
    const int axis = p.surface[1] - 'x';
    const double pos[] = {p.centroid_m.x, p.centroid_m.y, p.centroid_m.z};
    std::array<double, 2> lo{}, hi{};
    for (int d = 0; d < 2; ++d) {
        const int a = (axis + 1 + d) % 3;
        lo[d] = std::max(pos[a] - p.rectangle_m[d]/2, box.at("low_m")[a].get<double>()) - origin[a];
        hi[d] = std::min(pos[a] + p.rectangle_m[d]/2, box.at("high_m")[a].get<double>()) - origin[a];
        if (hi[d] <= lo[d]) return 0;
    }
    double area = (hi[0]-lo[0])*(hi[1]-lo[1]);
    for (const auto& h : holes) if (h.face == face) area -= h.overlap(lo, hi);
    return std::max(0., area);
}
inline void scale_contact(J& c, double ratio) {
    for (const auto* key : {"background_conductance_w_per_k", "active_conductance_w_per_k",
                           "destroyed_background_conductance_w_per_k", "destroyed_active_conductance_w_per_k"})
        if (c.at("connectivity").contains(key)) c["connectivity"][key] = c["connectivity"][key].get<double>() * ratio;
}

inline void rebuild_contacts(J& e, const std::string& id, const Scenario& scene, bool restore) {
    const auto& entry = e.at("metadata").at("editor_enclosures").at(id);
    const auto& spec = entry.at("explicit_mesh");
    const auto origin = spec.at("position_m").get<V>();
    const auto holes = parse_enclosure_openings(spec);
    auto source_spec = spec; source_spec["openings"] = entry.at("reference_openings");
    const auto source_holes = parse_enclosure_openings(source_spec);
    const auto& boxes = e.at("metadata").at("gas_volume_geometry");
    std::set<std::string> walls, owned_ports;
    for (const auto& wall : entry.at("reference_walls")) walls.insert(wall.at("id"));
    std::map<std::string, const PortConfig*> ports;
    for (const auto& p : scene.ports) {
        ports[p.id] = &p;
        if (walls.contains(p.cell_id)) owned_ports.insert(p.id);
    }
    auto& contacts = e["assembly"]["gas_network"]["thermal_contacts"];
    contacts.erase(std::remove_if(contacts.begin(), contacts.end(), [&](const J& c) {
        return owned_ports.contains(c.value("port_id", ""));
    }), contacts.end());
    if (restore) {
        for (const auto& c : entry.at("reference_contacts")) contacts.push_back(c);
        return;
    }
    // Rebuild each existing wall-face tiling. Whole-face service ports are not
    // included when the reference uses smaller thermal tiles.
    for (const auto& wid : walls) for (int side : {-1, 1}) {
        const int axis = wid.back() - 'x';
        const int wall_sign = wid == id + "-plus" + "xyz"[axis] ? 1 : -1;
        const std::string surface = std::string(side > 0 ? "+" : "-") + "xyz"[axis];
        const std::string face = std::string(wall_sign > 0 ? "+" : "-") + "xyz"[axis];
        const bool inner = side != wall_sign;
        std::map<std::string, J> templates;
        std::map<std::string, int> volumes;
        double max_port_area = 0;
        for (const auto& c : entry.at("reference_contacts")) {
            const auto& p = *ports.at(c.at("port_id"));
            if (p.cell_id != wid || p.surface != surface) continue;
            const bool already_cut = inner || !entry.value("reference_outer_contacts_uncut", false);
            const double area = contact_area(p, boxes.at(c.at("compartment")), origin,
                                             already_cut ? source_holes : std::vector<EnclosureOpening>{}, face);
            if (area <= 1e-16) continue;
            auto law = c; scale_contact(law, 1/area);
            templates[p.id] = law;
            volumes[p.id] = c.value("thermal_volume", -1);
            max_port_area = std::max(max_port_area, p.rectangle_m[0]*p.rectangle_m[1]);
        }
        if (templates.empty()) continue;
        for (const auto& p : scene.ports) {
            if (p.cell_id != wid || p.surface != surface ||
                p.rectangle_m[0]*p.rectangle_m[1] > max_port_area*(1+1e-8)) continue;
            const auto& wall = scene.cells[p.cell_index];
            const double pos[] = {p.centroid_m.x, p.centroid_m.y, p.centroid_m.z};
            const double center[] = {wall.position_m.x, wall.position_m.y, wall.position_m.z};
            int index[3];
            for (int d = 0; d < 3; ++d)
                index[d] = std::clamp(int((pos[d]-center[d]+wall.dimensions_m[d]/2) /
                                         wall.dimensions_m[d]*wall.thermal_divisions[d]),
                                      0, wall.thermal_divisions[d]-1);
            const int volume = volumes.contains(p.id) ? volumes.at(p.id) :
                index[0] + wall.thermal_divisions[0]*(index[1]+wall.thermal_divisions[1]*index[2]);
            for (const auto& [gas, box] : boxes.items()) {
                // Gas lies on the side pointed to by this surface normal.
                const double edge = box.at(side > 0 ? "low_m" : "high_m")[axis];
                if (std::abs(edge - pos[axis]) > 1e-10) continue;
                const double area = contact_area(p, box, origin, holes, face);
                if (area <= 1e-16) continue;
                auto c = templates.contains(p.id) ? templates.at(p.id) : templates.begin()->second;
                c["port_id"] = p.id; c["compartment"] = gas; c["thermal_volume"] = volume;
                scale_contact(c, area);
                contacts.push_back(c);
            }
        }
    }
}
} // namespace battery::explicit_case_detail

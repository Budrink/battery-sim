#include "contract_internal.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/thermal/object_grid.hpp"
#include <algorithm>
#include <cmath>
#include <map>

namespace battery::contract_detail {
namespace {
// A complete physical face has one connected fracture front, regardless of
// battery shape or whether its channels came from the editor or an import.
// Partial apertures and an explicit legacy opt-out retain their declared scope.
void connect_complete_faces(J& document) {
    auto& channels = document["gas_network"]["channels"];
    const auto scene = scenario_from_json(document);
    std::map<int, std::vector<std::size_t>> faces;
    for (std::size_t i = 0; i < scene.gas_channels.size(); ++i) {
        const auto& ch = scene.gas_channels[i];
        if (ch.shell_object >= 0)
            faces[ch.shell_object].push_back(i);
    }
    for (const auto& [object, indices] : faces) {
        const auto& face = scene.cells[object];
        if (face.shell_contents_id.empty()) continue;
        const auto n = face.thermal_divisions;
        std::vector<bool> covered(n[0] * n[1] * n[2], false);
        bool compatible = indices.size() == covered.size();
        for (auto i : indices) {
            const auto& ch = scene.gas_channels[i];
            const auto law = channels[i].value("damage_opening", J::object());
            compatible &= ch.opening.deterministic_aperture && law.value("whole_plate", true);
            const int tile = ch.shell_thermal_volume;
            if (tile < 0 || tile >= int(covered.size()) || covered[tile]) compatible = false;
            else covered[tile] = true;
        }
        if (compatible && std::all_of(covered.begin(), covered.end(), [](bool b) { return b; }))
            for (auto i : indices) channels[i]["damage_opening"]["whole_plate"] = true;
    }
}

// A resolved material pore must vent through the adjacent casing patch, not
// through the model's fallback gas compartment on the opposite side of a cell.
std::string pore_source(const Scenario& scene, const CellConfig& shell, int axis, bool positive,
                        const int xyz[3], const std::string& fallback) {
    auto contents = std::find_if(scene.cells.begin(), scene.cells.end(),
                                 [&](const auto& c) { return c.id == shell.shell_contents_id; });
    if (contents == scene.cells.end())
        return fallback;
    const int object = static_cast<int>(contents - scene.cells.begin());
    const bool bound = std::any_of(scene.gas_compartments.begin(), scene.gas_compartments.end(),
                                   [&](const auto& g) { return g.thermal_cell == object; });
    if (!bound)
        return fallback; // existing lumped-gas contracts remain supported
    const auto n = contents->thermal_divisions;
    int q[] = {xyz[0], xyz[1], xyz[2]};
    for (int d = 0; d < 3; ++d)
        if (d != axis) {
            if (shell.thermal_divisions[d] % n[d] != 0)
                fail("aperture", "Local pore and shell surface grids must align by integer refinement");
            q[d] /= shell.thermal_divisions[d] / n[d];
        }
    q[axis] = positive ? n[axis] - 1 : 0;
    const int local = q[0] + n[0] * (q[1] + n[1] * q[2]);
    auto gas = std::find_if(scene.gas_compartments.begin(), scene.gas_compartments.end(), [&](const auto& g) {
        return g.thermal_cell == object && g.thermal_volume == local;
    });
    if (gas == scene.gas_compartments.end())
        fail("aperture", "Missing local pore behind shell patch");
    return gas->id;
}
} // namespace
// Contract openings describe a physical face. Native aperture state is local to
// one thermal patch: expand here, not in the solver and not in the reusable model.
void resolve_vent_patches(J& out) {
    auto& channels = out["gas_network"]["channels"];
    const auto needs_patch = [](const J& ch) {
        return ch.value("shell_thermal_volume", -1) < 0 && ch.contains("damage_opening") &&
               ch.at("damage_opening").value("mode", "") == "deterministic_aperture";
    };
    if (std::none_of(channels.begin(), channels.end(), needs_patch)) {
        connect_complete_faces(out);
        return;
    }
    const auto scene = scenario_from_json(out);
    J resolved = J::array();
    for (const auto& ch : channels) {
        if (!needs_patch(ch)) {
            resolved.push_back(ch);
            continue;
        }
        const auto port = std::find_if(scene.ports.begin(), scene.ports.end(),
                                       [&](const auto& p) { return p.id == ch.at("port_id"); });
        if (port == scene.ports.end())
            fail("aperture", "Missing shell port");
        const auto& cell = scene.cells[port->cell_index];
        const auto n = cell.thermal_divisions;
        const int axis = port->surface[1] - 'x', u = (axis + 1) % 3, v = (axis + 2) % 3;
        if (n[axis] != 1)
            fail("aperture", "Shell must have one layer through thickness");
        const auto grid = object_grid(scene, cell);
        const auto& law = ch.at("damage_opening");
        int valves = 0;
        for (int z = 0; z < n[2]; ++z)
            for (int y = 0; y < n[1]; ++y)
                for (int x = 0; x < n[0]; ++x) {
                    const int xyz[] = {x, y, z}, local = x + n[0] * (y + n[1] * z);
                    const double w = grid[u][xyz[u] + 1] - grid[u][xyz[u]],
                                 h = grid[v][xyz[v] + 1] - grid[v][xyz[v]];
                    auto patch = ch;
                    patch["id"] = ch.at("id").get<std::string>() + "-patch-" + std::to_string(local);
                    patch["shell_thermal_volume"] = local;
                    patch["from"] = pore_source(scene, cell, axis, port->surface[0] == '+', xyz,
                                                ch.at("from").get<std::string>());
                    patch["area_m2"] = 0; // native mesh derives exact patch area
                    if (law.contains("valve")) {
                        const auto center = law.at("valve").at("center_m").get<std::array<double, 2>>();
                        const double du = center[0] - (grid[u][xyz[u]] + w / 2);
                        const double dv = center[1] - (grid[v][xyz[v]] + h / 2);
                        const double radius = law.at("valve").at("radius_m");
                        if (std::abs(du) + radius <= w / 2 + 1e-12 &&
                            std::abs(dv) + radius <= h / 2 + 1e-12) {
                            patch["damage_opening"]["valve"]["center_m"] = {du, dv};
                            ++valves;
                        } else
                            patch["damage_opening"].erase("valve");
                    }
                    resolved.push_back(patch);
                }
        if (law.contains("valve") && valves != 1)
            fail("aperture",
                 "Valve crosses thermal patch edges: adjust its position/radius or shell surface divisions");
    }
    channels = std::move(resolved);
    connect_complete_faces(out);
}
} // namespace battery::contract_detail

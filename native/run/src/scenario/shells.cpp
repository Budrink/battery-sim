#include "scenario_internal.hpp"
#include "battery_native/thermal/constitutive.hpp"
#include "battery_native/thermal/state_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include "../shell_face_material.hpp"

namespace battery::scenario_detail {
void parse_shells(const json& root, Scenario& scenario, CellIndex& cell_index) {
    for (const auto& value : root.value("shells", json::array())) {
        ShellConfig sh;
        sh.contents_id = value.at("contents_id");
        sh.material_id = value.at("material_id");
        sh.thickness_m = value.at("thickness_m");
        sh.density_kg_m3 = value.at("density_kg_m3");
        sh.specific_heat_j_kg_k = value.at("specific_heat_j_kg_k");
        sh.conductivity_w_per_m_k = value.at("conductivity_w_per_m_k").get<std::array<double, 3>>();
        sh.contents_contact_h = value.at("contents_contact_h");
        sh.seam_contact_h = value.at("seam_contact_h");
        sh.damage_onset_k = value.value("damage_onset_k", 500.);
        sh.damage_rate_per_s = value.value("damage_rate_per_s", 0.);
        sh.damage_scale_k = value.value("damage_scale_k", 25.);
        sh.destroyed_conductance_factor = value.value("destroyed_conductance_factor", .01);
        sh.initial_temperature_k = value.value("initial_temperature_k", -1.);
        if (value.contains("surface_divisions")) {
            sh.surface_divisions = value.at("surface_divisions").get<std::array<int, 3>>();
            for (int n : sh.surface_divisions)
                if (n <= 0)
                    throw std::runtime_error("Shell surface_divisions must be positive");
        }
        if (value.contains("initial_temperature_k") && sh.initial_temperature_k <= 0)
            throw std::runtime_error("Shell temperature must be positive");
        if (value.contains("face_materials"))
            for (const auto& [face, material] : value.at("face_materials").items()) {
                if (face != "-x" && face != "+x" && face != "-y" && face != "+y" &&
                    face != "-z" && face != "+z")
                    throw std::runtime_error("Invalid shell material face");
                auto& plate = sh.plates[2 * (face[1] - 'x') + (face[0] == '+')];
                const auto physical = [&](const char* key) {
                    if (!material.contains(key)) return 0.;
                    const double x = material.at(key);
                    if (!std::isfinite(x) || x <= 0.)
                        throw std::runtime_error(std::string("Invalid face parameter: ") + key);
                    return x;
                };
                plate.thickness_m = physical("thickness_m");
                plate.density_kg_m3 = physical("density_kg_m3");
                plate.specific_heat_j_kg_k = physical("specific_heat_j_kg_k");
            }
        add_physical_shell(scenario, sh);
        if (value.contains("contents_surface_connectivity") ||
            value.contains("contents_connectivity_per_m2")) {
            const auto response = parse_surface(value.value("contents_surface_connectivity", json::object()));
            int count = 0;
            for (auto& edge : scenario.interfaces)
                if (scenario.cells[edge.cell_a].id == sh.contents_id &&
                    scenario.cells[edge.cell_b].shell_contents_id == sh.contents_id) {
                    edge.has_endpoint_response = true;
                    if (value.contains("contents_connectivity_per_m2")) {
                        // Conductances supplied for one square metre; preserve
                        // exact generated contact areas and the common solver.
                        edge.law = parse_law(value.at("contents_connectivity_per_m2"));
                        if (edge.law.kind != TransitionKind::DirectedGeometry)
                            throw std::runtime_error("Contents contact requires directed geometry");
                        const auto port = std::find_if(scenario.ports.begin(), scenario.ports.end(),
                                                       [&](const auto& p) { return p.id == edge.port_a; });
                        if (port == scenario.ports.end())
                            throw std::runtime_error("Missing contents port");
                        const double area = port->area_m2;
                        edge.law.background_conductance_w_per_k *= area;
                        edge.law.active_conductance_w_per_k *= area;
                        edge.law.destroyed_background_conductance_w_per_k *= area;
                        edge.law.destroyed_active_conductance_w_per_k *= area;
                    }
                    edge.law.endpoint_a = edge.law.endpoint_b = response;
                    ++count;
                }
            if (count != 6)
                throw std::runtime_error("Expected six contents/casing contacts");
        }
        const auto frame = value.value("conductivity_frame", std::string("global"));
        if (frame != "global" && frame != "normal_tangential")
            throw std::runtime_error("Unknown casing conductivity frame");
        for (auto& panel : scenario.cells)
            if (panel.shell_contents_id == sh.contents_id) {
                const auto face = panel.id.substr(panel.id.size() - 2);
                const int axis = face[1] - 'x';
                if (frame == "normal_tangential")
                    for (int k = 0; k < 3; ++k)
                        panel.conductivity_w_per_m_k[(axis + k) % 3] = sh.conductivity_w_per_m_k[k];
                if (value.contains("surface_connectivity"))
                    panel.surface_connectivity = parse_surface(value.at("surface_connectivity"));
                if (value.contains("initial_temperature_fields_k") &&
                    value.at("initial_temperature_fields_k").contains(face))
                    panel.initial_temperature_field_k =
                        value.at("initial_temperature_fields_k").at(face).get<std::vector<double>>();
            }
    }
    for (const auto& shell : root.value("shells", json::array()))
        if (shell.contains("face_materials"))
            apply_shell_face_materials(scenario, shell.at("contents_id"), shell.at("face_materials"));
    for (std::size_t i = 0; i < scenario.cells.size(); ++i)
        cell_index[scenario.cells[i].id] = i;
}
} // namespace battery::scenario_detail

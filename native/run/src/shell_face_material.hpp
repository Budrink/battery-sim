#pragma once
#include "battery_native/model/model.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <stdexcept>

namespace battery {
// Per-face constitutive overrides. Plate geometry and rho*cp*V have already
// been built by add_physical_shell. Conductivity vectors use the global frame.
inline void apply_shell_face_materials(Scenario& scene, const std::string& contents,
                                       const nlohmann::json& faces) {
    if (!faces.is_object())
        throw std::runtime_error("face_materials must be an object");
    for (const auto& [face, material] : faces.items()) {
        if (face != "+x" && face != "-x" && face != "+y" && face != "-y" && face != "+z" && face != "-z")
            throw std::runtime_error("Invalid shell material face");
        if (!material.is_object())
            throw std::runtime_error("Invalid face material");
        for (const auto& item : material.items())
            if (item.key() != "material_id" && item.key() != "conductivity_w_per_m_k" &&
                item.key() != "damage_onset_k" && item.key() != "damage_rate_per_s" &&
                item.key() != "damage_scale_k" && item.key() != "destroyed_conductance_factor" &&
                item.key() != "thickness_m" && item.key() != "density_kg_m3" &&
                item.key() != "specific_heat_j_kg_k")
                throw std::runtime_error("Unknown face material property: " + item.key());
        const auto id = material.at("material_id").get<std::string>();
        const auto conductivity = material.at("conductivity_w_per_m_k").get<std::array<double, 3>>();
        if (id.empty())
            throw std::runtime_error("Empty face material id");
        for (double k : conductivity)
            if (!std::isfinite(k) || k <= 0.)
                throw std::runtime_error("Invalid face conductivity");
        bool found = false;
        for (auto& panel : scene.cells)
            if (panel.id == contents + ":case:" + face) {
                panel.material_id = id;
                panel.conductivity_w_per_m_k = conductivity;
                const auto parameter = [&](const char* key, double fallback, bool zero = false) {
                    const double x = material.value(key, fallback);
                    if (!std::isfinite(x) || (zero ? x < 0 : x <= 0))
                        throw std::runtime_error(std::string("Invalid face parameter: ") + key);
                    return x;
                };
                panel.damage_onset_k = parameter("damage_onset_k", panel.damage_onset_k);
                panel.damage_rate_at_onset_per_s =
                    parameter("damage_rate_per_s", panel.damage_rate_at_onset_per_s, true);
                panel.damage_temperature_scale_k =
                    parameter("damage_scale_k", panel.damage_temperature_scale_k);
                panel.bulk_connectivity.destroyed_background_conductance_w_per_k =
                    parameter("destroyed_conductance_factor",
                              panel.bulk_connectivity.destroyed_background_conductance_w_per_k);
                found = true;
            }
        if (!found)
            throw std::runtime_error("Missing shell face: " + contents + face);
    }
    // Keep generated contact material labels consistent with panel properties.
    for (auto& edge : scene.interfaces) {
        edge.material_a = scene.cells[edge.cell_a].material_id;
        edge.material_b = scene.cells[edge.cell_b].material_id;
    }
}
} // namespace battery

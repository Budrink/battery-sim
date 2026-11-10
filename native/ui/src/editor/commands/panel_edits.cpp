#include "editor/editor_internal.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include <algorithm>

namespace battery::ui {
void ExperimentEditor::Impl::commit_panel_edits() {
    using J = nlohmann::json;
    // Each form is independent. Keep rejected input visible, and never retry an
    // identical invalid candidate every frame. No drawing, GPU allocation or stepping.
    auto commit = [&](const char* key, bool& dirty, const J& fields, auto apply) {
        if (!dirty)
            return;
        if (panel_attempt.is_null())
            panel_attempt = J::object();
        const J signature = {fields, document.draft};
        if (panel_attempt.value(key, J()) == signature)
            return;
        panel_attempt[key] = signature;
        try {
            apply();
            dirty = false;
            panel_attempt.erase(key);
            panel_errors.erase(key);
        } catch (const std::exception& e) {
            panel_errors[key] = e.what();
        }
    };
    commit("Opening", opening_dirty,
           J{case_opening_selection.owner, case_opening_selection.id, case_opening_face, case_opening_shape,
             case_opening_size_mm[0], case_opening_size_mm[1], case_opening_center_mm[0],
             case_opening_center_mm[1]},
           [&] {
               auto openings =
                   editable_enclosures(document.draft).at(case_opening_selection.owner).at("openings");
               auto found = std::find_if(openings.begin(), openings.end(), [&](const auto& h) {
                   return h.at("id") == case_opening_selection.id;
               });
               if (found == openings.end())
                   throw std::runtime_error("The selected opening no longer exists.");
               (*found)["face"] = case_opening_face;
               (*found)["shape"] = case_opening_shape == 0 ? "circle" : "rectangle";
               (*found)["size_m"] = {case_opening_size_mm[0] / 1000, case_opening_size_mm[1] / 1000};
               (*found)["center_m"] = {case_opening_center_mm[0] / 1000, case_opening_center_mm[1] / 1000};
               document.draft =
                   set_enclosure_openings(document.draft, case_opening_selection.owner, openings);
               case_opening_error.clear();
               message = "Opening updated automatically.";
           });
    commit("Case", enclosure_dirty, J{enclosure_loaded, enclosure_properties, enclosure_gas}, [&] {
        const auto registry =
            document.draft.value("metadata", J::object()).value("editor_enclosures", J::object());
        const bool explicit_case =
            registry.contains(enclosure_loaded) && registry.at(enclosure_loaded).contains("explicit_mesh");
        auto properties = enclosure_properties;
        if (!explicit_case && properties.contains("volumetric_heat_capacity_j_m3_k")) {
            const double density = properties.at("density_kg_m3");
            if (!(density > 0))
                throw std::runtime_error("Case density must be positive.");
            properties["specific_heat_j_kg_k"] =
                properties.at("volumetric_heat_capacity_j_m3_k").get<double>() / density;
        }
        document.draft =
            explicit_case ? edit_explicit_enclosure_properties(document.draft, enclosure_loaded, properties)
                          : set_enclosure(document.draft, enclosure_loaded, enclosure_gas, properties);
        message = "Case updated automatically; existing openings retained.";
    });
    commit("Dimensions", shape_dirty,
           J{shape_owner, shape_kind, shape_dimensions_mm[0], shape_dimensions_mm[1], shape_dimensions_mm[2]},
           [&] {
               std::vector<double> dimensions;
               if (shape_kind)
                   dimensions = {shape_dimensions_mm[0] * .0005, shape_dimensions_mm[1] * .001};
               else
                   dimensions = {shape_dimensions_mm[0] * .001, shape_dimensions_mm[1] * .001,
                                 shape_dimensions_mm[2] * .001};
               document.draft = set_battery_geometry(document.draft, shape_owner,
                                                     shape_kind ? "cylinder" : "prismatic", dimensions);
               message = "Dimensions updated. Review external contacts after resizing.";
           });
    commit("Shell", shell_dirty,
           J{shell_owner, face_material, opening_kind, opening_pressure_kpa, area_rate, fracture_energy,
             weak_factor, weak_size_mm, valve_center_mm[0], valve_center_mm[1], gas_destination},
           [&] {
               J channel;
               if (opening_kind) {
                   channel = face_channel;
                   channel["opening_pressure_pa"] = opening_pressure_kpa * 1000;
                   J opening = channel.value("damage_opening", J::object());
                   const J defaults = {{"mode", "deterministic_aperture"},
                                       {"whole_plate", true},
                                       {"pressure_exponent", 1},
                                       {"max_damage_step", .005}};
                   for (const auto& [key, value] : defaults.items())
                       if (!opening.contains(key))
                           opening[key] = value;
                   opening["area_rate_per_s"] = area_rate;
                   opening["fracture_energy_j_per_m2"] = fracture_energy;
                   opening.erase("valve");
                   opening.erase("seam_width_m");
                   opening.erase("seam_strength_factor");
                   if (opening_kind == 2)
                       opening["valve"] = {
                           {"radius_m", weak_size_mm * .001},
                           {"center_m", {valve_center_mm[0] * .001, valve_center_mm[1] * .001}},
                           {"pressure_factor", weak_factor}};
                   if (opening_kind == 3) {
                       opening["seam_width_m"] = weak_size_mm * .001;
                       opening["seam_strength_factor"] = weak_factor;
                   }
                   channel["damage_opening"] = opening;
               }
               document.draft =
                   set_shell_face(document.draft, shell_owner, shell_source.at(1).get<std::string>(),
                                  face_material, channel, gas_destination);
               message = "Selected shell face updated automatically.";
           });
}
} // namespace battery::ui

#include "editor/editor_internal.hpp"
#include "shared/form_fields.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

namespace battery::ui {
void ExperimentEditor::Impl::draw_shell() {
    using J = nlohmann::json;
    const auto& a = document.draft.at("assembly");
    if (form::begin_combo("Battery", selected.c_str())) {
        for (const auto& b : a.at("batteries")) {
            const auto id = b.at("id").get<std::string>();
            if (ImGui::Selectable(id.c_str(), selected == id))
                selected = id;
        }
        ImGui::EndCombo();
    }
    form::combo("Face", &shell_face, "-X\0+X\0-Y\0+Y\0-Z bottom\0+Z top\0");
    const char* faces[] = {"-x", "+x", "-y", "+y", "-z", "+z"};
    const std::string f = faces[shell_face];
    const J* model = nullptr;
    for (const auto& b : a.at("batteries"))
        if (b.at("id") == selected)
            for (const auto& m : document.draft.at("battery_models"))
                if (m.at("id") == b.at("model_id") && m.at("revision") == b.at("model_revision"))
                    model = &m;
    if (!model || model->at("shell").is_null()) {
        ImGui::TextWrapped("Select a battery with a physical casing.");
        return;
    }
    const J source = {*model, f, a.at("vent_connections")};
    if (shell_owner != selected || source != shell_source) {
        if (shell_dirty) {
            commit_panel_edits();
            if (shell_dirty)
                ImGui::TextWrapped("Correct or discard the pending shell input before changing face.");
            return;
        }
        shell_owner = selected;
        shell_source = source;
        const auto& sh = model->at("shell");
        face_material = {{"material_id", sh.at("material_id")},
                         {"thickness_m", sh.at("thickness_m")},
                         {"density_kg_m3", sh.at("density_kg_m3")},
                         {"specific_heat_j_kg_k", sh.at("specific_heat_j_kg_k")},
                         {"conductivity_w_per_m_k", sh.at("conductivity_w_per_m_k")},
                         {"damage_onset_k", sh.value("damage_onset_k", 500.)},
                         {"damage_rate_per_s", sh.value("damage_rate_per_s", 0.)},
                         {"damage_scale_k", sh.value("damage_scale_k", 25.)},
                         {"destroyed_conductance_factor", sh.value("destroyed_conductance_factor", .01)}};
        if (sh.contains("face_materials") && sh.at("face_materials").contains(f))
            face_material.update(sh.at("face_materials").at(f));
        face_channel = {{"kind", "rupture"},
                        {"opening_pressure_pa", 80000},
                        {"strength_reference_k", 300},
                        {"strength_temperature_scale_k", 300},
                        {"discharge_coefficient", .7}};
        gas_destination.clear();
        opening_kind = 0;
        for (const auto& v : model->at("vents"))
            if (v.at("port_suffix") == "case:" + f + ":outer") {
                face_channel = v.at("channel");
                opening_kind = 1;
                for (const auto& r : a.at("vent_connections"))
                    if (r.at("battery_id") == selected && r.at("vent_id") == v.at("id"))
                        gas_destination = r.at("to");
            }
        const auto o = face_channel.value("damage_opening", J::object());
        opening_pressure_kpa = face_channel.value("opening_pressure_pa", 80000.) / 1000;
        area_rate = o.value("area_rate_per_s", 1.);
        fracture_energy = o.value("fracture_energy_j_per_m2", 200.);
        weak_factor = .4;
        weak_size_mm = 1;
        valve_center_mm[0] = valve_center_mm[1] = 0;
        if (o.contains("valve")) {
            opening_kind = 2;
            const auto& v = o.at("valve");
            weak_factor = v.at("pressure_factor");
            weak_size_mm = v.at("radius_m").get<double>() * 1000;
            for (int k = 0; k < 2; ++k)
                valve_center_mm[k] = v.at("center_m")[k].get<double>() * 1000;
        } else if (o.value("seam_width_m", 0.) > 0) {
            opening_kind = 3;
            weak_factor = o.value("seam_strength_factor", .4);
            weak_size_mm = o.at("seam_width_m").get<double>() * 1000;
        }
        if (gas_destination.empty())
            for (const auto& g : a.at("gas_network").at("compartments"))
                if (!g.contains("contents_id")) {
                    gas_destination = g.at("id");
                    break;
                }
    }
    const J fields_before = {face_material,      opening_kind,   opening_pressure_kpa, area_rate,
                             fracture_energy,    weak_factor,    weak_size_mm,         valve_center_mm[0],
                             valve_center_mm[1], gas_destination};
    ImGui::TextWrapped(
        "Only the selected battery and face. Other instances keep their model. No timed ignition.");
    auto number = [&](const char* label, const char* key, double scale = 1., double offset = 0.) {
        form::number(face_material, key, label, scale, offset);
    };
    auto material = face_material.at("material_id").get<std::string>();
    if (form::text_input("Material ID", &material))
        face_material["material_id"] = material;
    ImGui::TextWrapped("Physical plate: own temperature, heat capacity and irreversible damage. "
                       "Contact subdivisions are not separate shell materials.");
    number("Plate thickness mm", "thickness_m", 1000.);
    number("Plate density kg/m3", "density_kg_m3");
    number("Plate specific heat J/(kg K)", "specific_heat_j_kg_k");
    for (int k = 0; k < 3; ++k) {
        const char* labels[] = {"Thermal conductivity X [W/(m K)]", "Thermal conductivity Y [W/(m K)]",
                                "Thermal conductivity Z [W/(m K)]"};
        double x = face_material["conductivity_w_per_m_k"][k];
        if (form::input(labels[k], &x, "%.8g", "conductivity_w_per_m_k"))
            face_material["conductivity_w_per_m_k"][k] = x;
    }
    number("Damage onset C", "damage_onset_k", 1., -273.15);
    number("Damage rate 1/s", "damage_rate_per_s");
    number("Damage temperature scale K", "damage_scale_k");
    number("Destroyed conductivity factor", "destroyed_conductance_factor");
    ImGui::SeparatorText("Pressure opening / gas path");
    form::combo("Opening", &opening_kind,
                "Sealed (no gas path)\0Panel rupture\0Weak circular valve\0Weak perimeter seam\0");
    if (opening_kind) {
        form::input("Rupture pressure difference [kPa]", &opening_pressure_kpa, "%.8g",
                    "opening_pressure_pa");
        form::input("Opening rate [1/s]", &area_rate, "%.8g", "area_rate_per_s");
        form::input("Energy needed to rupture unit area [J/m2]", &fracture_energy, "%.8g",
                    "fracture_energy_j_per_m2");
        if (opening_kind >= 2) {
            form::input("Valve / seam strength relative to plate [0..1]", &weak_factor, "%.8g",
                        "pressure_factor");
            form::input(opening_kind == 2 ? "Valve radius [mm]" : "Perimeter seam width [mm]", &weak_size_mm,
                        "%.8g", opening_kind == 2 ? "radius_m" : "seam_width_m");
            if (opening_kind == 2) {
                form::input("Valve center U [mm]", &valve_center_mm[0], "%.8g", "valve_center");
                form::input("Valve center V [mm]", &valve_center_mm[1], "%.8g", "valve_center");
            }
        }
        if (form::begin_combo("Gas destination", gas_destination.c_str())) {
            if (ImGui::Selectable("outside_domain", gas_destination == "outside_domain"))
                gas_destination = "outside_domain";
            for (const auto& g : a.at("gas_network").at("compartments")) {
                if (g.value("contents_id", "") == selected)
                    continue;
                auto id = g.at("id").get<std::string>();
                if (ImGui::Selectable(id.c_str(), id == gas_destination))
                    gas_destination = id;
            }
            ImGui::EndCombo();
        }
        ImGui::TextWrapped("Opening uses one connected rupture front for this face. It starts where pressure "
                           "relative to remaining strength is highest, then advances to neighbouring tiles. "
                           "Valve/seam lowers local strength; thermal tiles keep their own temperature. "
                           "Unloading can arrest the front. Thermal damage does not heal.");
    }
    const J fields_after = {face_material,      opening_kind,   opening_pressure_kpa, area_rate,
                            fracture_energy,    weak_factor,    weak_size_mm,         valve_center_mm[0],
                            valve_center_mm[1], gas_destination};
    shell_dirty |= fields_before != fields_after;
    ImGui::TextWrapped("Valid face changes apply automatically when you leave the field.");
}
} // namespace battery::ui

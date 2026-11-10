#include "editor/editor_internal.hpp"
#include "shared/toolbar.hpp"
#include "shared/form_fields.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/contracts/enclosure_openings.hpp"
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"
#include <algorithm>
#include <cmath>
namespace battery::ui {
using J = nlohmann::json;
void ExperimentEditor::Impl::draw_assembly() {
    auto& a = document.draft["assembly"];
    ImGui::TextWrapped("Select an object. Position fields are in mm; contract values remain SI. Preview "
                       "updates automatically.");
    if (form::begin_combo("Element", selected.c_str())) {
        for (const auto* key : {"batteries", "passive_cells"})
            if (a.contains(key))
                for (auto& obj : a[key]) {
                    const auto id = obj.at("id").get<std::string>();
                    if (ImGui::Selectable(id.c_str(), selected == id)) {
                        selected = id;
                        case_opening_selection.id.clear();
                    }
                }
        ImGui::EndCombo();
    }
    ImGui::Separator();
    bool is_battery = false;
    std::string pad_preset;
    for (const auto* key : {"batteries", "passive_cells"})
        if (a.contains(key))
            for (auto& obj : a[key])
                if (obj.at("id") == selected) {
                    is_battery = std::string(key) == "batteries";
                    ImGui::TextWrapped("Object: %s", selected.c_str());
                    if (!is_battery && obj.contains("material_id")) {
                        const std::string material = obj.at("material_id");
                        const auto styles = document.draft.value("metadata", J::object())
                                                .value("visual_materials", J::object());
                        const auto overrides = document.draft.value("metadata", J::object())
                                                   .value("visual_objects", J::object());
                        const auto current =
                            overrides.value(selected, styles.value(material, std::string("unspecified")));
                        if (current == "insulation" || current == "thermal_conductor" ||
                            current == "unspecified") {
                            int kind = current == "insulation" ? 1 : current == "thermal_conductor" ? 2 : 0;
                            if (form::combo("Pad type", &kind,
                                            "Custom\0Insulation (k=0.06)\0Thermal conductor (k=2)\0") &&
                                kind != 0)
                                pad_preset = kind == 1 ? "insulation" : "thermal_conductor";
                            ImGui::TextWrapped("Selected object only. Type sets conductivity and appearance. "
                                               "Heat capacity, damage and contact laws are retained.");
                        }
                    }
                    if (!is_battery && obj.contains("thermal") &&
                        obj["thermal"].contains("conductivity_w_per_m_k")) {
                        auto& k = obj["thermal"]["conductivity_w_per_m_k"];
                        double values[3] = {k[0], k[1], k[2]};
                        ImGui::TextUnformatted("Conductivity XYZ, W/(m K)");
                        ImGui::SetNextItemWidth(-1);
                        if (ImGui::InputScalarN("##conductivity", ImGuiDataType_Double, values, 3, nullptr,
                                                nullptr, "%.6g"))
                            k = {values[0], values[1], values[2]};
                    }
                    if (obj.contains("pose"))
                        form::vector(obj.at("pose"), "position_m", "Position [mm]", 1000.);
                    if (obj.contains("initial_state"))
                        form::number(obj.at("initial_state"), "temperature_k", "Initial temperature [C]", 1.,
                                     -273.15);
                    if (!is_battery && obj.contains("thermal")) {
                        auto& thermal = obj.at("thermal");
                        form::number(thermal, "heat_capacity_j_per_k", "Element heat capacity [J/K]");
                        form::number(thermal, "initial_temperature_k", "Initial temperature [C]", 1.,
                                     -273.15);
                        form::vector(thermal, "divisions", "Thermal grid divisions", 1., true);
                    }
                    if (!is_battery && obj.contains("geometry"))
                        form::vector(obj.at("geometry"), "dimensions_m", "Dimensions [mm]", 1000.);
                    if (obj.contains("pose") && ImGui::CollapsingHeader("Rotation (advanced)"))
                        form::vector(obj.at("pose"), "orientation_xyzw",
                                     "Orientation quaternion (X, Y, Z, W)");
                    if (obj.contains("initial_state")) {
                        auto& initial = obj.at("initial_state");
                        form::number(initial, "damage", "Initial damage fraction [0..1]");
                        form::number(initial, "charged_fraction", "Initial state of charge [0..1]");
                    }
                }
    if (!pad_preset.empty()) {
        try {
            document.draft = set_pad_conductivity_preset(document.draft, selected, pad_preset);
            message = "Pad conductivity changed for " + selected + "; preview updates automatically.";
        } catch (const std::exception& e) {
            message = e.what();
        }
        return; // replacing a validated draft invalidates references into the old document
    }
    if (is_battery && draw_battery_shape())
        return; // the validated replacement invalidates references into the old document
    ImGui::BeginDisabled(selected.empty());
    const bool duplicate = ImGui::Button("Duplicate element");
    ImGui::EndDisabled();
    if (duplicate)
        try {
            document.draft = duplicate_element(document.draft, selected);
            selected = document.draft["assembly"][is_battery ? "batteries" : "passive_cells"].back().at("id");
            message = "Element duplicated. Review its position, contacts and wiring. Battery vents keep "
                      "their explicit destinations.";
            return;
        } catch (const std::exception& e) {
            message = e.what();
        }
    ImGui::TextWrapped("Case gas mesh updates automatically. Solid contacts and electrical wiring are not "
                       "inferred when moving or duplicating objects.");
    ImGui::TextWrapped("Connections and gas transport: Contacts. Global R3 combustion: Gas model / R3.");
}
} // namespace battery::ui

#include "editor_internal.hpp"
#include "form_fields.hpp"
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
    ImGui::BeginChild("objects", {160, 260}, true);
    for (const auto* key : {"batteries", "passive_cells"})
        if (a.contains(key))
            for (auto& obj : a[key]) {
                const auto id = obj.at("id").get<std::string>();
                if (ImGui::Selectable(id.c_str(), selected == id)) {
                    selected = id;
                    case_opening_selection.id.clear();
                }
            }
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("object properties", {0, 260}, true);
    bool is_battery = false;
    std::string pad_preset;
    for (const auto* key : {"batteries", "passive_cells"})
        if (a.contains(key))
            for (auto& obj : a[key])
                if (obj.at("id") == selected) {
                    is_battery = std::string(key) == "batteries";
                    ImGui::Text("Object: %s", selected.c_str());
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
                            if (ImGui::Combo("Pad type", &kind,
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
                    if (obj.contains("pose")) {
                        float p[3];
                        for (int i = 0; i < 3; ++i)
                            p[i] = float(obj["pose"]["position_m"][i].get<double>() * 1000);
                        if (ImGui::InputFloat3("Position mm", p))
                            for (int i = 0; i < 3; ++i)
                                obj["pose"]["position_m"][i] = double(p[i]) / 1000.;
                    }
                    if (obj.contains("initial_state")) {
                        double c = obj["initial_state"]["temperature_k"].get<double>() - 273.15;
                        if (ImGui::InputDouble("Initial C", &c))
                            obj["initial_state"]["temperature_k"] = c + 273.15;
                    }
                    if(!is_battery && obj.contains("thermal")) {
                        auto& thermal=obj.at("thermal");
                        form::number(thermal,"heat_capacity_j_per_k","Element heat capacity [J/K]");
                        form::number(thermal,"initial_temperature_k","Initial temperature [C]",1.,-273.15);
                        form::vector(thermal,"divisions","Thermal grid divisions",1.,true);
                    }
                    if(!is_battery && obj.contains("geometry"))
                        form::vector(obj.at("geometry"),"dimensions_m","Dimensions [mm]",1000.);
                    if(obj.contains("pose"))
                        form::vector(obj.at("pose"),"orientation_xyzw","Orientation quaternion (X, Y, Z, W)");
                    if(obj.contains("initial_state")) {
                        auto& initial=obj.at("initial_state");
                        form::number(initial,"damage","Initial damage fraction [0..1]");
                        form::number(initial,"charged_fraction","Initial state of charge [0..1]");
                    }
                }
    ImGui::EndChild();
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
    ImGui::TextWrapped("Connections and gas transport: Contacts. Global R3 combustion: Models.");
}
void ExperimentEditor::Impl::delete_selected() {
    try {
        if (selected.empty() && !case_opening_selection.id.empty()) {
            auto list =
                editable_enclosures(document.draft).at(enclosure_id).value("openings", J::array());
            list.erase(std::remove_if(list.begin(), list.end(),
                                      [&](const auto& h) { return h.at("id") == case_opening_selection.id; }),
                       list.end());
            document.draft = set_enclosure_openings(document.draft, enclosure_id, list);
            case_opening_selection.id.clear();
            message = "Opening deleted; Undo restores it.";
            return;
        }
        const auto id = selected;
        document.draft = delete_element(document.draft, id);
        selected.clear();
        message = "Deleted " + id +
                  " and dependent contacts/routes. Undo restores the edit. "
                  "Running simulation unchanged; review wiring and heater controls before RUN SNAPSHOT.";
    } catch (const std::exception& e) {
        message = e.what();
    }
}
void ExperimentEditor::Impl::wire() {
    auto next = document.draft;
    auto& a = next["assembly"];
    J circuit{{"mode", "solved_dc"},
              {"reference_node", "ground"},
              {"nodes", {"ground", "supply"}},
              {"branches", J::array()}};
    const auto count = a["batteries"].size();
    if (!count)
        throw std::runtime_error("Add a battery first");
    for (std::size_t i = 0; i < count; ++i) {
        const auto& b = a["batteries"][i];
        const auto bid = b.at("id").get<std::string>();
        const std::string from = parallel || i == 0 ? "supply" : "junction-" + std::to_string(i);
        const std::string to = parallel || i + 1 == count ? "ground" : "junction-" + std::to_string(i + 1);
        if (!parallel && i + 1 < count)
            circuit["nodes"].push_back(to);
        circuit["branches"].push_back({{"id", "branch-" + bid},
                                       {"kind", "battery"},
                                       {"cell_id", bid},
                                       {"from", from},
                                       {"to", to},
                                       {"emf_v", emf}});
        for (auto& model : next["battery_models"])
            if (model.at("id") == b.at("model_id") && model.at("revision") == b.at("model_revision"))
                if (!model["contents"].contains("electrical"))
                    model["contents"]["electrical"] = {{"resistance_ohm", .03}};
    }
    circuit["branches"].push_back({{"id", "source"},
                                   {"kind", voltage_source ? "voltage_source" : "current_source"},
                                   {"from", voltage_source ? "supply" : "ground"},
                                   {"to", voltage_source ? "ground" : "supply"}});
    a["electrical_circuit"] = circuit;
    next["metadata"]["editor_changes"].push_back(
        {{"kind", "rewire_dc"},
         {"emf_v", emf},
         {"assumption", "Fixed declared EMFs; missing electrical laws initialized to synthetic 0.03 ohm"}});
    if (next["provenance"]["status"] != "synthetic")
        next["provenance"]["status"] = "mixed";
    // Explicit replace-all action: the wizard connects every battery in this document.
    next["conditions"]["electrical_drives"] = J::array();
    next["conditions"]["source_drives"] = {
        {{"source_id", "source"}, {voltage_source ? "voltage_v" : "current_a", source_value}}};
    (void)compile_experiment(next);
    document.draft = std::move(next);
    message = "DC network created. Currents are solved from current material resistance; inspect assumptions "
              "before running.";
}
void ExperimentEditor::Impl::draw_circuit() {
    ImGui::TextWrapped("Solved DC: current redistribution with temperature/composition. Positive battery "
                       "current = charging. Reverse discharge is rejected, not invented.");
    ImGui::Checkbox("Parallel (unchecked: series)", &parallel);
    ImGui::Checkbox("Voltage source (unchecked: current source)", &voltage_source);
    ImGui::InputFloat(voltage_source ? "Total source V" : "Total source A", &source_value);
    ImGui::InputFloat("Battery EMF V", &emf);
    ImGui::TextWrapped(
        "Replace all wiring below. Existing electrical laws are preserved; models without one receive an "
        "explicit synthetic 0.03 ohm law. Series supply voltage is the TOTAL voltage.");
    if (ImGui::Button("Replace wiring using these settings"))
        ImGui::OpenPopup("Replace electrical network?");
    if (ImGui::BeginPopupModal("Replace electrical network?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("Replace current wiring and per-battery drives in the EDITOR draft? Running "
                           "simulation is unaffected.");
        if (ImGui::Button("Replace")) {
            try {
                wire();
            } catch (const std::exception& e) {
                message = e.what();
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel"))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    const auto& branches = document.draft["assembly"]["electrical_circuit"]["branches"];
    if (ImGui::BeginTable("Wiring", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Element");
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("From (+)");
        ImGui::TableSetupColumn("To (-)");
        ImGui::TableHeadersRow();
        for (const auto& b : branches) {
            ImGui::TableNextRow();
            int column = 0;
            for (const auto* key : {"id", "kind", "from", "to"}) {
                ImGui::TableSetColumnIndex(column++);
                ImGui::TextUnformatted(b.at(key).get<std::string>().c_str());
            }
        }
        ImGui::EndTable();
    }
    draw_circuit_properties();
}
} // namespace battery::ui

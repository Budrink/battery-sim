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
void ExperimentEditor::Impl::draw_circuit() {
    ImGui::TextWrapped("Solved DC: current redistribution with temperature/composition. Positive battery "
                       "current = charging. Reverse discharge is rejected, not invented.");
    ImGui::Checkbox("Parallel (unchecked: series)", &parallel);
    ImGui::Checkbox("Voltage source (unchecked: current source)", &voltage_source);
    double supply = source_value;
    if (form::input(voltage_source ? "Total supply voltage [V]" : "Total supply current [A]", &supply))
        source_value = float(supply);
    double battery_emf = emf;
    if (form::input("Battery electromotive force [V]", &battery_emf))
        emf = float(battery_emf);
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
                ImGui::TextWrapped("%s", b.at(key).get<std::string>().c_str());
            }
        }
        ImGui::EndTable();
    }
    draw_circuit_properties();
}
} // namespace battery::ui

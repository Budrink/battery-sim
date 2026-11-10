#include "editor/editor_internal.hpp"
#include "editor/forms/model_properties.hpp"
#include "imgui.h"
#include "shared/form_fields.hpp"
#include <cmath>

namespace battery::ui {
void ExperimentEditor::Impl::draw_models() {
    auto& models = document.draft.at("battery_models");
    if (models.empty()) {
        ImGui::TextWrapped("No battery models in this experiment.");
        return;
    }
    const auto& batteries = document.draft.at("assembly").at("batteries");
    // Follow viewport/assembly selection, but allow browsing the whole library independently.
    if (model_selected_battery != selected) {
        model_selected_battery = selected;
        for (const auto& b : batteries)
            if (b.at("id") == selected)
                for (std::size_t i = 0; i < models.size(); ++i)
                    if (models[i].at("id") == b.at("model_id") &&
                        models[i].at("revision") == b.at("model_revision"))
                        model_index = i;
    }
    if (model_index >= models.size())
        model_index = 0;
    const auto title = [](const nlohmann::json& m) {
        return m.at("id").get<std::string>() + " / revision " + m.at("revision").get<std::string>();
    };
    ImGui::SeparatorText("Battery material: R1 / R2");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##model", title(models[model_index]).c_str())) {
        for (std::size_t i = 0; i < models.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::Selectable(title(models[i]).c_str(), model_index == i))
                model_index = i;
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    auto& model = models[model_index];
    int count = 0;
    std::string instances;
    for (const auto& b : batteries)
        if (b.at("model_id") == model.at("id") && b.at("model_revision") == model.at("revision")) {
            ++count;
            if (!instances.empty())
                instances += ", ";
            instances += b.at("id").get<std::string>();
        }
    ImGui::TextWrapped("Shared by %d batteries (hover for IDs). Edits affect the next Run simulation.",
                       count);
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28);
        ImGui::TextUnformatted(title(model).c_str());
        ImGui::Text("Used by: %s", instances.empty() ? "none" : instances.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
    ImGui::PushID(static_cast<int>(model_index));
    auto& edit = model_edits.edit(model_index, model);
    if (edit.value != edit.source) {
        ImGui::TextWrapped("Unapplied input; the last valid model remains active.");
        if (!edit.error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1, .45f, .3f, 1});
            ImGui::TextWrapped("Cannot apply: %s", edit.error.c_str());
            ImGui::PopStyleColor();
        }
        if (ImGui::Button("Discard model input"))
            edit = {model, model, {}, {}};
    }
    draw_model_properties(edit.value);
    ImGui::SeparatorText("Model shape / contents dimensions");
    auto& shape = edit.value.at("geometry");
    int kind = shape.at("kind") == "cylinder" ? 1 : 0;
    ImGui::TextWrapped("Shape (all batteries using this model)");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##model_shape", &kind, "Rectangular prism\0Cylinder\0")) {
        const auto d = shape.at("dimensions_m").get<std::vector<double>>();
        shape["kind"] = kind ? "cylinder" : "prismatic";
        shape["dimensions_m"] =
            kind ? nlohmann::json{std::sqrt(d[0] * d[1] / std::acos(-1.)), d[2]}
                 : nlohmann::json{std::sqrt(std::acos(-1.)) * d[0], std::sqrt(std::acos(-1.)) * d[0], d[1]};
    }
    auto& dimensions = shape.at("dimensions_m");
    for (std::size_t i = 0; i < dimensions.size(); ++i) {
        const double factor = kind && i == 0 ? 2000. : 1000.;
        double value = dimensions[i].get<double>() * factor;
        const char* caption = kind ? (i == 0 ? "Cylinder diameter [mm]" : "Cylinder height [mm]")
                                   : (i == 0   ? "Prism width X [mm]"
                                      : i == 1 ? "Prism length Y [mm]"
                                               : "Prism height Z [mm]");
        if (form::input(caption, &value))
            dimensions[i] = value / factor;
    }
    ImGui::TextWrapped(
        "Dimensions exclude the shell. Mass and total heat capacity are retained. "
        "Cylinder uses an equal-volume thermal prism; wrap/disk heater contacts use cylinder area. "
        "Review other contacts and gas routes after resizing.");
    ImGui::PopID();
}
} // namespace battery::ui

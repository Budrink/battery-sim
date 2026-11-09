#pragma once
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <algorithm>
#include <string>

namespace battery::ui::form {
using J = nlohmann::json;
inline bool input(const char* label, double* value, const char* format = "%.8g") {
    ImGui::TextWrapped("%s", label);
    ImGui::SetNextItemWidth(-1);
    return ImGui::InputDouble((std::string("##") + label).c_str(), value, 0, 0, format);
}
inline bool number(J& object, const char* key, const char* label, double scale = 1., double offset = 0.) {
    if (!object.contains(key) || !object.at(key).is_number())
        return false;
    ImGui::PushID(key);
    ImGui::TextWrapped("%s", label);
    ImGui::SetNextItemWidth(-1);
    double value = object.at(key).get<double>() * scale + offset;
    const bool changed = ImGui::InputDouble("##value", &value, 0, 0, "%.8g") && std::isfinite(value);
    if (changed)
        object[key] = (value - offset) / scale;
    ImGui::PopID();
    return changed;
}
inline void text(J& object, const char* key, const char* label) {
    if (!object.contains(key) || !object.at(key).is_string())
        return;
    ImGui::PushID(key);
    ImGui::TextWrapped("%s", label);
    ImGui::SetNextItemWidth(-1);
    auto value = object.at(key).get<std::string>();
    if (ImGui::InputText("##value", &value))
        object[key] = value;
    ImGui::PopID();
}
inline void integer(J& object, const char* key, const char* label) {
    if (!object.contains(key) || !object.at(key).is_number_integer())
        return;
    ImGui::PushID(key);
    ImGui::TextWrapped("%s", label);
    ImGui::SetNextItemWidth(-1);
    int value = object.at(key);
    if (ImGui::InputInt("##value", &value))
        object[key] = value;
    ImGui::PopID();
}
inline void vector(J& object, const char* key, const char* label, double scale = 1., bool integral = false) {
    if (!object.contains(key) || !object.at(key).is_array())
        return;
    ImGui::PushID(key);
    ImGui::TextWrapped("%s", label);
    for (std::size_t i = 0; i < object.at(key).size(); ++i) {
        ImGui::PushID(int(i));
        ImGui::SetNextItemWidth(-1);
        ImGui::TextUnformatted(i == 0 ? "X" : i == 1 ? "Y" : i == 2 ? "Z" : "W");
        if (integral) {
            int v = object[key][i].get<int>();
            if (ImGui::InputInt("##value", &v))
                object[key][i] = v;
        } else {
            double v = object[key][i].get<double>() * scale;
            if (ImGui::InputDouble("##value", &v, 0, 0, "%.8g") && std::isfinite(v))
                object[key][i] = v / scale;
        }
        ImGui::PopID();
    }
    ImGui::PopID();
}
// One record at a time, not hundreds of expandable JSON children. Selection is
// UI-only state; viewing an empty/missing section never modifies the document.
inline J* record(const char* label, J& records, const char* id_key = "id") {
    if (!records.is_array() || records.empty()) {
        ImGui::TextWrapped("%s: none", label);
        return nullptr;
    }
    const ImGuiID id = ImGui::GetID(label);
    auto* storage = ImGui::GetStateStorage();
    int selected = std::clamp(storage->GetInt(id, 0), 0, int(records.size()) - 1);
    auto name = [&](int i) { return records[i].value(id_key, std::string("Item ") + std::to_string(i + 1)); };
    ImGui::TextWrapped("%s (%d)", label, int(records.size()));
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo((std::string("##") + label).c_str(), name(selected).c_str())) {
        for (int i = 0; i < int(records.size()); ++i) {
            ImGui::PushID(i);
            if (ImGui::Selectable(name(i).c_str(), i == selected)) {
                selected = i;
                storage->SetInt(id, i);
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    return &records[selected];
}
} // namespace battery::ui::form

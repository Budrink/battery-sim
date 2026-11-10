#pragma once
#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"
#include "editor/help/parameter_catalog.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdarg>

namespace battery::ui::form {
using J = nlohmann::json;
inline void caption(const char* label, const char* key = nullptr) {
    ImGui::TextWrapped("%s", label);
    if (const auto* help = parameter_help(key ? key : label); help && ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 32);
        ImGui::TextUnformatted(help->explanation);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}
inline bool text_input(const char* label, std::string* value, const char* key = nullptr) {
    caption(label, key);
    ImGui::SetNextItemWidth(-1);
    return ImGui::InputText((std::string("##") + label).c_str(), value);
}
inline bool begin_combo(const char* label, const char* preview) {
    caption(label);
    ImGui::SetNextItemWidth(-1);
    return ImGui::BeginCombo((std::string("##") + label).c_str(), preview);
}
inline bool combo(const char* label, int* value, const char* choices) {
    caption(label);
    ImGui::SetNextItemWidth(-1);
    return ImGui::Combo((std::string("##") + label).c_str(), value, choices);
}
inline void hint(const char* format, ...) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    va_list args;
    va_start(args, format);
    ImGui::TextWrappedV(format, args);
    va_end(args);
    ImGui::PopStyleColor();
}
inline bool input(const char* label, double* value, const char* format = "%.8g", const char* key = nullptr) {
    caption(label, key);
    ImGui::SetNextItemWidth(-1);
    return ImGui::InputDouble((std::string("##") + label).c_str(), value, 0, 0, format);
}
inline bool number(J& object, const char* key, const char* label, double scale = 1., double offset = 0.) {
    if (!object.contains(key) || !object.at(key).is_number())
        return false;
    ImGui::PushID(key);
    caption(label, key);
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
    caption(label, key);
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
    caption(label, key);
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
    caption(label, key);
    for (std::size_t i = 0; i < object.at(key).size(); ++i) {
        ImGui::PushID(int(i));
        ImGui::TextUnformatted(i == 0 ? "X" : i == 1 ? "Y" : i == 2 ? "Z" : "W");
        ImGui::SetNextItemWidth(-1);
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

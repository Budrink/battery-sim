#pragma once
#include "battery_native/render/renderer.hpp"
#include "imgui.h"
namespace battery::ui {
constexpr auto fixed = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar |
                       ImGuiWindowFlags_NoSavedSettings;
ImVec4 color(CellMode);
void legend(float width, const VisualSettings&);
} // namespace battery::ui

#pragma once
#include "imgui.h"

namespace battery::ui {
struct ViewportInput {
    float dx = 0, dy = 0, wheel = 0;
    bool clicked = false;
};
// Image alone does not own the mouse: it also drags its parent window.
// Keep the gesture active until release, including outside the image.
inline ViewportInput orbit_viewport(ImTextureID texture, ImVec2 size) {
    const auto origin = ImGui::GetCursorScreenPos();
    const bool released = ImGui::InvisibleButton("orbit viewport", size);
    const bool hovered = ImGui::IsItemHovered();
    ImGui::GetWindowDrawList()->AddImage(texture, origin, {origin.x + size.x, origin.y + size.y}, {0, 1},
                                         {1, 0});
    ViewportInput input;
    const auto& io = ImGui::GetIO();
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        input.dx = io.MouseDelta.x;
        input.dy = io.MouseDelta.y;
    }
    if (hovered)
        input.wheel = io.MouseWheel;
    if (released) {
        const auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        input.clicked = d.x * d.x + d.y * d.y < 16;
    }
    return input;
}
} // namespace battery::ui

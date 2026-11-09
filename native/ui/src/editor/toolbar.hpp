#pragma once
#include "imgui.h"

namespace battery::ui {
// Caller starts a new line before/after a toolbar. Each item wraps instead of
// being clipped beyond the right edge of a narrow editor window.
inline bool toolbar_button(const char* label) {
    const float width = ImGui::CalcTextSize(label).x + 2 * ImGui::GetStyle().FramePadding.x;
    if (width > ImGui::GetContentRegionAvail().x && ImGui::GetCursorPosX() > ImGui::GetCursorStartPos().x + 1)
        ImGui::NewLine();
    const bool clicked = ImGui::Button(label);
    ImGui::SameLine();
    return clicked;
}
} // namespace battery::ui

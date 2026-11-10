#include "viewer/widgets.hpp"
namespace battery::ui {
ImVec4 color(battery::CellMode mode) {
    if (mode == battery::CellMode::Spent)
        return {battery::destroyed_rgb[0], battery::destroyed_rgb[1], battery::destroyed_rgb[2], 1};
    if (mode == battery::CellMode::Runaway)
        return {1, .40f, .12f, 1};
    return {.29f, .78f, .71f, 1};
}
void legend(float width, const battery::VisualSettings& visual) {
    auto* draw = ImGui::GetWindowDrawList();
    const auto p = ImGui::GetCursorScreenPos();
    for (int i = 0; i < 160; ++i) {
        auto a = battery::field_rgb(visual.field_layer, i / 159.f);
        draw->AddRectFilled({p.x + width * i / 160, p.y}, {p.x + width * (i + 1) / 160 + 1, p.y + 8},
                            ImGui::ColorConvertFloat4ToU32({a[0], a[1], a[2], 1}));
    }
    ImGui::Dummy({width, 10});
}
} // namespace battery::ui

#include "viewer/viewer_window.hpp"
#include "viewer/widgets.hpp"
#include <algorithm>
namespace battery::ui {
void ViewerWindow::draw_header() {
    const auto& scenario = session.scenario();
    const auto& frame = session.frame();
    const auto& batteries = session.batteries();
    const bool replay = session.recorded();
    auto& inspector = state.inspector;
    auto& io = ImGui::GetIO();
    const float top = layout.top;
    int reached = 0, spent = 0, holes = 0;
    double tmax = 0, gasmax = 298.15;
    for (int i : batteries) {
        const auto& c = frame.cells[i];
        reached += c.runaway_started_s >= 0;
        spent += c.mode == battery::CellMode::Spent;
        tmax = std::max(tmax, c.temperature_k);
    }
    for (std::size_t i = 0; i < frame.process.channels.size(); ++i)
        if (scenario.gas_channels[i].rupture) {
            const auto& c = frame.process.channels[i];
            holes += c.effective_area_m2 > 0;
        }
    for (const auto& g : frame.process.gas)
        gasmax = std::max(gasmax, g.temperature_k);
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({io.DisplaySize.x, top});
    ImGui::Begin("header", nullptr, fixed);
    ImGui::TextColored({.38f, .90f, .82f, 1}, "BATTERY / SIM");
    ImGui::SameLine(220);
    if (ImGui::Button("EXPERIMENT EDITOR"))
        editor.show();
    ImGui::SameLine();
    ImGui::Text("%zu CELLS", batteries.size());
    ImGui::SameLine(620);
    ImGui::TextColored({1, .67f, .30f, 1}, "R2 %d/%zu   |   SPENT %d   |   OPEN PATCHES %d", reached,
                       batteries.size(), spent, holes);
    ImGui::TextDisabled("%s  |  T max %.0f C  |  gas %.0f C",
                        replay ? "RECORDED CUDA / time-interpolated" : "LIVE CUDA / committed physical steps",
                        tmax - 273.15, gasmax - 273.15);
    if (replay && ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", session.source_label().c_str());
    ImGui::SameLine(io.DisplaySize.x - 210);
    if (ImGui::SmallButton(inspector ? "Hide controls [Tab]" : "Show controls [Tab]"))
        inspector = !inspector;
    ImGui::TextDisabled("%s: %s", replay ? "Recording" : "Running snapshot", scenario.name.c_str());
    ImGui::End();
}
} // namespace battery::ui

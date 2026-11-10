#include "viewer/viewer_window.hpp"
#include "viewer/widgets.hpp"
#include <algorithm>
namespace battery::ui {
void ViewerWindow::draw_transport() {
    const auto& scenario = session.scenario();
    const auto& frame = session.frame();
    const bool replay = session.recorded();
    const double time = session.time();
    const double duration = session.duration();
    auto& paused = session.playback.paused;
    auto& rate = session.playback.rate;
    auto& visual = state.visual;
    auto& io = ImGui::GetIO();
    const float bottom = layout.bottom;
    ImGui::SetNextWindowPos({0, io.DisplaySize.y - bottom});
    ImGui::SetNextWindowSize({io.DisplaySize.x, bottom});
    ImGui::Begin("transport", nullptr, fixed);
    if (ImGui::Button(paused ? "PLAY  [Space]" : "PAUSE [Space]", {145, 0}))
        paused = !paused;
    ImGui::SameLine();
    if (ImGui::Button("Start"))
        session.restart();
    ImGui::SameLine();
    if (replay) {
        if (ImGui::Button("Next rupture"))
            session.next_rupture();
        ImGui::SameLine();
        ImGui::SetNextItemWidth(170);
        ImGui::SliderFloat("Speed", &rate, .25f, 120, "%.1fx", ImGuiSliderFlags_Logarithmic);
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(210);
    int auto_pause = int(session.playback.auto_pause);
    ImGui::Combo("Auto pause", &auto_pause, "Never\0Each rupture\0Each spent cell\0All cells spent\0");
    session.playback.auto_pause = AutoPause(auto_pause);
    ImGui::SameLine();
    ImGui::Text("%8.1f / %.0f s", time, duration);
    float scrub = float(time);
    ImGui::SetNextItemWidth(io.DisplaySize.x - 40);
    ImGui::BeginDisabled(!replay);
    if (ImGui::SliderFloat("##time", &scrub, 0, float(duration), "%.1f s")) {
        session.seek(scrub);
    }
    ImGui::EndDisabled();
    // Tick marks are exact saved event timestamps, not threshold guesses.
    const auto lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    auto* draw = ImGui::GetWindowDrawList();
    const auto& events = session.events();
    for (const auto& e : events)
        if (e.kind == "runaway_complete" || e.kind == "shell_rupture") {
            const float x = lo.x + (hi.x - lo.x) * float(e.time_s / duration);
            draw->AddLine({x, hi.y + 1}, {x, hi.y + 5},
                          e.kind == "shell_rupture" ? IM_COL32(80, 210, 240, 200)
                                                    : IM_COL32(90, 130, 255, 200));
        }
    legend(240, visual);
    ImGui::SameLine();
    if (visual.field_layer == battery::FieldLayer::Temperature)
        ImGui::TextDisabled("%.0f - %.0f C / blue - yellow - red", visual.temperature_min_k - 273.15,
                            visual.temperature_max_k - 273.15);
    else
        ImGui::TextDisabled("0 - 100%% / shell damage; contents and gas remain thermal (%.0f - %.0f C)",
                            visual.temperature_min_k - 273.15, visual.temperature_max_k - 273.15);
    ImGui::SameLine();
    ImGui::TextDisabled("Shell hatching = damage / black = rupture");
    const auto known =
        frame.known_channels.empty()
            ? scenario.gas_channels.size()
            : std::size_t(std::count(frame.known_channels.begin(), frame.known_channels.end(), true));
    ImGui::TextDisabled(
        "Gas flows: %zu / %zu%s. Motion scaled for display; not CFD velocity. Physics unchanged.", known,
        scenario.gas_channels.size(), replay ? " recorded" : " live");
    if (replay && session.source_label().find("PARTIAL") != std::string::npos) {
        ImGui::SameLine();
        ImGui::TextColored({1, .65f, .25f, 1}, "PARTIAL RECORDING");
    }
    ImGui::End();
}
} // namespace battery::ui

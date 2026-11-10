#include "viewer/viewer_window.hpp"
#include "imgui.h"
#include <algorithm>
namespace battery::ui {
ViewerWindow::ViewerWindow(ViewerSession& session, ExperimentEditor& editor, bool verify_gl)
    : session(session), editor(editor), hidden(verify_gl) {
    auto& visual = state.visual;
    visual.temperature_min_k = 298.15f;
    visual.temperature_max_k = 1273.15f;
    visual.field_opacity = .82f;
    visual.shell_opacity = .035f;
    visual.show_interfaces = false;
    visual.gas_opacity = .24f;
    visual.battery_surface_opacity = .90f;
    auto& orbit = state.orbit;
    orbit.yaw = -.87f;
    orbit.pitch = .60f;
    orbit.distance = .98f;
    state.initial_camera = orbit;

    state.selected = session.batteries().front();
    if (!renderer.init())
        throw std::runtime_error("Renderer initialization failed");
}
ViewerWindow::~ViewerWindow() {
    renderer.destroy();
}
void ViewerWindow::new_snapshot() {
    renderer.destroy();
    if (!renderer.init())
        throw std::runtime_error("Cannot restart scene renderer");
    state.selected = session.batteries().front();
    scene_texture = 0;
    rendered_time = -1;
}
void ViewerWindow::draw(double wall_dt) {
    if (state.auto_orbit)
        state.orbit.yaw += float(wall_dt * .10);
    auto& io = ImGui::GetIO();
    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Space))
        session.playback.paused = !session.playback.paused;
    layout.side = state.inspector ? 310.f : 0.f;
    draw_header();
    if (ImGui::IsKeyPressed(ImGuiKey_Tab))
        state.inspector = !state.inspector;
    if (state.inspector)
        draw_inspector();
    draw_scene();
    draw_transport();
}
} // namespace battery::ui

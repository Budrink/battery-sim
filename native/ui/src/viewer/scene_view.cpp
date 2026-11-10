#include "viewer/viewer_window.hpp"
#include "viewer/widgets.hpp"
#include "shared/viewport_input.hpp"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cmath>
namespace battery::ui {
void ViewerWindow::draw_scene() {
    const auto& scenario = session.scenario();
    const auto& frame = session.frame();
    const auto& mesh = session.mesh();
    const auto& geometry = session.geometry();
    const bool replay = session.recorded();
    const double time = session.time();
    auto& selected = state.selected;
    auto& visual = state.visual;
    auto& orbit = state.orbit;
    const auto& initial_camera = state.initial_camera;
    auto& io = ImGui::GetIO();
    const float side = layout.side;
    const float top = layout.top;
    const float bottom = layout.bottom;
    ImGui::SetNextWindowPos({side, top});
    ImGui::SetNextWindowSize({io.DisplaySize.x - side, io.DisplaySize.y - top - bottom});
    ImGui::Begin("scene", nullptr, fixed | ImGuiWindowFlags_NoScrollbar);
    ImGui::TextDisabled("Drag to orbit / click object / Alt+click case / Space to play");
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset view"))
        orbit = initial_camera;
    const auto origin = ImGui::GetCursorScreenPos(), available = ImGui::GetContentRegionAvail();
    const int w = std::max(64, int(available.x)), h = std::max(64, int(available.y));
    if (!scene_texture || rendered_w != w || rendered_h != h || rendered_time != time ||
        rendered_selected != selected || rendered_camera != orbit || rendered_settings != visual) {
        scene_texture = renderer.render(w, h, scenario, mesh, frame, geometry, orbit, visual, selected);
        if (hidden && glGetError() != GL_NO_ERROR)
            throw std::runtime_error("OpenGL error in hidden renderer verification");
        rendered_w = w;
        rendered_h = h;
        rendered_time = time;
        rendered_selected = selected;
        rendered_camera = orbit;
        rendered_settings = visual;
    }
    const auto gesture = battery::ui::orbit_viewport(ImTextureID(scene_texture), {float(w), float(h)});
    orbit.yaw -= gesture.dx * .008f;
    orbit.pitch = std::clamp(orbit.pitch + gesture.dy * .008f, -1.4f, 1.4f);
    orbit.distance = std::clamp(orbit.distance * std::exp(-gesture.wheel * .1f), .55f, 6.f);
    if (gesture.clicked) {
        const int hit = renderer.pick_cell(io.MousePos.x - origin.x, io.MousePos.y - origin.y, w, h, scenario,
                                           orbit, &geometry, &visual, io.KeyAlt);
        if (hit >= 0) {
            selected = hit;
            editor.select_object(scenario.cells[hit].id);
        }
    }
    auto* overlay = ImGui::GetWindowDrawList();
    overlay->AddText({origin.x + 18, origin.y + 18}, IM_COL32(210, 224, 235, 255),
                     replay ? "RECORDED FIELDS  /  time-interpolated" : "LIVE FIELDS  /  CUDA");
    overlay->AddText({origin.x + 18, origin.y + 42}, IM_COL32(133, 157, 178, 255),
                     "Volumetric gas / colour: temperature / black spots: shell rupture");
    if (!visual.show_far_gas)
        overlay->AddText({origin.x + 18, origin.y + 66}, IM_COL32(133, 157, 178, 255),
                         "Gas view: inside case only (outside gas hidden, physics unchanged)");
    if (visual.show_gas && geometry.unresolved_gas_regions > 0) {
        const auto warning = "Gas geometry missing: " + std::to_string(geometry.unresolved_gas_regions) +
                             " volumes cannot be drawn";
        overlay->AddText({origin.x + 18, origin.y + 90}, IM_COL32(255, 145, 70, 255), warning.c_str());
    }
    ImGui::End();
}
} // namespace battery::ui

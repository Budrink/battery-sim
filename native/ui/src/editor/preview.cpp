#include "editor_internal.hpp"
#include "battery_native/scenario_document.hpp"
#include "imgui.h"
#include "viewport_input.hpp"
#include <algorithm>
#include <cmath>

namespace battery::ui {
void ExperimentEditor::Impl::refresh_preview() {
    if (preview && appearance_only_change(preview_source, document.draft)) {
        auto compiled = preview_compiled;
        for (const auto* key : {"visual_objects", "visual_materials"})
            compiled["metadata"][key] = document.draft.value("metadata", nlohmann::json::object())
                                            .value(key, nlohmann::json::object());
        // Display edits do not compile the gas grid or allocate a new solver.
        const auto next = visual_geometry_from_json(compiled, preview_scenario);
        preview_geometry.solid_materials = next.solid_materials;
        preview_compiled = std::move(compiled);
        preview_source = document.draft;
        preview_updates.completed(document.draft);
        preview_error.clear();
        preview_texture = 0;
        return;
    }
    auto compiled = document.compile();
    auto next = scenario_from_json(compiled);
    auto geometry = visual_geometry_from_json(compiled, next);
    auto sim = std::make_unique<Simulation>(next);
    auto mesh = sim->thermal_mesh();
    auto frame = capture_visual_frame(*sim);
    // Renderer caches topology by address: invalidate it even when the voxel count is unchanged.
    if (renderer_ready)
        preview_renderer.destroy();
    renderer_ready = preview_renderer.init();
    if (!renderer_ready)
        throw std::runtime_error("Cannot initialize preview renderer");
    preview_scenario = std::move(next);
    preview_geometry = std::move(geometry);
    preview = std::move(sim);
    preview_mesh = std::move(mesh);
    preview_frame = std::move(frame);
    preview_source = document.draft;
    preview_compiled = std::move(compiled);
    preview_updates.completed(document.draft);
    preview_error.clear();
    preview_texture = 0;
}
void ExperimentEditor::Impl::update_preview() {
    const auto now = ImGui::GetTime();
    const bool changed = preview_updates.observe(document.draft, now);
    const bool appearance = changed && preview && appearance_only_change(preview_source, document.draft);
    if (!appearance && !preview_updates.ready(now))
        return;
    preview_updates.attempted();
    try {
        refresh_preview();
    } catch (const std::exception& e) {
        // Do not discard a usable view or repeatedly rebuild the same invalid
        // intermediate input. The next edit (including Undo) retries it.
        preview_error = e.what();
    }
}
void ExperimentEditor::Impl::draw_preview() {
    if (!preview_error.empty()) {
        ImGui::TextColored({1, .45f, .3f, 1}, "Preview cannot apply this draft:");
        ImGui::TextWrapped("%s", preview_error.c_str());
        if (preview)
            ImGui::TextDisabled("Showing the last valid state; correct the input or Undo.");
    }
    if (!preview) {
        ImGui::TextWrapped("Preview builds automatically from a valid draft. No physical time is advanced.");
        return;
    }
    const bool stale = preview_source != document.draft;
    ImGui::TextColored(stale ? ImVec4{1, .65f, .2f, 1} : ImVec4{.3f, .9f, .7f, 1}, "%s",
                       stale ? "DRAFT NOT YET APPLIED" : "LIVE PREVIEW / INITIAL STATE / NOT RUNNING");
    ImGui::TextDisabled("Drag to orbit / click object / Alt+click case");
    ImGui::TextColored(place_case_opening ? ImVec4{.3f, .9f, 1.f, 1.f} : ImVec4{.5f,.6f,.7f,1.f},
                       "%s", place_case_opening ? "OPENING TOOL: click wall to place / drag to orbit"
                                                 : "Select an object / drag to orbit");
    ImGui::TextDisabled("Gas network: %zu volumes, %zu channels", preview_scenario.gas_compartments.size(),
                        preview_scenario.gas_channels.size());
    ImGui::Checkbox("Ports", &visual.show_ports);
    ImGui::SameLine();
    ImGui::Checkbox("Contacts", &visual.show_interfaces);
    ImGui::SameLine();
    ImGui::Checkbox("Pads", &visual.show_pads);
    ImGui::SameLine();
    ImGui::Checkbox("Show heater", &visual.show_heaters);
    ImGui::TextWrapped("Yellow = heater enabled; black = cutoff; cyan rings = valves. Show heater changes visibility only.");
    ImGui::TextDisabled("Pads: brown insulation / white conductor; temperature overlay");
    const auto origin = ImGui::GetCursorScreenPos(), size = ImGui::GetContentRegionAvail();
    const int w = std::max(64, int(size.x)), h = std::max(64, int(size.y));
    int index = -1;
    for (std::size_t i = 0; i < preview_scenario.cells.size(); ++i)
        if (preview_scenario.cells[i].id == selected)
            index = int(i);
    auto preview_settings = visual;
    if (place_case_opening) {
        preview_settings.show_enclosure = true;
        preview_settings.enclosure_opacity = std::max(.2f, visual.enclosure_opacity);
    }
    if (!preview_texture || preview_width != w || preview_height != h || preview_selected != index ||
        rendered_camera != camera || rendered_visual != preview_settings) {
        preview_texture = preview_renderer.render(w, h, preview_scenario, preview_mesh, preview_frame,
                                                  preview_geometry, camera, preview_settings, index);
        preview_width = w;
        preview_height = h;
        preview_selected = index;
        rendered_camera = camera;
        rendered_visual = preview_settings;
    }
    const auto input = orbit_viewport(ImTextureID(preview_texture), {float(w), float(h)});
    camera.yaw -= input.dx * .008f;
    camera.pitch = std::clamp(camera.pitch + input.dy * .008f, -1.4f, 1.4f);
    camera.distance = std::clamp(camera.distance * std::exp(-input.wheel * .1f), .55f, 6.f);
    if (input.clicked) {
        const auto& io = ImGui::GetIO();
        if (place_case_opening) {
            Vec3 point;
            const int wall =
                preview_renderer.pick_case_surface(io.MousePos.x - origin.x, io.MousePos.y - origin.y, w, h,
                                                   preview_scenario, camera, preview_geometry, point);
            if (wall >= 0 && !stale)
                place_opening_at(wall, point);
            else {
                case_opening_error =
                    stale ? "Wait for preview to update before placing an opening." : "Click a case wall.";
                message = case_opening_error;
            }
            return;
        }
        const auto hit =
            preview_renderer.pick_cell(io.MousePos.x - origin.x, io.MousePos.y - origin.y, w, h,
                                       preview_scenario, camera, &preview_geometry, &visual, io.KeyAlt);
        if (hit >= 0) {
            const auto& c = preview_scenario.cells[hit];
            selected = c.shell_contents_id.empty() ? c.id : c.shell_contents_id;
            case_opening_selection.id.clear();
        }
    }
}
} // namespace battery::ui

#include "battery_native/model.hpp"
#include "battery_native/playback.hpp"
#include "battery_native/renderer.hpp"
#include "battery_native/simulation.hpp"

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

void apply_dark_style() {
    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.WindowRounding = 7.0f;
    style.ChildRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 5.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = {8.0f, 6.0f};
    style.WindowPadding = {11.0f, 10.0f};
    auto& colors = style.Colors;
    colors[ImGuiCol_WindowBg] = {0.018f,0.022f,0.028f,1.0f};
    colors[ImGuiCol_ChildBg] = {0.022f,0.027f,0.034f,1.0f};
    colors[ImGuiCol_TitleBg] = {0.028f,0.034f,0.042f,1.0f};
    colors[ImGuiCol_TitleBgActive] = {0.040f,0.048f,0.058f,1.0f};
    colors[ImGuiCol_FrameBg] = {0.055f,0.065f,0.077f,1.0f};
    colors[ImGuiCol_FrameBgHovered] = {0.085f,0.10f,0.115f,1.0f};
    colors[ImGuiCol_FrameBgActive] = {0.11f,0.12f,0.14f,1.0f};
    colors[ImGuiCol_Button] = {0.065f,0.076f,0.09f,1.0f};
    colors[ImGuiCol_ButtonHovered] = {0.10f,0.13f,0.15f,1.0f};
    colors[ImGuiCol_ButtonActive] = {0.15f,0.18f,0.20f,1.0f};
    colors[ImGuiCol_Header] = {0.075f,0.10f,0.115f,1.0f};
    colors[ImGuiCol_HeaderHovered] = {0.11f,0.15f,0.17f,1.0f};
    colors[ImGuiCol_HeaderActive] = {0.15f,0.19f,0.21f,1.0f};
    colors[ImGuiCol_CheckMark] = {0.22f,0.83f,0.90f,1.0f};
    colors[ImGuiCol_SliderGrab] = {0.25f,0.78f,0.84f,1.0f};
    colors[ImGuiCol_SliderGrabActive] = {0.98f,0.55f,0.14f,1.0f};
    colors[ImGuiCol_Border] = {0.12f,0.15f,0.18f,1.0f};

    auto& plot = ImPlot::GetStyle();
    plot.Colors[ImPlotCol_PlotBg] = {0.012f,0.015f,0.019f,1.0f};
    plot.Colors[ImPlotCol_FrameBg] = {0.018f,0.022f,0.028f,1.0f};
    plot.Colors[ImPlotCol_PlotBorder] = {0.12f,0.15f,0.18f,1.0f};
    plot.Colors[ImPlotCol_LegendBg] = {0.025f,0.030f,0.036f,0.94f};
}

std::string find_scenario(int argc, char** argv) {
    if (argc > 1) return argv[1];
    for (const char* candidate : {
             "examples/reaction-chain-demo.json",
             "../examples/reaction-chain-demo.json",
             "../../examples/reaction-chain-demo.json"}) {
        std::ifstream stream(candidate);
        if (stream.good()) return candidate;
    }
    return "examples/reaction-chain-demo.json";
}

ImVec4 mode_color(battery::CellMode mode) {
    switch (mode) {
        case battery::CellMode::Healthy: return {0.25f,0.88f,0.82f,1.0f};
        case battery::CellMode::Runaway: return {1.0f,0.25f,0.06f,1.0f};
        case battery::CellMode::Quenched: return {0.9f,0.65f,0.2f,1.0f};
        case battery::CellMode::Spent: return {
            battery::destroyed_rgb[0], battery::destroyed_rgb[1], battery::destroyed_rgb[2], 1.0f};
        case battery::CellMode::PartiallySpent: return {0.65f,0.45f,0.9f,1.0f};
    }
    return {1,1,1,1};
}

void field_legend(float width, battery::FieldLayer layer) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 start = ImGui::GetCursorScreenPos();
    constexpr int segments = 256;
    for (int index = 0; index < segments; ++index) {
        const auto a = battery::field_rgb(layer, static_cast<float>(index) / segments);
        const auto b = battery::field_rgb(layer, static_cast<float>(index + 1) / segments);
        const ImU32 left = ImGui::ColorConvertFloat4ToU32({a[0],a[1],a[2],1});
        const ImU32 right = ImGui::ColorConvertFloat4ToU32({b[0],b[1],b[2],1});
        const float x0 = start.x + width * index / segments;
        const float x1 = start.x + width * (index + 1) / segments;
        draw->AddRectFilledMultiColor({x0,start.y}, {x1,start.y+10.0f},
                                     left, right, right, left);
    }
    draw->AddRect(start, {start.x + width,start.y + 10.0f}, IM_COL32(85,85,85,255));
    ImGui::Dummy({width, 11.0f});
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string scenario_path = find_scenario(argc, argv);
        battery::Scenario scenario = battery::load_scenario(scenario_path);
        for (int arg=2;arg<argc;) {
            if (std::string(argv[arg])=="--isothermal") {
                for (auto& cell:scenario.cells) if(cell.is_battery)cell.isothermal=true;
                ++arg;
                continue;
            }
            if (std::string(argv[arg])!="--mesh-scale" || arg+1>=argc)
                throw std::runtime_error("usage: battery_sim_viewer SCENARIO [--mesh-scale N] [--isothermal]");
            const int scale=std::stoi(argv[arg+1]);
            if (scale<1 || scale>8) throw std::runtime_error("mesh-scale must be 1..8");
            battery::refine_environment_mesh(scenario,scale);
            arg+=2;
        }
        battery::Simulation simulation(scenario);

        glfwSetErrorCallback([](int code, const char* description) {
            std::fprintf(stderr, "GLFW error %d: %s\n", code, description);
        });
        if (!glfwInit()) {
            std::fprintf(stderr, "GLFW initialization failed\n");
            return 1;
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        if (std::getenv("BATTERY_SIM_EGL") != nullptr) {
            glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
        }
        GLFWwindow* window = glfwCreateWindow(1920, 1080,
            "battery-sim · Thermal / CET-L03", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            std::fprintf(stderr, "OpenGL window creation failed\n");
            return 1;
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        apply_dark_style();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
        ImGui::GetIO().FontGlobalScale = 1.05f;

        battery::SceneRenderer renderer;
        if (!renderer.init()) {
            std::fprintf(stderr, "3D renderer initialization failed\n");
            return 1;
        }

        bool paused = true;
        battery::AutoPauseMode auto_pause_mode = battery::AutoPauseMode::AllCellsDestroyed;
        battery::PauseReason pause_reason = battery::PauseReason::Manual;
        const bool large_module = scenario.cells.size() > 8;
        int steps_per_frame = large_module ? 16 : 2;
        int selected_cell = scenario.primary_failures.empty() ? 0 :
            static_cast<int>(scenario.primary_failures.front().cell_index);
        bool focus_neighborhood = large_module;
        battery::OrbitCamera orbit;
        battery::VisualSettings visual;
        visual.temperature_min_k = static_cast<float>(simulation.max_temperature_k());
        visual.temperature_max_k = 700.0f;
        if (large_module) {
            orbit.pitch = 0.92f;
            orbit.distance = 1.35f;
            visual.field_opacity = 0.65f;
            visual.temperature_max_k = 1050.0f;
        }
        const battery::OrbitCamera initial_camera = orbit;
        float max_dt = static_cast<float>(scenario.settings.max_dt_s);
        float conductance_scale = 1.0f;
        float ambient_scale = 1.0f;
        float damage_scale = 1.0f;
        float runaway_scale = 1.0f;
        std::vector<int> probed_cells;
        bool vsync = true;
        double solver_ms = 0;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            simulation.runtime().max_dt_s = max_dt;
            simulation.runtime().interface_conductance_scale = conductance_scale;
            simulation.runtime().ambient_conductance_scale = ambient_scale;
            simulation.runtime().damage_rate_scale = damage_scale;
            simulation.runtime().runaway_power_scale = runaway_scale;

            const auto solver_start = std::chrono::steady_clock::now();
            const double time_before_frame = simulation.time_s();
            if (!paused) {
                pause_reason = battery::advance_playback(simulation, steps_per_frame, auto_pause_mode);
                paused = pause_reason != battery::PauseReason::None;
            }
            solver_ms = std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-solver_start).count();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGuiIO& io = ImGui::GetIO();

            const float status_height = 42.0f;
            const float left_width = std::clamp(io.DisplaySize.x * 0.205f, 330.0f, 405.0f);
            const float right_width = std::clamp(io.DisplaySize.x * 0.225f, 360.0f, 450.0f);
            const float gap = 7.0f;

            int active_runaway = 0;
            int destroyed_cells = 0;
            for (std::size_t i=0;i<simulation.cells().size();++i) {
                if(!scenario.cells[i].is_battery)continue;
                const auto& cell=simulation.cells()[i];
                if (cell.mode == battery::CellMode::Runaway) ++active_runaway;
                if (cell.mode == battery::CellMode::Spent) ++destroyed_cells;
            }

            ImGui::SetNextWindowPos({0,0});
            ImGui::SetNextWindowSize({io.DisplaySize.x,status_height});
            ImGui::Begin("##status", nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar);
            ImGui::TextColored({0.30f,0.86f,0.92f,1}, "BATTERY-SIM");
            ImGui::SameLine();
            ImGui::Text("| %s | t = %.3f / %.1f s | step = %zu | Tmax = %.1f K",
                scenario.name.c_str(), simulation.time_s(), scenario.settings.duration_s,
                simulation.step_count(), simulation.max_temperature_k());
            ImGui::SameLine(ImGui::GetWindowWidth() - 255.0f);
            if (paused) {
                ImGui::TextColored({0.80f,0.85f,0.90f,1}, "[%s]",
                    simulation.finished() ? "FINISHED" : "PAUSED");
            } else if (active_runaway > 0) {
                ImGui::TextColored({1.0f,0.24f,0.05f,1}, "[RUNAWAY: %d ACTIVE]", active_runaway);
            } else if (simulation.propagated()) {
                ImGui::TextColored({1.0f,0.65f,0.15f,1}, "[CASCADE / SPENT]");
            } else {
                ImGui::TextColored({0.32f,0.88f,0.72f,1}, "[%s]", paused ? "PAUSED" : "RUNNING");
            }
            ImGui::End();

            ImGui::SetNextWindowPos({0,status_height});
            ImGui::SetNextWindowSize({left_width,io.DisplaySize.y-status_height});
            ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
            ImGui::Text("%zu batteries / %zu passive solids / %zu contacts", simulation.battery_count(), scenario.cells.size()-simulation.battery_count(), scenario.interfaces.size());
            ImGui::Text("CUDA: %s", simulation.device_name().c_str());
            const auto isothermal_count=std::count_if(scenario.cells.begin(),scenario.cells.end(),
                [](const auto& cell){return cell.isothermal;});
            ImGui::Text("Isothermal objects: %zu / %zu", static_cast<std::size_t>(isothermal_count),scenario.cells.size());
            ImGui::Text("%zu thermal volumes / %zu local links", simulation.thermal_mesh().volumes.size(), simulation.thermal_mesh().links.size());
            ImGui::Text("Actual dt: %.6g s", simulation.last_dt_s());
            ImGui::Text("Outer dt: %.4g s | %zu substeps", simulation.last_interval_s(),simulation.last_substep_count());
            ImGui::Text("Outer steps: %zu | substeps: %zu", simulation.outer_step_count(),simulation.step_count());
            ImGui::Text("%.1f FPS | solver %.3f ms/frame", io.Framerate, solver_ms);
            ImGui::Text("Simulation speed: %.1fx realtime", (simulation.time_s()-time_before_frame)/std::max(static_cast<double>(io.DeltaTime),1e-9));
            const auto& selected_macro = simulation.cells()[static_cast<std::size_t>(selected_cell)];
            ImGui::Text("Selected Tmin/Tmax: %.1f / %.1f K", selected_macro.min_temperature_k, selected_macro.max_temperature_k);
            ImGui::Text("Destroyed volume: %.1f%%", 100.0 * selected_macro.spent_fraction);
            ImGui::TextColored({1.0f,0.72f,0.25f,1.0f}, "%s", scenario.calibration_status.c_str());
            if (!scenario.geometry_status.empty()) ImGui::TextWrapped("%s", scenario.geometry_status.c_str());
            ImGui::Text("Runaway reached: %zu / %zu", simulation.runaway_order().size(), simulation.battery_count());
            ImGui::Text("Destroyed batteries: %d / %zu", destroyed_cells, simulation.battery_count());
            ImGui::BeginDisabled(simulation.finished());
            if (ImGui::Button(paused ? "Resume" : "Pause", {88,0})) {
                paused = !paused;
                pause_reason = paused ? battery::PauseReason::Manual : battery::PauseReason::None;
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            if (ImGui::Button("Reset", {72,0})) {
                simulation.reset();
                paused = false;
                pause_reason = battery::PauseReason::None;
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(simulation.finished());
            if (ImGui::Button("Step", {65,0})) {
                pause_reason = battery::advance_playback(simulation, 1, auto_pause_mode);
                if (pause_reason == battery::PauseReason::None)
                    pause_reason = battery::PauseReason::SingleStep;
                paused = true;
            }
            ImGui::EndDisabled();
            ImGui::SliderInt("Outer steps / frame", &steps_per_frame, 1, 256);
            if (ImGui::Checkbox("VSync", &vsync)) glfwSwapInterval(vsync ? 1 : 0);
            ImGui::TextUnformatted("Automatic pause");
            if (ImGui::RadioButton("All batteries destroyed",
                    auto_pause_mode == battery::AutoPauseMode::AllCellsDestroyed))
                auto_pause_mode = battery::AutoPauseMode::AllCellsDestroyed;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "Run continuously until every battery is spent, or the time limit.\n"
                "Passive pads are not counted as batteries.\n"
                "Surviving cells and quiet gaps do not imply the cascade is over.\n"
                "Resume after the last destruction to inspect cooldown.");
            ImGui::SameLine();
            if (ImGui::RadioButton("Each destruction",
                    auto_pause_mode == battery::AutoPauseMode::EachDestruction))
                auto_pause_mode = battery::AutoPauseMode::EachDestruction;
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                "Pause on battery, material-volume or interface destruction.\n"
                "Blue state, not the start of runaway. Simultaneous events pause together.");
            ImGui::TextDisabled("%s", battery::pause_description(pause_reason));
            ImGui::Checkbox("Focus plots / edges on selected cell", &focus_neighborhood);
            std::vector<bool> visible_cells(scenario.cells.size(), !focus_neighborhood);
            visible_cells[static_cast<std::size_t>(selected_cell)] = true;
            for (const auto& edge : scenario.interfaces) {
                if (edge.cell_a == static_cast<std::size_t>(selected_cell) ||
                    edge.cell_b == static_cast<std::size_t>(selected_cell)) {
                    visible_cells[edge.cell_a] = visible_cells[edge.cell_b] = true;
                }
            }

            // Include batteries across a directly adjacent pad in neighborhood plots.
            const auto direct_neighbors=visible_cells;
            for(const auto& edge:scenario.interfaces) {
                if((direct_neighbors[edge.cell_a] && !scenario.cells[edge.cell_a].is_battery) ||
                   (direct_neighbors[edge.cell_b] && !scenario.cells[edge.cell_b].is_battery))
                    visible_cells[edge.cell_a]=visible_cells[edge.cell_b]=true;
            }
            if (ImGui::CollapsingHeader("Solver", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Max substep [s]", &max_dt, 0.001f, 0.25f, "%.4f",
                                   ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Interface G scale", &conductance_scale, 0.05f, 5.0f, "%.3f",
                                   ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Ambient G scale", &ambient_scale, 0.05f, 5.0f, "%.3f",
                                   ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Damage rate scale", &damage_scale, 0.05f, 5.0f, "%.3f",
                                   ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat("Runaway power scale", &runaway_scale, 0.25f, 2.5f, "%.3f");
                ImGui::TextDisabled("Multipliers apply to the next live solver step.");
            }

            if (ImGui::CollapsingHeader("Cells", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::BeginTable("cells", 4,
                    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                    ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY,
                    {0, large_module ? 190.0f : 105.0f})) {
                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableSetupColumn("Cell");
                    ImGui::TableSetupColumn("T [K]");
                    ImGui::TableSetupColumn("D");
                    ImGui::TableSetupColumn("State");
                    ImGui::TableHeadersRow();
                    for (std::size_t index = 0; index < scenario.cells.size(); ++index) {
                        const auto& state = simulation.cells()[index];
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        const bool selected = selected_cell == static_cast<int>(index);
                        if (ImGui::Selectable(scenario.cells[index].id.c_str(), selected,
                                              ImGuiSelectableFlags_SpanAllColumns)) {
                            selected_cell = static_cast<int>(index);
                        }
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%.1f", state.temperature_k);
                        ImGui::TableSetColumnIndex(2);
                        ImGui::Text("%.3f", state.damage);
                        ImGui::TableSetColumnIndex(3);
                        ImGui::TextColored(mode_color(state.mode), "%s", battery::name(state.mode));
                    }
                    ImGui::EndTable();
                }
            }

            if (ImGui::CollapsingHeader("Interfaces", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::BeginChild("edge-list", {0, large_module ? 150.0f : 100.0f});
                for (std::size_t index = 0; index < scenario.interfaces.size(); ++index) {
                    const auto& config = scenario.interfaces[index];
                    if (focus_neighborhood && config.cell_a != static_cast<std::size_t>(selected_cell) &&
                        config.cell_b != static_cast<std::size_t>(selected_cell)) continue;
                    const auto& state = simulation.interfaces()[index];
                    const double ratio = state.l_m / std::max(state.l0_m, 1.0e-12);
                    ImGui::Text("%s", config.id.c_str());
                    if (!large_module) ImGui::SameLine(128.0f);
                    ImGui::TextColored(state.phase == battery::EdgePhase::Destroyed
                        ? mode_color(battery::CellMode::Spent)
                        : ImVec4{0.95f,0.78f,0.28f,1.0f},
                        "l/l0 %.3f  G %.3f", ratio, state.conductance_w_per_k);
                    ImGui::ProgressBar(static_cast<float>(std::min(ratio / 1.5, 1.0)), {-1,3}, "");
                }
                ImGui::EndChild();
            }

            if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Volume cutaway X", &visual.cutaway_fraction, 0.05f, 1.0f, "%.2f");
                ImGui::Text("Field layer");
                int layer = static_cast<int>(visual.field_layer);
                if (ImGui::RadioButton("Temperature", layer == 0)) {
                    visual.field_layer = battery::FieldLayer::Temperature;
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Degradation", layer == 1)) {
                    visual.field_layer = battery::FieldLayer::Degradation;
                }
                ImGui::SliderFloat("Temperature min [K]", &visual.temperature_min_k, 250.0f, 500.0f, "%.1f");
                ImGui::SliderFloat("Temperature max [K]", &visual.temperature_max_k, 400.0f, 1200.0f, "%.1f");
                ImGui::SliderFloat("Field opacity", &visual.field_opacity, 0.05f, 1.0f, "%.2f");
                ImGui::SliderFloat("Shell opacity", &visual.shell_opacity, 0.0f, 0.6f, "%.2f");
                ImGui::Checkbox("Show interfaces", &visual.show_interfaces);
                ImGui::Checkbox("Show ports", &visual.show_ports);
                field_legend(ImGui::GetContentRegionAvail().x, visual.field_layer);
                if (visual.field_layer == battery::FieldLayer::Temperature) {
                    ImGui::TextDisabled("Thermal: black -> red -> yellow -> white");
                    ImGui::TextColored(mode_color(battery::CellMode::Spent),
                                       "cobalt blue outline = destroyed");
                    ImGui::TextDisabled("CET-L03 / Peter Kovesi / CC BY 4.0");
                } else {
                    ImGui::TextDisabled("healthy: green  ->  critical: red");
                    ImGui::TextColored(mode_color(battery::CellMode::Spent),
                                       "cobalt blue = destroyed");
                }
            }
            ImGui::End();

            const float center_x = left_width + gap;
            const float center_width = std::max(io.DisplaySize.x - left_width - right_width - gap*2.0f, 220.0f);
            ImGui::SetNextWindowPos({center_x,status_height});
            ImGui::SetNextWindowSize({center_width,io.DisplaySize.y-status_height});
            ImGui::Begin("Realtime 3D module", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
            ImGui::TextDisabled("Click: select cell | drag: orbit | wheel: zoom | double-click: reset");
            int layer = static_cast<int>(visual.field_layer);
            if (ImGui::RadioButton("Temperature##scene", layer == 0)) visual.field_layer = battery::FieldLayer::Temperature;
            ImGui::SameLine();
            if (ImGui::RadioButton("Degradation##scene", layer == 1)) visual.field_layer = battery::FieldLayer::Degradation;
            ImGui::SameLine();
            ImGui::TextColored(mode_color(battery::CellMode::Spent), "| blue = destroyed");
            field_legend(std::min(240.0f, ImGui::GetContentRegionAvail().x * 0.35f), visual.field_layer);
            ImGui::SameLine();
            if (visual.field_layer == battery::FieldLayer::Temperature) {
                ImGui::TextDisabled("%.0f - %.0f K | CET-L03", visual.temperature_min_k,
                                    visual.temperature_max_k);
            } else {
                ImGui::TextDisabled("0 - 100%% degradation");
            }
            ImVec2 available = ImGui::GetContentRegionAvail();
            const ImVec2 image_origin = ImGui::GetCursorScreenPos();
            const int render_width = std::max(static_cast<int>(available.x), 64);
            const int render_height = std::max(static_cast<int>(available.y), 64);
            const unsigned int texture = renderer.render(render_width, render_height,
                scenario, simulation, orbit, visual, selected_cell, probed_cells);
            ImGui::Image(static_cast<ImTextureID>(texture),
                         {static_cast<float>(render_width), static_cast<float>(render_height)},
                         {0,1}, {1,0});
            if (ImGui::IsItemHovered()) {
                const int hovered_cell = renderer.pick_cell(io.MousePos.x-image_origin.x,
                    io.MousePos.y-image_origin.y, render_width, render_height, scenario, orbit);
                if (hovered_cell >= 0 && !ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    const auto& cell = simulation.cells()[static_cast<std::size_t>(hovered_cell)];
                    ImGui::SetTooltip("%s | %.1f K | D %.3f | %s",
                        scenario.cells[static_cast<std::size_t>(hovered_cell)].id.c_str(),
                        cell.temperature_k, cell.damage, battery::name(cell.mode));
                }
                const ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
                if (hovered_cell >= 0 && ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
                    drag.x*drag.x + drag.y*drag.y < 16.0f) selected_cell = hovered_cell;
                if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                    orbit.yaw -= io.MouseDelta.x * 0.009f;
                    orbit.pitch = std::clamp(orbit.pitch + io.MouseDelta.y * 0.009f,
                                             -1.45f, 1.45f);
                }
                if (io.MouseWheel != 0.0f) {
                    orbit.distance = std::clamp(orbit.distance * (1.0f - io.MouseWheel*0.09f),
                                                1.0f, 8.0f);
                }
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) orbit = initial_camera;
            }
            ImGui::End();

            const float right_x = center_x + center_width + gap;
            ImGui::SetNextWindowPos({right_x,status_height});
            ImGui::SetNextWindowSize({right_width,io.DisplaySize.y-status_height});
            ImGui::Begin("Diagnostics", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
            selected_cell = std::clamp(selected_cell, 0, static_cast<int>(scenario.cells.size())-1);
            const auto& selected_config = scenario.cells[static_cast<std::size_t>(selected_cell)];
            const auto& selected_state = simulation.cells()[static_cast<std::size_t>(selected_cell)];
            ImGui::TextDisabled("SELECTED CELL");
            ImGui::Text("%s", selected_config.id.c_str());
            ImGui::TextDisabled("%s | %s",selected_config.is_battery?"Battery":"Passive material",selected_config.material_id.c_str());
            ImGui::SameLine(ImGui::GetWindowWidth()-105.0f);
            ImGui::TextColored(mode_color(selected_state.mode), "%s", battery::name(selected_state.mode));
            const auto& divisions=selected_config.thermal_divisions;
            const int core_count=std::max(0,divisions[0]-2)*std::max(0,divisions[1]-2)*std::max(0,divisions[2]-2);
            const int volume_count=divisions[0]*divisions[1]*divisions[2];
            ImGui::TextDisabled("Grid %dx%dx%d | interior %d | surface %d",divisions[0],divisions[1],divisions[2],core_count,volume_count-core_count);
            if(!selected_config.shell_contents_id.empty())ImGui::TextDisabled("Physical casing of %s",selected_config.shell_contents_id.c_str());
            ImGui::SetWindowFontScale(1.35f);
            ImGui::Text("%.1f K", selected_state.temperature_k);
            ImGui::SetWindowFontScale(1.0f);
            ImGui::TextDisabled("%.1f C | damage %.2f%% | released %.1f kJ",
                selected_state.temperature_k - 273.15, selected_state.damage*100.0,
                selected_state.released_energy_j/1000.0);
            if(selected_config.runaway_pulse.profile==battery::RunawayProfile::Kinetic) {
                ImGui::Text("Chemical reserve %.2f kJ | power %.1f W",selected_state.chemical_energy_remaining_j/1000.,selected_state.chemical_power_w);
                ImGui::TextDisabled("Discharged heat %.2f kJ | unreacted %.2f kJ",selected_state.vent_heat_out_j/1000.,selected_state.ejected_chemical_energy_j/1000.);
            }
            ImGui::Separator();

            if(!scenario.gas_compartments.empty() && ImGui::CollapsingHeader("Gas / physical casing",ImGuiTreeNodeFlags_DefaultOpen)) {
                const auto p=simulation.process_diagnostics();
                for(std::size_t i=0;i<p.gas.size();++i)ImGui::Text("%s: %.2f kPa | %.1f K | %.3f g",scenario.gas_compartments[i].id.c_str(),p.gas[i].pressure_pa/1000.,p.gas[i].temperature_k,p.gas[i].mass_kg*1000.);
                for(std::size_t i=0;i<p.channels.size();++i)ImGui::Text("%s: %s | %.1f%% | %.3f mm2 | %.3f g/s",scenario.gas_channels[i].id.c_str(),p.channels[i].open?"OPEN":"closed",100*p.channels[i].opening_fraction,1e6*p.channels[i].effective_area_m2,p.channels[i].mass_flow_kg_s*1000.);
                ImGui::TextDisabled("Tracked mass residual %.3g kg",p.mass_residual_kg);
                ImGui::TextDisabled("Outside: %.3f g | heat %.2f J | chemical %.2f J",p.escaped_mass_kg*1000.,p.escaped_energy_j,p.escaped_chemical_j);
                ImGui::TextDisabled("Gas chemical reserve is transported, not burned.");
            }

            const auto& history = simulation.history();
            if (!history.time_s.empty() && ImPlot::BeginPlot("Cell temperature", {-1,205})) {
                ImPlot::SetupAxes("t [s]", "T [K]", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                const ImVec4 palette[] = {
                    {1.0f,0.26f,0.06f,1}, {1.0f,0.70f,0.10f,1},
                    {1.0f,0.94f,0.46f,1}, {0.30f,0.84f,0.90f,1},
                };
                for (std::size_t index = 0; index < scenario.cells.size(); ++index) {
                    if (!visible_cells[index]) continue;
                    ImPlot::SetNextLineStyle(palette[index % 4], index == static_cast<std::size_t>(selected_cell) ? 2.4f : 1.2f);
                    ImPlot::PlotLine(scenario.cells[index].id.c_str(), history.time_s.data(),
                        history.temperature_k[index].data(), static_cast<int>(history.time_s.size()));
                }
                ImPlot::EndPlot();
            }
            if (!history.time_s.empty() && !scenario.interfaces.empty() &&
                ImPlot::BeginPlot("Interface l / l0", {-1,155})) {
                ImPlot::SetupAxes("t [s]", "ratio", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                for (std::size_t index = 0; index < scenario.interfaces.size(); ++index) {
                    const auto& edge = scenario.interfaces[index];
                    if (focus_neighborhood && edge.cell_a != static_cast<std::size_t>(selected_cell) &&
                        edge.cell_b != static_cast<std::size_t>(selected_cell)) continue;
                    ImPlot::PlotLine(scenario.interfaces[index].id.c_str(), history.time_s.data(),
                        history.edge_l_ratio[index].data(), static_cast<int>(history.time_s.size()));
                }
                ImPlot::EndPlot();
            }

            if (ImGui::CollapsingHeader("Energy balance", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::BulletText("Solid stored: %.3f kJ", simulation.stored_energy_j()/1000.0);
                ImGui::BulletText("Runaway source: %.3f kJ", simulation.released_energy_j()/1000.0);
                ImGui::BulletText("Boundary out: %.3f kJ", simulation.boundary_energy_out_j()/1000.0);
                ImGui::BulletText("Vent heat out: %.3f kJ",simulation.vent_heat_out_j()/1000.);
                ImGui::BulletText("Solid chemical reserve: %.3f kJ | legacy ejected: %.3f kJ",simulation.chemical_energy_remaining_j()/1000.,simulation.ejected_chemical_energy_j()/1000.);
                ImGui::TextDisabled("Chemical ledger residual: %.4g J",simulation.chemical_balance_residual_j());
                const double residual = simulation.energy_balance_residual_j();
                ImGui::TextColored(std::abs(residual) < 1.0e-5
                    ? ImVec4{0.30f,0.88f,0.72f,1} : ImVec4{1.0f,0.25f,0.08f,1},
                    "Residual: %.4g J", residual);
            }

            if (ImGui::CollapsingHeader("Event stream", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::BeginChild("events", {-1,150}, true);
                if (simulation.events().empty()) ImGui::TextDisabled("No events yet");
                for (auto iterator = simulation.events().rbegin();
                     iterator != simulation.events().rend(); ++iterator) {
                    const bool hot = iterator->kind.find("runaway") != std::string::npos ||
                                     iterator->kind.find("destroyed") != std::string::npos;
                    ImGui::TextColored(hot ? ImVec4{1.0f,0.34f,0.10f,1}
                                           : ImVec4{0.28f,0.82f,0.90f,1},
                                       "%7.2fs", iterator->time_s);
                    ImGui::SameLine();
                    ImGui::TextWrapped("%s · %s", iterator->kind.c_str(), iterator->entity_id.c_str());
                }
                ImGui::EndChild();
            }
            ImGui::End();

            ImGui::Render();
            int framebuffer_width = 0;
            int framebuffer_height = 0;
            glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, framebuffer_width, framebuffer_height);
            glClearColor(0.005f,0.007f,0.010f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }

        renderer.destroy();
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "battery_sim_viewer: %s\n", error.what());
        return 2;
    }
}

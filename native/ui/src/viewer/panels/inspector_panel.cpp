#include "viewer/viewer_window.hpp"
#include "viewer/widgets.hpp"
#include <algorithm>
#include <cmath>
namespace battery::ui {
void ViewerWindow::draw_inspector() {
    const auto& scenario = session.scenario();
    const auto& frame = session.frame();
    const auto& geometry = session.geometry();
    const auto& batteries = session.batteries();
    const bool replay = session.recorded();
    auto& selected = state.selected;
    auto& visual = state.visual;
    auto& auto_orbit = state.auto_orbit;
    auto& io = ImGui::GetIO();
    const float side = layout.side;
    const float top = layout.top;
    const float bottom = layout.bottom;
    const auto& error = session.error();
    const double solver_ms = session.timing().solver_ms;
    double area = 0, flow = 0;
    for (std::size_t i = 0; i < frame.process.channels.size(); ++i)
        if (scenario.gas_channels[i].rupture) {
            area += frame.process.channels[i].effective_area_m2;
            flow += std::max(0., frame.process.channels[i].mass_flow_kg_s);
        }
    ImGui::SetNextWindowPos({0, top});
    ImGui::SetNextWindowSize({side, io.DisplaySize.y - top - bottom});
    ImGui::Begin("inspector", nullptr, fixed);
    if (session.can_switch_gas()) {
        int method = session.kawasaki() ? 1 : 0;
        if (ImGui::Combo("Gas solver", &method, "Network (deterministic)\0Kawasaki MC (experimental)\0")) {
            session.set_kawasaki(method == 1);
        }
        ImGui::TextDisabled("Live switch / same gas inventory");
        if (session.timing().solver_s > 0. && session.timing().steps) {
            ImGui::Text("Compute: %.2fx realtime", session.timing().simulated_s / session.timing().solver_s);
            ImGui::Text("With UI: %.2fx realtime", session.timing().simulated_s / session.timing().wall_s);
            ImGui::TextDisabled("%.2f ms/step; %.2f s simulated",
                                1000 * session.timing().solver_s / session.timing().steps,
                                session.timing().simulated_s);
        }
        if (method)
            ImGui::TextWrapped("Bounded pair sweeps; physical time. Reaction/transport splitting "
                               "differs from network solver.");
        ImGui::TextDisabled("Flow memory: %s", scenario.gas_model.directional_inertia
                                                   ? "physical momentum"
                                                   : "off in this scenario");
        ImGui::Separator();
    }
    ImGui::TextDisabled("LAYERS");
    int layer = int(visual.field_layer);
    if (ImGui::RadioButton("Temperature", layer == 0))
        visual.field_layer = battery::FieldLayer::Temperature;
    ImGui::SameLine();
    if (ImGui::RadioButton("Shell damage", layer == 1))
        visual.field_layer = battery::FieldLayer::Degradation;
    ImGui::Checkbox("Gas mixture volumes", &visual.show_gas);
    ImGui::Checkbox("R3 flames", &visual.show_combustion);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Local R3 energy release per simulation interval, not temperature.\n"
                          "Stylized flame texture; not a resolved combustion front. Pauses with simulation.");
    if (visual.show_combustion) {
        ImGui::SetNextItemWidth(140);
        ImGui::SliderFloat("Flame intensity", &visual.flame_opacity, 0.f, .4f, "%.2f");
    }
    ImGui::Checkbox("Shell openings", &visual.show_ruptures);
    ImGui::SameLine();
    ImGui::Checkbox("Gas motion", &visual.show_flux);
    ImGui::Checkbox("Shell damage marks", &visual.show_damage);
    ImGui::Checkbox("Enclosure", &visual.show_enclosure);
    ImGui::Checkbox("Show heater", &visual.show_heaters);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Visibility only. Yellow: controller enabled; black: cutoff latched.");
    ImGui::Checkbox("Pads / insulation", &visual.show_pads);
    if (visual.show_pads) {
        ImGui::SliderFloat("Pad opacity", &visual.pad_opacity, 0, 1, "%.2f");
        ImGui::TextDisabled("Pads: brown insulation / white conductor; temperature overlay");
    }
    ImGui::SameLine();
    ImGui::Checkbox("Orbit", &auto_orbit);
    ImGui::SetNextItemWidth(140);
    ImGui::SliderFloat("Gas opacity", &visual.gas_opacity, .02f, .8f, "%.2f");
    ImGui::SetNextItemWidth(140);
    ImGui::SliderFloat("Battery density", &visual.battery_surface_opacity, .1f, 1.f, "%.2f");
    ImGui::SetNextItemWidth(140);
    ImGui::SliderFloat("Cutaway X", &visual.cutaway_fraction, .05f, 1, "%.2f");
    if (ImGui::CollapsingHeader("Display range / geometry")) {
        ImGui::Checkbox("Gas outside case", &visual.show_far_gas);
        ImGui::Checkbox("Hot gas only (filter)", &visual.hot_gas_only);
        ImGui::TextWrapped("Mixture includes initial air. Colour = temperature; opacity = relative density. "
                           "Flow markers include observed two-way mixing, not CFD velocities.");
        float max_c = visual.temperature_max_k - 273.15f;
        if (ImGui::SliderFloat("Max C", &max_c, 200, 1800, "%.0f"))
            visual.temperature_max_k = max_c + 273.15f;
        if (visual.hot_gas_only)
            ImGui::SliderFloat("Gas threshold K", &visual.gas_threshold_k, 1, 100, "%.0f");
        ImGui::SliderFloat("Enclosure opacity", &visual.enclosure_opacity, 0, .5f, "%.2f");
        ImGui::TextWrapped("%zu mapped gas regions; %zu omitted (no geometry).", geometry.gas.size(),
                           geometry.unresolved_gas_regions);
        ImGui::TextWrapped("Geometry: equivalent-volume prismatic cells; no invented cylindrical contacts.");
    }
    ImGui::Separator();
    ImGui::TextDisabled("CASCADE MAP / SELECT A CELL");
    for (std::size_t n = 0; n < batteries.size(); ++n) {
        const int i = batteries[n];
        const auto& c = frame.cells[i];
        auto rgb = battery::field_rgb(battery::FieldLayer::Temperature,
                                      float((c.temperature_k - visual.temperature_min_k) /
                                            (visual.temperature_max_k - visual.temperature_min_k)),
                                      false);
        ImGui::PushID(i);
        ImGui::PushStyleColor(ImGuiCol_Button,
                              ImVec4{rgb[0] * .65f + .04f, rgb[1] * .65f + .04f, rgb[2] * .65f + .04f, 1});
        if (ImGui::Button(scenario.cells[i].id.c_str(), {45, 28}))
            selected = i;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%.1f C | %s", c.temperature_k - 273.15, battery::name(c.mode));
        ImGui::PopStyleColor();
        ImGui::PopID();
        if (n % 5 != 4 && n + 1 < batteries.size())
            ImGui::SameLine();
    }
    ImGui::Separator();
    const auto& chosen = frame.cells[selected];
    ImGui::Text("%s", scenario.cells[selected].id.c_str());
    ImGui::SameLine();
    ImGui::TextColored(color(chosen.mode), "%s", battery::name(chosen.mode));
    ImGui::SetWindowFontScale(1.6f);
    ImGui::Text("%.1f C", chosen.temperature_k - 273.15);
    ImGui::SetWindowFontScale(1);
    ImGui::TextDisabled("Contents mean (not heater / shell)");
    ImGui::Text("Contents range %.1f - %.1f C", chosen.min_temperature_k - 273.15,
                chosen.max_temperature_k - 273.15);
    double shell_min = 1e30, shell_max = 0;
    for (std::size_t i = 0; i < scenario.cells.size(); ++i)
        if (scenario.cells[i].shell_contents_id == scenario.cells[selected].id) {
            shell_min = std::min(shell_min, frame.cells[i].min_temperature_k);
            shell_max = std::max(shell_max, frame.cells[i].max_temperature_k);
        }
    if (shell_max > 0)
        ImGui::Text("Shell range %.1f - %.1f C", shell_min - 273.15, shell_max - 273.15);
    ImGui::Text("R1 degradation   %.2f %%", chosen.damage * 100);
    if (!replay && scenario.cells[selected].electrical.enabled &&
        ImGui::CollapsingHeader("Electrical state")) {
        const auto states = session.electrical_states();
        const auto& e = states.at(selected);
        ImGui::Text("I %.4g A  |  R %.4g ohm", e.current_a, e.resistance_ohm);
        ImGui::Text("Joule %.4g W", e.power_w);
        ImGui::TextDisabled("%s", scenario.electrical_circuit.branches.empty() ? "Prescribed drive"
                                                                               : "Solved DC network");
    }
    double shell_damage = 0, opened_area = 0, shell_area = 0;
    for (std::size_t k = 0; k < scenario.gas_channels.size(); ++k) {
        const auto& c = scenario.gas_channels[k];
        if (c.shell_object < 0 ||
            scenario.cells[c.shell_object].shell_contents_id != scenario.cells[selected].id)
            continue;
        shell_damage = std::max(shell_damage, frame.process.channels[k].max_thermal_damage);
        opened_area += frame.process.channels[k].effective_area_m2;
        shell_area += battery::gas_channel_area(scenario, c);
    }
    ImGui::Text("Shell damage     %.1f %%", shell_damage * 100);
    ImGui::ProgressBar(float(shell_damage), {-1, 6}, "");
    ImGui::Text("Rupture area     %.1f %%", shell_area > 0 ? 100 * opened_area / shell_area : 0);
    ImGui::Text("R2 power         %.3g W", chosen.chemical_power_w);
    ImGui::Text("Chemical reserve %.2f kJ", chosen.chemical_energy_remaining_j / 1000);
    for (std::size_t i = 0; i < scenario.heating_protocols.size() && i < frame.heaters.size(); ++i) {
        const auto& p = scenario.heating_protocols[i];
        if (p.sensor_object != selected &&
            scenario.cells[p.sensor_object].shell_contents_id != scenario.cells[selected].id)
            continue;
        const auto& h = frame.heaters[i];
        ImGui::Text("Heater %s / %.2f W", h.stopped ? "CUTOFF" : "enabled", h.commanded_power_w);
        ImGui::Text("Control sensor %.1f C", h.previous_temperature_k - 273.15);
    }
    ImGui::Text("Open shell area  %.1f mm2", area * 1e6);
    ImGui::Text("Outward flux     %.3f g/s", flow * 1e3);
    ImGui::TextDisabled("%0.0f FPS%s", io.Framerate, replay ? " / playback" : "");
    if (!replay)
        ImGui::TextDisabled("Solver %.1f ms/frame", solver_ms);
    if (!error.empty())
        ImGui::TextWrapped("SOLVER PAUSED: %s", error.c_str());
    ImGui::End();
}
} // namespace battery::ui

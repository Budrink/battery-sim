#include "editor/forms/heater_controller_panel.hpp"
#include <nlohmann/json.hpp>
#include "imgui.h"
#include "shared/form_fields.hpp"
#include "misc/cpp/imgui_stdlib.h"
#include <cmath>

namespace battery::ui {
namespace {
using J = nlohmann::json;
void number(const char* label, J& protocol, const char* key, double scale = 1., double offset = 0.,
            bool positive = false) {
    double value = protocol.at(key).get<double>() * scale + offset;
    const bool changed = form::input(label, &value, "%.6g", key);
    const double stored = (value - offset) / scale;
    if (changed && std::isfinite(stored))
        protocol[key] = stored;
    if (!std::isfinite(stored) || (positive ? stored <= 0. : stored < 0.)) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1, .45f, .3f, 1});
        ImGui::TextWrapped("Invalid value: %s", offset != 0. ? "temperature must be above absolute zero."
                                                : positive   ? "must be greater than zero."
                                                             : "must be nonnegative.");
        ImGui::PopStyleColor();
    }
}
} // namespace

// Caller supplies the protocol belonging to this heater and scopes widget IDs
// with the heater ID. No shared selection/index into the controller array.
void draw_heater_controller(nlohmann::json& protocol) {
    ImGui::SeparatorText("Controller");
    auto sensor = protocol.at("sensor_id").get<std::string>();
    if (form::text_input("Temperature sensor", &sensor, "sensor_id"))
        protocol["sensor_id"] = sensor;
    bool sensor_initial = !protocol.contains("initial_temperature_k");
    if (ImGui::Checkbox("Initial target from sensor", &sensor_initial)) {
        if (sensor_initial)
            protocol.erase("initial_temperature_k");
        else
            protocol["initial_temperature_k"] = 298.15;
    }
    if (!sensor_initial)
        number("Initial target C", protocol, "initial_temperature_k", 1., -273.15, true);
    else
        ImGui::TextWrapped("Initial target: sensor's initial temperature");
    number("Heating rate C/min", protocol, "ramp_k_per_s", 60.);
    number("Controller gain W/K", protocol, "proportional_w_per_k");
    double maximum = protocol.at("maximum_power_w");
    if (form::input("Maximum power W", &maximum, "%.6g", "maximum_power_w") && std::isfinite(maximum))
        protocol["maximum_power_w"] = maximum;
    double base = protocol.at("feedforward_w");
    if (form::input("Base power W", &base, "%.6g", "feedforward_w") && std::isfinite(base))
        protocol["feedforward_w"] = base;
    if (maximum <= 0. || base < 0. || base > maximum) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1, .45f, .3f, 1});
        ImGui::TextWrapped(
            "Invalid power limits: maximum must be positive; base must be between zero and maximum.");
        ImGui::PopStyleColor();
    }
    ImGui::TextWrapped("Base power must not exceed maximum. Gain 0 = constant base power.");
    number("Cutoff temperature C", protocol, "cutoff_temperature_k", 1., -273.15, true);
    number("Cutoff minimum rise C/s", protocol, "cutoff_rate_k_per_s");
    ImGui::TextWrapped("0 C/s = temperature only; otherwise temperature AND rise rate. "
                       "No automatic restart after cutoff.");
}

void draw_heater_parameters(nlohmann::json& apparatus, nlohmann::json* protocol, nlohmann::json* contact,
                            double area) {
    auto& thermal = apparatus.at("thermal");
    number("Heater capacity J/K", thermal, "heat_capacity_j_per_k", 1., 0., true);
    double initial = thermal.value("initial_temperature_k", 298.15) - 273.15;
    if (form::input("Heater initial temperature C", &initial, "%.6g", "initial_temperature_k") &&
        std::isfinite(initial))
        thermal["initial_temperature_k"] = initial + 273.15;
    if (initial <= -273.15)
        ImGui::TextWrapped("Invalid temperature: must be above absolute zero.");
    if (contact && area > 0.) {
        auto& law = contact->at("law");
        const double activation = 1. / (1. + std::exp(-law.value("geometry_reference_logit", 0.)));
        const double reference = law.value("background_conductance_w_per_k", 0.) +
                                 activation * law.value("active_conductance_w_per_k", 0.);
        double h = reference / area;
        if (form::input("Heater-to-battery heat transfer [W/(m2 K)]", &h, "%.6g", "heater_contact_h") &&
            std::isfinite(h) && h > 0. && reference > 0.) {
            const double scale = h * area / reference;
            for (const auto* key :
                 {"background_conductance_w_per_k", "active_conductance_w_per_k",
                  "destroyed_background_conductance_w_per_k", "destroyed_active_conductance_w_per_k"})
                if (law.contains(key))
                    law[key] = law.at(key).get<double>() * scale;
        }
        ImGui::TextWrapped("Reference contact; temperature/damage response is preserved.");
    } else {
        ImGui::TextWrapped("Contact h: no explicit heater coupling found");
    }
    auto& electrical = apparatus.at("electrical");
    const double resistance = electrical.at("resistance_ohm");
    double power = std::pow(electrical.at("current_a").get<double>(), 2) * resistance;
    ImGui::BeginDisabled(protocol != nullptr);
    if (form::input("Uncontrolled power W", &power, "%.6g", "heater_power_w") && std::isfinite(power) &&
        power >= 0. && resistance > 0.) {
        const double current = std::sqrt(power / resistance);
        electrical["current_a"] = current;
        electrical["voltage_profile_v"] = {{0, current * resistance}};
    }
    ImGui::EndDisabled();
    if (protocol) {
        ImGui::TextWrapped("Active power is set by Base power, Gain and Maximum power below.");
        draw_heater_controller(*protocol);
    }
}
} // namespace battery::ui

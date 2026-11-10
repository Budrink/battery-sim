#pragma once
#include <nlohmann/json_fwd.hpp>

namespace battery::ui {
// Caller must scope ImGui IDs to the owning heater.
void draw_heater_controller(nlohmann::json& protocol);
// The only parameter form for both just-created and imported apparatus.
// Contact is the explicit heater/battery coupling, not another nearby contact.
void draw_heater_parameters(nlohmann::json& apparatus, nlohmann::json* protocol, nlohmann::json* contact,
                            double contact_area_m2);
} // namespace battery::ui

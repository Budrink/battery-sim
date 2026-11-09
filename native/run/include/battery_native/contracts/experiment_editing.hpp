#pragma once
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>
namespace battery {
// Explicit apparatus creation, never a fitted material change. Returns a validated copy.
nlohmann::json add_external_heater(const nlohmann::json&, const std::string& battery_id,
                                   const std::string& face, double power_w, double contact_h,
                                   double capacity_j_k, bool temperature_control, double ramp_k_s,
                                   double cutoff_temperature_k = 773.15, double cutoff_rate_k_s = 10.,
                                   bool constant_power_cutoff = false);
// One thermal reservoir and controller; wrap distributes its contact over the
// four lateral shell faces. A cylinder's +/-z mount is a circular end disk.
nlohmann::json set_heater_mount(const nlohmann::json&, const std::string& heater_id,
                                const std::string& battery_id, const std::string& face);
// Update an existing controller, or attach a latched cutoff to a constant heater.
nlohmann::json set_heater_cutoff(const nlohmann::json&, const std::string& heater_id,
                                 double cutoff_temperature_k, double cutoff_rate_k_s);
nlohmann::json duplicate_battery(const nlohmann::json&, const std::string& battery_id);
// Object operations return a validated copy; shared material models are retained.
nlohmann::json duplicate_element(const nlohmann::json&, const std::string& object_id);
nlohmann::json delete_element(const nlohmann::json&, const std::string& object_id);
// Object-local conductivity preset, not a measured material replacement.
// Keeps heat capacity, damage law, dimensions and contact resistances unchanged.
nlohmann::json set_pad_conductivity_preset(const nlohmann::json&, const std::string& object_id,
                                           const std::string& kind);
nlohmann::json set_enclosure(const nlohmann::json&, const std::string& id, const std::string& gas_compartment,
                             const nlohmann::json& properties);
nlohmann::json set_shell_face(const nlohmann::json&, const std::string& battery_id, const std::string& face,
                              const nlohmann::json& material, const nlohmann::json& opening_channel,
                              const std::string& destination);
// Dimensions: prism [x,y,z], cylinder [radius,height], all metres.
// Isolates the selected instance's model. Does not infer new gas/contact routes.
nlohmann::json set_battery_geometry(const nlohmann::json&, const std::string& battery_id,
                                    const std::string& kind, const std::vector<double>& dimensions_m);
} // namespace battery

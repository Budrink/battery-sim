#pragma once

#include "battery_native/model/model.hpp"
#include <nlohmann/json_fwd.hpp>
#include <string>

namespace battery {
// Resolve an experiment ONCE; both physics and visualization consume this
// native snapshot, including generated gas-volume geometry metadata.
nlohmann::json load_scenario_document(const std::string& path);
// Same native parser as load_scenario, for adapters which already own a JSON snapshot.
Scenario scenario_from_json(const nlohmann::json& document);
} // namespace battery

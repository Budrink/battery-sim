#pragma once

#include "battery_native/model/model.hpp"
#include <string>

namespace battery {
// File parsing belongs to the run adapter. The solver accepts a typed Scenario
// and does not know whether it came from JSON, a test, or a future editor.
Scenario load_scenario(const std::string& path);
} // namespace battery

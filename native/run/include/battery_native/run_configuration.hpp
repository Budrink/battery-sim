#pragma once

#include "battery_native/model/model.hpp"
#include <string>

namespace battery {
// Explicit launch overrides, not material laws or renderer preferences.
// Defaults preserve the scenario exactly, including all calibrated parameters.
struct RunConfiguration {
    int environment_mesh_scale = 1;
    bool isothermal_batteries = false;
};

void validate_run_configuration(const RunConfiguration& configuration);
Scenario configure_run(Scenario scenario, const RunConfiguration& configuration);
Scenario load_run(const std::string& path, const RunConfiguration& configuration = {});
} // namespace battery

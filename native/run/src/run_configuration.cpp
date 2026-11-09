#include "battery_native/run_configuration.hpp"
#include "battery_native/scenario_io.hpp"
#include <stdexcept>

namespace battery {
void validate_run_configuration(const RunConfiguration& configuration) {
    if (configuration.environment_mesh_scale < 1 || configuration.environment_mesh_scale > 8)
        throw std::invalid_argument("mesh-scale must be 1..8");
}

Scenario configure_run(Scenario scenario, const RunConfiguration& configuration) {
    validate_run_configuration(configuration);
    if (configuration.environment_mesh_scale != 1)
        refine_environment_mesh(scenario, configuration.environment_mesh_scale);
    if (configuration.isothermal_batteries)
        for (auto& cell : scenario.cells)
            if (cell.is_battery)
                cell.isothermal = true;
    return scenario;
}

Scenario load_run(const std::string& path, const RunConfiguration& configuration) {
    return configure_run(load_scenario(path), configuration);
}
} // namespace battery

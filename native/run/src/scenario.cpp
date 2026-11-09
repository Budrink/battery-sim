#include "battery_native/scenario_io.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/model/validation.hpp"
#include "scenario/scenario_internal.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace battery {
using namespace scenario_detail;

json load_scenario_document(const std::string& path) {
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot open scenario: " + path);
    json root;
    stream >> root;
    if (root.value("format", "") == "battery-sim.experiment")
        root = compile_experiment(root);
    return root;
}

Scenario load_scenario(const std::string& path) {
    return scenario_from_json(load_scenario_document(path));
}

Scenario scenario_from_json(const json& root) {
    Scenario scenario;
    scenario.name = root.at("name").get<std::string>();
    scenario.source = root.value("source", "unspecified");
    const auto metadata = root.value("metadata", json::object());
    scenario.calibration_status = metadata.value("calibration_status", "unspecified");
    scenario.geometry_status = metadata.value("geometry_status", "");
    const auto& simulation = root.at("simulation");
    scenario.settings.duration_s = simulation.at("duration_s").get<double>();
    scenario.settings.max_dt_s = simulation.at("max_dt_s").get<double>();
    scenario.settings.output_interval_s = simulation.at("output_interval_s").get<double>();
    scenario.settings.stability_factor = simulation.value("stability_factor", 0.2);
    scenario.settings.step_interval_s = simulation.value("step_interval_s", 0.0);
    scenario.settings.minimum_dt_s = simulation.value("minimum_dt_s", 0.0);
    if (!std::isfinite(scenario.settings.minimum_dt_s) || scenario.settings.minimum_dt_s < 0. ||
        (scenario.settings.minimum_dt_s > 0. && scenario.settings.minimum_dt_s < .005) ||
        scenario.settings.minimum_dt_s > scenario.settings.max_dt_s)
        throw std::runtime_error("minimum_dt_s must be zero (reference mode) or >=0.005 and <=max_dt_s");

    std::unordered_map<std::string, std::size_t> cell_index;
    parse_cells(root, scenario, cell_index);
    parse_shells(root, scenario, cell_index);
    parse_circuit(root, scenario, cell_index);
    const auto port_index = parse_thermal_links(root, scenario, cell_index);
    parse_gas_network(root, scenario, cell_index, port_index);

    for (const auto& value : root.value("primary_failures", json::array())) {
        FailureEvent event;
        event.cell_id = value.at("cell_id").get<std::string>();
        if (!cell_index.contains(event.cell_id)) {
            throw std::runtime_error("unknown primary failure cell: " + event.cell_id);
        }
        event.cell_index = cell_index.at(event.cell_id);
        event.time_s = value.at("time_s").get<double>();
        event.thermal_volume = value.value("thermal_volume", -1);
        scenario.primary_failures.push_back(std::move(event));
    }
    std::sort(scenario.primary_failures.begin(), scenario.primary_failures.end(),
              [](const FailureEvent& a, const FailureEvent& b) { return a.time_s < b.time_s; });

    if (root.contains("observations")) {
        const auto& observations = root.at("observations");
        if (observations.contains("propagated") && observations.at("propagated").is_boolean()) {
            scenario.has_expected_propagation = true;
            scenario.expected_propagation = observations.at("propagated").get<bool>();
        }
        scenario.expected_runaway_order = observations.value("runaway_order", std::vector<std::string>{});
    }
    validate_material_models(scenario);
    return scenario;
}

} // namespace battery

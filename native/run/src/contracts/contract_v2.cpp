#include "battery_native/contracts/experiment_contract.hpp"
#include "contract_internal.hpp"

namespace battery::contract_detail {
J lower_experiment_v2(const J& e) {
    J lowered = e;
    auto& assembly = lowered.at("assembly");
    auto& conditions = lowered.at("conditions");
    if (assembly.at("electrical_circuit").value("mode", "") == "solved_dc")
        (void)compile_dc_circuit(assembly, conditions);
    else
        validate_electrical_circuit(assembly, conditions);
    lower_local_ports(assembly);
    assembly.erase("electrical_circuit");
    conditions.erase("source_drives");
    lowered["version"] = 1;
    // The existing compiler remains the single authority for material/native validation.
    return lowered;
}
} // namespace battery::contract_detail

namespace battery {
nlohmann::json upgrade_experiment(const nlohmann::json& experiment) {
    (void)compile_experiment(experiment); // fail before changing the caller's document
    if (experiment.at("version") == 2)
        return experiment;
    auto upgraded = experiment;
    upgraded["version"] = 2;
    for (auto& port : upgraded["assembly"]["ports"])
        port["frame"] = "world"; // do not reinterpret old coordinates as local
    upgraded["assembly"]["electrical_circuit"] = {{"mode", "unmapped_drives"},
                                                  {"nodes", nlohmann::json::array()},
                                                  {"branches", nlohmann::json::array()}};
    upgraded["conditions"]["source_drives"] = nlohmann::json::array();
    (void)compile_experiment(upgraded);
    return upgraded;
}
} // namespace battery

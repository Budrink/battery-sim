#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace battery {
namespace {
using J = nlohmann::json;
// Added explicit parser defaults are allowed; every supplied value must survive.
void preserved(const J& source, const J& result, const std::string& path) {
    if (source.is_object() && result.is_object()) {
        for (const auto& [key, value] : source.items()) {
            if (!result.contains(key))
                throw std::runtime_error("Native import cannot represent " + path + "/" + key);
            preserved(value, result.at(key), path + "/" + key);
        }
    } else if (source.is_array() && result.is_array() && source.size() == result.size()) {
        for (std::size_t i = 0; i < source.size(); ++i)
            preserved(source[i], result[i], path + "/" + std::to_string(i));
    } else if (source != result) {
        throw std::runtime_error("Native import would change " + path + "; input left unchanged");
    }
}
J take(J& object, const char* key, const J& fallback) {
    J value = object.value(key, fallback);
    object.erase(key);
    return value;
}
} // namespace

J experiment_from_native(const J& native) {
    (void)scenario_from_json(native);
    const J provenance{
        {"status", "assumed"},
        {"notes",
         "Imported native inputs without parameter fitting. Original calibration metadata retained."}};
    J e{{"format", "battery-sim.experiment"},
        {"version", 1},
        {"id", "imported-native"},
        {"name", native.at("name")},
        {"provenance", provenance},
        {"battery_models", J::array()},
        {"assembly",
         {{"batteries", J::array()},
          {"passive_cells", J::array()},
          {"ports", native.value("ports", J::array())},
          {"interfaces", native.value("interfaces", J::array())},
          {"gas_network", native.at("gas_network")},
          {"vent_connections", J::array()}}},
        {"conditions",
         {{"boundaries", native.value("boundaries", J::array())},
          {"heating_protocols", native.value("heating_protocols", J::array())},
          {"electrical_drives", J::array()}}},
        {"numerics", {{"simulation", native.at("simulation")}, {"batteries", J::object()}}},
        {"measurements", J::array()},
        {"metadata", native.value("metadata", J::object())}};
    for (const auto& cell : native.at("cells")) {
        if (cell.value("role", "battery") == "passive") {
            e["assembly"]["passive_cells"].push_back(cell);
            continue;
        }
        const std::string bid = cell.at("id");
        J contents = cell;
        for (const auto* key : {"id", "role", "pose", "geometry", "runaway"})
            contents.erase(key);
        auto& thermal = contents.at("thermal");
        J state{{"temperature_k", take(thermal, "initial_temperature_k", 298.15)}};
        J mesh{{"divisions", take(thermal, "divisions", J::array({4, 4, 4}))},
               {"internal_heat_transfer", take(thermal, "internal_heat_transfer", "finite_volume")}};
        if (!thermal.contains("conductivity_w_per_m_k"))
            thermal["conductivity_w_per_m_k"] = {1, 1, 1};
        auto degradation = contents.value("degradation", J::object());
        state["damage_fraction"] = take(degradation, "initial_damage", 0.);
        state["aging_fraction"] = take(degradation, "aging_fraction", 0.);
        contents["degradation"] = degradation;
        auto& conversion = contents.at("conversion");
        state["precursor_fraction"] = take(conversion, "initial_precursor_fraction", 0.);
        state["charged_fraction"] = take(conversion, "initial_charged_fraction", 0.);
        const J compartment = take(conversion, "gas_compartment", nullptr);
        for (const auto* block : {"conversion", "electrical"}) {
            if (!contents.contains(block))
                continue;
            auto& law = contents.at(block);
            for (const auto* key : {"max_fraction_per_step", "max_temperature_step_k"})
                if (law.contains(key))
                    mesh[block][key] = take(law, key, nullptr);
        }
        if (contents.contains("electrical")) {
            auto& law = contents.at("electrical");
            J drive{{"cell_id", bid}};
            for (const auto* key : {"current_a", "current_profile_a", "voltage_profile_v", "current_input"})
                if (law.contains(key))
                    drive[key] = take(law, key, nullptr);
            if (!drive.contains("current_a") && !drive.contains("current_profile_a"))
                drive["current_a"] = 0.;
            e["conditions"]["electrical_drives"].push_back(drive);
        }
        J shell;
        for (const auto& candidate : native.value("shells", J::array()))
            if (candidate.at("contents_id") == bid) {
                if (!shell.is_null())
                    throw std::runtime_error("Duplicate shell for " + bid);
                shell = candidate;
                shell.erase("contents_id");
                state["shell_temperature_k"] =
                    take(shell, "initial_temperature_k", state.at("temperature_k"));
                if (shell.contains("surface_divisions"))
                    mesh["shell_surface_divisions"] = take(shell, "surface_divisions", nullptr);
            }
        const std::string mid = "imported-model-" + std::to_string(e.at("battery_models").size());
        e["battery_models"].push_back({{"format", "battery-sim.battery"},
                                       {"version", 1},
                                       {"id", mid},
                                       {"revision", "1"},
                                       {"provenance", provenance},
                                       {"geometry", cell.at("geometry")},
                                       {"contents", contents},
                                       {"shell", shell},
                                       {"vents", J::array()}});
        e["assembly"]["batteries"].push_back(
            {{"id", bid},
             {"model_id", mid},
             {"model_revision", "1"},
             {"pose", cell.value("pose", J{{"position_m", {0, 0, 0}}, {"orientation_xyzw", {0, 0, 0, 1}}})},
             {"initial_state", state},
             {"gas_compartment", compartment}});
        e["numerics"]["batteries"][bid] = mesh;
    }
    const auto round_trip = compile_experiment(e);
    // These identify provenance rather than the physical problem. All other
    // fields (including unknown ones) must round-trip, or import fails atomically.
    auto physical = native;
    physical.erase("source");
    physical.erase("metadata");
    preserved(physical, round_trip, "scenario");
    return upgrade_experiment(e);
}
} // namespace battery

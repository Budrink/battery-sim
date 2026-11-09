#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/thermal/thermal_mesh.hpp"
#include "battery_native/simulation/simulation.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using J = nlohmann::json;
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
J read(const char* path) {
    std::ifstream in(path);
    J j;
    in >> j;
    return j;
}
void rejects(const J& source, const std::function<void(J&)>& edit) {
    auto candidate = source;
    edit(candidate);
    bool rejected = false;
    try {
        battery::compile_experiment(candidate);
    } catch (const std::exception&) {
        rejected = true;
    }
    require(rejected, "Invalid or misplaced input accepted");
}
void check_contract(const J& e, J old) {
    const auto immutable = e.dump();
    auto compiled = battery::compile_experiment(e);
    require(e.dump() == immutable, "Compilation mutated input");
    require(compiled == battery::compile_experiment(e), "Compilation is nondeterministic");
    require(compiled.at("metadata").at("experiment_contract") == e, "Incomplete provenance snapshot");
    auto& original = old["cells"][0];
    original["pose"] = {{"position_m", {0, 0, 0}}, {"orientation_xyzw", {0, 0, 0, 1}}};
    original["degradation"] = {{"initial_damage", 0}, {"aging_fraction", 0}};
    original["conversion"]["initial_charged_fraction"] = 0;
    original["thermal"]["internal_heat_transfer"] = "finite_volume";
    old["heating_protocols"] = J::array();
    // Only metadata/source and explicitly expanded defaults differ from the original demo.
    compiled.erase("metadata");
    compiled.erase("source");
    old.erase("metadata");
    old.erase("source");
    require(compiled == old, "Compiler changed physical/numerical parameters of the existing demo");
    const auto parsed = battery::scenario_from_json(compiled);
    const auto mesh = battery::build_thermal_mesh(parsed);
    require(!mesh.volumes.empty(), "Compiled scenario has no thermal mesh");

    // A direct Scenario and both JSON entry points must enforce the same
    // material constraints before CUDA allocation, not fail only after meshing.
    auto unsupported = parsed;
    unsupported.cells[0].initial_damage = .1;
    bool direct_rejected = false;
    try {
        (void)battery::build_thermal_mesh(unsupported);
    } catch (const std::exception&) {
        direct_rejected = true;
    }
    require(direct_rejected, "Direct Scenario bypassed material validation");
    auto native_input = compiled;
    native_input["cells"][0]["degradation"]["initial_damage"] = .1;
    bool native_rejected = false;
    try {
        (void)battery::scenario_from_json(native_input);
    } catch (const std::exception&) {
        native_rejected = true;
    }
    require(native_rejected, "Native loader accepted unsupported material state");

    rejects(e, [](J& j) { j["version"] = 2; });
    rejects(e, [](J& j) { j["verison"] = 1; });
    rejects(e, [](J& j) { j["battery_models"][0]["contents"]["thermal"]["initial_temperature_k"] = 900; });
    rejects(
        e, [](J& j) { j["battery_models"][0]["contents"]["conversion"]["initial_precursor_fraction"] = .8; });
    rejects(e, [](J& j) { j["battery_models"][0]["contents"]["conversion"]["max_temperature_step_k"] = 1; });
    rejects(e, [](J& j) { j["assembly"]["batteries"][0]["model_revision"] = "unknown"; });
    rejects(e, [](J& j) { j["battery_models"].push_back(j["battery_models"][0]); });
    rejects(e, [](J& j) { j["assembly"]["batteries"].push_back(j["assembly"]["batteries"][0]); });
    rejects(e, [](J& j) { j["assembly"]["batteries"][0]["initial_state"]["temperature_k"] = -1; });
    rejects(e, [](J& j) { j["assembly"]["batteries"][0]["initial_state"]["damage_fraction"] = .1; });
    rejects(e, [](J& j) { j["battery_models"][0]["contents"]["degradation"]["rate_at_onset_per_s"] = .01; });
    rejects(e, [](J& j) { j["assembly"]["batteries"][0]["initial_state"]["charged_fraction"] = .7; });
    rejects(e, [](J& j) { j["assembly"]["batteries"][0]["pose"]["orientation_xyzw"] = {0, 0, 0, 2}; });
    rejects(e, [](J& j) { j["numerics"]["batteries"]["B"]["divisions"] = {4294967297ULL, 3, 3}; });
    rejects(e, [](J& j) {
        j["numerics"]["simulation"]["duration_s"] = std::numeric_limits<double>::quiet_NaN();
    });
    rejects(e, [](J& j) { j["numerics"]["simulation"]["minimum_dt_s"] = .001; });
    rejects(e, [](J& j) { j["numerics"]["batteries"]["ghost"] = j["numerics"]["batteries"]["B"]; });
    rejects(e, [](J& j) { j["assembly"]["passive_cells"] = J::object(); });
    rejects(e, [](J& j) { j["assembly"]["gas_network"]["compartments"][0]["contents_id"] = "ghost"; });
    rejects(e, [](J& j) { j["assembly"]["vent_connections"][0]["to"] = "ghost"; });
    rejects(e, [](J& j) { j["assembly"]["vent_connections"] = J::array(); });
    rejects(e,
            [](J& j) { j["assembly"]["vent_connections"].push_back(j["assembly"]["vent_connections"][0]); });
    rejects(e, [](J& j) { j["battery_models"][0]["vents"][0]["channel"]["from"] = "cavity"; });
    rejects(e, [](J& j) { j["measurements"][0]["mapping"]["cell_id"] = "B"; });
    rejects(e, [](J& j) { j["measurements"][0]["unit"] = "Pa"; });
    rejects(e,
            [](J& j) { j["measurements"][0]["mapping"] = {{"status", "confirmed"}, {"cell_id", "ghost"}}; });
    rejects(e, [](J& j) { j["conditions"]["primary_failures"] = {{{"cell_id", "B"}, {"time_s", 1}}}; });

    auto cylindrical = e;
    cylindrical["battery_models"][0]["geometry"] = {{"kind", "cylinder"},
                                                    {"dimensions_m", {.02 / std::sqrt(std::acos(-1.)), .02}}};
    rejects(cylindrical, [](J&) {});
    cylindrical["numerics"]["batteries"]["B"]["geometry_approximation"] = {
        {"geometry", e["battery_models"][0]["geometry"]},
        {"reason", "Equal-volume synthetic mesh regression"}};
    battery::compile_experiment(cylindrical);
    rejects(cylindrical, [](J& j) {
        j["numerics"]["batteries"]["B"]["geometry_approximation"]["geometry"]["dimensions_m"][0] = .04;
    });

    auto two = e;
    auto second = two["assembly"]["batteries"][0];
    second["id"] = "B2";
    second["gas_compartment"] = "inside-B2";
    second["pose"]["position_m"] = {.04, 0, 0};
    second["initial_state"]["temperature_k"] = 310;
    two["assembly"]["batteries"].push_back(second);
    two["numerics"]["batteries"]["B2"] = two["numerics"]["batteries"]["B"];
    auto gas = two["assembly"]["gas_network"]["compartments"][0];
    gas["id"] = "inside-B2";
    gas["contents_id"] = "B2";
    two["assembly"]["gas_network"]["compartments"].push_back(gas);
    two["assembly"]["vent_connections"].push_back(
        {{"id", "second-vent"}, {"battery_id", "B2"}, {"vent_id", "top"}, {"to", "cavity"}});
    const auto t = battery::compile_experiment(two);
    require(t["cells"][0]["conversion"]["reaction_rate_per_s"] ==
                t["cells"][1]["conversion"]["reaction_rate_per_s"],
            "Shared model changed coefficients");
    require(t["cells"][0]["thermal"]["initial_temperature_k"] == 500 &&
                t["cells"][1]["thermal"]["initial_temperature_k"] == 310,
            "Instance states aliased");
    require(t["gas_network"]["channels"][1]["port_id"] == "B2:case:+z:outer", "Vent binding not namespaced");

    auto electric = e;
    electric["battery_models"][0]["contents"]["electrical"] = {{"resistance_ohm", .01}};
    auto undriven = battery::compile_experiment(electric);
    require(undriven["cells"][0]["electrical"]["current_a"] == 0, "Unconnected battery energized");
    electric["conditions"]["electrical_drives"].push_back({{"cell_id", "B"},
                                                           {"current_profile_a", {{0, 2}, {1, 4}}},
                                                           {"voltage_profile_v", {{0, 4}, {1, 4.1}}}});
    auto driven = battery::compile_experiment(electric);
    require(!driven["cells"][0]["electrical"].contains("current_a"), "Constant drive conflicts with profile");
    require(driven["cells"][0]["electrical"]["resistance_ohm"] == .01, "Drive overwrote material resistance");
    rejects(electric, [](J& j) { j["conditions"]["electrical_drives"][0]["resistance_ohm"] = 5; });
    rejects(electric, [](J& j) { j["conditions"]["electrical_drives"][0]["current_a"] = 2; });
    std::cout << "Contract v1: legacy equivalence, immutable models, bindings, states, geometry, drives and "
                 "rejection checks passed\n";
}

void check_cuda(const J& experiment, const J& original) {
    auto a = battery::scenario_from_json(battery::compile_experiment(experiment));
    auto b = battery::scenario_from_json(original);
    a.settings.duration_s = b.settings.duration_s = .05;
    auto upgraded =
        battery::scenario_from_json(battery::compile_experiment(battery::upgrade_experiment(experiment)));
    upgraded.settings.duration_s = .05;
    battery::Simulation one(a), two(b), three(upgraded);
    one.run_to_end();
    two.run_to_end();
    three.run_to_end();
    require(one.time_s() == two.time_s(), "CUDA end time changed");
    // Full committed state, including thermal, reaction and gas inventories, uses a common test contract.
    require(one.save_checkpoint("contract-regression") == two.save_checkpoint("contract-regression"),
            "CUDA state changed after contract compilation");
    require(one.save_checkpoint("contract-regression") == three.save_checkpoint("contract-regression"),
            "CUDA state changed after v2 migration");
    std::cout << "CUDA v1/v2 vs legacy: identical committed checkpoint after 0.05 s\n";
}

void check_v2(const J& original) {
    const auto v2 = battery::upgrade_experiment(original);
    require(v2.at("version") == 2, "Import did not upgrade version");
    require(battery::upgrade_experiment(v2) == v2, "Repeated import changes v2");
    auto before = battery::compile_experiment(original), after = battery::compile_experiment(v2);
    require(after["metadata"]["experiment_contract"] == v2, "Lost original v2 snapshot");
    for (const auto* field : {"source", "metadata"}) {
        before.erase(field);
        after.erase(field);
    }
    require(before == after, "v1 import changed physical/numerical inputs");

    auto local = v2;
    local["assembly"]["ports"] = {{{"id", "local"},
                                   {"cell_id", "B"},
                                   {"frame", "local"},
                                   {"area_m2", .0001},
                                   {"centroid_m", {.01, 0, 0}},
                                   {"normal", {1, 0, 0}}}};
    local["assembly"]["batteries"][0]["pose"]["position_m"] = {.1, .2, .3};
    require(std::abs(battery::compile_experiment(local)["ports"][0]["centroid_m"][0].get<double>() - .11) <
                1e-12,
            "Port did not follow translated shell-backed battery");
    // The shell builder itself only supports axis-aligned contents; exercise rotation without it.
    local["battery_models"][0]["shell"] = nullptr;
    local["battery_models"][0]["vents"] = J::array();
    local["assembly"]["vent_connections"] = J::array();
    local["assembly"]["batteries"][0]["initial_state"].erase("shell_temperature_k");
    // Quarter-turn around Z: local +X becomes world +Y. Native mesh limitations are unchanged.
    local["assembly"]["batteries"][0]["pose"]["orientation_xyzw"] = {0, 0, std::sqrt(.5), std::sqrt(.5)};
    const auto port = battery::compile_experiment(local)["ports"][0];
    require(std::abs(port["centroid_m"][0].get<double>() - .1) < 1e-12 &&
                std::abs(port["centroid_m"][1].get<double>() - .21) < 1e-12 &&
                std::abs(port["normal"][1].get<double>() - 1.) < 1e-12,
            "Local port transformation incorrect");
    local["assembly"]["ports"][0]["frame"] = "world";
    require(battery::compile_experiment(local)["ports"][0]["centroid_m"] == J({.01, 0, 0}),
            "World port moved with object");
    auto legacy = local;
    legacy["version"] = 1;
    legacy["assembly"].erase("electrical_circuit");
    legacy["conditions"].erase("source_drives");
    legacy["assembly"]["ports"][0].erase("frame");
    require(battery::upgrade_experiment(legacy)["assembly"]["ports"][0]["frame"] == "world",
            "Imported port reinterpreted as local");
    rejects(local, [](J& j) { j["assembly"]["ports"][0]["frame"] = "bad"; });
    rejects(local, [](J& j) { j["assembly"]["ports"][0]["normal"] = {0, 0, 0}; });
    rejects(local, [](J& j) { j["assembly"]["ports"][0]["unexpected"] = 1; });
    rejects(local, [](J& j) {
        j["assembly"]["ports"][0]["frame"] = "local";
        j["assembly"]["ports"][0]["cell_id"] = "ghost";
    });

    auto electric = v2;
    electric["battery_models"][0]["contents"]["electrical"] = {{"resistance_ohm", .01}};
    electric["assembly"]["electrical_circuit"] = {
        {"mode", "prescribed_currents"},
        {"nodes", {"pos", "neg"}},
        {"branches",
         {{{"id", "cell-B"}, {"kind", "battery"}, {"cell_id", "B"}, {"from", "pos"}, {"to", "neg"}},
          {{"id", "charger"}, {"kind", "current_source"}, {"from", "neg"}, {"to", "pos"}}}}};
    electric["conditions"]["electrical_drives"] = {
        {{"cell_id", "B"}, {"current_profile_a", {{0, 2}, {1, 4}}}}};
    electric["conditions"]["source_drives"] = {
        {{"source_id", "charger"}, {"current_profile_a", {{0, 2}, {.5, 3}, {1, 4}}}}};
    const auto driven = battery::compile_experiment(electric);
    require(driven["cells"][0]["electrical"]["current_profile_a"] == J({{0, 2}, {1, 4}}),
            "Circuit changed measured current");
    rejects(electric, [](J& j) { j["conditions"]["source_drives"][0]["current_profile_a"][1][1] = 4; });
    rejects(electric, [](J& j) { j["conditions"]["source_drives"][0]["current_profile_a"][1][0] = 0; });
    rejects(electric, [](J& j) { j["conditions"]["source_drives"][0]["source_id"] = "ghost"; });
    rejects(electric, [](J& j) { j["assembly"]["electrical_circuit"]["branches"][0]["to"] = "pos"; });
    rejects(electric, [](J& j) { j["assembly"]["electrical_circuit"]["branches"][0]["cell_id"] = "ghost"; });
    rejects(electric, [](J& j) {
        j["conditions"]["electrical_drives"][0]["current_input"] = "voltage_limited_setpoint";
    });
    rejects(electric, [](J& j) { j["assembly"]["electrical_circuit"]["mode"] = "unmapped_drives"; });
    rejects(electric, [](J& j) { j["assembly"]["electrical_circuit"]["mode"] = "automatic"; });
    rejects(electric, [](J& j) {
        j["assembly"]["electrical_circuit"]["branches"].push_back(
            j["assembly"]["electrical_circuit"]["branches"][0]);
    });
    rejects(electric, [](J& j) { j["version"] = 3; });

    auto pair = electric;
    auto second = pair["assembly"]["batteries"][0];
    second["id"] = "B2";
    second["gas_compartment"] = "inside-B2";
    second["pose"]["position_m"] = {.04, 0, 0};
    pair["assembly"]["batteries"].push_back(second);
    pair["numerics"]["batteries"]["B2"] = pair["numerics"]["batteries"]["B"];
    auto compartment = pair["assembly"]["gas_network"]["compartments"][0];
    compartment["id"] = "inside-B2";
    compartment["contents_id"] = "B2";
    pair["assembly"]["gas_network"]["compartments"].push_back(compartment);
    pair["assembly"]["vent_connections"].push_back(
        {{"id", "second-vent"}, {"battery_id", "B2"}, {"vent_id", "top"}, {"to", "cavity"}});
    auto drive2 = pair["conditions"]["electrical_drives"][0];
    drive2["cell_id"] = "B2";
    pair["conditions"]["electrical_drives"].push_back(drive2);
    auto& circuit = pair["assembly"]["electrical_circuit"];
    circuit["nodes"].push_back("mid");
    circuit["branches"][0]["to"] = "mid";
    circuit["branches"].push_back({{"id", "cell-B2"}, {"kind", "battery"}, {"cell_id", "B2"},
                                   {"from", "mid"}, {"to", "neg"}});
    require(battery::compile_experiment(pair)["cells"][1]["electrical"]["current_profile_a"] ==
                driven["cells"][0]["electrical"]["current_profile_a"],
            "Series circuit changed equal branch currents");
    circuit["nodes"] = {"pos", "neg"};
    circuit["branches"][0]["to"] = "neg";
    circuit["branches"][2]["from"] = "pos";
    rejects(pair, [](J&) {}); // Parallel input current must be the sum, not the series current.
    pair["conditions"]["source_drives"][0]["current_profile_a"] = {{0, 4}, {.5, 6}, {1, 8}};
    battery::compile_experiment(pair);

    J editor{{"format", "battery-sim.editor"},
             {"version", 1},
             {"experiment_id", v2.at("id")},
             {"hidden_objects", {"B"}},
             {"layout", {{"inspector", true}}}};
    battery::validate_editor_document(editor, v2);
    for (const auto& invalid : {J{{"version", 2}}, J{{"experiment_id", "other"}},
                                J{{"hidden_objects", {"ghost"}}}, J{{"physics", 1}}}) {
        auto bad = editor;
        bad.update(invalid);
        bool rejected = false;
        try {
            battery::validate_editor_document(bad, v2);
        } catch (const std::exception&) {
            rejected = true;
        }
        require(rejected, "Invalid editor sidecar accepted");
    }
    require(battery::compile_experiment(v2)["metadata"]["experiment_contract"] == v2,
            "Editor state leaked into executable snapshot");
    std::cout << "Contract v2: migration, poses, circuits, KCL and isolated editor state passed\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4)
            throw std::runtime_error("Expected experiment and legacy scenario paths, optionally --cuda");
        const auto experiment = read(argv[1]), original = read(argv[2]);
        if (argc == 4) {
            require(std::string(argv[3]) == "--cuda", "Unknown option");
            check_cuda(experiment, original);
        } else {
            check_contract(experiment, original);
            check_v2(experiment);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

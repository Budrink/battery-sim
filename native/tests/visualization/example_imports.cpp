#include "battery_ui/editor_document.hpp"
#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/simulation/simulation.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

using J = nlohmann::json;
void require(bool ok, const char* why) {
    if (!ok)
        throw std::runtime_error(why);
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected examples directory");
        const std::filesystem::path root = argv[1];
        battery::ui::EditorDocument doc;
        for (const auto* relative :
             {"reaction-chain-demo.json", "jobs/reaction-chain.json", "jobs/module-4x4.json",
              "contracts/reaction-chain-experiment.json", "contracts/editor-experiment.json",
              "contracts/laboratory-demo.json", "contracts/module-4x4.json",
              "contracts/fsri30-shell-020mm.json", "contracts/fsri30-shell-020mm-wrap.json",
              "contracts/module-4x4.json.editor.json",
              "contracts/fsri30-shell-020mm-wrap.json.editor.json"}) {
            doc.load(root / relative);
            const auto compiled = doc.compile();
            const auto scenario = battery::scenario_from_json(compiled);
            require(!battery::build_thermal_mesh(scenario).volumes.empty(), "Empty imported mesh");
            require(doc.draft.at("format") == "battery-sim.experiment", "Import did not produce a contract");
            std::cout << "Loaded " << relative << ": " << doc.draft.at("assembly").at("batteries").size()
                      << " batteries\n";
        }
        const auto good = doc.draft;
        for (const auto* relative :
             {"contracts/synthetic-battery-model.json", "contracts/editor-state.json",
              "contracts/fsri30-material-preset.json", "contracts/fsri30-heater-preset.json",
              "contracts/fsri30-gas-transport-preset.json"}) {
            bool rejected = false;
            try {
                doc.load(root / relative);
            } catch (const std::exception& e) {
                rejected = std::string(e.what()).find("no lossy") == std::string::npos;
            }
            require(rejected && doc.draft == good, "Non-experiment replaced draft or gave opaque error");
        }
        J raw;
        std::ifstream(root / "reaction-chain-demo.json") >> raw;
        const auto immutable = raw;
        const auto imported = battery::experiment_from_native(raw);
        require(raw == immutable, "Native importer mutated input");
        auto unsupported = raw;
        unsupported["unsupported_physics"] = 123;
        bool rejected = false;
        try {
            (void)battery::experiment_from_native(unsupported);
        } catch (const std::exception& e) {
            rejected = std::string(e.what()).find("unsupported_physics") != std::string::npos;
        }
        require(rejected, "Native importer silently discarded an unknown field");
        battery::Simulation a(battery::scenario_from_json(raw));
        battery::Simulation b(battery::scenario_from_json(battery::compile_experiment(imported)));
        for (int i = 0; i < 20; ++i) {
            a.step();
            b.step();
            require(a.time_s() == b.time_s(), "Native import changed physical time");
            require(a.cells().size() == b.cells().size(), "Native import changed cell count");
            for (std::size_t j = 0; j < a.cells().size(); ++j)
                require(a.cells()[j].temperature_k == b.cells()[j].temperature_k &&
                            a.cells()[j].damage == b.cells()[j].damage,
                        "Native import changed trajectory");
        }
        std::cout << "11 example imports; 5 explicit non-experiment rejections; native CUDA trajectory "
                     "unchanged for 20 steps\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

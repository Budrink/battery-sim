#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const bool mount = argc == 6 && std::string(argv[2]) == "--mount-heater";
        if (!mount && (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--upgrade-v2")))
            throw std::invalid_argument("usage: battery_sim_compile_experiment EXPERIMENT.json "
                                        "[--upgrade-v2 | --mount-heater HEATER BATTERY wrap|+z|-z] "
                                        "(JSON on stdout; no simulation)");
        std::ifstream input(argv[1]);
        if (!input)
            throw std::runtime_error("Cannot open experiment");
        nlohmann::json document;
        input >> document;
        std::cout << (mount       ? battery::set_heater_mount(document, argv[3], argv[4], argv[5])
                      : argc == 3 ? battery::upgrade_experiment(document)
                                  : battery::compile_experiment(document))
                         .dump(2)
                  << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}

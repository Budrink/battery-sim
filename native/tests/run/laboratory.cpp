#include "battery_native/contracts/experiment_contract.hpp"
#include "battery_native/contracts/experiment_editing.hpp"
#include "battery_native/scenario_document.hpp"
#include "battery_native/simulation/simulation.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
using J = nlohmann::json;
void require(bool b, const char* message) {
    if (!b)
        throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected laboratory contract");
        std::ifstream in(argv[1]);
        J doc;
        in >> doc;
        auto two = battery::duplicate_battery(doc, "B");
        require(two["assembly"]["batteries"].size() == 2, "Duplicate failed");
        auto cylinder = battery::set_battery_geometry(doc, "B", "cylinder", {.009, .065});
        auto cylindrical_scenario = battery::scenario_from_json(battery::compile_experiment(cylinder));
        cylindrical_scenario.settings.duration_s = .02;
        battery::Simulation cylindrical(cylindrical_scenario);
        cylindrical.run_to_end();
        require(std::abs(cylindrical.energy_balance_residual_j()) < 1e-7,
                "Cylinder approximation heat balance failed");
        auto heated = battery::add_external_heater(doc, "B", "+x", 5, 500, 5, false, .1);
        auto compiled = battery::compile_experiment(heated);
        auto s = battery::scenario_from_json(compiled);
        require(s.interfaces.back().port_b == "B:case:+x:outer", "Heater bypassed casing");
        s.settings.duration_s = 1;
        battery::Simulation sim(s);
        sim.run_to_end();
        require(sim.joule_energy_j() > 5 && sim.joule_energy_j() < 5.1, "Heater/circuit power ledger wrong");
        require(std::abs(sim.energy_balance_residual_j()) < 1e-7, "External heat balance failed");
        require(sim.cells()[0].temperature_k > 298.15, "No heat reached the battery");
        auto controlled = battery::add_external_heater(doc, "B", "-x", 5, 500, 5, true, .1);
        s = battery::scenario_from_json(battery::compile_experiment(controlled));
        s.settings.duration_s = .1;
        battery::Simulation control(s);
        control.run_to_end();
        require(control.heating_protocol_states().size() == 1, "Controller not connected");
        require(control.joule_energy_j() > 0, "Controller does not heat");
        require(doc["assembly"]["passive_cells"].empty(), "Editing mutated source snapshot");
        std::cout << "Laboratory: duplicate, external casing contact, controlled/constant heater, CUDA "
                     "circuit and balance passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

// Read-only diagnostic of the same compiled contract and solver used by the UI.
#include "battery_native/scenario_io.hpp"
#include "battery_native/simulation/simulation.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::runtime_error("usage: battery_sim_heating_reaction_probe SCENARIO UNTIL_S");
        auto scenario = battery::load_scenario(argv[1]);
        scenario.settings.duration_s = std::stod(argv[2]);
        battery::Simulation sim(scenario);
        const auto start = std::chrono::steady_clock::now();
        auto sample = [&] {
            nlohmann::json row{{"time_s", sim.time_s()}, {"batteries", nlohmann::json::array()},
                {"thermal_residual_j", sim.energy_balance_residual_j()},
                {"chemical_residual_j", sim.chemical_balance_residual_j()},
                {"wall_s", std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
            for (std::size_t i = 0; i < scenario.cells.size(); ++i) {
                const auto& c = sim.cells()[i];
                if (scenario.cells[i].is_battery)
                    row["batteries"].push_back({{"id", scenario.cells[i].id},
                        {"temperature_c", c.temperature_k-273.15}, {"max_c", c.max_temperature_k-273.15},
                        {"r2_power_w", c.chemical_power_w}, {"remaining_j", c.chemical_energy_remaining_j},
                        {"released_j", c.released_energy_j}, {"mode", battery::name(c.mode)}});
            }
            row["heaters"] = nlohmann::json::array();
            for (std::size_t i = 0; i < scenario.heating_protocols.size(); ++i) {
                const auto& p = scenario.heating_protocols[i];
                const auto& h = sim.heating_protocol_states()[i];
                row["heaters"].push_back({{"id", p.id}, {"sensor", p.sensor_id},
                    {"state_power_w", h.commanded_power_w}, {"cutoff", h.stopped},
                    {"sensor_c", h.previous_temperature_k-273.15}, {"cutoff_s", h.cutoff_time_s}});
            }
            std::cout << row.dump() << std::endl;
        };
        sample();
        for (double next = 100; !sim.finished(); next += 100) {
            const double end = std::min(next, scenario.settings.duration_s);
            while (sim.time_s() < end-1e-9) sim.step(end-sim.time_s());
            sample();
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

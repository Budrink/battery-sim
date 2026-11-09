#include "battery_native/run_configuration.hpp"
#include "battery_native/scenario_io.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected a scenario path");
        const auto original = battery::load_scenario(argv[1]);
        const auto unchanged = battery::load_run(argv[1]);
        require(original.cells.size() == unchanged.cells.size(), "Defaults changed topology");
        for (std::size_t i = 0; i < original.cells.size(); ++i) {
            require(original.cells[i].thermal_divisions == unchanged.cells[i].thermal_divisions,
                    "Defaults changed discretization");
            require(original.cells[i].conversion.reaction_rate_per_s ==
                        unchanged.cells[i].conversion.reaction_rate_per_s,
                    "Defaults changed chemistry");
        }
        auto fixture = original;
        battery::CellConfig passive;
        passive.id = "passive";
        passive.is_battery = false;
        passive.thermal_divisions = {2, 3, 1};
        fixture.cells.push_back(passive);
        battery::RunConfiguration settings;
        settings.environment_mesh_scale = 2;
        settings.isothermal_batteries = true;
        const auto configured = battery::configure_run(fixture, settings);
        require(configured.cells.back().thermal_divisions == std::array<int, 3>{4, 6, 2},
                "Passive mesh override not applied");
        require(configured.cells.front().thermal_divisions == fixture.cells.front().thermal_divisions,
                "Environment refinement changed battery topology");
        require(configured.cells.front().isothermal && !configured.cells.back().isothermal,
                "Battery override leaked into a passive material");
        require(fixture.cells.back().thermal_divisions == passive.thermal_divisions,
                "Run configuration mutated the reusable seed");
        for (int scale : {0, 9}) {
            settings.environment_mesh_scale = scale;
            bool rejected = false;
            try {
                battery::configure_run(fixture, settings);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "Invalid mesh scale accepted");
        }
        std::cout << "Run adapter: unchanged defaults, isolated overrides, immutable seed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

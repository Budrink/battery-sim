#include "battery_native/model.hpp"
#include "battery_native/simulation.hpp"

#include <cmath>
#include <chrono>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: battery_sim_headless SCENARIO.json [--lumped-limit|--isothermal] [--duration S] [--max-dt S] [--outer-step S] [--mesh-scale N]\n";
        return 2;
    }
    try {
        auto scenario = battery::load_scenario(argv[1]);
        for(int arg=2;arg<argc;++arg) {
        const std::string key=argv[arg];
        if (key == "--isothermal") {
            for (auto& cell : scenario.cells) if(cell.is_battery)cell.isothermal = true;
            std::cout << "test_mode=isothermal-full-mesh\n";
        } else if (key == "--lumped-limit") {
            // The old cascade fixture describes the isothermal/high-k limit,
            // not a measured spatial-model outcome. Keep that regression explicit.
            for (auto& cell : scenario.cells) {
                cell.thermal_divisions = {1,1,1};
                cell.conductivity_w_per_m_k = {1e12,1e12,1e12};
                cell.initial_temperature_field_k.clear();
            }
            std::cout << "test_mode=isothermal-high-conductivity-limit\n";
        } else {
            if(arg+1>=argc)throw std::runtime_error("Missing option value");
            const std::string value=argv[++arg];
            if(key=="--duration")scenario.settings.duration_s=std::stod(value);
            else if(key=="--max-dt")scenario.settings.max_dt_s=std::stod(value);
            else if(key=="--outer-step")scenario.settings.step_interval_s=std::stod(value);
            else if(key=="--mesh-scale")battery::refine_environment_mesh(scenario,std::stoi(value));
            else throw std::runtime_error("Unknown argument: "+key);
        }
        }
        battery::Simulation simulation(scenario);
        const auto start=std::chrono::steady_clock::now();
        simulation.run_to_end();
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::size_t spent=0,active=0;
        for(std::size_t i=0;i<scenario.cells.size();++i)if(scenario.cells[i].is_battery) {
            spent+=simulation.cells()[i].mode==battery::CellMode::Spent;
            active+=simulation.cells()[i].mode==battery::CellMode::Runaway;
        }
        std::cout<<"batteries="<<simulation.battery_count()<<" passive="<<scenario.cells.size()-simulation.battery_count()
            <<" volumes="<<simulation.thermal_mesh().volumes.size()<<" links="<<simulation.thermal_mesh().links.size()
            <<" t_s="<<simulation.time_s()<<" outer_steps="<<simulation.outer_step_count()<<" substeps="<<simulation.step_count()
            <<" destroyed_batteries="<<spent<<" active_batteries="<<active<<" reached="<<simulation.runaway_order().size()
            <<" tmax_k="<<simulation.max_temperature_k()<<" termination=time_limit"
            <<" wall_s="<<elapsed<<" simulated_s_per_wall_s="<<simulation.time_s()/elapsed<<'\n';

        std::cout << "scenario=" << scenario.name
                  << " propagated=" << (simulation.propagated() ? "true" : "false")
                  << " order=";
        bool first = true;
        for (const auto index : simulation.runaway_order()) {
            if (!first) std::cout << ',';
            first = false;
            std::cout << scenario.cells[index].id;
        }
        std::cout << " residual_j=" << simulation.energy_balance_residual_j() << '\n';

        bool valid = true;
        if (scenario.has_expected_propagation) {
            valid = valid && simulation.propagated() == scenario.expected_propagation;
        }
        if (!scenario.expected_runaway_order.empty()) {
            valid = valid && scenario.expected_runaway_order.size() == simulation.runaway_order().size();
            for (std::size_t index = 0;
                 valid && index < scenario.expected_runaway_order.size(); ++index) {
                valid = scenario.expected_runaway_order[index] ==
                    scenario.cells[simulation.runaway_order()[index]].id;
            }
        }
        valid = valid && std::abs(simulation.energy_balance_residual_j()) < 1.0e-5;
        return valid ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "battery_sim_headless: " << error.what() << '\n';
        return 2;
    }
}

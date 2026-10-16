#include "battery_native/simulation.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc,char** argv) {try {
    if(argc<2)throw std::runtime_error("usage: battery_sim_benchmark SCENARIO [--scale N] [--steps N] [--repeats N] [--field-every N] [--output FILE]");
    int scale=1,steps=500,repeats=3,field_every=0;std::string output;
    for(int i=2;i<argc;i+=2) {
        if(i+1>=argc)throw std::runtime_error("Missing option value");
        std::string key=argv[i];
        if(key=="--scale")scale=std::stoi(argv[i+1]);
        else if(key=="--steps")steps=std::stoi(argv[i+1]);
        else if(key=="--repeats")repeats=std::stoi(argv[i+1]);
        else if(key=="--field-every")field_every=std::stoi(argv[i+1]);
        else if(key=="--output")output=argv[i+1];
        else throw std::runtime_error("Unknown option: "+key);
    }
    if(scale<1||steps<1||repeats<1||field_every<0)throw std::runtime_error("Invalid benchmark dimensions");
    auto scenario=battery::load_scenario(argv[1]);
    battery::refine_environment_mesh(scenario,scale);
    scenario.settings.duration_s=1e8;
    // Synthetic hot field exercises conduction, contact transitions, thermal
    // damage and finite runaway sources. It is NOT a measured experiment.
    for(auto& c:scenario.cells) {
        const auto n=c.thermal_divisions;
        c.initial_temperature_field_k.clear();
        for(int z=0;z<n[2];++z)for(int y=0;y<n[1];++y)for(int x=0;x<n[0];++x)
            c.initial_temperature_field_k.push_back(450+60*std::sin(3.141592653589793*(x+0.5)/n[0]));
    }
    using Clock=std::chrono::steady_clock;
    auto construction_start=Clock::now();battery::Simulation simulation(scenario);
    double setup_s=std::chrono::duration<double>(Clock::now()-construction_start).count();
    nlohmann::json report={{"device",simulation.device_name()},{"scenario",scenario.name},
        {"workload","synthetic-hot-field"},{"scale",scale},{"volumes",simulation.thermal_mesh().volumes.size()},
        {"links",simulation.thermal_mesh().links.size()},{"steps",steps},{"warmup_steps",50},
        {"field_every",field_every},{"setup_s",setup_s},{"samples",nlohmann::json::array()}};
    std::vector<double> elapsed;
    for(int run=0;run<repeats;++run) {
        simulation.reset();for(int i=0;i<50;++i)simulation.step();
        double start_time=simulation.time_s();auto start=Clock::now();
        for(int i=0;i<steps;++i) {
            if(!simulation.step())throw std::runtime_error("Benchmark exhausted simulation");
            if(field_every && (i+1)%field_every==0) (void)simulation.thermal_states();
        }
        double wall=std::chrono::duration<double>(Clock::now()-start).count();elapsed.push_back(wall);
        double simulated=simulation.time_s()-start_time;
        double relative=std::abs(simulation.energy_balance_residual_j())/
            std::max({1.0,simulation.initial_stored_energy_j(),simulation.released_energy_j()});
        if(!std::isfinite(relative)||relative>1e-9)throw std::runtime_error("Benchmark energy balance failed");
        report["samples"].push_back({{"wall_s",wall},{"steps_per_s",steps/wall},
            {"volume_updates_per_s",simulation.thermal_mesh().volumes.size()*static_cast<double>(steps)/wall},
            {"simulated_s",simulated},{"realtime_factor",simulated/wall},{"relative_energy_residual",relative}});
    }
    std::sort(elapsed.begin(),elapsed.end());double median=elapsed[elapsed.size()/2];
    report["median_step_ms"]=median*1000/steps;report["median_steps_per_s"]=steps/median;
    report["median_volume_updates_per_s"]=simulation.thermal_mesh().volumes.size()*static_cast<double>(steps)/median;
    std::cout<<report.dump(2)<<'\n';
    if(!output.empty()) {std::ofstream stream(output);stream<<report.dump(2)<<'\n';if(!stream)throw std::runtime_error("Cannot write benchmark report");}
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}

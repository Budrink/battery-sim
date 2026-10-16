#include "battery_native/gas_experiment.hpp"
#include "battery_native/simulation.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>

using namespace battery;
namespace gx=battery::gas_experiment;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}

int main(int argc,char** argv) {try {
    if(argc!=8)throw std::runtime_error("Usage: gas_patch_benchmark SCENE BACKEND DURATION CAP SAMPLE_INTERVAL REPEATS PATCH_SIZE\n"
        "BACKEND: cpu-global, cuda-global, cpu-local, cuda-local, production");
    const auto scenario=load_scenario(argv[1]);const std::string mode=argv[2];
    const double duration=std::stod(argv[3]),cap=std::stod(argv[4]),interval=std::stod(argv[5]);
    const int repeats=std::stoi(argv[6]),patch_size=std::stoi(argv[7]);
    for(double x:{duration,cap,interval})if(!(x>0) || !std::isfinite(x))throw std::runtime_error("Invalid time controls");
    if(repeats<1 || repeats>20)throw std::runtime_error("Invalid repetitions");
    const bool cpu=mode=="cpu-global" || mode=="cpu-local",local=mode=="cpu-local" || mode=="cuda-local";
    if(!cpu && mode!="cuda-global" && mode!="cuda-local" && mode!="production")throw std::runtime_error("Unknown backend");
    const auto planning=Clock::now();const auto network=gx::make_network(scenario,patch_size);
    Json out={{"backend",mode},{"duration_s",duration},{"max_dt_s",cap},{"sample_interval_s",interval},
        {"repeats",repeats},{"nodes",network.initial.size()},{"edges",network.edges.size()},
        {"patch_size",patch_size},{"patches",network.patches.size()},{"colors",network.colors.size()},
        {"plan_wall_s",seconds(planning)},{"gas_state_bytes",network.initial.size()*sizeof(GasState)},
        {"scope","Fixed gas-only operator experiment, not a coupled thermal/destruction solver"}};
    std::vector<int> sizes;for(const auto& c:network.colors)sizes.push_back(c.size());out["patches_per_color"]=sizes;
    std::ifstream info("/proc/cpuinfo");std::string line;
    while(std::getline(info,line))if(line.starts_with("model name")){out["cpu"]=line.substr(line.find(':')+2);break;}
    const auto setup=Clock::now();std::unique_ptr<gx::Cpu> host;std::unique_ptr<gx::Cuda> device;
    std::unique_ptr<CudaSolver> production;const auto mesh=build_thermal_mesh(scenario);RuntimeParameters runtime;
    if(cpu)host=std::make_unique<gx::Cpu>(network);
    else if(mode=="production")production=std::make_unique<CudaSolver>(scenario,mesh);
    else device=std::make_unique<gx::Cuda>(network);
    out["setup_wall_s"]=seconds(setup);
    // Warm code paths and CUDA initialization, excluded from all measured trials.
    if(host){if(local)host->local_window(1e-12);else host->global_step(1e-12);}
    if(device){if(local)device->local_window(1e-12);else device->global_step(1e-12);device->synchronize();}
    if(production)production->step(0,1e-12,scenario.settings.stability_factor,runtime);
    if(device)out["full_gas_state_roundtrip_s"]=device->state_roundtrip_seconds();
    std::vector<double> timings;Json trials=Json::array();
    std::vector<GasState> first_final;
    for(int repeat=0;repeat<repeats;++repeat) {
        if(host)host->reset();if(device)device->reset();if(production)production->reset();
        Json history={{"time_s",Json::array()},{"gas",Json::object()},{"channels",Json::object()}};
        std::vector<GasState> states;std::vector<gx::Packet> totals;
        double time=0,solve=0,download=0,max_mass_residual=0,max_energy_residual=0,max_chemical_residual=0,max_oxidizer_residual=0;
        unsigned long long steps=0;
        auto sample=[&] {
            const auto start=Clock::now();
            if(host){states=host->states();totals=host->transfers();}
            if(device)device->download(states,totals);
            if(production) {
                std::vector<GasChannelState> channels;production->download_gas_states(states,channels);
                totals.resize(channels.size());
                for(std::size_t j=0;j<channels.size();++j)totals[j]={channels[j].transferred_kg,channels[j].transferred_energy_j,
                    channels[j].transferred_chemical_j,channels[j].transferred_oxidizer_kg};
            }
            download+=seconds(start);
            long double mass=0,energy=0,chemical=0,oxidizer=0;
            history["time_s"].push_back(time);
            for(std::size_t i=0;i<states.size();++i) {
                const auto& s=states[i];const auto& initial=network.initial[i];
                mass+=static_cast<long double>(s.mass_kg)-initial.mass_kg;
                energy+=static_cast<long double>(s.energy_j)-initial.energy_j;
                chemical+=static_cast<long double>(s.chemical_j)-initial.chemical_j;
                oxidizer+=static_cast<long double>(s.oxidizer_kg)-initial.oxidizer_kg;
                auto& row=history["gas"][scenario.gas_compartments[i].id];
                row["mass_kg"].push_back(s.mass_kg);row["energy_j"].push_back(s.energy_j);
                row["pressure_pa"].push_back(s.pressure_pa);row["temperature_k"].push_back(s.temperature_k);
                if(!std::isfinite(s.pressure_pa) || s.mass_kg<=0 || s.temperature_k<=0)throw std::runtime_error("Invalid benchmark state");
            }
            for(std::size_t j=0;j<totals.size();++j) {
                auto& row=history["channels"][scenario.gas_channels[j].id];
                row["transferred_mass_kg"].push_back(totals[j].mass);row["transferred_energy_j"].push_back(totals[j].energy);
            }
            max_mass_residual=std::max(max_mass_residual,static_cast<double>(std::abs(mass)));
            max_energy_residual=std::max(max_energy_residual,static_cast<double>(std::abs(energy)));
            max_chemical_residual=std::max(max_chemical_residual,static_cast<double>(std::abs(chemical)));
            max_oxidizer_residual=std::max(max_oxidizer_residual,static_cast<double>(std::abs(oxidizer)));
        };
        sample();std::size_t tick=1;
        while(duration-time>duration*1e-13) {
            const double target=std::min(duration,tick*interval);const auto start=Clock::now();
            while(target-time>duration*1e-13) {
                const double request=std::min(cap,target-time);double dt=request;
                if(host){if(local)host->local_window(request);else dt=host->global_step(request);}
                if(device){if(local)device->local_window(request);else dt=device->global_step(request);}
                if(production)dt=production->step(time,request,scenario.settings.stability_factor,runtime);
                if(time+dt==time || ++steps>2000000)throw std::runtime_error("Benchmark timestep/count limit");
                time+=dt;
            }
            if(device)device->synchronize();solve+=seconds(start);sample();++tick;
        }
        gx::Work work;if(host)work=host->work();if(device)work=device->work();
        if(production){work.steps=steps;work.edge_evaluations=steps*network.edges.size();}
        double replay_error=0;
        if(repeat==0)first_final=states;
        else for(std::size_t i=0;i<states.size();++i)replay_error=std::max(replay_error,std::abs(states[i].temperature_k-first_final[i].temperature_k));
        trials.push_back({{"solve_wall_s",solve},{"sample_download_s",download},{"steps",work.steps},
            {"edge_evaluations",work.edge_evaluations},{"patch_substeps",work.patch_substeps},{"kernel_launches",work.launches},
            {"max_mass_residual_kg",max_mass_residual},{"max_energy_residual_j",max_energy_residual},
            {"max_chemical_residual_j",max_chemical_residual},{"max_oxidizer_residual_kg",max_oxidizer_residual},
            {"repeat_final_temperature_difference_k",replay_error}});
        timings.push_back(solve);if(repeat==repeats-1)out.update(history);
        std::cerr<<mode<<" repeat "<<repeat+1<<": "<<solve<<" s, steps="<<work.steps<<", edge evaluations="<<work.edge_evaluations<<'\n';
    }
    std::sort(timings.begin(),timings.end());out["solve_wall_median_s"]=timings[timings.size()/2];out["trials"]=trials;
    std::cout<<out.dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

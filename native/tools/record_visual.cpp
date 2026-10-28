#include "battery_native/simulation.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
volatile std::sig_atomic_t stop_requested=0;
void stop(int){stop_requested=1;}
void save_json(const std::filesystem::path& path,const nlohmann::json& value) {
    {std::ofstream f(path.string()+".tmp");f<<value.dump(2)<<'\n';if(!f)throw std::runtime_error("Cannot write recording metadata");}
    std::filesystem::rename(path.string()+".tmp",path);
}
}
// An observer, not a different solver: retain the original 5s integration
// landing points; dense samples observe already committed substeps only.
int main(int argc,char** argv){try {
    if(argc<3 || argc>4)throw std::runtime_error("record_visual SCENARIO OUTPUT [UNTIL_S]");
    const auto source=std::filesystem::absolute(argv[1]);
    const auto output=std::filesystem::absolute(argv[2]);
    if(std::filesystem::exists(output))throw std::runtime_error("Recording destination already exists");
    const auto scenario=battery::load_scenario(source.string());
    std::ifstream input(source);const auto root=nlohmann::json::parse(input);
    const double until=argc==4?std::stod(argv[3]):3000.;
    if(until<=0 || until>scenario.settings.duration_s)throw std::runtime_error("Invalid record horizon");
    const auto times=root.at("failure_sample_times_s").get<std::vector<double>>();
    if(std::none_of(times.begin(),times.end(),[&](double t){return std::abs(t-until)<1e-9;}))
        throw std::runtime_error("End must be an original observation time");
    std::filesystem::create_directories(output);
    std::signal(SIGTERM,stop);std::signal(SIGINT,stop);
    battery::Simulation sim(scenario);
    using J=nlohmann::json;
    J manifest={{"format","BATTVIEW2"},{"scenario_path",source.string()},
        {"source_label",source.parent_path().filename().string()+" / full gas-flow recording"},{"time_s",J::array()},
        {"objects",J::array()},{"channels",J::array()},{"gas",J::array()},
        {"channel_fields",12},{"gas_fields",6},{"events",J::array()},
        {"solid_sampling","object means"},{"gas_sampling","committed control volumes; all accepted graph flows"},
        {"physics_changed",false},{"partial",true},{"validation_passed",false}};
    for(auto& c:scenario.cells)manifest["objects"].push_back(c.id);
    for(auto& c:scenario.gas_channels)manifest["channels"].push_back(c.id);
    for(auto& g:scenario.gas_compartments)manifest["gas"].push_back(g.id);
    std::ofstream frames(output/"frames.bin",std::ios::binary);
    const auto start=std::chrono::steady_clock::now();auto report=start;
    double last=-1e9;
    auto sample=[&] {
        if(sim.time_s()<=last+1e-9)return;
        const auto state=sim.process_diagnostics();
        if(std::abs(state.mass_residual_kg)>1e-10 || std::abs(state.oxidizer_residual_kg)>1e-10 ||
           std::abs(sim.energy_balance_residual_j())>1e-4 || std::abs(sim.chemical_balance_residual_j())>1e-4)
            throw std::runtime_error("Recording stopped: strict conservation ledger failed");
        std::vector<float> values;values.reserve(scenario.cells.size()*4+state.channels.size()*12+state.gas.size()*6);
        for(auto& c:sim.cells())values.insert(values.end(),{float(c.temperature_k),float(c.damage),float(c.chemical_power_w),float(c.chemical_energy_remaining_j)});
        for(auto& c:state.channels) {
            values.insert(values.end(),{float(c.opening_fraction),float(c.effective_area_m2),float(c.mass_flow_kg_s),float(c.aperture_radius_m)});
            for(double strip:c.aperture_strips_m)values.push_back(float(strip));
            values.insert(values.end(),{float(c.transferred_kg),float(c.mixed_mass_kg),float(c.max_thermal_damage),float(c.max_bulk_damage)});
        }
        for(auto& g:state.gas)values.insert(values.end(),{float(g.temperature_k),float(g.chemical_j),float(g.oxidizer_kg),float(g.mass_kg),float(g.pressure_pa),float(g.released_energy_j)});
        frames.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
        frames.flush();if(!frames)throw std::runtime_error("Recording write failed");
        manifest["time_s"].push_back(sim.time_s());last=sim.time_s();
    };
    auto publish=[&] {
        manifest["events"]=J::array();
        for(const auto& e:sim.events())if(e.kind=="shell_rupture" || e.kind=="runaway_primary" || e.kind=="runaway_secondary" || e.kind=="runaway_complete")
            manifest["events"].push_back({{"time_s",e.time_s},{"kind",e.kind},{"id",e.entity_id}});
        manifest["wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        save_json(output/"manifest.json",manifest);
        save_json(output/"status.json",{{"time_s",sim.time_s()},{"until_s",until},{"frames",manifest["time_s"].size()},
            {"wall_s",manifest["wall_s"]},{"partial",manifest["partial"]},{"minimum_regular_dt",sim.profile().minimum_regular_step_s}});
    };
    sample();publish();
    for(double target:times) {
        if(target<=0 || target>until+1e-9 || stop_requested)continue;
        while(sim.time_s()<target-1e-10 && !stop_requested) {
            sim.step(target-sim.time_s());
            if(sim.time_s()>=2380 && sim.time_s()-last>=.5)sample();
            const auto now=std::chrono::steady_clock::now();
            if(std::chrono::duration<double>(now-report).count()>15) {
                publish();report=now;std::cerr<<"Visual recording: "<<sim.time_s()<<" / "<<until<<" s\n";
            }
        }
        sample();
    }
    manifest["partial"]=sim.time_s()<until-1e-8;publish();
    std::cout<<"Recorded "<<sim.time_s()<<" s, "<<manifest["time_s"].size()<<" frames\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

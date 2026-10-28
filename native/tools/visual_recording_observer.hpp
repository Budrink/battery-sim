#pragma once
#include "battery_native/simulation.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>

// BATTVIEW2 observer of accepted states only. Never changes solver landing
// points or invokes a second simulation. Partial captures remain playable.
class VisualRecordingObserver {
    using J=nlohmann::json;
    std::filesystem::path directory_;
    std::ofstream frames_;
    J manifest_;
    double last_=-1e30;
    std::chrono::steady_clock::time_point started_=std::chrono::steady_clock::now();
    void publish(const battery::Simulation& sim,bool finished) {
        manifest_["partial"]=!finished;
        manifest_["events"]=J::array();
        for(const auto& e:sim.events())
            if(e.kind=="shell_rupture" || e.kind=="runaway_primary" ||
               e.kind=="runaway_secondary" || e.kind=="runaway_complete")
                manifest_["events"].push_back({{"time_s",e.time_s},{"kind",e.kind},{"id",e.entity_id}});
        manifest_["wall_s"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started_).count();
        const auto path=directory_/"manifest.json";
        {std::ofstream f(path.string()+".tmp");f<<manifest_.dump(2);if(!f)throw std::runtime_error("Cannot publish visual recording");}
        std::filesystem::rename(path.string()+".tmp",path);
    }
public:
    VisualRecordingObserver(const std::string& path,const std::string& source,const battery::Scenario& s) {
        if(path.empty())return;
        directory_=std::filesystem::absolute(path);
        if(std::filesystem::exists(directory_))throw std::runtime_error("Visual recording destination already exists");
        std::filesystem::create_directories(directory_);
        frames_.open(directory_/"frames.bin",std::ios::binary);
        if(!frames_)throw std::runtime_error("Cannot open visual recording");
        manifest_={{"format","BATTVIEW2"},{"scenario_path",std::filesystem::absolute(source).string()},
            {"source_label",directory_.parent_path().filename().string()},
            {"time_s",J::array()},{"objects",J::array()},{"channels",J::array()},{"gas",J::array()},
            {"channel_fields",12},{"gas_fields",6},{"events",J::array()},
            {"solid_sampling","object means"},{"gas_sampling","committed volumes and accepted graph flows"},
            {"physics_changed",false},{"partial",true},{"validation_passed",false},
            {"dense_sampling_from_s",1600.},{"dense_interval_s",.5}};
        for(const auto& c:s.cells)manifest_["objects"].push_back(c.id);
        for(const auto& c:s.gas_channels)manifest_["channels"].push_back(c.id);
        for(const auto& g:s.gas_compartments)manifest_["gas"].push_back(g.id);
    }
    void observe(const battery::Simulation& sim,bool force=false,bool finished=false) {
        if(directory_.empty())return;
        const double interval=sim.time_s()<1600.?5.:.5;
        if(sim.time_s()>last_+1e-9 && (force || sim.time_s()-last_>=interval)) {
            const auto state=sim.process_diagnostics();
            if(std::abs(state.mass_residual_kg)>1e-10 || std::abs(state.oxidizer_residual_kg)>1e-10 ||
               std::abs(sim.energy_balance_residual_j())>1e-4 || std::abs(sim.chemical_balance_residual_j())>1e-4)
                throw std::runtime_error("Visual recording: strict conservation failed");
            std::vector<float> values;
            values.reserve(sim.cells().size()*4+state.channels.size()*12+state.gas.size()*6);
            for(const auto& c:sim.cells())values.insert(values.end(),{float(c.temperature_k),float(c.damage),float(c.chemical_power_w),float(c.chemical_energy_remaining_j)});
            for(const auto& c:state.channels) {
                values.insert(values.end(),{float(c.opening_fraction),float(c.effective_area_m2),float(c.mass_flow_kg_s),float(c.aperture_radius_m)});
                for(double strip:c.aperture_strips_m)values.push_back(float(strip));
                values.insert(values.end(),{float(c.transferred_kg),float(c.mixed_mass_kg),float(c.max_thermal_damage),float(c.max_bulk_damage)});
            }
            for(const auto& g:state.gas)values.insert(values.end(),{float(g.temperature_k),float(g.chemical_j),float(g.oxidizer_kg),float(g.mass_kg),float(g.pressure_pa),float(g.released_energy_j)});
            frames_.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
            frames_.flush();if(!frames_)throw std::runtime_error("Cannot write visual frame");
            manifest_["time_s"].push_back(sim.time_s());last_=sim.time_s();
            publish(sim,finished);
        }else if(finished)publish(sim,true);
    }
};

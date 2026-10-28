#include "battery_native/model.hpp"
#include "battery_native/thermal_mesh.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Parser bounds only: no GPU evolution or changed convergence criterion.
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    namespace fs=std::filesystem;
    const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path=fs::temp_directory_path()/("battery-pressure-controls-"+std::to_string(stamp)+".json");
    try {
        nlohmann::json root;std::ifstream(argv[1])>>root;
        root["gas_network"]["network_integrator"]="averaged";
        const auto load=[&](const nlohmann::json& input) {
            {std::ofstream out(path);out<<input;}
            const auto scene=battery::load_scenario(path.string());
            battery::build_thermal_mesh(scene);
            return scene.gas_model.averaged_pressure_iterations;
        };
        if(load(root)!=512)throw std::runtime_error("Default pressure budget changed");
        for(int limit:{4,512,1024,2048,8192}) {
            root["gas_network"]["averaged_pressure_iterations"]=limit;
            if(load(root)!=limit)throw std::runtime_error("Valid pressure budget not retained");
        }
        for(int limit:{-1,0,3,8193}) {
            root["gas_network"]["averaged_pressure_iterations"]=limit;
            bool rejected=false;
            try{load(root);}catch(const std::runtime_error&){rejected=true;}
            if(!rejected)throw std::runtime_error("Invalid pressure budget accepted");
        }
        fs::remove(path);
        std::cout<<"Pressure iteration controls: default 512, explicit 4..8192; physical solver unchanged\n";
    }catch(const std::exception& e){fs::remove(path);std::cerr<<e.what()<<'\n';return 1;}
}

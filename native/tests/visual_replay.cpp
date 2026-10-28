#include "battery_native/visual_frame.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char** argv) {try {
    if(argc!=2)throw std::runtime_error("usage: visual_replay_test manifest.json");
    battery::VisualReplay replay(argv[1]);const auto s=battery::load_scenario(replay.scenario_path());
    const auto mesh=battery::build_thermal_mesh(s);const auto geometry=battery::load_visual_geometry(replay.scenario_path(),s);
    auto cold=replay.sample(replay.times().front(),s,mesh);
    require(cold.samples.size()==mesh.volumes.size(),"Solid sample map mismatch");
    require(cold.process.gas.size()==s.gas_compartments.size(),"Gas sample map mismatch");
    require(geometry.gas.size()>100,"Missing resolved exterior gas geometry");
    for(const auto& g:geometry.gas) {
        require(g.size.x>0 && g.size.y>0 && g.size.z>0,"Invalid gas extent");
        require(s.gas_compartments[g.compartment].owner_cell<0,"Internal inventory drawn as exterior plume");
    }
    auto first=std::find_if(replay.events().begin(),replay.events().end(),[](auto& e){return e.kind=="shell_rupture";});
    require(first!=replay.events().end(),"Recording has no rupture");
    auto before=replay.sample(first->time_s-1e-5,s,mesh);
    for(std::size_t i=0;i<s.gas_channels.size();++i)if(s.gas_channels[i].rupture)
        require(before.process.channels[i].effective_area_m2==0,"Aperture anticipated its event");
    const auto end=replay.sample(replay.times().back(),s,mesh);
    int complete=0;for(std::size_t i=0;i<s.cells.size();++i)if(s.cells[i].is_battery)complete+=end.cells[i].mode==battery::CellMode::Spent;
    require(complete==30,"Full saved cascade not reproduced");
    auto reset=replay.sample(replay.times().front(),s,mesh);
    for(std::size_t i=0;i<cold.samples.size();++i)require(cold.samples[i]==reset.samples[i],"Reverse seek retains future state");
    std::cout<<"Recorded-frame contract: 30 spent cells, causal rupture, reversible seek, "<<geometry.gas.size()<<" exterior regions; no CUDA launch\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

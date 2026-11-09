#include "battery_native/data/visual_frame.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
#include <cmath>
void require(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv) {try {
    using namespace battery;
    GasVisualRegion gas; gas.center={0,0,0};gas.size={4,4,4};
    const VisualBox box{{-1,-1,-1},{1,1,1}};
    clip_gas_to_cases(gas,{box});
    require(gas.case_clipped && gas.case_clips.size()==1,"Cross-wall cell must clip, not disappear");
    require(gas.case_clips[0].low==box.low && gas.case_clips[0].high==box.high,"Incorrect wall clipping");
    // Refining exterior cells must not make them visible.
    for(double side:{1.,.1,.001}) {
        gas.center={2,0,0};gas.size={side,side,side};clip_gas_to_cases(gas,{box});
        require(gas.case_clipped && gas.case_clips.empty(),"Fine exterior gas leaked into default view");
    }
    gas.center={0,0,0};gas.size={4,4,4};
    clip_gas_to_cases(gas,{box,box,{{0,-1,-1},{2,1,1}}});
    double volume=0;
    for(const auto& p:gas.case_clips)volume+=(p.high[0]-p.low[0])*(p.high[1]-p.low[1])*(p.high[2]-p.low[2]);
    require(std::abs(volume-12)<1e-10,"Overlapping cases duplicate gas opacity");
    clip_gas_to_cases(gas,{});
    require(!gas.case_clipped,"A scene without a case must retain gas visibility");
    for(int arg=1;arg<argc;++arg) {
        auto root=load_scenario_document(argv[arg]);
        auto scenario=scenario_from_json(root);
        auto geometry=visual_geometry_from_json(root,scenario);
        int inside=0,outside=0;
        for(const auto& g:geometry.gas) {
            require(g.case_clipped,"Fixture case not registered with common display filter");
            (g.case_clips.empty()?outside:inside)++;
        }
        require(inside>0,"Fixture lost all interior gas");
        if(arg==2) require(outside>0,"Imported fixture did not cover exterior gas");
        std::cout<<argv[arg]<<": inside="<<inside<<" hidden exterior="<<outside<<'\n';
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "battery_native/gas_display.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
double distance(battery::Vec3 a,battery::Vec3 b){return std::hypot(a.x-b.x,a.y-b.y,a.z-b.z);}
}
int main(){try {
    using namespace battery;
    Scenario s;s.gas_compartments.resize(2);
    GasChannel channel;channel.from=0;channel.to=1;channel.area_m2=.0001;
    s.gas_channels.push_back(channel);
    ThermalMesh mesh;VisualGeometry geometry;
    GasVisualRegion a;a.compartment=0;a.center={-.01,0,0};a.size={.02,.02,.02};
    GasVisualRegion b=a;b.compartment=1;b.center={.01,0,0};
    geometry.gas={a,b};
    VisualFrame frame;frame.process.gas.resize(2);frame.process.channels.resize(1);
    frame.process.gas[0].temperature_k=800;frame.process.gas[1].temperature_k=400;
    auto& flow=frame.process.channels[0];flow.effective_area_m2=.0001;flow.mass_flow_kg_s=.0001;
    flow.transferred_kg=.0000125;
    GasDisplayModel model(s,mesh,geometry);
    const auto base=model.sample(frame),paused=model.sample(frame);
    require(!base.puffs.empty(),"Accepted hot flow must be visible");
    require(base.puffs.size()==paused.puffs.size(),"Pause changes particle count");
    for(std::size_t i=0;i<base.puffs.size();++i) {
        require(distance(base.puffs[i].position,paused.puffs[i].position)==0,"Paused particles moved");
        require(base.puffs[i].temperature_k>=400 && base.puffs[i].temperature_k<=800,"Flow temperature overshoot");
        require(in_gas_region(a,base.puffs[i].position)||in_gas_region(b,base.puffs[i].position),"Particle outside gas support");
    }
    for(const auto& r:base.regions) {
        require(r.minimum_k==400 && r.maximum_k==800,"Connected gradient bounds lost");
        require(r.temperature_gradient.x<0 && r.temperature_gradient.y==0,"Temperature gradient has wrong direction");
    }
    flow.transferred_kg+=.000005;
    const auto forward=model.sample(frame);
    require(forward.puffs.front().position.x>base.puffs.front().position.x,"Accepted transfer did not advance flow");
    require(forward.regions[0].displacement.x>base.regions[0].displacement.x,"Cloud did not advect");
    flow.transferred_kg-=.00001;flow.mass_flow_kg_s=-.0001;
    const auto reverse=model.sample(frame);
    require(reverse.puffs.front().position.x<base.puffs.front().position.x,"Reversed transfer did not reverse motion");
    frame.known_channels={false};
    require(model.sample(frame).puffs.empty(),"Unrecorded channels invented gas flow");
    frame.known_channels={true};flow.mass_flow_kg_s=0;
    require(model.sample(frame).puffs.empty(),"Zero accepted flow animated particles");
    require(distance(model.sample(frame).regions[0].displacement,reverse.regions[0].displacement)==0,"Stopped cloud jumped backwards");
    s.gas_channels[0].rupture=true;flow.effective_area_m2=0;
    GasDisplayModel barrier(s,mesh,geometry);
    const auto closed=barrier.sample(frame);
    require(closed.puffs.empty() && closed.regions[0].minimum_k==800 && closed.regions[1].maximum_k==400,"Closed barrier leaks interpolation/flow");
    a.excluded_center=a.center;a.excluded_size={.01,.01,.01};
    require(!in_gas_region(a,a.center),"Solid occupied region treated as gas");
    require(!in_gas_region(a,{1,1,1}),"Outside volume treated as gas");
    a.excluded_size={};a.polygon={{{-.02f,-.01f}},{{0,-.01f}},{{-.02f,.01f}}};
    require(!in_gas_region(a,{-.001,.009,0}),"Outside polygon treated as gas");
    std::cout<<"Gas display: pause, signed accepted motion, bounded gradients, unknown flows and solid/barrier masking passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

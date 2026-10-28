#include "battery_native/visual_frame.hpp"
#include "battery_native/gas_display.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv){try {
    if(argc!=2)throw std::runtime_error("visual_cylinders SCENARIO");
    const auto s=battery::load_scenario(argv[1]);
    const auto g=battery::load_visual_geometry(argv[1],s);
    require(g.solid_proxies.size()==s.cells.size(),"Proxy index mapping lost");
    int batteries=0,panels=0;
    for(std::size_t i=0;i<s.cells.size();++i) {
        const auto& c=s.cells[i];const auto& p=g.solid_proxies[i];
        require(c.id==p.id && c.thermal_divisions==p.thermal_divisions,"Numerical sample mapping changed");
        if(c.is_battery) {
            ++batteries;
            require(c.shape==battery::ShapeKind::Prismatic,"Expected unchanged numerical prism");
            require(p.shape==battery::ShapeKind::Cylinder,"Battery not displayed as cylinder");
            require(std::abs(p.dimensions_m[0]-.009)<1e-12 && std::abs(p.dimensions_m[1]-.065)<1e-12,"Wrong display dimensions");
            require(p.position_m.x==c.position_m.x && p.position_m.y==c.position_m.y && p.position_m.z==c.position_m.z,"Battery moved");
        }
        if(g.hidden_planar_shells[i]) {
            ++panels;
            const auto q=battery::project_visual_shell_point(s,g,int(i),c.position_m);
            const auto parent=std::find_if(g.solid_proxies.begin(),g.solid_proxies.end(),[&](const auto& b){return b.id==c.shell_contents_id;});
            const double r=std::hypot(q.x-parent->position_m.x,q.y-parent->position_m.y);
            const double z=std::abs(q.z-parent->position_m.z);
            require(std::abs(r-.009)<1e-10 || std::abs(z-.0325)<1e-10,"Rupture projected off the can");
        }
    }
    require(batteries==30 && panels==180,"Missing batteries or planar shell masks");
    battery::GasVisualRegion region;region.center={0,0,0};region.size={.04,.04,.08};
    region.excluded_cylinders.push_back({{0,0,0},.009,.065});
    require(!battery::in_gas_region(region,{.008,0,0}),"Gas inside round can");
    require(battery::in_gas_region(region,{.008,.008,0}),"Square exclusion erases free corner");
    require(battery::in_gas_region(region,{0,0,.034}),"Gas above lid erased");
    std::cout<<"30 display cylinders 18x65mm; 180 shell projections; circular gas mask; numerical prisms unchanged\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

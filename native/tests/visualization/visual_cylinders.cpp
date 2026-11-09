#include "battery_native/scenario_io.hpp"
#include "battery_native/scenario_document.hpp"
#include <nlohmann/json.hpp>
#include "battery_native/data/visual_frame.hpp"
#include "battery_native/fields/gas_display.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char** argv){try {
    if(argc!=2)throw std::runtime_error("visual_cylinders SCENARIO");
    const auto s=battery::load_scenario(argv[1]);
    const auto g=battery::load_visual_geometry(argv[1],s);
    const auto document=battery::load_scenario_document(argv[1]);
    double radius=.009,height=.065;
    if(document.at("metadata").contains("experiment_contract")) {
        const auto& model=document.at("metadata").at("experiment_contract").at("battery_models")[0];
        radius=model.at("geometry").at("dimensions_m")[0];
        height=model.at("geometry").at("dimensions_m")[1];
    }
    require(g.solid_proxies.size()==s.cells.size(),"Proxy index mapping lost");
    int batteries=0,panels=0;
    for(std::size_t i=0;i<s.cells.size();++i) {
        const auto& c=s.cells[i];const auto& p=g.solid_proxies[i];
        require(c.id==p.id && c.thermal_divisions==p.thermal_divisions,"Numerical sample mapping changed");
        if(c.is_battery) {
            ++batteries;
            require(c.shape==battery::ShapeKind::Prismatic,"Expected unchanged numerical prism");
            require(p.shape==battery::ShapeKind::Cylinder,"Battery not displayed as cylinder");
            require(std::abs(p.dimensions_m[0]-radius)<1e-12 && std::abs(p.dimensions_m[1]-height)<1e-12,"Wrong display dimensions");
            require(p.position_m.x==c.position_m.x && p.position_m.y==c.position_m.y && p.position_m.z==c.position_m.z,"Battery moved");
        }
        if(g.hidden_planar_shells[i]) {
            ++panels;
            const auto q=battery::project_visual_shell_point(s,g,int(i),c.position_m);
            const auto parent=std::find_if(g.solid_proxies.begin(),g.solid_proxies.end(),[&](const auto& b){return b.id==c.shell_contents_id;});
            const double r=std::hypot(q.x-parent->position_m.x,q.y-parent->position_m.y);
            const double z=std::abs(q.z-parent->position_m.z);
            require(std::abs(r-radius)<1e-10 || std::abs(z-height/2)<1e-10,"Rupture projected off the can");
            const int axis=c.id.back()-'x',u=(axis+1)%3,v=(axis+2)%3;
            require(axis>=0 && axis<3,"Expected physical shell face");
            const double center[]={c.position_m.x,c.position_m.y,c.position_m.z};
            auto point=[&](double a,double b) {
                double xyz[]={center[0],center[1],center[2]};
                xyz[u]+=a*c.dimensions_m[u]/2;xyz[v]+=b*c.dimensions_m[v]/2;
                return battery::map_visual_shell_point(s,g,int(i),{xyz[0],xyz[1],xyz[2]}, {},1e-5);
            };
            for(int y=0;y<=8;++y)for(int x=0;x<=8;++x) {
                const auto mapped=point(x/4.-1,y/4.-1);
                const auto n=mapped.normal;
                require(std::abs(n.x*n.x+n.y*n.y+n.z*n.z-1)<1e-10,"Non-unit curved surface normal");
                if(axis==2)
                    require(std::abs(std::abs(mapped.position.z-parent->position_m.z)-height/2-1e-5)<1e-10,"Cap offset lost in projection");
                else
                    require(std::abs(std::hypot(mapped.position.x-parent->position_m.x,mapped.position.y-parent->position_m.y)-radius-1e-5)<1e-10,"Side offset lost in projection");
                if(x<8 && y<8) {
                    const auto a=point((x+1)/4.-1,y/4.-1).position;
                    const auto b=point(x/4.-1,(y+1)/4.-1).position;
                    const auto p=mapped.position;
                    const double ax=a.x-p.x,ay=a.y-p.y,az=a.z-p.z;
                    const double bx=b.x-p.x,by=b.y-p.y,bz=b.z-p.z;
                    const double cx=ay*bz-az*by,cy=az*bx-ax*bz,cz=ax*by-ay*bx;
                    require(cx*cx+cy*cy+cz*cz>1e-22,"Surface mapping collapses rupture triangles");
                }
            }
        }
    }
    require(batteries==30 && panels==180,"Missing batteries or planar shell masks");
    battery::GasVisualRegion region;region.center={0,0,0};region.size={.04,.04,.08};
    region.excluded_cylinders.push_back({{0,0,0},.009,.065});
    require(!battery::in_gas_region(region,{.008,0,0}),"Gas inside round can");
    require(battery::in_gas_region(region,{.008,.008,0}),"Square exclusion erases free corner");
    require(battery::in_gas_region(region,{0,0,.034}),"Gas above lid erased");
    std::cout<<"30 display cylinders; 180 shell projections, normals and noncollapsed patches; circular gas mask; numerical prisms unchanged\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

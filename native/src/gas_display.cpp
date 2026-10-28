#include "battery_native/gas_display.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace battery {
namespace {
// Presentation constants only. They never enter the solver or a calibration.
constexpr float minimum_visible_temperature_k = 323.15f;
Vec3 add(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
Vec3 sub(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
Vec3 mul(Vec3 a,double k){return {a.x*k,a.y*k,a.z*k};}
double dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
double length(Vec3 a){return std::sqrt(dot(a,a));}
Vec3 anchor(const GasVisualRegion& g) {
    if(in_gas_region(g,g.center))return g.center;
    Vec3 best=g.center;double distance=1e30;
    for(int x=0;x<9;++x)for(int y=0;y<9;++y)for(int z=0;z<3;++z) {
        const Vec3 p=add(g.center,{g.size.x*((x+.5)/9.-.5),g.size.y*((y+.5)/9.-.5),g.size.z*((z+.5)/3.-.5)});
        const double d=length(sub(p,g.center));
        if(d<distance && in_gas_region(g,p)){distance=d;best=p;}
    }
    if(distance==1e30)throw std::runtime_error("No visible free gas support");
    return best;
}
float hash(unsigned n){n^=n>>16;n*=0x7feb352d;n^=n>>15;n*=0x846ca68b;n^=n>>16;return (n&0xffffff)/float(0x1000000);}
}
bool in_gas_region(const GasVisualRegion& g,Vec3 p) {
    const auto d=sub(p,g.center);
    if(std::abs(d.x)>g.size.x*.5+1e-9 || std::abs(d.y)>g.size.y*.5+1e-9 || std::abs(d.z)>g.size.z*.5+1e-9)return false;
    const auto e=sub(p,g.excluded_center);
    for(const auto& c:g.excluded_cylinders) {
        const auto q=sub(p,c.center);
        if(q.x*q.x+q.y*q.y<c.radius_m*c.radius_m && std::abs(q.z)<c.height_m*.5)return false;
    }
    if(g.excluded_size.x>0 && std::abs(e.x)<g.excluded_size.x*.5 &&
       std::abs(e.y)<g.excluded_size.y*.5 && std::abs(e.z)<g.excluded_size.z*.5)return false;
    bool positive=false,negative=false;
    for(std::size_t i=0;i<g.polygon.size();++i) {
        const auto a=g.polygon[i],b=g.polygon[(i+1)%g.polygon.size()];
        const double v=(b[0]-a[0])*(p.y-a[1])-(b[1]-a[1])*(p.x-a[0]);
        positive|=v>1e-9;negative|=v< -1e-9;
    }
    return !(positive&&negative);
}
GasDisplayModel::GasDisplayModel(const Scenario& s,const ThermalMesh& mesh,const VisualGeometry& geometry):geometry_(geometry) {
    // Cache topology once, independently of changing temperatures and flows.
    gas_to_region_.assign(s.gas_compartments.size(),-1);
    std::vector<Vec3> anchors;
    for(std::size_t i=0;i<geometry.gas.size();++i){gas_to_region_[geometry.gas[i].compartment]=int(i);anchors.push_back(anchor(geometry.gas[i]));}
    for(std::size_t k=0;k<s.gas_channels.size();++k) {
        const auto& c=s.gas_channels[k];if(c.from<0 || c.to<0)continue;
        const int a=gas_to_region_[c.from],b=gas_to_region_[c.to];
        if(a>=0 && b>=0 && c.area_m2>0)links_.push_back({int(k),a,b,anchors[a],anchors[b],c.area_m2,c.rupture});
        else if(c.rupture && b>=0 && c.shell_object>=0 && c.shell_thermal_volume>=0) {
            const auto port=std::find_if(s.ports.begin(),s.ports.end(),[&](auto& p){return p.id==c.port_id;});
            if(port==s.ports.end())continue;
            const auto& patch=mesh.volumes.at(mesh.object_offsets[c.shell_object]+c.shell_thermal_volume);
            auto p=add(s.cells[c.shell_object].position_m,patch.local_center_m);
            if(std::abs(port->normal.x)>.5)p.x=port->centroid_m.x+port->normal.x*1e-6;
            if(std::abs(port->normal.y)>.5)p.y=port->centroid_m.y+port->normal.y*1e-6;
            if(std::abs(port->normal.z)>.5)p.z=port->centroid_m.z+port->normal.z*1e-6;
            p=project_visual_shell_point(s,geometry,c.shell_object,p);
            links_.push_back({int(k),-1,b,p,anchors[b],gas_channel_area(s,c),true});
        }
    }
    for(auto& link:links_) {
        const double b=s.gas_compartments[geometry.gas[link.to].compartment].initial_mass_kg;
        const double a=link.from>=0?s.gas_compartments[geometry.gas[link.from].compartment].initial_mass_kg:0;
        const double reference=a>0 && b>0?std::min(a,b):std::max(a,b);
        // A fixed reference inventory per link, not a changing current mass:
        // tiny near-cell cavities should not need a whole 0.5g to show motion.
        // Fixed normalization also avoids cloud jumps when the inventory drops.
        if(reference>0)link.display_mass_kg=std::clamp(reference,5e-6,5e-4);
    }
}
GasDisplayState GasDisplayModel::sample(const VisualFrame& frame) const {
    GasDisplayState out;out.regions.resize(geometry_.gas.size());
    struct Sum {Vec3 gradient;double weights[3]{};};std::vector<Sum> sums(geometry_.gas.size());
    for(std::size_t i=0;i<geometry_.gas.size();++i)out.regions[i].minimum_k=out.regions[i].maximum_k=
        float(frame.process.gas.at(geometry_.gas[i].compartment).temperature_k);
    for(const auto& link:links_) {
        const auto& c=frame.process.channels.at(link.channel);
        if(link.rupture && c.effective_area_m2<=0)continue;
        auto& to=out.regions[link.to];
        const auto& gas_b=frame.process.gas[geometry_.gas[link.to].compartment];
        // Reconstruction uses connected neighbours only. Final shader samples
        // are clamped to these bounds, so display interpolation cannot create
        // temperatures outside the input field's local range.
        if(link.from>=0) {
            const auto& gas_a=frame.process.gas[geometry_.gas[link.from].compartment];
            const auto d=sub(geometry_.gas[link.to].center,geometry_.gas[link.from].center);
            const double w=1./std::max(dot(d,d),1e-12),dt=gas_b.temperature_k-gas_a.temperature_k;
            for(int i:{link.from,link.to}) {
                sums[i].gradient=add(sums[i].gradient,mul(d,dt*w));
                sums[i].weights[0]+=d.x*d.x*w;sums[i].weights[1]+=d.y*d.y*w;sums[i].weights[2]+=d.z*d.z*w;
                out.regions[i].minimum_k=std::min({out.regions[i].minimum_k,float(gas_a.temperature_k),float(gas_b.temperature_k)});
                out.regions[i].maximum_k=std::max({out.regions[i].maximum_k,float(gas_a.temperature_k),float(gas_b.temperature_k)});
            }
        }
        if(!frame.known_channels.empty() && !frame.known_channels[link.channel])continue;
        const double flow=c.mass_flow_kg_s;
        // Parcel markers use an explicit DISPLAY reference-inventory scale,
        // not an assertion of resolved gas velocity. Cumulative accepted mass
        // gives deterministic pause/seek and the correct reversal direction.
        const double phase=c.transferred_kg/link.display_mass_kg;
        const auto direction=sub(link.b,link.a);const double route_length=length(direction);
        const auto drift=mul(direction,phase);
        to.displacement=add(to.displacement,mul(drift,.5));
        if(link.from>=0)out.regions[link.from].displacement=add(out.regions[link.from].displacement,mul(drift,.5));
        if(std::abs(flow)<1e-10)continue;
        const float strength=std::clamp(float(std::log1p(std::abs(flow)*1e6)/6.),0.f,1.f);
        const int count=link.rupture?18:8;
        const float radius=std::clamp(float(std::sqrt(std::max(c.effective_area_m2,link.area)/3.14159265)*.4),.0006f,.0035f);
        for(int k=0;k<count;++k) {
            const double raw=phase+k/double(count),u=raw-std::floor(raw);
            auto p=add(link.a,mul(direction,u));
            const unsigned seed=unsigned(link.channel)*7919u+unsigned(k)*109u;
            const double jitter=std::min(double(radius),route_length*.12);
            const auto displaced=add(p,{(hash(seed)-.5)*jitter,(hash(seed+1)-.5)*jitter,(hash(seed+2)-.5)*jitter});
            auto valid=[&](Vec3 x){return in_gas_region(geometry_.gas[link.to],x)||
                (link.from>=0 && in_gas_region(geometry_.gas[link.from],x));};
            if(valid(displaced))p=displaced;else if(!valid(p))continue;
            const double ta=link.from>=0?frame.process.gas[geometry_.gas[link.from].compartment].temperature_k:gas_b.temperature_k;
            const float temperature=float(ta+(gas_b.temperature_k-ta)*u);
            if(temperature<minimum_visible_temperature_k)continue;
            out.puffs.push_back({p,radius,temperature,(.07f+strength*.30f)*float(std::sin(3.14159265*u))});
        }
    }
    for(std::size_t i=0;i<out.regions.size();++i) {
        auto& g=out.regions[i].temperature_gradient;const auto& sum=sums[i];
        g={sum.gradient.x/std::max(sum.weights[0],1.),sum.gradient.y/std::max(sum.weights[1],1.),sum.gradient.z/std::max(sum.weights[2],1.)};
    }
    return out;
}
}

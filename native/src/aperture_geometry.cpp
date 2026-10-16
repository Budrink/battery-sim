#include "battery_native/thermal_mesh.hpp"
#include "battery_native/shell_opening.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace battery {
namespace {
using V = std::array<double,3>;
V xyz(Vec3 p) { return {p.x,p.y,p.z}; }
struct Rectangle { double u0,u1,v0,v1; };
double area(Rectangle r) { return std::max(0.,r.u1-r.u0)*std::max(0.,r.v1-r.v0); }
Rectangle intersection(Rectangle a,Rectangle b) {
    return {std::max(a.u0,b.u0),std::min(a.u1,b.u1),std::max(a.v0,b.v0),std::min(a.v1,b.v1)};
}
// Remove an occluding rectangle, retaining disjoint uncovered rectangles.
// Unlike centre rays this preserves thin obstacles and mixed grid resolutions.
void subtract(Rectangle r,Rectangle cut,std::vector<Rectangle>& out) {
    const auto c=intersection(r,cut);
    if(area(c)<=0) { out.push_back(r);return; }
    for(auto part:{Rectangle{r.u0,c.u0,r.v0,r.v1},Rectangle{c.u1,r.u1,r.v0,r.v1},
                  Rectangle{c.u0,c.u1,r.v0,c.v0},Rectangle{c.u0,c.u1,c.v1,r.v1}})
        if(area(part)>0)out.push_back(part);
}
bool axis_aligned(const CellConfig& c) {
    const auto& q=c.orientation_xyzw;
    return std::abs(q[0])+std::abs(q[1])+std::abs(q[2])<1e-12 && std::abs(q[3])>1e-12;
}
Rectangle footprint(const ThermalVolume& v,const CellConfig& c,int u,int w) {
    const auto x=xyz(v.local_center_m),d=xyz(v.size_m),o=xyz(c.position_m);
    return {o[u]+x[u]-d[u]/2,o[u]+x[u]+d[u]/2,o[w]+x[w]-d[w]/2,o[w]+x[w]+d[w]/2};
}
struct Hit { int volume;double distance;Rectangle rectangle; };
std::vector<Hit> receiving_faces(const Scenario& s,const ThermalMesh& m,int owner,int shell,
                                 int axis,int sign,double plane,Rectangle aperture) {
    std::vector<Hit> hits;
    const int u=(axis+1)%3,v=(axis+2)%3;
    for(std::size_t object=0;object<s.cells.size();++object) {
        if(static_cast<int>(object)==owner || static_cast<int>(object)==shell)continue;
        const auto& cell=s.cells[object];
        const auto position=xyz(cell.position_m);
        const double distance=sign*(position[axis]-plane)-cell.dimensions_m[axis]/2;
        if(distance< -1e-9) {
            const double front=sign*(position[axis]-plane)+cell.dimensions_m[axis]/2;
            const Rectangle bounds{position[u]-cell.dimensions_m[u]/2,position[u]+cell.dimensions_m[u]/2,
                                   position[v]-cell.dimensions_m[v]/2,position[v]+cell.dimensions_m[v]/2};
            if(front>1e-9 && area(intersection(bounds,aperture))>1e-18)
                throw std::runtime_error("A solid overlaps the aperture plane: "+cell.id);
            continue;
        }
        for(int k=m.object_offsets[object];k<m.object_offsets[object+1];++k) {
            const auto& volume=m.volumes[k];
            const auto center=xyz(volume.local_center_m),size=xyz(volume.size_m);
            if(std::abs(center[axis]-sign*size[axis]/2+sign*cell.dimensions_m[axis]/2)>1e-9)continue;
            const auto r=intersection(footprint(volume,cell,u,v),aperture);
            if(area(r)>0)hits.push_back({k,std::max(0.,distance),r});
        }
    }
    std::stable_sort(hits.begin(),hits.end(),[](const Hit& a,const Hit& b){return a.distance<b.distance;});
    return hits;
}
void append_channel(const Scenario& s,ThermalMesh& m,int index) {
    const auto& channel=s.gas_channels[index];
    const auto port=std::find_if(s.ports.begin(),s.ports.end(),[&](const auto& p){return p.id==channel.port_id;});
    if(port==s.ports.end())throw std::runtime_error("Surface link has no aperture port");
    const int owner=s.gas_compartments[channel.from].owner_cell;
    if(channel.to<0 || owner<0 || !channel.opening.enabled)
        throw std::runtime_error("Surface link requires progressive shell opening and an explicit receiving gas compartment");
    const auto& shell=s.cells[channel.shell_object];
    const int axis=port->surface[1]-'x',sign=port->surface[0]=='+'?1:-1,u=(axis+1)%3,v=(axis+2)%3;
    const auto origin=xyz(port->centroid_m);
    // Progressive openings already require a full, one-volume-thick panel.
    const Rectangle aperture{origin[u]-shell.dimensions_m[u]/2,origin[u]+shell.dimensions_m[u]/2,
                             origin[v]-shell.dimensions_m[v]/2,origin[v]+shell.dimensions_m[v]/2};
    const auto hits=receiving_faces(s,m,owner,channel.shell_object,axis,sign,origin[axis],aperture);
    for(int k=m.object_offsets[channel.shell_object];k<m.object_offsets[channel.shell_object+1];++k) {
        const auto& volume=m.volumes[k];const auto center=xyz(volume.local_center_m),size=xyz(volume.size_m);
        const double seam=shell_seam_fraction(center[u],center[v],size[u],size[v],shell.dimensions_m[u],shell.dimensions_m[v],channel.opening.seam_width_m);
        std::vector<Rectangle> remaining{intersection(footprint(volume,shell,u,v),aperture)};
        for(const auto& hit:hits) {
            std::vector<Rectangle> next;
            for(const auto& r:remaining) {
                const double overlap=area(intersection(r,hit.rectangle));
                if(overlap>0) {
                    if(s.cells[m.volumes[hit.volume].object].isothermal)
                        throw std::runtime_error("Aperture surface target requires a spatial solid");
                    // Same panel-local opening closure as the discharge model.
                    // No geometric ray is assigned to more than one receiver.
                    const double fraction=overlap/port->area_m2;
                    m.gas_patches.push_back({hit.volume,channel.to,
                        channel.surface_link_h_w_per_m2_k*gas_channel_area(s,channel)*fraction,index,k,fraction,seam});
                }
                subtract(r,hit.rectangle,next);
            }
            remaining=std::move(next);
            if(remaining.empty())break;
        }
    }
}
} // namespace
void append_aperture_surface_links(const Scenario& s,ThermalMesh& m) {
    bool enabled=false;
    for(const auto& channel:s.gas_channels) {
        if(!std::isfinite(channel.surface_link_h_w_per_m2_k) || channel.surface_link_h_w_per_m2_k<0)
            throw std::runtime_error("Invalid surface-link coefficient");
        enabled|=channel.surface_link_h_w_per_m2_k>0;
    }
    if(!enabled)return;
    // Explicit restriction, never silently ray-trace a rotated bounding box.
    // Other thermal transport retains its existing orientation support.
    for(const auto& cell:s.cells)if(!axis_aligned(cell) || cell.shape==ShapeKind::Cylinder)
        throw std::runtime_error("Aperture surface links currently require axis-aligned rectangular solids");
    for(std::size_t i=0;i<s.gas_channels.size();++i)
        if(s.gas_channels[i].surface_link_h_w_per_m2_k>0)append_channel(s,m,static_cast<int>(i));
}
} // namespace battery

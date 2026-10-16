#include "battery_native/model.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace battery {
void add_physical_shell(Scenario& s,const ShellConfig& spec) {
    auto it=std::find_if(s.cells.begin(),s.cells.end(),[&](const auto& c){return c.id==spec.contents_id;});
    if(it==s.cells.end())throw std::runtime_error("Unknown shell contents: "+spec.contents_id);
    const int core=static_cast<int>(it-s.cells.begin());const auto c=*it;
    for(double x:{spec.thickness_m,spec.density_kg_m3,spec.specific_heat_j_kg_k,
        spec.contents_contact_h,spec.seam_contact_h,spec.damage_scale_k,spec.destroyed_conductance_factor})
        if(!std::isfinite(x)||x<=0)throw std::runtime_error("Physical shell requires positive material/contact data");
    if(c.shape==ShapeKind::Cylinder || c.dimensions_m.size()!=3 || !c.is_battery || spec.material_id.empty() ||
        c.orientation_xyzw!=std::array<double,4>{0,0,0,1})
        throw std::runtime_error("Shell builder currently requires axis-aligned prismatic/pouch contents and an explicit material");
    if(!std::isfinite(spec.damage_rate_per_s)||spec.damage_rate_per_s<0 || !std::isfinite(spec.damage_onset_k)||spec.damage_onset_k<=0)
        throw std::runtime_error("Invalid shell damage law");
    if(!std::isfinite(spec.initial_temperature_k) || (spec.initial_temperature_k!=-1 && spec.initial_temperature_k<=0))
        throw std::runtime_error("Invalid shell initial temperature");
    const bool own_grid=spec.surface_divisions!=std::array<int,3>{0,0,0};
    if(own_grid && std::any_of(spec.surface_divisions.begin(),spec.surface_divisions.end(),[](int n){return n<=0;}))
        throw std::runtime_error("Shell surface_divisions must be three positive counts");
    const double t=spec.thickness_m;const double origin[3]{c.position_m.x,c.position_m.y,c.position_m.z};
    int panels[3][2];
    auto port=[&](int object,int axis,int sign,const double center[3],double du,double dv,const std::string& id) {
        if(std::any_of(s.ports.begin(),s.ports.end(),[&](const auto& p){return p.id==id;}))throw std::runtime_error("Duplicate shell port");
        PortConfig p;p.id=id;p.cell_index=object;p.cell_id=s.cells[object].id;p.kind="contact";
        p.surface=std::string(sign>0?"+":"-")+"xyz"[axis];p.area_m2=du*dv;p.rectangle_m={du,dv};
        p.centroid_m={center[0],center[1],center[2]};
        if(axis==0)p.normal.x=sign;else if(axis==1)p.normal.y=sign;else p.normal.z=sign;
        s.ports.push_back(p);return id;
    };
    auto link=[&](int a,int b,const std::string& pa,const std::string& pb,double area,double h) {
        InterfaceConfig e;e.id=pa+"--"+pb;e.cell_a=a;e.cell_b=b;e.port_a=pa;e.port_b=pb;
        e.material_a=s.cells[a].material_id;e.material_b=s.cells[b].material_id;
        e.interface_material="physical-casing-contact";
        e.point_a_m=std::find_if(s.ports.begin(),s.ports.end(),[&](const auto& p){return p.id==pa;})->centroid_m;
        e.point_b_m=std::find_if(s.ports.begin(),s.ports.end(),[&](const auto& p){return p.id==pb;})->centroid_m;
        auto& l=e.law;l.background_conductance_w_per_k=h*area;l.endpoint_damage_weight=1.;
        l.destroyed_background_conductance_w_per_k=h*area*spec.destroyed_conductance_factor;
        l.l0_m=l.l_low_m=l.l_high_m=l.destroyed_l0_m=l.destroyed_l_m=t;
        e.initial_l_m=t;s.interfaces.push_back(e);
    };
    for(int axis=0;axis<3;++axis)for(int side=0;side<2;++side) {
        int sign=side?1:-1;CellConfig panel;panel.id=c.id+":case:"+(side?"+":"-")+"xyz"[axis];
        if(std::any_of(s.cells.begin(),s.cells.end(),[&](const auto& x){return x.id==panel.id;}))throw std::runtime_error("Duplicate physical shell");
        panel.is_battery=false;panel.material_id=spec.material_id;panel.shell_contents_id=c.id;
        panel.dimensions_m=c.dimensions_m;panel.thermal_divisions=c.thermal_divisions;panel.thermal_divisions[axis]=1;
        if(own_grid) {panel.thermal_divisions=spec.surface_divisions;panel.thermal_divisions[axis]=1;}
        for(int k=0;k<3;++k)panel.dimensions_m[k]=k==axis?t:c.dimensions_m[k]+(k>axis?2*t:0.);
        double pos[3]{origin[0],origin[1],origin[2]};pos[axis]+=sign*(c.dimensions_m[axis]+t)/2;
        panel.position_m={pos[0],pos[1],pos[2]};panel.initial_temperature_k=spec.initial_temperature_k<0?c.initial_temperature_k:spec.initial_temperature_k;
        panel.heat_capacity_j_per_k=spec.density_kg_m3*spec.specific_heat_j_kg_k;
        for(double d:panel.dimensions_m)panel.heat_capacity_j_per_k*=d;
        panel.conductivity_w_per_m_k=spec.conductivity_w_per_m_k;
        panel.damage_onset_k=spec.damage_onset_k;panel.damage_rate_at_onset_per_s=spec.damage_rate_per_s;panel.damage_temperature_scale_k=spec.damage_scale_k;
        panel.has_bulk_connectivity=true;auto& b=panel.bulk_connectivity;
        b.background_conductance_w_per_k=1.;b.destroyed_background_conductance_w_per_k=spec.destroyed_conductance_factor;
        b.endpoint_damage_weight=1.;b.l0_m=b.l_low_m=b.destroyed_l0_m=b.destroyed_l_m=t;
        int pi=static_cast<int>(s.cells.size());panels[axis][side]=pi;s.cells.push_back(panel);
        const int u=(axis+1)%3,v=(axis+2)%3;double center[3]{origin[0],origin[1],origin[2]};center[axis]+=sign*c.dimensions_m[axis]/2;
        const auto cp=port(core,axis,sign,center,c.dimensions_m[u],c.dimensions_m[v],panel.id+":contents");
        const auto ip=port(pi,axis,-sign,center,c.dimensions_m[u],c.dimensions_m[v],panel.id+":inner");
        link(core,pi,cp,ip,c.dimensions_m[u]*c.dimensions_m[v],spec.contents_contact_h);
        center[axis]+=sign*t;
        port(pi,axis,sign,center,panel.dimensions_m[u],panel.dimensions_m[v],panel.id+":outer");
    }
    // Exact shared rectangles; x panels own corners, then y, then z. No overlap
    // or missing corner mass. These seams allow heat to travel around contents.
    for(int a=0;a<3;++a)for(int b=a+1;b<3;++b)for(int sa=0;sa<2;++sa)for(int sb=0;sb<2;++sb) {
        int ia=panels[a][sa],ib=panels[b][sb],as=sa?1:-1;
        const auto& cb=s.cells[ib];double center[3]{cb.position_m.x,cb.position_m.y,cb.position_m.z};
        center[a]=origin[a]+as*c.dimensions_m[a]/2;
        const int u=(a+1)%3,v=(a+2)%3;double du=cb.dimensions_m[u],dv=cb.dimensions_m[v];
        std::string label=c.id+":seam:"+std::to_string(a)+std::to_string(b)+std::to_string(sa)+std::to_string(sb);
        auto pa=port(ia,a,-as,center,du,dv,label+":a");auto pb=port(ib,a,as,center,du,dv,label+":b");
        link(ia,ib,pa,pb,du*dv,spec.seam_contact_h);
    }
}
} // namespace battery

#pragma once
#include "battery_native/thermal_mesh.hpp"
#include <stdexcept>

namespace battery {
// Separate electrical geometry: no thermal contact, TIM or casing edge leaks
// into this graph. A terminal is an ideal equipotential full face, not a voxel.
struct ElectricalEdge {
    int a,b; // b < 0: terminal; potential is 1 at inlet, 0 at outlet
    double half_a,half_b,terminal_potential;
};
struct ElectricalMesh {
    std::vector<ElectricalEdge> edges;
    std::vector<int> objects,edge_offsets,offsets,adjacency;
    std::vector<double> initial_potential;
};
inline ElectricalMesh build_electrical_mesh(const Scenario& s,const ThermalMesh& mesh) {
    ElectricalMesh out;std::vector<std::vector<int>> rows(mesh.volumes.size());
    out.initial_potential.resize(mesh.volumes.size(),0.);out.edge_offsets.push_back(0);
    for(std::size_t object=0;object<s.cells.size();++object) {
        const auto& c=s.cells[object];
        if(!c.electrical.enabled)continue;
        out.objects.push_back(static_cast<int>(object));
        const auto n=c.thermal_divisions;const int axis=c.electrical.terminal_axis;
        const auto& d=c.dimensions_m;const double length=d[axis],area=d[(axis+1)%3]*d[(axis+2)%3];
        const int begin=mesh.object_offsets[object];
        if(mesh.object_offsets[object+1]-begin!=n[0]*n[1]*n[2])throw std::runtime_error("Electrical geometry requires regular core mesh");
        const double h[3]{d[0]/n[0],d[1]/n[1],d[2]/n[2]};
        auto edge=[&](int a,int b,double half,double potential) {
            const int id=static_cast<int>(out.edges.size());out.edges.push_back({a,b,half,b>=0?half:0.,potential});
            rows[a].push_back(id);if(b>=0)rows[b].push_back(id);
        };
        for(int z=0;z<n[2];++z)for(int y=0;y<n[1];++y)for(int x=0;x<n[0];++x) {
            const int xyz[]{x,y,z},stride[]{1,n[0],n[0]*n[1]};const int i=begin+x+n[0]*(y+n[1]*z);
            out.initial_potential[i]=1.-(xyz[axis]+.5)/n[axis];
            for(int a=0;a<3;++a) {
                // R_half = R_whole(T,composition) * geometric half factor.
                const double half=area/length*h[a]/(2.*h[(a+1)%3]*h[(a+2)%3]);
                if(xyz[a]+1<n[a])edge(i,i+stride[a],half,0.);
                if(a==axis && xyz[a]==0)edge(i,-1,half,1.);
                if(a==axis && xyz[a]+1==n[a])edge(i,-1,half,0.);
            }
        }
        out.edge_offsets.push_back(static_cast<int>(out.edges.size()));
    }
    out.offsets.push_back(0);
    for(const auto& row:rows) {out.adjacency.insert(out.adjacency.end(),row.begin(),row.end());out.offsets.push_back(static_cast<int>(out.adjacency.size()));}
    return out;
}
struct ElectricalState {
    double time_s=0.,resistance_ohm=0.,current_a=0.,outlet_current_a=0.;
    double ohmic_drop_v=0.,power_w=0.,kcl_relative_residual=0.,power_relative_residual=0.;
    int iterations=0,error=0;
    // Transactional drive ledger, used ONLY for the explicit setpoint mode.
    // Candidate intervals replace, rather than accumulate on, rejected trials.
    double positive_charge_c=0.,terminal_work_j=0.,voltage_limited_time_s=0.;
    double step_charge_c=0.,step_i2_a2_s=0.,step_work_j=0.;
};
} // namespace battery

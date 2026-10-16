#include "battery_native/mass_mesh.hpp"
#include <cmath>
#include <stdexcept>

namespace battery {
MassMesh build_mass_mesh(const Scenario& s,const ThermalMesh& m) {
    MassMesh out;
    out.bindings.resize(s.gas_compartments.size());
    out.gas_of_thermal_volume.assign(m.volumes.size(),-1);
    out.source_gas.assign(m.volumes.size(),-1);
    for(std::size_t i=0;i<s.gas_compartments.size();++i) {
        const auto& g=s.gas_compartments[i];auto& b=out.bindings[i];
        if(g.thermal_cell< -1)throw std::runtime_error("Invalid local material index");
        if(g.thermal_cell<0) {
            if(g.reactant_density_kg_m3!=0 || g.residue_density_kg_m3!=0)
                throw std::runtime_error("Occupied-volume law requires a local material binding");
            continue;
        }
        if(!s.gas_model.split_transport)throw std::runtime_error("Local gas/material mixtures require split_local_equilibrium transport");
        const auto& c=s.cells.at(g.thermal_cell);
        if(g.owner_cell>=0 && g.owner_cell!=g.thermal_cell)
            throw std::runtime_error("Gas owner and local material binding disagree");
        if(c.isothermal)throw std::runtime_error("Local mixtures require finite_volume material cells");
        const int begin=m.object_offsets.at(g.thermal_cell),end=m.object_offsets.at(g.thermal_cell+1);
        if(g.thermal_volume<0 || begin+g.thermal_volume>=end)throw std::runtime_error("Invalid local mixture volume");
        b.thermal_volume=begin+g.thermal_volume;const auto& v=m.volumes[b.thermal_volume];
        if(out.gas_of_thermal_volume[b.thermal_volume]>=0)throw std::runtime_error("Duplicate mobile phase in a material volume");
        out.gas_of_thermal_volume[b.thermal_volume]=static_cast<int>(i);
        b.geometric_volume_m3=v.size_m.x*v.size_m.y*v.size_m.z;
        if(g.volume_m3>b.geometric_volume_m3*(1+1e-10))throw std::runtime_error("Gas pore volume exceeds geometric volume");
        if(std::abs(g.initial_temperature_k-v.initial_temperature_k)>1e-8)
            throw std::runtime_error("Local equilibrium requires a common initial gas/material temperature");
        const double rd=g.reactant_density_kg_m3,pd=g.residue_density_kg_m3;
        if(!std::isfinite(rd)||!std::isfinite(pd)||rd<0||pd<0||(rd==0)!=(pd==0))
            throw std::runtime_error("Specify both positive inventory densities, or neither");
        b.initial_reactant_kg=c.conversion.enabled?c.conversion.mass_kg*v.capacity_j_per_k/c.heat_capacity_j_per_k:0.;
        b.reactant_density=rd;b.residue_density=pd;
        if(rd>0 && (!c.conversion.enabled || g.volume_m3+b.initial_reactant_kg/rd>b.geometric_volume_m3*(1+1e-10)))
            throw std::runtime_error("Gas and initial condensed inventory do not fit in material volume");
    }
    for(std::size_t i=0;i<m.volumes.size();++i) {
        const auto& c=s.cells[m.volumes[i].object].conversion;
        if(c.enabled)out.source_gas[i]=out.gas_of_thermal_volume[i]>=0?out.gas_of_thermal_volume[i]:c.gas_compartment;
    }
    if(s.gas_model.split_transport) {
        for(const auto& c:s.gas_channels) {
            if(c.surface_link_h_w_per_m2_k>0)
                throw std::runtime_error("Split mass transport forbids legacy heat-only aperture shortcuts; route mass to a receiving pore node");
            if(c.internal) {
                const auto& a=m.volumes.at(out.bindings.at(c.from).thermal_volume);
                const auto& b=m.volumes.at(out.bindings.at(c.to).thermal_volume);
                const double delta[3]{std::abs(a.local_center_m.x-b.local_center_m.x),std::abs(a.local_center_m.y-b.local_center_m.y),std::abs(a.local_center_m.z-b.local_center_m.z)};
                const double size[3]{a.size_m.x,a.size_m.y,a.size_m.z};int axis=-1;
                for(int k=0;k<3;++k)if(delta[k]>1e-10) {
                    if(axis>=0 || std::abs(delta[k]-size[k])>1e-10)throw std::runtime_error("Nonadjacent internal pore path");
                    axis=k;
                }
                if(axis<0 || c.area_m2>size[(axis+1)%3]*size[(axis+2)%3]*(1+1e-10))
                    throw std::runtime_error("Internal pore path exceeds common face");
            }
            // Local source bindings must not circumvent the casing by omitting
            // the legacy owner_cell field. Check both ends, including backflow.
            for(int endpoint:{c.from,c.to}) {
                if(endpoint<0)continue;
                const int object=s.gas_compartments.at(endpoint).thermal_cell;
                if(object<0 || !s.cells[object].is_battery)continue;
                const int other=endpoint==c.from?c.to:c.from;
                if(other>=0 && s.gas_compartments[other].thermal_cell==object)continue;
                bool enclosed=false;
                for(const auto& cell:s.cells)enclosed|=cell.shell_contents_id==s.cells[object].id;
                if(enclosed && s.cells.at(c.shell_object).shell_contents_id!=s.cells[object].id)
                    throw std::runtime_error("Local mass channel bypasses the battery's physical casing");
            }
        }
        for(const auto& patch:m.gas_patches)
            if(out.gas_of_thermal_volume[patch.volume]==patch.compartment)
                throw std::runtime_error("Local LTE must not also have a gas/material film contact");
    }
    return out;
}
}

#include "battery_native/kawasaki_gas_math.hpp"
#include <algorithm>
#include <stdexcept>

namespace battery::kawasaki {
void validate(const Network& n) {
    const auto& s=n.settings;
    auto positive=[](double x){return std::isfinite(x) && x>0;};
    if(n.initial.empty() || !positive(s.cv) || !positive(s.R) || !positive(s.packet_fraction) ||
       s.packet_fraction>1 || !positive(s.equilibrium_atol_pa) || !positive(s.equilibrium_rtol) ||
       !std::isfinite(s.tail_atol_pa) || s.tail_atol_pa<0 || !positive(s.tail_rtol) || s.tail_rtol>1e-2 ||
       !std::isfinite(s.thermal_stirring_coefficient)||s.thermal_stirring_coefficient<0.||!positive(s.mixing_length_cap_m)||
       !positive(s.thermal_stirring_length_m))
        throw std::invalid_argument("Invalid Kawasaki gas controls");
    for(const auto& v:n.initial)if(!positive(v.mass) || !positive(v.energy) || !positive(v.volume) ||
        !positive(temperature(v,s)) || !positive(pressure(v,s)) ||
        !(std::isfinite(v.solid_capacity) && v.solid_capacity>=0 && std::isfinite(v.chemical) && v.chemical>=0 &&
          std::isfinite(v.oxidizer) && v.oxidizer>=0 && v.oxidizer<=v.mass))
        throw std::invalid_argument("Invalid Kawasaki inventory; vacuum/variable capacity not supported yet");
    for(const auto& e:n.edges)if(e.a<0 || e.b<0 || e.a==e.b || e.a>=int(n.initial.size()) || e.b>=int(n.initial.size()) ||
        !std::isfinite(e.area) || e.area<0 || !std::isfinite(e.cd) || e.cd<0 ||
        !positive(e.mixing_length) || !positive(e.hydraulic_resistance_factor) || e.geometry_axis<0 || e.geometry_axis>2)
        throw std::invalid_argument("Invalid Kawasaki edge");
    std::vector<int> seen(n.edges.size());
    for(const auto& color:n.colors) {
        std::vector<bool> occupied(n.initial.size());
        for(int id:color) {
            if(id<0 || id>=int(n.edges.size()) || ++seen[id]!=1)throw std::invalid_argument("Invalid pair coverage");
            const auto& e=n.edges[id];
            if(occupied[e.a] || occupied[e.b])throw std::invalid_argument("Pairs share a writable endpoint");
            occupied[e.a]=occupied[e.b]=true;
        }
    }
    for(int count:seen)if(count!=1)throw std::invalid_argument("Missing pair");
}
void schedule(Network& n) {
    // Validate incidence before indexing it. Temporary singleton colors suffice.
    n.colors.clear();for(int id=0;id<int(n.edges.size());++id)n.colors.push_back({id});validate(n);
    n.colors.clear();std::vector<std::vector<bool>> occupied;
    for(int id=0;id<int(n.edges.size());++id) {
        const auto& e=n.edges[id];int c=0;
        for(;c<int(occupied.size());++c)if(!occupied[c][e.a] && !occupied[c][e.b])break;
        if(c==int(occupied.size())){occupied.emplace_back(n.initial.size(),false);n.colors.emplace_back();}
        occupied[c][e.a]=occupied[c][e.b]=true;n.colors[c].push_back(id);
    }
    validate(n);
}
} // namespace battery::kawasaki

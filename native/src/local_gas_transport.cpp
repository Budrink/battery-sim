#include "battery_native/local_gas_transport.hpp"
#include "battery_native/mixture_equilibrium.hpp"
#include "battery_native/gas_timestep.hpp"
#include "battery_native/gas_combustion.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

namespace battery {
namespace {
struct Packet {double mass=0,energy=0,chemical=0,oxidizer=0;};
struct Rate {double flow=0,cap=0,dt=1e300;};
void apply(GasState& g,const Packet& q,double sign) {
    g.mass_kg+=sign*q.mass;g.energy_j+=sign*q.energy;
    g.chemical_j+=sign*q.chemical;g.oxidizer_kg+=sign*q.oxidizer;
}
}
LocalGasTransport::LocalGasTransport(const Scenario& s,const ThermalMesh& t,const MassMesh& m):
    scenario_(s),thermal_(t),mass_(m) {
    std::vector<std::pair<int,int>> ends;
    for(const auto& c:s.gas_channels)ends.emplace_back(c.from,c.to);
    partition_=partition_gas(s.gas_compartments.size(),ends);
}
double LocalGasTransport::capacity(int id,const GasState& g,const std::vector<ThermalState>& solid) const {
    double c=g.mass_kg*scenario_.gas_model.cv_j_per_kg_k;const int j=mass_.bindings[id].thermal_volume;
    if(j>=0) {
        const auto& v=thermal_.volumes[j];const auto& m=scenario_.cells[v.object];
        c+=m.conversion.enabled?conversion_capacity(m.conversion,solid[j],v.capacity_j_per_k,v.capacity_j_per_k/m.heat_capacity_j_per_k):
            v.capacity_j_per_k*capacity_factor(m.heat_capacity_law,solid[j].temperature_k,solid[j].damage);
    }
    return c;
}
void LocalGasTransport::equilibrate(int id,GasState& g,std::vector<ThermalState>& solid) const {
    const auto& model=scenario_.gas_model;const int j=mass_.bindings[id].thermal_volume;
    if(j>=0) {
        auto& s=solid[j];const auto& v=thermal_.volumes[j];const auto& m=scenario_.cells[v.object];
        const double total=s.stored_energy_j+g.energy_j;
        const double t=mixture_temperature(m.heat_capacity_law,m.conversion,s,g,v.capacity_j_per_k,
            v.capacity_j_per_k/m.heat_capacity_j_per_k,model);
        g.temperature_k=s.temperature_k=t;g.energy_j=g.mass_kg*model.cv_j_per_kg_k*(t-model.reference_temperature_k);
        s.stored_energy_j=total-g.energy_j;s.mobile_capacity_j_per_k=g.mass_kg*model.cv_j_per_kg_k;
    } else g.temperature_k=model.reference_temperature_k+g.energy_j/(g.mass_kg*model.cv_j_per_kg_k);
    g.pressure_pa=g.mass_kg*model.specific_r_j_per_kg_k*g.temperature_k/g.free_volume_m3;
    if(!(g.mass_kg>0 && g.temperature_k>0 && std::isfinite(g.pressure_pa)) ||
       g.chemical_j< -1e-10 || g.oxidizer_kg< -1e-14)
        throw std::runtime_error("Invalid local CPU gas state");
}
void LocalGasTransport::group(const GasLocalGroup& p,double duration,std::vector<GasState>& gas,
    std::vector<ThermalState>& solid,std::vector<GasChannelState>& channels,GasTransportStats& stats) const {
    ++stats.groups;
    const auto& model=scenario_.gas_model;const double r=model.specific_r_j_per_kg_k,cv=model.cv_j_per_kg_k;
    std::array<GasState,9> state,next;std::array<Rate,64> rates;std::array<Packet,64> totals{};
    for(int i=0;i<int(p.nodes.size());++i)state[i]=gas[p.nodes[i]];
    double elapsed=0;int count=0;
    while(elapsed<duration) {
        double dt=duration-elapsed;std::array<double,9> outgoing{},power{},caps{};
        for(int i=0;i<int(p.nodes.size());++i)caps[i]=capacity(p.nodes[i],state[i],solid);
        bool flowing=false;
        for(int j=0;j<int(p.edges.size());++j) {
            rates[j]={};const auto& c=scenario_.gas_channels[p.edges[j]];const auto& ch=channels[p.edges[j]];
            if(!ch.open || ch.effective_area_m2<=0)continue;
            const auto [a,b]=p.ends[j];const double pb=b<0?c.outside_pressure_pa:state[b].pressure_pa;
            const bool forward=state[a].pressure_pa>=pb;
            if(b<0 && !forward)continue; // external reservoir is outflow-only
            const int d=forward?a:b,v=forward?b:a;const auto& donor=state[d];
            auto& q=rates[j];q.flow=(forward?1.:-1.)*gas_orifice_flow(fmax(state[a].pressure_pa,pb),
                fmin(state[a].pressure_pa,pb),donor.temperature_k,ch.effective_area_m2,c.discharge_coefficient,model);
            if(q.flow==0)continue;
            flowing=true;
            const double sd=r*donor.temperature_k*(1+donor.mass_kg*r/caps[d])/donor.free_volume_m3;
            double sr=0.;
            if(v>=0) {
                const auto& recv=state[v];const double f=recv.mass_kg*cv/caps[v];
                sr=r*((1-f)*recv.temperature_k+f*(1+r/cv)*donor.temperature_k)/recv.free_volume_m3;
            }
            q.cap=fabs(state[a].pressure_pa-pb)/(p.degree[d]*sd+(v>=0?p.degree[v]*sr:0));
            q.dt=gas_pressure_dt(state[a].pressure_pa,pb,q.flow,q.cap,model.pressure_relaxation_fraction,
                model.pressure_absolute_tolerance_pa,model.pressure_relative_tolerance);
            dt=fmin(dt,q.dt);outgoing[d]+=fabs(q.flow);power[d]+=fabs(q.flow)*r*donor.temperature_k;
            if(v>=0)power[v]+=fabs(q.flow)*fabs(gas_enthalpy(donor,model)-cv*(state[v].temperature_k-model.reference_temperature_k));
        }
        if(!flowing)break;
        for(int i=0;i<int(p.nodes.size());++i) {
            if(outgoing[i]>0)dt=fmin(dt,model.max_fraction_per_step*state[i].mass_kg/outgoing[i]);
            if(power[i]>0)dt=fmin(dt,fmin(model.max_temperature_step_k,.05*state[i].temperature_k)*caps[i]/power[i]);
        }
        if(!(dt>0) || elapsed+dt==elapsed || ++count>2000000)throw std::runtime_error("CPU local gas substep budget exhausted");
        next=state;
        for(int j=0;j<int(p.edges.size());++j) {
            const auto& q=rates[j];if(q.flow==0)continue;
            auto [a,b]=p.ends[j];const auto& donor=state[q.flow>=0?a:b];
            const double dm=copysign(fmin(fabs(q.flow)*dt,q.cap),q.flow);
            const Packet packet{dm,dm*gas_enthalpy(donor,model),dm*donor.chemical_j/donor.mass_kg,dm*donor.oxidizer_kg/donor.mass_kg};
            apply(next[a],packet,-1);if(b>=0)apply(next[b],packet,1);
            auto& sum=totals[j];sum.mass+=packet.mass;sum.energy+=packet.energy;sum.chemical+=packet.chemical;sum.oxidizer+=packet.oxidizer;
        }
        for(int i=0;i<int(p.nodes.size());++i)equilibrate(p.nodes[i],next[i],solid);
        state=next;elapsed+=dt;
        ++stats.explicit_substeps;
    }
    for(int i=0;i<int(p.nodes.size());++i)gas[p.nodes[i]]=state[i];
    for(int j=0;j<int(p.edges.size());++j) {
        auto& c=channels[p.edges[j]];const auto& q=totals[j];
        c.transferred_kg+=q.mass;c.transferred_energy_j+=q.energy;
        c.transferred_chemical_j+=q.chemical;c.transferred_oxidizer_kg+=q.oxidizer;
        c.mass_flow_kg_s+=q.mass;
    }
}
void LocalGasTransport::advance(std::vector<GasState>& gas,std::vector<ThermalState>& solid,
    std::vector<GasChannelState>& channels,double duration,GasTransportStats* counters) const {
    GasTransportStats scratch;auto& stats=counters?*counters:scratch;
    if(!(duration>0 && std::isfinite(duration)))throw std::invalid_argument("Invalid local gas interval");
    const auto reaction_input=gas;
    for(auto& c:channels)c.mass_flow_kg_s=0;
    double elapsed=0;
    while(elapsed<duration) {
        const double h=std::min(duration-elapsed,scenario_.gas_model.local_exchange_interval_s);
        // Each edge acts for h, not color_count*h. Local Euler integration is
        // first order; the palindromic ordering does not imply second order.
        ++stats.windows;
        const auto evolve=[&](int id) {
            if(scenario_.gas_model.implicit_local_transport)implicit_group(partition_.groups[id],h/2,gas,solid,channels,stats);
            else group(partition_.groups[id],h/2,gas,solid,channels,stats);
        };
        for(const auto& color:partition_.colors)for(int id:color)evolve(id);
        for(auto it=partition_.colors.rbegin();it!=partition_.colors.rend();++it)
            for(int id:*it)evolve(id);
        if(elapsed+h==elapsed)throw std::runtime_error("Local exchange interval underflow");
        elapsed+=h;
    }
    for(auto& c:channels)c.mass_flow_kg_s/=duration;
    // Preserve the existing old-state kinetic proposal. Newly arriving fuel
    // cannot enlarge it, and exported stocks cannot be spent twice.
    for(int i=0;i<int(gas.size());++i) {
        auto& g=gas[i];const auto& c=scenario_.gas_compartments[i];
        const auto& old=reaction_input[i];
        const double rate=combustion_rate(gas_combustion_law(c),old.temperature_k);
        if(rate>0) {
            const double available=fmin(old.chemical_j,old.oxidizer_kg*c.oxidizer_heat_j_per_kg);
            const double q=fmax(0.,fmin(available*(-expm1(-rate*duration)),
                fmin(g.chemical_j,g.oxidizer_kg*c.oxidizer_heat_j_per_kg)));
            g.chemical_j-=q;g.energy_j+=q;g.released_energy_j+=q;
            g.oxidizer_kg-=q/c.oxidizer_heat_j_per_kg;g.consumed_oxidizer_kg+=q/c.oxidizer_heat_j_per_kg;
        }
        equilibrate(i,g,solid);
    }
}
}

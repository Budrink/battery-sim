#include "battery_native/gas_experiment_math.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace battery::gas_experiment {
Network make_network(const Scenario& s,int patch_size) {
    if(patch_size<2 || patch_size>max_patch_nodes)throw std::runtime_error("Patch size must be 2..9");
    if(!s.gas_model.split_transport || !s.interfaces.empty() || !s.boundaries.empty() ||
       !s.gas_contacts.empty() || !s.primary_failures.empty() || s.gas_compartments.empty())
        throw std::runtime_error("Experiment requires an isolated closed gas-only scenario");
    for(const auto& c:s.cells)if(c.is_battery || c.conversion.enabled || c.runaway_energy_j!=0 || c.damage_rate_at_onset_per_s!=0)
        throw std::runtime_error("Active solid in gas-only experiment");
    // Validate native numeric controls and gas geometry without initializing CUDA.
    (void)build_thermal_mesh(s);
    Network n;n.model=s.gas_model;
    for(const auto& c:s.gas_compartments) {
        if(c.owner_cell>=0 || c.thermal_cell>=0 || c.reaction_rate_per_s!=0)
            throw std::runtime_error("LTE, gas production or chemistry is not supported by this experiment");
        GasState g;g.mass_kg=c.initial_mass_kg;g.free_volume_m3=c.volume_m3;
        g.energy_j=g.mass_kg*n.model.cv_j_per_kg_k*(c.initial_temperature_k-n.model.reference_temperature_k);
        g.chemical_j=c.initial_chemical_j;g.oxidizer_kg=g.mass_kg*c.initial_oxidizer_mass_fraction;
        refresh(g,n.model);n.initial.push_back(g);
    }
    std::vector<std::vector<int>> rows(n.initial.size());
    for(const auto& c:s.gas_channels) {
        if(!c.passage || c.to<0 || c.opening.enabled || c.rupture || c.opening_pressure_pa!=0 || c.surface_link_h_w_per_m2_k!=0)
            throw std::runtime_error("Only fixed unobstructed internal passages are supported");
        const int j=n.edges.size();n.edges.push_back({c.from,c.to,c.area_m2,c.discharge_coefficient});
        rows[c.from].push_back(j);rows[c.to].push_back(j);
    }
    n.offsets.push_back(0);
    for(const auto& row:rows){n.adjacent.insert(n.adjacent.end(),row.begin(),row.end());n.offsets.push_back(n.adjacent.size());}
    // Edge-disjoint connected patches. Prefer low-degree seed endpoints so a
    // high-degree chamber does not swallow unrelated local neighborhoods.
    std::vector<bool> used(n.edges.size());
    for(int seed=0;seed<static_cast<int>(n.edges.size());++seed)if(!used[seed]) {
        Patch p;auto local=[&](int id){for(int i=0;i<p.nodes;++i)if(p.node[i]==id)return i;return -1;};
        auto add=[&](int j) {
            const auto& e=n.edges[j];int a=local(e.from),b=local(e.to);
            if(a<0)a=p.nodes,p.node[p.nodes++]=e.from;
            if(b<0)b=p.nodes,p.node[p.nodes++]=e.to;
            p.edge[p.edges]=j;p.from[p.edges]=a;p.to[p.edges++]=b;++p.degree[a];++p.degree[b];used[j]=true;
        };
        add(seed);
        while(p.edges<max_patch_edges) {
            int chosen=-1,best=1000000000;
            for(int i=0;i<p.nodes;++i)for(int j:rows[p.node[i]])if(!used[j]) {
                const auto& e=n.edges[j];const int missing=(local(e.from)<0)+(local(e.to)<0);
                if(p.nodes+missing>patch_size)continue;
                const int score=missing*1000000+static_cast<int>(rows[p.node[i]].size());
                if(score<best || (score==best && j<chosen)){best=score;chosen=j;}
            }
            if(chosen<0)break;
            add(chosen);
        }
        n.patches.push_back(p);
    }
    // Same-color patches have no shared node, so they can write without atomics.
    std::vector<std::unordered_set<int>> occupied;
    for(int i=0;i<static_cast<int>(n.patches.size());++i) {
        const auto& p=n.patches[i];std::size_t color=0;
        for(;color<occupied.size();++color) {
            bool conflict=false;for(int j=0;j<p.nodes;++j)conflict|=occupied[color].contains(p.node[j]);
            if(!conflict)break;
        }
        if(color==occupied.size()){occupied.emplace_back();n.colors.emplace_back();}
        n.colors[color].push_back(i);for(int j=0;j<p.nodes;++j)occupied[color].insert(p.node[j]);
    }
    validate_partition(n);return n;
}
void validate_partition(const Network& n) {
    std::vector<int> ownership(n.edges.size()),scheduled(n.patches.size());
    for(const auto& p:n.patches) {
        if(p.nodes<2 || p.nodes>max_patch_nodes || p.edges<1 || p.edges>max_patch_edges)throw std::runtime_error("Invalid patch size");
        int degree[max_patch_nodes]{};std::unordered_set<int> unique;
        for(int i=0;i<p.nodes;++i)if(p.node[i]<0 || p.node[i]>=static_cast<int>(n.initial.size()) || !unique.insert(p.node[i]).second)
            throw std::runtime_error("Invalid patch node");
        for(int j=0;j<p.edges;++j) {
            if(p.edge[j]<0 || p.edge[j]>=static_cast<int>(n.edges.size()))throw std::runtime_error("Invalid patch edge");
            const auto& e=n.edges[p.edge[j]];++ownership[p.edge[j]];
            if(p.from[j]<0 || p.from[j]>=p.nodes || p.to[j]<0 || p.to[j]>=p.nodes ||
               p.node[p.from[j]]!=e.from || p.node[p.to[j]]!=e.to)throw std::runtime_error("Patch incidence mismatch");
            ++degree[p.from[j]];++degree[p.to[j]];
        }
        for(int i=0;i<p.nodes;++i)if(degree[i]!=p.degree[i])throw std::runtime_error("Patch degree mismatch");
    }
    for(const auto& color:n.colors) {
        std::unordered_set<int> nodes;
        for(int id:color) {
            if(id<0 || id>=static_cast<int>(n.patches.size()))throw std::runtime_error("Invalid patch schedule");
            ++scheduled[id];const auto& p=n.patches[id];
            for(int i=0;i<p.nodes;++i)if(!nodes.insert(p.node[i]).second)throw std::runtime_error("Concurrent patches share a node");
        }
    }
    for(int v:ownership)if(v!=1)throw std::runtime_error("Edge omitted or duplicated");
    for(int v:scheduled)if(v!=1)throw std::runtime_error("Patch omitted or duplicated");
}
struct Cpu::Buffers {std::vector<Proposal> proposed;};
Cpu::Cpu(const Network& n):network_(n),buffers_(std::make_unique<Buffers>()) {reset();}
Cpu::~Cpu()=default;
void Cpu::reset() {
    state_=network_.initial;next_=state_;total_.assign(network_.edges.size(),{});packet_=total_;
    buffers_->proposed.resize(total_.size());work_={};
}
double Cpu::global_step(double cap) {
    if(!(cap>0) || !std::isfinite(cap))throw std::runtime_error("Invalid global timestep");
    const auto& n=network_;auto& q=buffers_->proposed;double dt=cap;
    for(std::size_t j=0;j<n.edges.size();++j) {
        const auto& e=n.edges[j];q[j]=propose(state_[e.from],state_[e.to],n.offsets[e.from+1]-n.offsets[e.from],
            n.offsets[e.to+1]-n.offsets[e.to],e,n.model);
    }
    for(std::size_t i=0;i<state_.size();++i) {
        Budget b;
        for(int k=n.offsets[i];k<n.offsets[i+1];++k) {
            const int j=n.adjacent[k];const auto& e=n.edges[j];const int donor=q[j].rate>=0?e.from:e.to;
            add_budget(b,state_[i],state_[donor],donor==static_cast<int>(i),q[j],n.model);
        }
        dt=std::min(dt,limit(b,state_[i],n.model));
    }
    if(!(dt>0) || !std::isfinite(dt))throw std::runtime_error("Invalid CPU gas timestep");
    for(std::size_t j=0;j<n.edges.size();++j) {
        const auto& e=n.edges[j];packet_[j]=transfer(q[j],state_[e.from],state_[e.to],dt,n.model);accumulate(total_[j],packet_[j]);
    }
    for(std::size_t i=0;i<state_.size();++i) {
        auto s=state_[i];
        for(int k=n.offsets[i];k<n.offsets[i+1];++k) {
            const int j=n.adjacent[k];apply(s,packet_[j],n.edges[j].from==static_cast<int>(i)?-1.:1.);
        }
        refresh(s,n.model);if(!valid(s))throw std::runtime_error("Invalid CPU gas state");next_[i]=s;
    }
    state_.swap(next_);++work_.steps;work_.edge_evaluations+=n.edges.size();return dt;
}
void Cpu::patch_step(const Patch& p,double duration) {
    std::array<GasState,max_patch_nodes> state,next;
    std::array<Proposal,max_patch_edges> q;
    std::array<Packet,max_patch_edges> packet,total{};
    const auto& n=network_;for(int i=0;i<p.nodes;++i)state[i]=state_[p.node[i]];
    double elapsed=0;unsigned long long count=0;
    while(elapsed<duration) {
        double dt=duration-elapsed;
        for(int j=0;j<p.edges;++j)q[j]=propose(state[p.from[j]],state[p.to[j]],p.degree[p.from[j]],p.degree[p.to[j]],n.edges[p.edge[j]],n.model);
        for(int i=0;i<p.nodes;++i) {
            Budget b;
            for(int j=0;j<p.edges;++j)if(p.from[j]==i || p.to[j]==i) {
                const int donor=q[j].rate>=0?p.from[j]:p.to[j];add_budget(b,state[i],state[donor],donor==i,q[j],n.model);
            }
            dt=std::min(dt,limit(b,state[i],n.model));
        }
        if(!(dt>0) || elapsed+dt==elapsed || ++count>2000000)throw std::runtime_error("CPU patch timestep/count limit");
        for(int j=0;j<p.edges;++j) {packet[j]=transfer(q[j],state[p.from[j]],state[p.to[j]],dt,n.model);accumulate(total[j],packet[j]);}
        for(int i=0;i<p.nodes;++i) {
            auto s=state[i];for(int j=0;j<p.edges;++j) {
                if(p.from[j]==i)apply(s,packet[j],-1.);else if(p.to[j]==i)apply(s,packet[j],1.);
            }
            refresh(s,n.model);if(!valid(s))throw std::runtime_error("Invalid CPU patch state");next[i]=s;
        }
        state=next;elapsed+=dt;
    }
    for(int i=0;i<p.nodes;++i)state_[p.node[i]]=state[i];
    for(int j=0;j<p.edges;++j)accumulate(total_[p.edge[j]],total[j]);
    work_.patch_substeps+=count;work_.edge_evaluations+=count*p.edges;
}
void Cpu::local_window(double duration) {
    if(!(duration>0) || !std::isfinite(duration))throw std::runtime_error("Invalid patch interval");
    // Palindromic operator composition. Each edge evolves for exactly duration;
    // no physical time is added for the other colors. Inner solvers are explicit
    // first order, so symmetry alone does NOT guarantee second-order accuracy.
    for(const auto& color:network_.colors)for(int id:color)patch_step(network_.patches[id],duration/2);
    for(auto c=network_.colors.rbegin();c!=network_.colors.rend();++c)for(int id:*c)patch_step(network_.patches[id],duration/2);
    ++work_.steps;
}
} // namespace battery::gas_experiment

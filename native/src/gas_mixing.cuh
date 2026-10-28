#pragma once
#include "battery_native/gas_mixing.hpp"
// Included inside batched_gas. Three snapshot/gather passes, one writer per
// node; the SAME transfer record is used at both endpoints. No float atomics.
__device__ inline void mix_snapshot(cooperative_groups::grid_group grid,int tid,int stride,
    int node_count,int edge_count,const int* offsets,const int* incidence,const kg::Edge* edges,
    kg::Node* nodes,Proposal* proposals,Budget* budgets,GasChannelState* channels,
    const kg::Settings& settings,const GasModel& model,double step) {
    for(int e=tid;e<edge_count;e+=stride) {
        auto edge=edges[e];edge.area=channels[e].open?channels[e].effective_area_m2:0.;
        proposals[e].mixing=MixingTransfer{};
        proposals[e].mixing.mass_each_way=mixing_request(nodes[edge.a],nodes[edge.b],edge,settings,
            model.mixing_diffusivity_m2_s,model.mixing_flow_fraction,proposals[e].mass/step,step);
    }
    grid.sync();
    for(int i=tid;i<node_count;i+=stride) {
        double out=0.;
        for(int j=offsets[i];j<offsets[i+1];++j)out+=proposals[incidence[j]].mixing.mass_each_way;
        budgets[i].scale=out>0.?fmin(1.,.5*nodes[i].mass/out):1.;
    }
    grid.sync();
    for(int e=tid;e<edge_count;e+=stride) {
        const auto& edge=edges[e];auto& p=proposals[e];
        const double dm=p.mixing.mass_each_way*fmin(budgets[edge.a].scale,budgets[edge.b].scale);
        p.mixing=mixing_transfer(nodes[edge.a],nodes[edge.b],settings,dm);
        auto& c=channels[e];c.mixed_mass_kg+=dm;
        c.transferred_energy_j+=p.mixing.energy;c.transferred_chemical_j+=p.mixing.chemical;
        c.transferred_oxidizer_kg+=p.mixing.oxidizer;
        c.transferred_vapor_kg+=p.mixing.vapor;c.transferred_latent_j+=p.mixing.latent;
    }
    grid.sync();
    // All payloads have been evaluated before ANY endpoint can be written.
    for(int i=tid;i<node_count;i+=stride) {
        auto node=nodes[i];
        for(int j=offsets[i];j<offsets[i+1];++j) {
            const int e=incidence[j];const auto& p=proposals[e].mixing;
            const double sign=edges[e].a==i?-1.:1.;
            node.energy+=sign*p.energy;node.chemical+=sign*p.chemical;node.oxidizer+=sign*p.oxidizer;
            node.vapor+=sign*p.vapor;node.latent+=sign*p.latent;
        }
        nodes[i]=node;
    }
    grid.sync();
}

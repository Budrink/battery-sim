#pragma once
#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

namespace battery {
// Edge-disjoint, connected subgraphs. Negative endpoint denotes an external
// sink, not a shared numerical node. Colors have disjoint writable nodes.
struct GasLocalGroup {
    std::vector<int> nodes, edges;
    std::vector<std::pair<int,int>> ends;
    std::vector<int> degree;
};
struct GasPartition {
    std::vector<GasLocalGroup> groups;
    std::vector<std::vector<int>> colors;
};
inline GasPartition partition_gas(int count,const std::vector<std::pair<int,int>>& edges,int size=9) {
    if(size<2 || size>9)throw std::invalid_argument("Gas group size must be 2..9");
    GasPartition result;std::vector<std::vector<int>> rows(count);
    for(int j=0;j<int(edges.size());++j) {
        auto [a,b]=edges[j];
        if(a<0 || a>=count || b>=count || a==b)throw std::invalid_argument("Invalid gas incidence");
        rows[a].push_back(j);if(b>=0)rows[b].push_back(j);
    }
    std::vector<bool> used(edges.size());
    for(int seed=0;seed<int(edges.size());++seed)if(!used[seed]) {
        GasLocalGroup g;
        auto local=[&](int id){auto it=std::find(g.nodes.begin(),g.nodes.end(),id);return it==g.nodes.end()?-1:int(it-g.nodes.begin());};
        auto add=[&](int j){
            auto [a,b]=edges[j];int u=local(a),v=local(b);
            if(u<0){u=g.nodes.size();g.nodes.push_back(a);g.degree.push_back(0);}
            if(v<0 && b>=0){v=g.nodes.size();g.nodes.push_back(b);g.degree.push_back(0);}
            g.edges.push_back(j);g.ends.emplace_back(u,v);++g.degree[u];if(v>=0)++g.degree[v];used[j]=true;
        };
        add(seed);
        while(g.edges.size()<64) {
            int chosen=-1,best=1000000000;
            for(int node:g.nodes)for(int j:rows[node])if(!used[j]) {
                auto [a,b]=edges[j];const int missing=(local(a)<0)+(b>=0 && local(b)<0);
                if(int(g.nodes.size())+missing>size)continue;
                int score=missing*1000000+int(rows[node].size());
                if(score<best || (score==best && j<chosen)){chosen=j;best=score;}
            }
            if(chosen<0)break;
            add(chosen);
        }
        result.groups.push_back(std::move(g));
    }
    std::vector<std::vector<bool>> occupied;
    for(int id=0;id<int(result.groups.size());++id) {
        const auto& g=result.groups[id];int color=0;
        for(;color<int(occupied.size());++color) {
            bool conflict=false;for(int i:g.nodes)conflict|=occupied[color][i];
            if(!conflict)break;
        }
        if(color==int(occupied.size())){occupied.emplace_back(count,false);result.colors.emplace_back();}
        result.colors[color].push_back(id);for(int i:g.nodes)occupied[color][i]=true;
    }
    return result;
}
}

#pragma once
#include "battery_native/cuda_solver.hpp"
#include <memory>
#include <vector>

// Experimental CLOSED gas-only transport. No thermal/LTE, sources, reactions,
// changing apertures or external sinks. Not selected by the production solver.
namespace battery::gas_experiment {
constexpr int max_patch_nodes=9,max_patch_edges=64;
struct Edge {int from,to;double area,cd;};
struct Packet {double mass=0,energy=0,chemical=0,oxidizer=0;};
struct Patch {
    int nodes=0,edges=0;
    int node[max_patch_nodes]{},degree[max_patch_nodes]{};
    int edge[max_patch_edges]{},from[max_patch_edges]{},to[max_patch_edges]{};
};
struct Network {
    GasModel model;
    std::vector<GasState> initial;
    std::vector<Edge> edges;
    std::vector<int> offsets,adjacent;
    std::vector<Patch> patches;
    std::vector<std::vector<int>> colors;
};
struct Work {unsigned long long steps=0,edge_evaluations=0,patch_substeps=0,launches=0;};
Network make_network(const Scenario&,int patch_size=9);
void validate_partition(const Network&);

class Cpu {
public:
    explicit Cpu(const Network&);
    void reset();
    double global_step(double cap);
    void local_window(double duration);
    const std::vector<GasState>& states() const {return state_;}
    const std::vector<Packet>& transfers() const {return total_;}
    Work work() const {return work_;}
private:
    void patch_step(const Patch&,double duration);
    const Network& network_;
    std::vector<GasState> state_,next_;
    std::vector<Packet> total_,packet_;
    struct Buffers;
    std::unique_ptr<Buffers> buffers_;
    Work work_;
public:
    ~Cpu();
};
class Cuda {
public:
    explicit Cuda(const Network&);
    ~Cuda();
    void reset();
    double global_step(double cap);
    void local_window(double duration);
    void synchronize() const;
    void download(std::vector<GasState>&,std::vector<Packet>&) const;
    Work work() const;
    double state_roundtrip_seconds(int repetitions=100);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace battery::gas_experiment

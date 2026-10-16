#pragma once
#include <cstdint>
#include <memory>
#include <vector>

// Isolated experimental pressure transport. Constant-cv ideal pseudo-gas,
// optional constant-capacity immobile material in local thermal equilibrium.
// This standalone API has no chemistry or rupture. kawasaki_transport.hpp
// provides the opt-in full-model CUDA adapter, including the batched mode.
namespace battery::kawasaki {
struct Node {
    double mass=0,energy=0,volume=0,solid_capacity=0; // absolute sensible energy
    double chemical=0,oxidizer=0;
    // Per-adapter-step reaction diagnostics; these are local history, not advected.
    double reaction_released_j=0,reaction_oxidizer_kg=0;
};
struct Edge {int a=0,b=0;double area=0,cd=.7;};
struct Settings {
    double cv=1000,R=287;
    double packet_fraction=.1; // fraction of pressure-equalizing mass, not a time fit
    double equilibrium_atol_pa=1e-6,equilibrium_rtol=1e-10;
    // Deterministic small-dp tail; zero selects the slower event-only reference.
    // Both limits must hold; the relative bound keeps the sqrt(dp) expansion local.
    double tail_atol_pa=100,tail_rtol=1e-3;
};
struct Clock {
    std::uint64_t batches=0,limited_batches=0; // whole-graph tau-leap diagnostics
    double hazard=0;
    std::uint64_t events=0,snaps=0;
    double snapped_mass=0;
    std::uint64_t tails=0;
    double tail_mass=0;
    bool initialized=false;
};
struct Network {
    Settings settings;
    std::vector<Node> initial;
    std::vector<Edge> edges;
    std::vector<std::vector<int>> colors;
};
void schedule(Network&);
void validate(const Network&);
class Cuda {
public:
    Cuda(const Network&,int replicas=1,std::uint64_t seed=1,int block_size=128);
    ~Cuda();
    void reset();
    void advance(double duration);
    void synchronize() const;
    void download(std::vector<Node>&,std::vector<Clock>&) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace battery::kawasaki

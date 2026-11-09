#pragma once
#include "gas_direction_memory.cuh"

namespace battery::batched_gas {

// Bounded stochastic quadrature of the physical finite-time pair flux. Unlike
// the legacy event loop this makes ONE proposal per matching, even for stiff
// open passages. Rare events have probability proportional to elapsed seconds.
// E[packet] = mean; saturation is a finite-volume approximation, not a time fit.
__device__ inline double sample_pair_mass(double mean, double cap, double fraction, std::uint64_t seed,
                                          int edge, std::uint64_t draw) {
    if (!(mean > 0. && cap > 0.))
        return 0.;
    const double quantum = fraction * cap;
    const double u = kg::uniform(seed, edge, draw, 0);
    if (mean < .5 * quantum)
        return u < 2. * mean / quantum ? quantum * kg::uniform(seed, edge, draw, 1) : 0.;
    const double width = fmin(fmin(mean, cap - mean), .5 * quantum);
    return mean + (2. * u - 1.) * fmax(0., width);
}

__device__ inline void colored_pair(int e, kg::Node* nodes, const kg::Edge& topology,
                                    GasChannelState& channel, kg::Clock& clock, const kg::Settings& settings,
                                    const GasModel& model, double dt, double duration, Proposal& accumulated,
                                    const gas_direction::Direction* directions, gas_direction::Memory* memory,
                                    const gas_direction::Arrival* arrivals) {
    auto edge = topology;
    edge.area = channel.open ? channel.effective_area_m2 : 0.;
    const auto a = nodes[edge.a], b = nodes[edge.b];
    const double pa = kg::pressure(a, settings), pb = kg::pressure(b, settings);
    const double old_k = memory ? memory[e].kinetic_j : 0.;
    double q = 0.;
    if (edge.area > 0. && edge.cd > 0.) {
        if (memory) {
            const auto op = momentum_operator(a, b, pa, pb, edge, settings, model, directions[e], memory[e],
                                              arrivals[edge.a], arrivals[edge.b], dt);
            q = op.mobility * (pa - pb) + op.bias;
        } else
            q = pa >= pb ? kg::flow(a, b, edge, settings) : -kg::flow(b, a, edge, settings);
    }
    const bool forward = q >= 0.;
    const auto& donor = forward ? a : b;
    const auto& receiver = forward ? b : a;
    const bool downhill = forward ? pa >= pb : pb >= pa;
    const double pressure_cap = downhill ? kg::cap(donor, receiver, settings) : 0.;
    const double cap =
        q == 0. ? 0. : (memory ? fmin(.2 * donor.mass, fmax(pressure_cap, 2. * fabs(q) * dt)) : pressure_cap);
    const double mean = cap > 0. ? -cap * expm1(-fabs(q) * dt / cap) : 0.;
    double mass = sample_pair_mass(mean, cap, settings.packet_fraction, kg::mix(model.kawasaki_seed), e,
                                   clock.batches++);
    const double entropy_before = kg::entropy(a, settings) + kg::entropy(b, settings);
    // Physical kinetic energy is paid out of sensible energy. The guard is
    // bounded work and cannot inject heat or hide a negative inventory.
    for (int attempt = 0; attempt <= 12; ++attempt) {
        if (attempt == 12)
            mass = 0.;
        auto na = a, nb = b;
        auto& d = forward ? na : nb;
        auto& r = forward ? nb : na;
        const double f = mass / donor.mass;
        const double energy = mass > 0. ? kg::packet_energy(donor, mass, settings) : 0.;
        d.mass -= mass;
        r.mass += mass;
        d.energy -= energy;
        r.energy += energy;
        d.chemical -= donor.chemical * f;
        r.chemical += donor.chemical * f;
        d.oxidizer -= donor.oxidizer * f;
        r.oxidizer += donor.oxidizer * f;
        d.vapor -= donor.vapor * f;
        r.vapor += donor.vapor * f;
        d.latent -= donor.latent * f;
        r.latent += donor.latent * f;
        const double rho = .5 * (na.mass / na.volume + nb.mass / nb.volume);
        const double kinetic =
            memory && mass > 0.
                ? gas_direction::kinetic_energy(mass / dt, directions[e].length_m, edge.area, rho)
                : 0.;
        na.energy += .5 * (old_k - kinetic);
        nb.energy += .5 * (old_k - kinetic);
        const bool valid = na.mass > 0. && nb.mass > 0. && na.energy > 0. && nb.energy > 0. &&
                           kg::entropy(na, settings) + kg::entropy(nb, settings) >= entropy_before - 1e-12;
        if (valid) {
            nodes[edge.a] = na;
            nodes[edge.b] = nb;
            const double sign = forward ? 1. : -1.;
            if (memory)
                memory[e] = {sign * mass / dt, kinetic};
            accumulated.mass += mass;
            accumulated.rate += .5 * fabs(q);
            channel.mass_flow_kg_s += sign * mass / duration;
            channel.transferred_kg += sign * mass;
            channel.transferred_energy_j +=
                sign * (energy - mass * settings.cv * model.reference_temperature_k);
            channel.transferred_chemical_j += sign * donor.chemical * f;
            channel.transferred_oxidizer_kg += sign * donor.oxidizer * f;
            channel.transferred_vapor_kg += sign * donor.vapor * f;
            channel.transferred_latent_j += sign * donor.latent * f;
            if (mass > 0.)
                ++clock.events;
            return;
        }
        mass *= .5;
        ++clock.limited_batches;
    }
}

__device__ inline void colored_transport(cooperative_groups::grid_group grid, int tid, int stride, int count,
                                         int edge_count, int colors, const int* color_offsets,
                                         const int* color_edges, const int* offsets, const int* incidence,
                                         const kg::Edge* edges, const kg::Node* current, kg::Node* next,
                                         Proposal* proposals, GasChannelState* channels, kg::Clock* clocks,
                                         const kg::Settings& settings, const GasModel& model, double dt,
                                         double duration, const gas_direction::Direction* directions,
                                         gas_direction::Memory* memory, gas_direction::Arrival* arrivals) {
    for (int i = tid; i < count; i += stride)
        next[i] = current[i];
    for (int e = tid; e < edge_count; e += stride)
        proposals[e] = {};
    grid.sync();
    // Same ownership rule as the plasma solver: each matching has disjoint
    // endpoints. Reverse sweep cancels a fixed preferred traversal direction.
    // Every open edge gets two half-intervals, NOT dt divided by color count.
    for (int pass = 0; pass < 2; ++pass) {
        if (memory)
            gather_momentum(grid, tid, stride, count, offsets, incidence, edges, next, channels, directions,
                            memory, arrivals);
        for (int k = 0; k < colors; ++k) {
            const int color = pass ? colors - 1 - k : k;
            for (int j = color_offsets[color] + tid; j < color_offsets[color + 1]; j += stride) {
                const int e = color_edges[j];
                colored_pair(e, next, edges[e], channels[e], clocks[e], settings, model, dt / 2, duration,
                             proposals[e], directions, memory, arrivals);
            }
            grid.sync();
        }
    }
}
} // namespace battery::batched_gas

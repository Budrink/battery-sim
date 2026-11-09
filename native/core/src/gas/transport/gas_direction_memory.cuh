#pragma once
#include "gas_device_context.cuh"

namespace battery::batched_gas {

// Snapshot incoming momentum flux before the pressure solve. One writer per
// node, no pair-of-edges tensor, atomics or fitted lifetime/strength.
__device__ inline void gather_momentum(cooperative_groups::grid_group grid, int tid, int stride, int count,
                                       const int* offsets, const int* incidence, const kg::Edge* edges,
                                       const kg::Node* nodes, const GasChannelState* channels,
                                       const gas_direction::Direction* directions,
                                       const gas_direction::Memory* memory,
                                       gas_direction::Arrival* arrivals) {
    for (int i = tid; i < count; i += stride) {
        gas_direction::Arrival a;
        for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
            const int e = incidence[j];
            const auto& edge = edges[e];
            const double q = memory[e].flow_kg_s;
            const bool forward = q >= 0.;
            if ((forward ? edge.b : edge.a) != i || !channels[e].open ||
                !(channels[e].effective_area_m2 > 0. && edge.cd > 0.))
                continue;
            const double rho =
                .5 * (nodes[edge.a].mass / nodes[edge.a].volume + nodes[edge.b].mass / nodes[edge.b].volume);
            const double weight = q * q / (rho * channels[e].effective_area_m2);
            const double sign = forward ? 1. : -1.;
            a.weight += weight;
            a.x += sign * weight * directions[e].x;
            a.y += sign * weight * directions[e].y;
            a.z += sign * weight * directions[e].z;
        }
        arrivals[i] = a;
    }
    grid.sync();
}

__device__ inline gas_direction::Operator
momentum_operator(const kg::Node& a, const kg::Node& b, double pa, double pb, const kg::Edge& open,
                  const kg::Settings& settings, const GasModel& model,
                  const gas_direction::Direction& geometry, const gas_direction::Memory& memory,
                  const gas_direction::Arrival& incoming_a, const gas_direction::Arrival& incoming_b,
                  double dt) {
    if (!(open.area > 0. && open.cd > 0.))
        return {};
    const double rho = .5 * (a.mass / a.volume + b.mass / b.volume);
    const double inertance = geometry.length_m / open.area;
    const bool forward = pa - pb + inertance * memory.flow_kg_s / dt >= 0.;
    const double turning = gas_direction::turn_loss(forward ? incoming_a : incoming_b, geometry, forward);
    // Existing compressible orifice law supplies the quasi-steady quadratic loss.
    // Away from exactly zero dp it has a finite quadratic limit.
    auto donor = pa >= pb ? a : b, receiver = pa >= pb ? b : a;
    const double high = fmax(pa, pb), low = fmin(pa, pb);
    const double gap = fmax(high - low, fmax(1e-6, high * 1e-6));
    const double td = kg::temperature(donor, settings), tr = kg::temperature(receiver, settings);
    donor.mass = (low + gap) * donor.volume / (settings.R * td);
    receiver.mass = low * receiver.volume / (settings.R * tr);
    donor.energy = (donor.solid_capacity + settings.cv * donor.mass) * td;
    receiver.energy = (receiver.solid_capacity + settings.cv * receiver.mass) * tr;
    const double steady = kg::flow(donor, receiver, open, settings);
    const double quadratic = gap / (steady * steady) + turning / (2. * rho * open.area * open.area);
    const double linear =
        gas_direction::laminar_resistance(model.dynamic_viscosity_pa_s, geometry.length_m, open.area, rho);
    return gas_direction::implicit_flow(memory.flow_kg_s, pa - pb, dt, inertance, linear, quadratic);
}

// K uses the actual reconciled transfer, not an unlimited proposal.
__device__ inline double accepted_kinetic(const Proposal& p, const kg::Edge& edge,
                                          const GasChannelState& channel, const kg::Node* next,
                                          const gas_direction::Direction& direction, double dt) {
    if (!(channel.open && channel.effective_area_m2 > 0. && edge.cd > 0.))
        return 0.;
    const double rho =
        .5 * (next[edge.a].mass / next[edge.a].volume + next[edge.b].mass / next[edge.b].volume);
    return gas_direction::kinetic_energy(p.mass / dt, direction.length_m, channel.effective_area_m2, rho);
}

} // namespace battery::batched_gas

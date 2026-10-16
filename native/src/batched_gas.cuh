#pragma once
#include "battery_native/cuda_solver.hpp"
#include "battery_native/kawasaki_gas_math.hpp"
#include "battery_native/gas_combustion.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Whole-graph, snapshot-based conservative transport. Edge proposals support
// stochastic tau-leaping, fixed deterministic sweeps, and a one-pass analytic
// network variant. CSR gathers give every node one writer.
namespace battery::batched_gas {
namespace kg = kawasaki;
struct Proposal {
    int donor = 0;
    double cap = 0, rate = 0, mass = 0;
    double energy = 0, chemical = 0, oxidizer = 0;
    double donor_pressure_weight = 0, receiver_pressure_weight = 0;
    bool tail = false;
};
struct Budget { double scale = 1, outgoing = 0, energy = 0; };
struct Control {
    double entropy_by_block[32];
    double pressure_gap_by_block[32];
    double last_max_pressure_gap_pa;
    unsigned active_combustion=0, reaction_intervals=1;
};

// Extra kinetic subcycling is needed only where reactants can meet and ignite.
// Include upwind arrivals (and their enthalpy), not only the resident snapshot.
// A cold or oxygen-free fuel stock behind a closed barrier must not make every
// mass step expensive merely because a distant reactor has a large rate.
__device__ inline bool reactive_neighbourhood(int i,const kg::Node* nodes,
    const GasCombustionLaw& law,const kg::Settings& settings,const kg::Edge* edges,
    const GasChannelState* channels,const int* offsets,const int* incidence) {
    if(!(law.rate_per_s>0.))return false;
    const auto& n=nodes[i];
    bool fuel=resolved_gas_fuel(n),oxygen=n.oxidizer>0.;
    double temperature=kg::temperature(n,settings);
    const double pressure=kg::pressure(n,settings);
    for(int j=offsets[i];j<offsets[i+1];++j) {
        const int e=incidence[j];const auto& edge=edges[e];const auto& channel=channels[e];
        if(!channel.open || !(channel.effective_area_m2>0 && edge.cd>0))continue;
        const int other=edge.a==i?edge.b:edge.a;const auto& incoming=nodes[other];
        const double upstream=kg::pressure(incoming,settings);
        if(upstream-pressure<=settings.equilibrium_atol_pa+settings.equilibrium_rtol*upstream)continue;
        fuel|=resolved_gas_fuel(incoming);oxygen|=incoming.oxidizer>0.;
        // Incoming enthalpy can heat a compressed receiving volume above the
        // donor temperature. gamma*T is a conservative ignition predictor.
        temperature=fmax(temperature,(1+settings.R/settings.cv)*kg::temperature(incoming,settings));
    }
    return fuel && oxygen && temperature>=law.ignition_temperature_k;
}

// Exact Poisson count for small means. Large means use the explicitly selected
// deterministic closure, not a costly loop over individual transfers.
__device__ inline unsigned poisson(double mean, std::uint64_t seed, int edge,
                                   std::uint64_t batch) {
    const double u = kg::uniform(seed, edge, batch, 0);
    double probability = exp(-mean), cumulative = probability;
    unsigned count = 0;
    while (u > cumulative && count < 256) {
        probability *= mean / ++count;
        cumulative += probability;
    }
    return count;
}

// All stages read a common snapshot. No node state is modified by an edge.
// A resident cooperative grid avoids repeated launches while allowing several
// SMs to evaluate edges concurrently. Residency is checked by launch().
__global__ void advance(int node_count, int edge_count, const int* offsets,
    const int* incidence, const kg::Edge* edges, kg::Node* initial, kg::Node* scratch,
    Proposal* proposals, Budget* budgets, kg::Clock* clocks, GasChannelState* channels,
    kg::Settings settings, GasModel model, double duration, int* error, Control* control,
    GasCombustionView combustion) {
    const auto grid = cooperative_groups::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    __shared__ double reduction[256];
    auto* current = initial;
    auto* next = scratch;
    double elapsed = 0;
    unsigned iterations = 0;
    if (tid == 0) {control->last_max_pressure_gap_pa = 0;control->active_combustion=0;}
    grid.sync();
    if(combustion.maximum_rate_per_s>0.) {
        bool active=false;
        for(int i=tid;i<node_count;i+=stride)
            active|=reactive_neighbourhood(i,current,combustion.laws[i],settings,edges,channels,offsets,incidence);
        if(active)atomicOr(&control->active_combustion,1u);
        grid.sync();
    }
    const double chemistry_interval=control->active_combustion?
        combustion_interval(combustion.maximum_rate_per_s,model.max_fraction_per_step):duration;
    const double requested_sweeps=ceil(duration/fmin(duration,chemistry_interval));
    if(!(requested_sweeps>=1 && requested_sweeps<=200000)) {
        if(tid==0)*error=18;return;
    }
    const unsigned reactive_sweeps=static_cast<unsigned>(requested_sweeps);
    if(tid==0)control->reaction_intervals=reactive_sweeps;
    const unsigned requested_fixed=static_cast<unsigned>(model.deterministic_sweeps_per_step);
    const unsigned fixed_sweeps = model.cuda_deterministic_transport ?
        (requested_fixed>reactive_sweeps?requested_fixed:reactive_sweeps) : 0u;
    const bool network = model.cuda_network_transport;
    while (network ? iterations < reactive_sweeps :
           (model.cuda_deterministic_transport ? iterations < fixed_sweeps : elapsed < duration)) {
        if (++iterations > 200000) { if (tid == 0) *error = network ? 18 : 14; return; }
        // Tau-leap on the configured physical window. A stiff edge is limited
        // by the common-snapshot pressure/inventory budgets below; its tiny
        // characteristic time no longer stalls every other edge in the graph.
        const double step = network ? duration/reactive_sweeps : (model.cuda_deterministic_transport ?
            duration / fixed_sweeps : fmin(duration - elapsed,fmin(model.local_exchange_interval_s,chemistry_interval)));
        if(combustion.laws) {
            // Symmetric reaction/transport integration: old resident gas reacts
            // before leaving, new arrivals react in the SAME physical interval.
            // Every subsequent flux sees the released heat and changed pressure.
            for(int i=tid;i<node_count;i+=stride)
                burn_gas(current[i],combustion.laws[i],settings,step/2);
            grid.sync();
        }
        for (int e = tid; e < edge_count; e += stride) {
            auto edge = edges[e];
            edge.area = channels[e].open ? channels[e].effective_area_m2 : 0;
            Proposal p;
            p.donor = kg::pressure(current[edge.a], settings) >= kg::pressure(current[edge.b], settings) ? edge.a : edge.b;
            const int receiver = p.donor == edge.a ? edge.b : edge.a;
            const auto& d = current[p.donor]; const auto& r = current[receiver];
            if (edge.area > 0 && edge.cd > 0) {
                p.rate = kg::flow(d, r, edge, settings);
                const double pressure = kg::pressure(d, settings);
                const double roundoff = settings.equilibrium_atol_pa + settings.equilibrium_rtol * pressure;
                const double tolerance = fmax(roundoff,
                    fmin(settings.tail_atol_pa, settings.tail_rtol * pressure));
                p.tail = pressure - kg::pressure(r, settings) <= tolerance;
                const double gap = pressure - kg::pressure(r, settings);
                // A machine-scale pressure difference must not consume the
                // same node budget as a macroscopically active channel.
                if(gap <= roundoff) {p.rate=0;p.cap=0;}
                if(p.rate > 0 && gap > 0) {
                    const double td=kg::temperature(d,settings),tr=kg::temperature(r,settings);
                    const double cd=d.solid_capacity+settings.cv*d.mass,cr=r.solid_capacity+settings.cv*r.mass;
                    p.donor_pressure_weight=settings.R*td/d.volume*(1+d.mass*settings.R/cd)/gap;
                    p.receiver_pressure_weight=settings.R/r.volume*(tr+r.mass*((settings.cv+settings.R)*td-settings.cv*tr)/cr)/gap;
                    if(network) {
                        // Linearized equalizing inventory from local pressure
                        // sensitivities; this avoids a per-edge nonlinear root.
                        const double sensitivity=p.donor_pressure_weight+p.receiver_pressure_weight;
                        p.cap=sensitivity>0 ? 1./sensitivity : 0.;
                    } else p.cap=kg::cap(d,r,settings);
                }
            }
            if (!(p.cap > 0)) p.rate = 0;
            proposals[e] = p;
        }
        grid.sync();

        if (!(step > 0) || elapsed + step == elapsed) { if (tid == 0) *error = 15; return; }

        for (int e = tid; e < edge_count; e += stride) {
            auto& p = proposals[e]; auto& clock = clocks[e];
            if (p.rate == 0) continue;
            const double quantum = .5 * settings.packet_fraction * p.cap;
            const double mean = p.rate * step / quantum;
            if (model.cuda_deterministic_transport) {
                // Deterministic finite-volume request. Receiver pressure is
                // already present in p.rate; cap prevents pairwise pressure
                // inversion, and the node gather below reconciles competing
                // paths without edge-order dependence.
                p.mass = p.rate * step;
            } else if(network) {
                const auto& edge=edges[e];
                const double pa=kg::pressure(current[p.donor],settings);
                const int receiver=p.donor==edge.a?edge.b:edge.a;
                const double ratio=kg::pressure(current[receiver],settings)/pa;
                const double gamma=1+settings.R/settings.cv;
                const double critical=pow(2/(gamma+1),gamma/(gamma-1));
                if(ratio<=critical) {
                    // Choked interval: rate is nearly donor-controlled; hold
                    // the initial rate over this short accepted thermal step.
                    p.mass=fmin(p.cap,p.rate*step);
                } else {
                    // Subcritical flow approaches zero as the edge equalizes.
                    // Integrate the local sqrt(Δp) closure analytically.
                    const double f=fmin(1.,p.rate*step/(2*p.cap));
                    p.mass=p.cap*f*(2-f);
                }
            } else if (p.tail) {
                const double f = fmin(1., p.rate * step / (2 * p.cap));
                p.mass = p.cap * f * (2 - f);
                ++clock.tails;
            } else if (mean >= model.batch_mean_threshold) {
                p.mass = p.rate * step;
                ++clock.tails;
            } else {
                const auto count = poisson(mean, kg::mix(model.kawasaki_seed), e, clock.batches);
                p.mass = quantum * count;
                clock.events += count;
            }
            ++clock.batches;
            if (p.mass > p.cap) { p.mass = p.cap; ++clock.limited_batches; }
        }
        grid.sync();

        // Reconcile all requests, including many simultaneous incoming streams.
        // Each node limits its total predicted pressure change to a configured
        // fraction of incident gaps. Equal-and-opposite packets stay conserved.
        for (int i = tid; i < node_count; i += stride) {
            double fractions = 0, outgoing = 0;
            for (int j = offsets[i]; j < offsets[i+1]; ++j) {
                const auto& p = proposals[incidence[j]];
                fractions += p.mass * (p.donor == i ? p.donor_pressure_weight : p.receiver_pressure_weight);
                if (p.donor == i) outgoing += p.mass;
            }
            Budget b;
            const double pressure_fraction = model.cuda_deterministic_transport ?
                model.deterministic_pressure_fraction : (network ? model.network_pressure_fraction : model.batch_pressure_fraction);
            if (fractions > pressure_fraction)
                b.scale = pressure_fraction / fractions;
            if (outgoing > 0) b.scale = fmin(b.scale, .5 * current[i].mass / outgoing);
            budgets[i] = b;
        }
        grid.sync();
        for (int e = tid; e < edge_count; e += stride) {
            const auto& edge = edges[e]; auto& p = proposals[e];
            const double scale = fmin(budgets[edge.a].scale, budgets[edge.b].scale);
            if (p.mass > 0 && scale < 1) ++clocks[e].limited_batches;
            p.mass *= scale;
        }
        grid.sync();
        for (int i = tid; i < node_count; i += stride) {
            double out = 0;
            for (int j = offsets[i]; j < offsets[i+1]; ++j) {
                const auto& p = proposals[incidence[j]];
                if (p.donor == i) out += p.mass;
            }
            budgets[i].outgoing = out;
            budgets[i].energy = out > 0 ? kg::packet_energy(current[i], out, settings) : 0;
        }
        grid.sync();
        for (int e = tid; e < edge_count; e += stride) {
            auto& p = proposals[e]; if (!(p.mass > 0)) continue;
            const auto& donor = current[p.donor]; const auto& b = budgets[p.donor];
            p.energy = b.energy * (p.mass / b.outgoing);
            p.chemical = donor.chemical * (p.mass / donor.mass);
            p.oxidizer = donor.oxidizer * (p.mass / donor.mass);
            const double sign = p.donor == edges[e].a ? 1 : -1;
            auto& channel = channels[e];
            channel.transferred_kg += sign * p.mass;
            channel.transferred_energy_j += sign * (p.energy - p.mass * settings.cv * model.reference_temperature_k);
            channel.transferred_chemical_j += sign * p.chemical;
            channel.transferred_oxidizer_kg += sign * p.oxidizer;
            channel.mass_flow_kg_s += sign * p.mass / duration;
        }
        grid.sync();
        double entropy_change = 0;
        for (int i = tid; i < node_count; i += stride) {
            auto node = current[i];
            for (int j = offsets[i]; j < offsets[i+1]; ++j) {
                const auto& p = proposals[incidence[j]];
                const double sign = p.donor == i ? -1 : 1;
                node.mass += sign * p.mass; node.energy += sign * p.energy;
                node.chemical += sign * p.chemical; node.oxidizer += sign * p.oxidizer;
            }
            if (!(node.mass > 0 && node.energy > 0 && node.chemical >= 0 && node.oxidizer >= 0)) atomicExch(error, 16);
            entropy_change += kg::entropy(node, settings) - kg::entropy(current[i], settings);
            // Audit transport entropy separately; reaction heat must not hide a
            // nonphysical entropy loss in the conservative transfer operator.
            if(combustion.laws)burn_gas(node,combustion.laws[i],settings,step/2);
            next[i] = node;
        }
        reduction[threadIdx.x] = entropy_change;
        __syncthreads();
        for (int width = blockDim.x/2; width; width /= 2) {
            if (threadIdx.x < width) reduction[threadIdx.x] += reduction[threadIdx.x+width];
            __syncthreads();
        }
        // Entropy is audited, never repaired by adding unaccounted heat.
        if (threadIdx.x == 0) control->entropy_by_block[blockIdx.x] = reduction[0];
        grid.sync();
        if (tid == 0) {
            double entropy=0;for(int k=0;k<int(gridDim.x);++k)entropy+=control->entropy_by_block[k];
            if(entropy < -1e-8)*error=17;
        }
        grid.sync();
        if (*error) return;
        auto* swap = current; current = next; next = swap;
        elapsed += step;
        if ((model.cuda_deterministic_transport && iterations == fixed_sweeps) ||
            (network && iterations == reactive_sweeps)) {
            double gap = 0;
            for (int e = tid; e < edge_count; e += stride) {
                if (channels[e].open && channels[e].effective_area_m2 > 0) {
                    const auto& edge = edges[e];
                    gap = fmax(gap, fabs(kg::pressure(current[edge.a], settings) -
                                         kg::pressure(current[edge.b], settings)));
                }
            }
            reduction[threadIdx.x] = gap;
            __syncthreads();
            for (int width = blockDim.x/2; width; width /= 2) {
                if (threadIdx.x < width) reduction[threadIdx.x] =
                    fmax(reduction[threadIdx.x], reduction[threadIdx.x+width]);
                __syncthreads();
            }
            if (threadIdx.x == 0) control->pressure_gap_by_block[blockIdx.x] = reduction[0];
            grid.sync();
            if (tid == 0) {
                double max_gap = 0;
                for (int b = 0; b < int(gridDim.x); ++b)
                    max_gap = fmax(max_gap, control->pressure_gap_by_block[b]);
                control->last_max_pressure_gap_pa = max_gap;
            }
            grid.sync();
        }
    }
    if (current != initial)
        for (int i = tid; i < node_count; i += stride) initial[i] = current[i];
}
inline cudaError_t launch(int node_count, int edge_count, const int* offsets,
    const int* incidence, const kg::Edge* edges, kg::Node* initial, kg::Node* scratch,
    Proposal* proposals, Budget* budgets, kg::Clock* clocks, GasChannelState* channels,
    kg::Settings settings, GasModel model, double duration, int* error, Control* control,
    int block=256, int requested_blocks=0, GasCombustionView combustion={}) {
    int device=0,supported=0,resident=0,sms=0;
    auto code=cudaGetDevice(&device);if(code!=cudaSuccess)return code;
    code=cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device);if(code!=cudaSuccess)return code;
    if(!supported)return cudaErrorNotSupported;
    code=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(code!=cudaSuccess)return code;
    code=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,advance,block,0);if(code!=cudaSuccess)return code;
    int blocks=std::max(1,std::min(8,(std::max(node_count,edge_count)+block-1)/block));
    if(requested_blocks>0)blocks=requested_blocks;
    if(blocks>32 || blocks>resident*sms)return cudaErrorCooperativeLaunchTooLarge;
    void* arguments[]{&node_count,&edge_count,&offsets,&incidence,&edges,&initial,&scratch,
        &proposals,&budgets,&clocks,&channels,&settings,&model,&duration,&error,&control,&combustion};
    return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(advance),blocks,block,arguments);
}
} // namespace battery::batched_gas

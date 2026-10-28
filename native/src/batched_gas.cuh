#pragma once
#include "battery_native/cuda_solver.hpp"
#include "battery_native/kawasaki_gas_math.hpp"
#include "battery_native/gas_combustion.hpp"
#include "battery_native/reaction_transport_limits.hpp"
#include "battery_native/gas_mixing.hpp"
#include "battery_native/gas_local_reactor.hpp"
#include "battery_native/gas_surface_exchange.hpp"
#include "battery_native/reactive_transit.hpp"
#include "battery_native/time_step_policy.hpp"
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
    double next_pressure_cap = 0.;
    double advective_energy_delta = 0.;
    double energy = 0, chemical = 0, oxidizer = 0;
    double donor_pressure_weight = 0, receiver_pressure_weight = 0;
    double vapor=0,latent=0;
    MixingTransfer mixing;
    bool tail = false;
};
struct Budget { double scale = 1, outgoing = 0, energy = 0; };
struct Control {
    double entropy_by_block[32];
    double pressure_gap_by_block[32];
    double reaction_step_by_block[32],reaction_step;
    double accepted_reaction_step=0.;
    double last_max_pressure_gap_pa;
    unsigned reaction_intervals=1;
    unsigned max_local_reactor_iterations=0;
    double averaged_scale=1.;
    unsigned averaged_backtracks=0;
    double averaged_min_scale=1.;
    unsigned averaged_windows=0,averaged_limited_windows=0,averaged_entropy_backtracks=0;
    // Failure-only diagnostics. No host transfer on an accepted window.
    unsigned averaged_invalid_mask[8]{};
    int averaged_first_invalid[8]{};
    double averaged_entropy[8]{},averaged_attempt_scale[8]{};
    double averaged_window_s=0.,averaged_elapsed_s=0.;
    bool averaged_old_is_initial=true;
    unsigned averaged_pressure_used=0;
    double averaged_pressure_residual=0.;
    unsigned averaged_nonlinear_used=0;
    unsigned averaged_first_linear_used=0;
    double averaged_flow_defect=0.;
    unsigned enthalpy_iterations=0;
    double enthalpy_residual=0.;
    bool pressure_pattern_valid=false;
    unsigned pressure_pattern_attempts=0,pressure_pattern_accepted=0;
    unsigned mixing_iterations=0;
    double mixing_residual=0.;
    double mixing_raw_residual=0.;
    double mixing_solid_entropy=0.;
    unsigned reactive_iterations=0;
    double reactive_residual=0.;
};
struct AverageState {
    double pressure=0.,mass=0.,chemical=0.,oxygen=0.,vapor=0.,latent=0.,temperature=0.;
    double compliance=0.,diagonal=0.,residual=0.,direction=0.,matvec=0.,correction=0.;
    double pressure_pattern=0.;
    int pressure_peer=-1;
    double pressure_coupling=0.;
    struct Reactive {
        double incoming=0.,outgoing=0.,exposure=0.,rate_dt=0.,heat=0.,vapor_burned=0.;
        double fuel=0.,oxygen=0.,vapor=0.,latent=0.; // integrated outgoing amounts
    } reaction;
};
#include "averaged_gas.cuh"
#include "gas_mixing.cuh"
#include "implicit_gas_mixing.cuh"

// Extra kinetic subcycling is needed only where reactants can meet and ignite.
// Include upwind arrivals (and their enthalpy), not only the resident snapshot.
// A cold or oxygen-free fuel stock behind a closed barrier must not make every
// mass step expensive merely because a distant reactor has a large rate.
__device__ inline double neighbourhood_reaction_step(int i,const kg::Node* nodes,
    const GasCombustionLaw& law,const kg::Settings& settings,const kg::Edge* edges,
    const GasChannelState* channels,const int* offsets,const int* incidence,GasModel model,
    const GasCombustionLaw* laws,double horizon,const GasFilmState* films=nullptr) {
    if(!(law.rate_per_s>0.))return 1e300;
    const auto& n=nodes[i];
    ReactionArrivals arrivals;
    double rise=model.max_temperature_step_k,resident=-1.;
    if(films) {
        if(films[i].conductance>0.)rise=gas_coupling_rise(n,settings,films[i],rise);
        else if(n.solid_capacity==0.)rise*=fmax(1.,.02*kg::temperature(n,settings));
        const double budget=.5*rise*(n.solid_capacity+settings.cv*n.mass);
        resident=1e300;
        if(combustion_available(n,law)>budget) {
            // A cheap upper rate up to the allowed rise. Do NOT run an
            // adaptive hypothetical ignition for the entire remaining window
            // every time the graph accepts a much shorter interval.
            const double rate=combustion_rate(law,kg::temperature(n,settings)+rise);
            if(rate>0.)resident=budget/(rate*combustion_available(n,law));
        }
        // The coupled local ODE resolves REAL arrivals after the conservative
        // pressure/mixing gather, in this same interval. Do not impose the
        // lifetime of speculative, uncapped future orifice streams globally.
        // Transport itself has an independent refined coupling-window cap.
        return resident;
    }
    horizon=fmin(horizon,resident>=0.?resident:
        resident_reaction_step(n,law,settings,model.max_fraction_per_step,model.max_temperature_step_k));
    for(int j=offsets[i];j<offsets[i+1];++j) {
        const int e=incidence[j];const auto& edge=edges[e];const auto& channel=channels[e];
        if(!channel.open || !(channel.effective_area_m2>0 && edge.cd>0))continue;
        const int other=edge.a==i?edge.b:edge.a;const auto& incoming=nodes[other];
        auto open_edge=edge;open_edge.area=channel.effective_area_m2;
        bound_reaction_arrival(arrivals,incoming,n,laws[other],law,open_edge,settings,horizon,
            model.max_temperature_step_k,model.max_fraction_per_step);
        bound_mixing_arrival(arrivals,incoming,n,laws[other],law,open_edge,settings,model,horizon);
    }
    return arrival_reaction_step(n,law,settings,arrivals,horizon,model.max_fraction_per_step,rise,resident);
}

__device__ inline bool local_process(kg::Node& node,const GasCombustionLaw& law,
    const kg::Settings& settings,const GasModel& model,double step,GasFilmState* film,Control* control) {
    if(model.minimum_step_s>0.) {
        // Coarse mode: one exact frozen-coefficient stage, NEVER hidden
        // adaptive microcycling. Strang half-stages are algebraic evaluations.
        atomicMax(&control->max_local_reactor_iterations,1u);
        if(film && film->conductance>0.) {
            frozen_gas_reactor(node,law,settings,step,*film);
            return isfinite(node.energy) && node.energy>0.;
        }
        return frozen_gas_combustion(node,law,settings,step);
    }
    if(film && film->conductance>0.) {
        unsigned iterations=0;
        const bool ok=advance_gas_reactor(node,law,settings,step,*film,
            .05*model.max_temperature_step_k,model.max_fraction_per_step,&iterations);
        if(iterations)atomicMax(&control->max_local_reactor_iterations,iterations);
        return ok;
    }
    return burn_gas(node,law,settings,step,model.max_fraction_per_step,model.max_temperature_step_k);
}

__device__ inline bool thermal_reaction_stage(int i,kg::Node& node,GasCombustionView chemistry,
    const kg::Settings& settings,const GasModel& model,double step,Control* control) {
    if(!chemistry.exchange.patches || model.joint_gas_surface_mixing)
        return local_process(node,chemistry.laws[i],settings,model,step,nullptr,control);
    auto film=evaluate_gas_surface_contacts(i,kg::temperature(node,settings),chemistry.exchange);
    const bool ok=local_process(node,chemistry.laws[i],settings,model,step,&film,control);
    record_gas_surface_heat(i,chemistry.exchange,film,step);
    chemistry.films[i]=film;
    return ok;
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
static __global__ void advance(int node_count, int edge_count, const int* offsets,
    const int* incidence, const kg::Edge* edges, kg::Node* initial, kg::Node* scratch,
    Proposal* proposals, Budget* budgets, kg::Clock* clocks, GasChannelState* channels,
    kg::Settings settings, GasModel model, double duration, int* error, Control* control,
    GasCombustionView combustion,AverageState* average_a,AverageState* average_b,MixSolveState* mix_work) {
    const auto grid = cooperative_groups::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    __shared__ double reduction[256];
    auto* current = initial;
    auto* next = scratch;
    double elapsed = 0;
    unsigned iterations = 0;
    if (tid == 0){control->last_max_pressure_gap_pa=0;control->max_local_reactor_iterations=0;
        // Cache is local to this launch: retries/checkpoint restores cannot
        // accidentally inherit a pattern from an uncommitted future state.
        control->pressure_pattern_valid=false;
        control->pressure_pattern_attempts=control->pressure_pattern_accepted=0;
        control->averaged_min_scale=1.;control->averaged_windows=control->averaged_limited_windows=control->averaged_entropy_backtracks=0;}
    grid.sync();
    const unsigned requested_fixed=static_cast<unsigned>(model.deterministic_sweeps_per_step);
    const bool network = model.cuda_network_transport;
    while (elapsed < duration) {
        if (++iterations > 200000) { if (tid == 0) *error = network ? 18 : 14; return; }
        // Recompute on the current snapshot, including possible hot arrivals.
        // A reference k(T_ref) is not an upper bound for an Arrhenius reactor.
        const double transport_interval=network?(model.coupled_gas_films?
            fmin(duration,model.coupled_transport_interval_s):duration):(model.cuda_deterministic_transport?
            duration/requested_fixed:model.local_exchange_interval_s);
        double chemistry_interval=1e300;
        if(combustion.enabled)for(int i=tid;i<node_count;i+=stride)
            chemistry_interval=fmin(chemistry_interval,neighbourhood_reaction_step(i,current,
                combustion.laws[i],settings,edges,channels,offsets,incidence,model,
                combustion.laws,fmin(duration-elapsed,transport_interval),combustion.films));
        reduction[threadIdx.x]=chemistry_interval;__syncthreads();
        for(int width=blockDim.x/2;width;width/=2) {
            if(threadIdx.x<width)reduction[threadIdx.x]=fmin(reduction[threadIdx.x],reduction[threadIdx.x+width]);
            __syncthreads();
        }
        if(threadIdx.x==0)control->reaction_step_by_block[blockIdx.x]=reduction[0];
        grid.sync();
        if(tid==0) {
            double limit=1e300;
            for(int b=0;b<int(gridDim.x);++b)limit=fmin(limit,control->reaction_step_by_block[b]);
            control->reaction_step=limit;control->reaction_intervals=iterations;
        }
        grid.sync();
        // Tau-leap on the configured physical window. A stiff edge is limited
        // by the common-snapshot pressure/inventory budgets below; its tiny
        // characteristic time no longer stalls every other edge in the graph.
        const double desired=fmin(transport_interval,control->reaction_step);
        const double step=balanced_transport_step(duration-elapsed,desired,model.minimum_step_s);
        if(tid==0)control->accepted_reaction_step=step;
        if (!(step > 0) || elapsed + step == elapsed) { if (tid == 0) *error = 15; return; }
        if(combustion.laws && !model.reactive_transit) {
            // Symmetric reaction/transport integration: old resident gas reacts
            // before leaving, new arrivals react in the SAME physical interval.
            // Every subsequent flux sees the released heat and changed pressure.
            for(int i=tid;i<node_count;i+=stride)
                if(!thermal_reaction_stage(i,current[i],combustion,settings,model,step/2,control))atomicExch(error,19);
            grid.sync();
            if(*error)return;
        }
        if(network && model.averaged_network_transport) {
            if(tid==0){control->averaged_window_s=step;control->averaged_elapsed_s=elapsed;
                control->averaged_old_is_initial=current==initial;}
            if(!average_transport(grid,tid,stride,node_count,edge_count,offsets,incidence,edges,
                current,next,proposals,channels,settings,model,step,duration,control,
                average_a,average_b,reduction,error,combustion))return;
        } else {
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
            p.vapor=donor.vapor*(p.mass/donor.mass);p.latent=donor.latent*(p.mass/donor.mass);
            const double sign = p.donor == edges[e].a ? 1 : -1;
            auto& channel = channels[e];
            channel.transferred_kg += sign * p.mass;
            channel.transferred_energy_j += sign * (p.energy - p.mass * settings.cv * model.reference_temperature_k);
            channel.transferred_chemical_j += sign * p.chemical;
            channel.transferred_oxidizer_kg += sign * p.oxidizer;
            channel.transferred_vapor_kg+=sign*p.vapor;channel.transferred_latent_j+=sign*p.latent;
            channel.mass_flow_kg_s += sign * p.mass / duration;
        }
        grid.sync();
        for (int i = tid; i < node_count; i += stride) {
            auto node = current[i];
            for (int j = offsets[i]; j < offsets[i+1]; ++j) {
                const auto& p = proposals[incidence[j]];
                const double sign = p.donor == i ? -1 : 1;
                node.mass += sign * p.mass; node.energy += sign * p.energy;
                node.chemical += sign * p.chemical; node.oxidizer += sign * p.oxidizer;
                node.vapor+=sign*p.vapor;node.latent+=sign*p.latent;
            }
            if (!(node.mass > 0 && node.energy > 0 && node.chemical >= 0 && node.oxidizer >= 0)) atomicExch(error, 16);
            // Audit transport entropy separately; reaction heat must not hide a
            // nonphysical entropy loss in the conservative transfer operator.
            next[i] = node;
        }
        grid.sync();
        }
        if(model.mixing_diffusivity_m2_s>0. || model.mixing_flow_fraction>0. || model.joint_gas_surface_mixing){
            if(model.implicit_network_mixing){
                if(!implicit_mix_snapshot(grid,tid,stride,node_count,edge_count,offsets,incidence,edges,next,
                    proposals,channels,settings,model,step,mix_work,control,reduction,error,combustion.exchange))return;
            }else mix_snapshot(grid,tid,stride,node_count,edge_count,offsets,incidence,edges,next,
                proposals,budgets,channels,settings,model,step);
        }
        // Audit the full conservative operator, including zero-net-mass mixing.
        double entropy_change=model.joint_gas_surface_mixing && tid==0?control->mixing_solid_entropy:0.;
        for(int i=tid;i<node_count;i+=stride) {
            auto& node=next[i];
            if(!(node.mass>0 && node.energy>0 && node.chemical>=0 && node.oxidizer>=0 &&
                node.vapor>=0 && node.vapor<=node.mass+1e-14 && node.latent>=0))atomicExch(error,16);
            entropy_change+=kg::entropy(node,settings)-kg::entropy(current[i],settings);
            if(combustion.laws && !model.reactive_transit && !thermal_reaction_stage(i,node,combustion,settings,model,step/2,control))atomicExch(error,19);
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
        if ((model.cuda_deterministic_transport || network) && elapsed>=duration) {
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
    int block=256, int requested_blocks=0, GasCombustionView combustion={},
    AverageState* average_a=nullptr,AverageState* average_b=nullptr,MixSolveState* mix_work=nullptr) {
    if(model.averaged_network_transport && (!average_a || !average_b))return cudaErrorInvalidValue;
    if(model.implicit_network_mixing && !mix_work)return cudaErrorInvalidValue;
    if(requested_blocks==0 && model.implicit_network_mixing && node_count<=4096)block=64;
    int device=0,supported=0,resident=0,sms=0;
    auto code=cudaGetDevice(&device);if(code!=cudaSuccess)return code;
    code=cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device);if(code!=cudaSuccess)return code;
    if(!supported)return cudaErrorNotSupported;
    code=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(code!=cudaSuccess)return code;
    code=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,advance,block,0);if(code!=cudaSuccess)return code;
    int blocks=std::max(1,std::min(8,(std::max(node_count,edge_count)+block-1)/block));
    if(requested_blocks==0 && model.implicit_network_mixing && node_count<=4096)
        blocks=std::max(1,std::min({32,sms,(std::max(node_count,edge_count)+block-1)/block}));
    if(requested_blocks>0)blocks=requested_blocks;
    if(blocks>32 || blocks>resident*sms)return cudaErrorCooperativeLaunchTooLarge;
    void* arguments[]{&node_count,&edge_count,&offsets,&incidence,&edges,&initial,&scratch,
        &proposals,&budgets,&clocks,&channels,&settings,&model,&duration,&error,&control,&combustion,&average_a,&average_b,&mix_work};
    return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(advance),blocks,block,arguments);
}
} // namespace battery::batched_gas

#include "battery_native/cuda_solver.hpp"
#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include "battery_native/failure.hpp"
#include "battery_native/reaction.hpp"
#include "battery_native/conversion.hpp"
#include "battery_native/joule.hpp"
#include "battery_native/gas_timestep.hpp"
#include "battery_native/shell_opening.hpp"
#include "battery_native/deterministic_aperture.hpp"
#include "battery_native/aperture_transfer.hpp"
#include "battery_native/mass_mesh.hpp"
#include "battery_native/local_gas_transport.hpp"
#include "battery_native/kawasaki_transport.hpp"
#include "battery_native/mixture_equilibrium.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace battery {
namespace {
void check(cudaError_t result, const char* context) {
    if(result!=cudaSuccess) throw std::runtime_error(std::string(context)+": "+cudaGetErrorString(result));
}
template<class T> struct Buffer {
    T* p=nullptr;std::size_t n=0;
    explicit Buffer(std::size_t count):n(count) { if(n) check(cudaMalloc(&p,n*sizeof(T)),"CUDA allocation"); }
    ~Buffer() { if(p) cudaFree(p); }
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void upload(const std::vector<T>& v) { if(n) check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice),"CUDA upload"); }
    void download(std::vector<T>& v) const { v.resize(n);if(n) check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost),"CUDA download"); }
};
struct DeviceRows { const int* offsets;const int* ids; };
struct StepResult {double dt;unsigned invalid;};
// Stable CSR incidence lists: gather touches neighbours only, in input order.
// No device-side searching, appending, sorting or floating-point atomics.
struct Incidence {
    Buffer<int> offsets;std::unique_ptr<Buffer<int>> ids;
    explicit Incidence(const std::vector<std::vector<int>>& rows):offsets(rows.size()+1) {
        std::vector<int> starts{0},flat;
        for(const auto& row:rows) {flat.insert(flat.end(),row.begin(),row.end());starts.push_back(static_cast<int>(flat.size()));}
        offsets.upload(starts);ids=std::make_unique<Buffer<int>>(flat.size());ids->upload(flat);
    }
    DeviceRows view() const {return {offsets.p,ids->p};}
};
struct Material {
    double capacity, energy, duration, onset, rate, temperature_scale, aging_multiplier;
    bool isothermal;
    bool is_battery;
    HeatCapacityLaw capacity_law;
    RunawayPulse pulse;
    BulkBoundaryPhase boundary_phase;
    ReactionKinetics kinetics;
    ReactionStructure structure;
    EnergyOutletLaw outlet;
    ConversionLaw conversion;
    GasModel gas_model;
    ElectricalLaw electrical;
    CurrentView current;
};
__device__ double maximum_structure(const Material& m,int axis) {
    if(m.conversion.enabled)return fmax(1.,m.conversion.residue_conductivity[axis]);
    return m.structure.enabled?fmax(1.,fmax(m.structure.active_factor[axis],m.structure.residue_factor[axis])):1.;
}
// Geometry and verbose law records are cold data; kernels stream only this
// compact connectivity record. Dynamic material laws are shared by index.
struct DeviceVolume { int object; double capacity_j_per_k; };
// Close derived constitutive fields from authoritative energy/composition.
// Used after thermal evolution AND after discrete structural events.
__device__ void close_thermal_state(const Material& m,const DeviceVolume& v,ThermalState& s,double speed) {
    if(m.conversion.enabled) {
        const double tref=m.gas_model.reference_temperature_k;
        s.temperature_k=tref+(s.stored_energy_j-v.capacity_j_per_k*tref)/
            conversion_capacity(m.conversion,s,v.capacity_j_per_k,v.capacity_j_per_k/m.capacity);
        refresh_conversion(m.conversion,s,v.capacity_j_per_k/m.capacity,speed);
    } else {
        if(m.capacity_law.factor.count)
            s.temperature_k=thermal_temperature(m.capacity_law,v.capacity_j_per_k,s.stored_energy_j,s.damage);
        if(m.pulse.profile==RunawayProfile::Kinetic)
            refresh_reaction(m.kinetics,m.structure,s,m.energy*v.capacity_j_per_k/m.capacity,speed);
    }
}
struct DeviceLink {
    int a,b,interface_index,law_index;
    double weight,series_resistance_k_per_w,boundary_temperature_k,maximum_g;
    int boundary_index;
    int geometry_axis;
    double series_resistance_a_k_per_w;
};
struct ProfilePoint { double time,temperature; };
__device__ double boundary_temperature(const DeviceLink& e,double time,const int* offsets,const ProfilePoint* points) {
    if(e.boundary_index<0)return e.boundary_temperature_k;
    int lo=offsets[e.boundary_index],hi=offsets[e.boundary_index+1];
    if(hi==lo)return e.boundary_temperature_k;
    if(time<=points[lo].time)return points[lo].temperature;
    if(time>=points[hi-1].time)return points[hi-1].temperature;
    --hi;
    while(hi-lo>1) {int mid=(lo+hi)/2;if(points[mid].time<=time)lo=mid;else hi=mid;}
    const auto a=points[lo],b=points[hi];
    return a.temperature+(b.temperature-a.temperature)*(time-a.time)/(b.time-a.time);
}
auto law_key(const ConnectivityLaw& l) {
    std::vector<double> key{static_cast<double>(l.kind),l.l0_m,l.l_low_m,l.l_high_m,l.transition_temperature_k,
        l.transition_width_k,l.hysteresis_k,l.relaxation_time_s,l.background_conductance_w_per_k,
        l.active_conductance_w_per_k,l.activation_width_m,l.damage_onset_k,l.damage_rate_at_onset_per_s,
        l.damage_temperature_scale_k,l.damage_l_shift_m,l.damage_conductance_factor_at_one,l.destruction_damage,
        l.destroyed_l0_m,l.destroyed_l_m,l.destroyed_background_conductance_w_per_k,l.destroyed_active_conductance_w_per_k,l.endpoint_damage_weight,l.geometry_reference_logit,
        l.destroyed_geometry_reference_logit};
    for(const auto& r:{l.endpoint_a,l.endpoint_b}) {
        key.push_back(r.reference_temperature_k);key.push_back(r.temperature_scale_k);
        for(const auto v:{r.linear,r.quadratic,r.damage,r.temperature_contrast,r.damage_contrast})for(double x:{v.x,v.y,v.z})key.push_back(x);
        key.push_back(r.temperature_log_response.count);
        for(int i=0;i<r.temperature_log_response.count;++i) {
            key.push_back(r.temperature_log_response.temperature_k[i]);key.push_back(r.temperature_log_response.value[i]);
        }
    }
    return key;
}
__device__ double clamp(double x,double lo,double hi) { return fmin(fmax(x,lo),hi); }
__device__ double logistic(double x) { return 1.0/(1.0+exp(-clamp(x,-80,80))); }
__device__ double damage_rate(double t,double onset,double rate,double scale) {
    return t<onset||rate<=0?0:rate*exp(clamp((t-onset)/scale,-80,80));
}
__device__ void minimum_positive(double* address,double value) {
    auto ptr=reinterpret_cast<unsigned long long*>(address);
    unsigned long long old=atomicCAS(ptr,0ULL,0ULL),assumed;
    while(value<__longlong_as_double(old)) {
        assumed=old;old=atomicCAS(ptr,assumed,__double_as_longlong(value));
        if(old==assumed) break;
    }
}
__device__ void block_minimum(double value,double* output) {
    __shared__ double partial[128];partial[threadIdx.x]=value;__syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(threadIdx.x<stride)partial[threadIdx.x]=fmin(partial[threadIdx.x],partial[threadIdx.x+stride]);
        __syncthreads();
    }
    if(threadIdx.x==0)minimum_positive(output,partial[0]);
}
__global__ void cfl_kernel(int count,const DeviceVolume* volume,
    const DeviceLink* links,const int* offsets,const int* adjacency,
    const Material* materials,RuntimeParameters runtime,double safety,double* cfl,DeviceRows patch_rows,const GasThermalPatch* patches) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    const bool spatial=i<count && !materials[volume[i].object].isothermal;
    double sum=0;
    for(int j=spatial?offsets[i]:0;j<(spatial?offsets[i+1]:0);++j) {
        const auto& e=links[adjacency[j]];
        double scale=e.b<0?runtime.ambient_conductance_scale:(e.interface_index>=0?runtime.interface_conductance_scale:1.0);
        double g=e.maximum_g*scale;
        const auto pa=materials[volume[e.a].object].boundary_phase;
        const auto pb=e.b>=0?materials[volume[e.b].object].boundary_phase:BulkBoundaryPhase{};
        if(g>0) g=1.0/(1.0/g+minimum_phase_resistance(e.series_resistance_k_per_w,e.series_resistance_a_k_per_w,pa,pb,
            maximum_structure(materials[volume[e.a].object],e.geometry_axis),
            e.b>=0?maximum_structure(materials[volume[e.b].object],e.geometry_axis):1.));
        sum+=g;
    }
    // Ordinary gas contacts are static. Rupture links are bounded separately
    // from their CURRENT opening and flow, never from a fully open panel.
    if(spatial)for(int j=patch_rows.offsets[i];j<patch_rows.offsets[i+1];++j) {
        const auto& patch=patches[patch_rows.ids[j]];
        if(patch.channel<0)sum+=patch.conductance_w_per_k;
    }
    const double factor=spatial?minimum_capacity_factor(materials[volume[i].object].capacity_law):1.;
    double capacity=spatial?volume[i].capacity_j_per_k*factor:1.;
    if(spatial && materials[volume[i].object].conversion.enabled) {
        const auto& m=materials[volume[i].object];capacity=conversion_min_capacity(m.conversion,volume[i].capacity_j_per_k,volume[i].capacity_j_per_k/m.capacity);
    }
    double limit=sum>0?safety*capacity/sum:1e300;
    block_minimum(limit,cfl);
}
// Isothermal objects exchange heat through their WHOLE heat capacity, not the
// capacity of a boundary voxel. The bound must not shrink when the grid changes.
__global__ void isothermal_cfl_kernel(const DeviceVolume* volumes,const DeviceLink* links,
    const int* object_offsets,const int* offsets,const int* adjacency,const Material* materials,
    RuntimeParameters runtime,double safety,double* cfl) {
    const int object=blockIdx.x,t=threadIdx.x;
    if(!materials[object].isothermal)return;
    double sum=0;
    for(int i=object_offsets[object]+t;i<object_offsets[object+1];i+=blockDim.x)
        for(int j=offsets[i];j<offsets[i+1];++j) {
            const auto& e=links[adjacency[j]];
            double scale=e.b<0?runtime.ambient_conductance_scale:runtime.interface_conductance_scale;
            double g=e.maximum_g*scale;
            const auto pa=materials[volumes[e.a].object].boundary_phase;
            const auto pb=e.b>=0?materials[volumes[e.b].object].boundary_phase:BulkBoundaryPhase{};
            sum+=g>0?1.0/(1.0/g+minimum_phase_resistance(e.series_resistance_k_per_w,e.series_resistance_a_k_per_w,pa,pb,
                maximum_structure(materials[volumes[e.a].object],e.geometry_axis),
                e.b>=0?maximum_structure(materials[volumes[e.b].object],e.geometry_axis):1.)):0;
        }
    __shared__ double partial[128];partial[t]=sum;__syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(t<stride)partial[t]+=partial[t+stride];
        __syncthreads();
    }
    if(t==0 && partial[0]>0)minimum_positive(cfl,safety*materials[object].capacity/partial[0]);
}
__global__ void stability_kernel(int count,const ThermalState* state,const DeviceVolume* volume,
    const Material* materials,RuntimeParameters runtime,const double* cfl,double* dt,double time,
    double safety,DeviceRows patch_rows,const GasThermalPatch* patches,const double* conductance) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;double limit=*cfl;
    if(i<count) {
        const auto& m=materials[volume[i].object];const auto p=m.boundary_phase;
        double joule_power=0.;
        if(m.electrical.enabled) {
            const double current=maximum_current(m.electrical,m.current,time,time+runtime.max_dt_s);
            joule_power=current*current*electrical_resistance(m.electrical,state[i].temperature_k,state[i].damage)*volume[i].capacity_j_per_k/m.capacity;
            const double capacity=m.conversion.enabled?
                conversion_min_capacity(m.conversion,volume[i].capacity_j_per_k,volume[i].capacity_j_per_k/m.capacity):
                volume[i].capacity_j_per_k*minimum_capacity_factor(m.capacity_law);
            if(joule_power>0)limit=fmin(limit,m.electrical.max_temperature_step_k*capacity/joule_power);
        }
        double aperture_g=0.;
        for(int j=patch_rows.offsets[i];j<patch_rows.offsets[i+1];++j) {
            const int patch=patch_rows.ids[j];
            if(patches[patch].channel>=0)aperture_g+=conductance[patch];
        }
        if(aperture_g>0.) {
            const double capacity=m.conversion.enabled?
                conversion_min_capacity(m.conversion,volume[i].capacity_j_per_k,volume[i].capacity_j_per_k/m.capacity):
                volume[i].capacity_j_per_k*minimum_capacity_factor(m.capacity_law);
            // 1/cfl bounds every static G/(safety*C). ADD the current
            // aperture contribution: min(dt_static,dt_aperture) is not enough
            // when both exchanges consume the same solid heat capacity.
            limit=1./(1./limit+aperture_g/(safety*capacity));
        }
        if(m.conversion.enabled) {
            const auto& c=m.conversion;const auto& s=state[i];const auto& g=m.gas_model;
            const double current=m.electrical.enabled?maximum_charging_current(m.electrical,m.current,time,time+runtime.max_dt_s):0.;
            const double ka=(degradation_rate(c,s.temperature_k)+overcharge_r1_rate(c,m.electrical,s.temperature_k,current))*runtime.runaway_power_scale;
            const double kb=completion_rate(c,s.temperature_k)*runtime.runaway_power_scale;
            if(s.reactant_kg>c.mass_kg*1e-15 && ka>0)limit=fmin(limit,-log1p(-c.max_fraction_per_step)/ka);
            const double fraction=volume[i].capacity_j_per_k/m.capacity;
            const double capmin=conversion_min_capacity(c,volume[i].capacity_j_per_k,fraction);
            const double ra=ka*s.reactant_kg,rb=kb*s.activated_kg+ra;
            const double yield=maximum_gas_yield(c);
            const double capacity_change=fmax(std::abs(c.residue_cp-c.activated_cp),std::abs((1-yield)*c.residue_cp-c.activated_cp));
            // Bound total transformed mass, not the fractional turnover of an
            // arbitrarily small quasi-steady intermediate. Include new feed in
            // the heat bound because it can complete within this same substep.
            if(ra+rb>0)limit=fmin(limit,c.max_fraction_per_step*c.mass_kg*fraction/(ra+rb));
            const double power=joule_power+ra*std::abs(c.activated_cp-c.reactant_cp)*std::abs(s.temperature_k-g.reference_temperature_k)+
                rb*(c.net_heat_j_per_kg+yield*g.cv_j_per_kg_k*std::abs(s.temperature_k-g.reference_temperature_k)+
                    capacity_change*std::abs(s.temperature_k-g.reference_temperature_k));
            if(power>0)limit=fmin(limit,fmin(c.max_temperature_step_k,.1*s.temperature_k)*capmin/power);
        }
        if(m.pulse.profile==RunawayProfile::Kinetic && state[i].mode!=CellMode::Spent) {
            const auto& s=state[i];const auto& k=m.kinetics;
            const double aging=damage_rate(s.temperature_k,m.onset,m.rate,m.temperature_scale)*m.aging_multiplier*runtime.damage_rate_scale;
            if(aging>0 && s.damage<1)limit=fmin(limit,k.max_fraction_per_step/aging);
            if(reaction_live(s) && runtime.runaway_power_scale>0) {
                const double rate=reaction_rate(k,s.temperature_k,s.damage)*runtime.runaway_power_scale;
                const double budget=m.energy*volume[i].capacity_j_per_k/m.capacity;
                const double opened=outlet_opening(m.outlet,reaction_conversion(s,budget));
                const double total_rate=rate*(1+opened*m.outlet.ejected_energy_per_released_energy);
                limit=fmin(limit,-log1p(-k.max_fraction_per_step)/total_rate);
                const double power=rate*s.chemical_energy_remaining_j;
                if(power>0)limit=fmin(limit,k.max_temperature_rise_per_step_k*volume[i].capacity_j_per_k*
                    minimum_capacity_factor(m.capacity_law)/power);
            }
        }
        if(p.relaxation_time_s>0 && state[i].runaway_started_s>=0) {
            const double left=6*p.relaxation_time_s-(time-state[i].runaway_started_s);
            if(left>1e-12)limit=fmin(limit,fmin(.1*p.relaxation_time_s,left));
        }
    }
    if(i<count && state[i].mode==CellMode::Runaway && runtime.runaway_power_scale>0) {
        const auto& m=materials[volume[i].object];
        if(m.pulse.profile==RunawayProfile::Kinetic) { /* limited above, including quenched nodes */ }
        else if(m.pulse.profile==RunawayProfile::BurstTail) {
            // Resolve a rapid pulse in the coupled transport, even though its
            // energy integral itself is exact for arbitrarily large dt.
            const double age=state[i].reaction_progress_s;
            const double tau=m.pulse.fast_fraction>0 && age<6*m.duration?m.duration:m.pulse.tail_time_s;
            const double left=pulse_end(m.pulse,m.duration)-age;
            if(left>1e-12)limit=fmin(limit,fmin(.1*tau,left)/runtime.runaway_power_scale);
        } else {
        double energy=m.energy*volume[i].capacity_j_per_k/m.capacity;
        double remaining=fmax(0.0,energy-state[i].released_energy_j);
        double power=energy/m.duration*runtime.runaway_power_scale;
        if(power>0 && remaining>energy*1e-12) limit=fmin(limit,remaining/power);
        }
    }
    block_minimum(limit,dt);
}
__device__ InterfaceState evolve(const DeviceLink& edge,const ConnectivityLaw& l,InterfaceState s,double t,double damage,double dt,double scale) {
    s.damage=fmax(s.damage,l.endpoint_damage_weight*damage);
    s.damage=fmin(1.0,s.damage+damage_rate(t,l.damage_onset_k,l.damage_rate_at_onset_per_s,l.damage_temperature_scale_k)*dt);
    if(s.damage>=l.destruction_damage) {s.phase=EdgePhase::Destroyed;s.l0_m=l.destroyed_l0_m;}
    double target=l.l_low_m;
    if(s.phase==EdgePhase::Destroyed) target=l.destroyed_l_m;
    else {
        if(l.kind==TransitionKind::Smooth) target+= (l.l_high_m-l.l_low_m)*logistic((t-l.transition_temperature_k)/l.transition_width_k);
        if(l.kind==TransitionKind::FirstOrder) {
            if(s.phase==EdgePhase::Low && t>=l.transition_temperature_k+l.hysteresis_k/2)s.phase=EdgePhase::High;
            else if(s.phase==EdgePhase::High && t<=l.transition_temperature_k-l.hysteresis_k/2)s.phase=EdgePhase::Low;
            target=s.phase==EdgePhase::High?l.l_high_m:l.l_low_m;
        }
        if(l.kind==TransitionKind::SecondOrder) {
            double reduced=fmax(0.0,(t-l.transition_temperature_k)/l.transition_width_k);
            target+=(l.l_high_m-l.l_low_m)*fmin(1.0,sqrt(reduced));
            s.phase=reduced>0?EdgePhase::High:EdgePhase::Low;
        }
        target+=l.damage_l_shift_m*s.damage;
    }
    target=fmax(0.0,target);
    s.l_m=l.relaxation_time_s>0?s.l_m+(target-s.l_m)*(-expm1(-dt/l.relaxation_time_s)):target;
    bool destroyed=s.phase==EdgePhase::Destroyed;
    double bg=destroyed?l.destroyed_background_conductance_w_per_k:l.background_conductance_w_per_k;
    double active=destroyed?l.destroyed_active_conductance_w_per_k:l.active_conductance_w_per_k;
    double g=(bg+active*logistic((s.l_m-s.l0_m)/l.activation_width_m))*pow(l.damage_conductance_factor_at_one,s.damage)*edge.weight*scale;
    s.conductance_w_per_k=g>0?1.0/(1.0/g+edge.series_resistance_k_per_w):0;
    s.destroyed_fraction=destroyed?1.0:0.0;
    return s;
}
// Dispatch is done once on the host into fixed/directed/state/transition pools.
template<int Kind>
__global__ void edge_kernel(int count,const int* ids,const DeviceLink* links,const InterfaceState* old,InterfaceState* states,
    const ThermalState* volumes,RuntimeParameters runtime,const double* dt,double* flux,const double* constant_g,const ConnectivityLaw* laws,
    double time,const int* profile_offsets,const ProfilePoint* profile_points,const DeviceVolume* mesh_volumes,const Material* materials) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;const int i=ids[j];
    states[i]=old[i];
    const auto& e=links[i];
    const double series_r=phase_resistance(e.series_resistance_k_per_w,e.series_resistance_a_k_per_w,
        volumes[e.a].damage,e.b>=0?volumes[e.b].damage:0.,materials[mesh_volumes[e.a].object].boundary_phase,
        e.b>=0?materials[mesh_volumes[e.b].object].boundary_phase:BulkBoundaryPhase{},
        time,volumes[e.a].runaway_started_s,e.b>=0?volumes[e.b].runaway_started_s:-1.,
        volumes[e.a].conductivity_factor[e.geometry_axis],e.b>=0?volumes[e.b].conductivity_factor[e.geometry_axis]:1.);
    double ta=volumes[e.a].temperature_k;
    double tb=e.b>=0?volumes[e.b].temperature_k:boundary_temperature(e,time,profile_offsets,profile_points);
    double scale=e.b<0?runtime.ambient_conductance_scale:(e.interface_index>=0?runtime.interface_conductance_scale:1.0);
    if(e.b>=0 && mesh_volumes[e.a].object==mesh_volumes[e.b].object)
        scale*=reaction_bulk_factor(volumes[e.a],volumes[e.b],e.geometry_axis);
    double g;
    if constexpr(Kind==0) {
        g=constant_g[i]*scale;
        g=g>0?1.0/(1.0/g+series_r):0;
        states[i].conductance_w_per_k=g;
    } else {
        const double damage=e.b>=0?0.5*(volumes[e.a].damage+volumes[e.b].damage):volumes[e.a].damage;
        const auto& law=laws[e.law_index];
        if constexpr(Kind==1) {
            const double multiplier=e.b>=0 && mesh_volumes[e.a].object==mesh_volumes[e.b].object?
                bulk_failure_multiplier(materials[mesh_volumes[e.a].object].boundary_phase,ta,tb,
                    volumes[e.a].damage,volumes[e.b].damage,volumes[e.a].runaway_started_s,volumes[e.b].runaway_started_s,time):1.;
            states[i]=evaluate_directed_geometry(law,states[i],e.geometry_axis,ta,tb,
                volumes[e.a].damage,e.b>=0?volumes[e.b].damage:0.,*dt,e.weight,series_r,scale,0,multiplier);
        }
        else if constexpr(Kind==2)
            states[i]=evaluate_state_geometry(law,states[i],e.geometry_axis,ta,tb,
                volumes[e.a].damage,e.b>=0?volumes[e.b].damage:0.,*dt,e.weight,series_r,scale);
        else {auto changed=e;changed.series_resistance_k_per_w=series_r;states[i]=evolve(changed,law,states[i],0.5*(ta+tb),damage,*dt,scale);}
        g=states[i].conductance_w_per_k;
    }
    flux[i]=g*(tb-ta); // one flux, gathered with opposite signs
}
#include "gas_kernels.cuh"
#include "mixture_kernels.cuh"
#include "shell_kernels.cuh"
#include "shell_feedback.cuh"
#include "aperture_kernels.cuh"
#include "shell_work_kernels.cuh"
#include "shell_event_kernels.cuh"
#include "leading_shell_kernels.cuh"
__device__ double release_source(const Material& m,ThermalState& s,double budget,double dt,double speed) {
    if(m.pulse.profile==RunawayProfile::Kinetic)return advance_reaction(m.kinetics,m.outlet,s,budget,dt,speed);
    if(s.mode!=CellMode::Runaway)return 0.;
    double source;
    if(m.pulse.profile==RunawayProfile::ConstantPower)
        source=fmin(fmax(0.,budget-s.released_energy_j),budget/m.duration*speed*dt);
    else {
        s.reaction_progress_s=fmin(pulse_end(m.pulse,m.duration),s.reaction_progress_s+speed*dt);
        const double target=budget*pulse_fraction(m.pulse,m.duration,s.reaction_progress_s);
        source=fmin(fmax(0.,budget-s.released_energy_j),fmax(0.,target-s.released_energy_j));
    }
    s.released_energy_j+=source;
    s.chemical_energy_remaining_j=fmax(0.,budget-s.released_energy_j);
    if(s.released_energy_j>=budget*(1-1e-12)) {s.mode=CellMode::Spent;s.damage=1;}
    return source;
}
// Local mechanisms own proposal buffers, never the committed node state.
// Gas production is an explicit transfer packet, not a read of next[] by a
// different subsystem. Exact two-stage chemistry is one coupled mechanism.
template<bool Conversion>
__global__ void source_kernel(int count,const int* ids,const DeviceVolume* volumes,const Material* materials,
    const ThermalState* old,ThermalState* local,double* heat,GasTransfer* gas,
    RuntimeParameters runtime,double time,const double* step) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;const int i=ids[j];
    const auto& v=volumes[i];const auto& m=materials[v.object];auto s=old[i];
    double q;
    if constexpr(Conversion) {
        const double charging=m.electrical.enabled?positive_current_integral(m.electrical,m.current,time,*step)/(*step):0.;
        const double acceleration=overcharge_r1_rate(m.conversion,m.electrical,s.temperature_k,charging);
        q=advance_conversion(m.conversion,m.gas_model,s,v.capacity_j_per_k/m.capacity,*step,runtime.runaway_power_scale,time,acceleration);
        gas[i]={s.generated_gas_kg,s.generated_gas_energy_j,s.generated_gas_chemical_j,0.};
    } else {
        q=release_source(m,s,m.energy*v.capacity_j_per_k/m.capacity,*step,runtime.runaway_power_scale);
        gas[i]={};
    }
    if(m.electrical.enabled) {
        const double joule=electrical_resistance(m.electrical,old[i].temperature_k,old[i].damage)*
            squared_current_integral(m.electrical,m.current,time,*step)*v.capacity_j_per_k/m.capacity;
        s.joule_energy_j+=joule;q+=joule;
    }
    local[i]=s;heat[i]=q;
}
__global__ void volume_kernel(int count,const DeviceVolume* volumes,const Material* materials,
    const DeviceLink* links,const int* offsets,const int* adjacency,const double* flux,
    const ThermalState* old,ThermalState* next,RuntimeParameters runtime,double time,const double* step,
    DeviceRows patch_rows,const double* gas_flux,
    const ThermalState* local,const double* heat) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const double dt=*step;const auto& v=volumes[i];const auto& m=materials[v.object];
    if(m.isothermal)return;
    ThermalState s=local[i];double power=0,boundary_power=0;
    for(int j=offsets[i];j<offsets[i+1];++j) {
        int k=adjacency[j];const auto& e=links[k];
        power+=(e.a==i?1.0:-1.0)*flux[k];
        if(e.b<0) boundary_power-=flux[k];
    }
    for(int j=patch_rows.offsets[i];j<patch_rows.offsets[i+1];++j)power+=gas_flux[patch_rows.ids[j]];
    const double budget=m.energy*v.capacity_j_per_k/m.capacity;
    const double source=heat[i];
    if(m.conversion.enabled) {
        s.stored_energy_j+=power*dt+source;
        s.boundary_energy_out_j+=boundary_power*dt;
        close_thermal_state(m,v,s,runtime.runaway_power_scale);
        next[i]=s;return;
    } else if(m.capacity_law.factor.count) {
        s.stored_energy_j+=power*dt+source;
        s.temperature_k=thermal_temperature(m.capacity_law,v.capacity_j_per_k,s.stored_energy_j,old[i].damage);
    } else s.temperature_k+=(power*dt+source)/v.capacity_j_per_k;
    s.boundary_energy_out_j+=boundary_power*dt;
    // A bound mixture must age at its shared temperature, not at the
    // intermediate solid-only temperature before local equilibration.
    if(old[i].mobile_capacity_j_per_k==0 &&
        (s.mode==CellMode::Healthy || (m.pulse.profile==RunawayProfile::Kinetic && reaction_live(s)))) {
        double t=0.5*(s.temperature_k+old[i].temperature_k);
        s.damage=fmin(1.0,s.damage+damage_rate(t,m.onset,m.rate,m.temperature_scale)*m.aging_multiplier*runtime.damage_rate_scale*dt);
        const double threshold=m.pulse.profile==RunawayProfile::Kinetic?m.kinetics.activation_damage:1.;
        if(s.mode==CellMode::Healthy && s.damage>=threshold) {
            if(m.is_battery)s.runaway_started_s=time+dt;
            s.mode=m.is_battery && budget>0?CellMode::Runaway:CellMode::Spent;
        }
    }
    if(!m.capacity_law.factor.count)s.stored_energy_j=v.capacity_j_per_k*s.temperature_k;
    close_thermal_state(m,v,s,runtime.runaway_power_scale);
    next[i]=s;
}
// Infinite internal conductivity as an algebraic constraint. Contact fluxes
// still use old-state temperatures; sum all heat before evaluating local damage.
// Each thread stages only its own volumes, and the block synchronizes before
// broadcasting the common temperature. No in-place field reads or atomic sums.
__global__ void isothermal_volume_kernel(const DeviceVolume* volumes,const Material* materials,
    const DeviceLink* links,const int* object_offsets,const int* offsets,const int* adjacency,
    const double* flux,const ThermalState* old,ThermalState* next,
    RuntimeParameters runtime,double time,const double* step,const ThermalState* local,const double* heat) {
    const int object=blockIdx.x,t=threadIdx.x;
    const auto& m=materials[object];if(!m.isothermal)return;
    const double dt=*step;
    double capacity=0,delta=0;
    for(int i=object_offsets[object]+t;i<object_offsets[object+1];i+=blockDim.x) {
        const auto& v=volumes[i];ThermalState s=local[i];double power=0,boundary_power=0;
        for(int j=offsets[i];j<offsets[i+1];++j) {
            const int k=adjacency[j];const auto& e=links[k];
            power+=(e.a==i?1.0:-1.0)*flux[k];
            if(e.b<0)boundary_power-=flux[k];
        }
        const double source=heat[i];
        s.boundary_energy_out_j+=boundary_power*dt;next[i]=s;
        capacity+=v.capacity_j_per_k;delta+=power*dt+source;
    }
    __shared__ double capacities[128],deltas[128];
    capacities[t]=capacity;deltas[t]=delta;__syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(t<stride) {capacities[t]+=capacities[t+stride];deltas[t]+=deltas[t+stride];}
        __syncthreads();
    }
    // Initial fields are validated uniform, and every update preserves this.
    // Avoid roundoff from repeatedly multiplying/dividing the stored baseline.
    const double temperature=old[object_offsets[object]].temperature_k+deltas[0]/capacities[0];
    for(int i=object_offsets[object]+t;i<object_offsets[object+1];i+=blockDim.x) {
        ThermalState s=next[i];s.temperature_k=temperature;
        s.stored_energy_j=volumes[i].capacity_j_per_k*temperature;
        if(s.mode==CellMode::Healthy) {
            const double average=0.5*(old[i].temperature_k+temperature);
            s.damage=fmin(1.0,s.damage+damage_rate(average,m.onset,m.rate,m.temperature_scale)*m.aging_multiplier*runtime.damage_rate_scale*dt);
            if(s.damage>=1) {if(m.is_battery)s.runaway_started_s=time+dt;s.mode=m.is_battery && m.energy>0?CellMode::Runaway:CellMode::Spent;}
        }
        next[i]=s;
    }
}
__global__ void ignition_kernel(ThermalState* state,const DeviceVolume* volumes,const Material* materials,int begin,int end,double time) {
    int i=begin+blockIdx.x*blockDim.x+threadIdx.x;if(i>=end)return;
    if(state[i].mode==CellMode::Healthy) {
        const auto& m=materials[volumes[i].object];
        state[i].mode=CellMode::Runaway;state[i].runaway_started_s=time;
        state[i].damage=m.pulse.profile==RunawayProfile::Kinetic?fmax(state[i].damage,m.kinetics.activation_damage):1.;
        const auto& l=m.capacity_law;
        if(l.factor.count)state[i].temperature_k=thermal_temperature(l,volumes[i].capacity_j_per_k,state[i].stored_energy_j,state[i].damage);
        if(m.pulse.profile==RunawayProfile::Kinetic)
            refresh_reaction(m.kinetics,m.structure,state[i],m.energy*volumes[i].capacity_j_per_k/m.capacity,1.);
    }
}
struct Aggregate { double capacity,energy,damage,max_damage,tmin,tmax,spent,active,released,boundary,start,weighted_temperature,reference_capacity;
    double chemical,ejected,vent,power,quenched,joule; };
__global__ void macro_kernel(const DeviceVolume* volumes,const Material* materials,const ThermalState* state,const int* offsets,CellState* macros) {
    const int object=blockIdx.x,t=threadIdx.x;
    Aggregate a{};a.tmin=1e300;a.start=1e300;
    for(int i=offsets[object]+t;i<offsets[object+1];i+=blockDim.x) {
        const auto& s=state[i];double c=volumes[i].capacity_j_per_k;
        const auto& l=materials[object].capacity_law;
        const auto& mat=materials[object];
        const double actual=(mat.conversion.enabled?conversion_capacity(mat.conversion,s,c,c/mat.capacity):c*capacity_factor(l,s.temperature_k,s.damage))+s.mobile_capacity_j_per_k;
        a.capacity+=actual;a.reference_capacity+=c;a.weighted_temperature+=actual*s.temperature_k;
        a.energy+=l.factor.count||mat.conversion.enabled?s.stored_energy_j:c*s.temperature_k;a.damage+=c*s.damage;a.max_damage=fmax(a.max_damage,s.damage);
        a.tmin=fmin(a.tmin,s.temperature_k);a.tmax=fmax(a.tmax,s.temperature_k);
        a.spent+=s.mode==CellMode::Spent?c:0;a.active+=s.mode==CellMode::Runaway?c:0;
        a.released+=s.released_energy_j;a.boundary+=s.boundary_energy_out_j;
        a.joule+=s.joule_energy_j;
        a.chemical+=s.chemical_energy_remaining_j;a.ejected+=s.ejected_chemical_energy_j;
        a.vent+=s.vent_heat_out_j;a.power+=s.chemical_power_w;a.quenched+=s.mode==CellMode::Quenched?c:0.;
        if(s.runaway_started_s>=0)a.start=fmin(a.start,s.runaway_started_s);
    }
    __shared__ Aggregate partial[128];partial[t]=a;__syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(t<stride) {
            auto& x=partial[t];const auto& y=partial[t+stride];
            x.capacity+=y.capacity;x.energy+=y.energy;x.damage+=y.damage;x.max_damage=fmax(x.max_damage,y.max_damage);
            x.reference_capacity+=y.reference_capacity;x.weighted_temperature+=y.weighted_temperature;
            x.tmin=fmin(x.tmin,y.tmin);x.tmax=fmax(x.tmax,y.tmax);x.spent+=y.spent;x.active+=y.active;
            x.released+=y.released;x.boundary+=y.boundary;x.start=fmin(x.start,y.start);
            x.joule+=y.joule;
            x.chemical+=y.chemical;x.ejected+=y.ejected;x.vent+=y.vent;x.power+=y.power;x.quenched+=y.quenched;
        }
        __syncthreads();
    }
    if(t==0) {
        const auto& x=partial[0];CellState out;
        out.temperature_k=x.weighted_temperature/x.capacity;out.damage=x.damage/x.reference_capacity;
        out.min_temperature_k=x.tmin;out.max_temperature_k=x.tmax;out.max_damage=x.max_damage;
        out.spent_fraction=x.spent/x.reference_capacity;out.runaway_fraction=x.active/x.reference_capacity;
        out.mode=x.active>0?CellMode::Runaway:(x.quenched>0?CellMode::Quenched:
            (out.spent_fraction>1-1e-12?CellMode::Spent:(x.spent>0?CellMode::PartiallySpent:CellMode::Healthy)));
        out.runaway_started_s=x.start<1e299?x.start:-1;
        out.released_energy_j=x.released;out.stored_energy_j=x.energy;out.boundary_energy_out_j=x.boundary;out.heat_capacity_j_per_k=x.capacity;
        out.joule_energy_j=x.joule;
        out.chemical_energy_remaining_j=x.chemical;out.ejected_chemical_energy_j=x.ejected;
        out.vent_heat_out_j=x.vent;out.chemical_power_w=x.power;out.quenched_fraction=x.quenched/x.reference_capacity;
        macros[object]=out;
    }
}
__global__ void interface_macro_kernel(int count,const DeviceLink* links,const InterfaceState* states,
    const int* offsets,const int* ids,InterfaceState* macros) {
    int i=blockIdx.x,t=threadIdx.x;
    InterfaceState out{};double high=0;
    for(int j=offsets[i]+t;j<offsets[i+1];j+=blockDim.x) {
        int k=ids[j];const auto& s=states[k];double w=links[k].weight;
        out.l_m+=w*s.l_m;out.l0_m+=w*s.l0_m;out.damage+=w*s.damage;
        out.conductance_w_per_k+=s.conductance_w_per_k;out.destroyed_fraction+=w*s.destroyed_fraction;
        if(s.phase==EdgePhase::High)high+=w;
    }
    __shared__ InterfaceState partial[128];__shared__ double highs[128];
    partial[t]=out;highs[t]=high;__syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(t<stride) {
            auto& a=partial[t];const auto& b=partial[t+stride];
            a.l_m+=b.l_m;a.l0_m+=b.l0_m;a.damage+=b.damage;
            a.conductance_w_per_k+=b.conductance_w_per_k;a.destroyed_fraction+=b.destroyed_fraction;highs[t]+=highs[t+stride];
        }
        __syncthreads();
    }
    if(t==0) {
        out=partial[0];out.phase=out.destroyed_fraction>1-1e-12?EdgePhase::Destroyed:(highs[0]>0?EdgePhase::High:EdgePhase::Low);
        macros[i]=out;
    }
}
int blocks(std::size_t count) {return static_cast<int>((count+127)/128);}
// A rejected candidate is never published. Only integer error flags are atomic;
// no floating-point inventory/energy reduction depends on scheduling.
__global__ void validate_candidate_kernel(int nn,const ThermalState* nodes,int ng,const GasState* gas,
    int ne,const InterfaceState* edges,const double* dt,StepResult* result) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;unsigned error=0;
    if(i==0)result->dt=*dt;
    if(i<nn) {
        const auto& s=nodes[i];
        if(!isfinite(s.temperature_k)||s.temperature_k<=0||!isfinite(s.stored_energy_j)||!isfinite(s.damage)||s.damage<0||s.damage>1||
            !isfinite(s.reactant_kg)||s.reactant_kg<0||!isfinite(s.activated_kg)||s.activated_kg<0||!isfinite(s.residue_kg)||s.residue_kg<0||
            !isfinite(s.joule_energy_j)||s.joule_energy_j<0)error|=1;
    }
    if(i<ng) {
        const auto& g=gas[i];
        if(!isfinite(g.temperature_k)||g.temperature_k<=0||!isfinite(g.pressure_pa)||g.pressure_pa<0||!isfinite(g.mass_kg)||g.mass_kg<=0||
            !isfinite(g.energy_j)||!isfinite(g.chemical_j)||g.chemical_j< -1e-10||!isfinite(g.oxidizer_kg)||g.oxidizer_kg< -1e-14||g.oxidizer_kg>g.mass_kg+1e-14)error|=2;
    }
    if(i<ne) {
        const auto& e=edges[i];
        if(!isfinite(e.conductance_w_per_k)||e.conductance_w_per_k<0||!isfinite(e.damage)||!isfinite(e.l_m))error|=4;
    }
    if(error)atomicOr(&result->invalid,error);
}
}

struct CudaSolver::Impl {
    const Scenario& scenario;const ThermalMesh& mesh;
    MassMesh mass_mesh;
    SolverProfile profiling;
    std::unique_ptr<LocalGasTransport> cpu_gas;
    std::unique_ptr<CudaKawasakiTransport> mc_gas;
    std::vector<GasState> host_gas;
    std::vector<ThermalState> host_solid;
    std::vector<GasChannelState> host_channels;
    std::string device;
    Buffer<DeviceVolume> volumes;Buffer<DeviceLink> links;
    std::unique_ptr<Buffer<ConnectivityLaw>> laws;
    Buffer<Material> materials;
    std::unique_ptr<Buffer<CurrentPoint>> current_points;
    Buffer<ThermalState> state_a,state_b,local_state;
    Buffer<InterfaceState> edge_state,edge_next,macro_edges;
    std::unique_ptr<Buffer<int>> conversion_ids,legacy_ids;
    std::array<std::unique_ptr<Buffer<int>>,4> edge_ids;
    Buffer<double> local_heat;
    Buffer<GasTransfer> source_transfer,channel_transfer,gas_reaction;
    Buffer<CellState> macro_cells;
    Buffer<int> offsets,adjacency,object_offsets,interface_offsets,interface_ids;
    Buffer<double> flux,dt,constant_g,cfl;
    Buffer<StepResult> result;
    Buffer<int> profile_offsets;
    std::unique_ptr<Buffer<ProfilePoint>> profile_points;
    Buffer<GasState> gas_a,gas_b;
    std::unique_ptr<Buffer<GasState>> gas_stage;
    std::unique_ptr<Buffer<MassBinding>> mass_bindings;
    std::unique_ptr<Buffer<double>> mass_dt;
    Buffer<DeviceGas> gas_config;
    Buffer<DeviceChannel> channels;
    Buffer<GasChannelState> channel_state,channel_next;
    Buffer<GasThermalPatch> gas_patches;
    Buffer<double> gas_flux,gas_conductance,channel_mass_cap;
    std::unique_ptr<Buffer<int>> aperture_patch_ids;
    std::unique_ptr<Incidence> aperture_channel_rows;
    std::unique_ptr<Buffer<DeviceShellPatch>> shell_patches;
    std::unique_ptr<Buffer<ShellDamageState>> shell_damage,shell_next,shell_rates;
    std::unique_ptr<Buffer<double>> shell_work_power;
    std::unique_ptr<Buffer<GasState>> shell_event_pressure_snapshot;
    // Scratch inputs for the leading aperture operator. Committed buffers are
    // never modified by prepare/retry; pointer swaps are restored before commit.
    std::unique_ptr<Buffer<ThermalState>> prefix_solid;
    std::unique_ptr<Buffer<GasState>> prefix_gas;
    std::unique_ptr<Buffer<GasChannelState>> prefix_channels;
    std::unique_ptr<Buffer<ShellDamageState>> prefix_shell;
    std::vector<DeviceShellPatch> shell_geometry;
    bool prefix_inputs_active=false,prefix_applied=false;
    std::unique_ptr<Incidence> volume_patches,gas_patches_rows,gas_channels_rows,gas_sources_rows,shell_channels_rows;
    double cached_interface_scale=-1,cached_ambient_scale=-1,cached_safety=-1;
    bool has_isothermal=false,has_spatial=false;
    bool prepared=false;
    int work_threads;
    int work_blocks(std::size_t n) const {return static_cast<int>((n+work_threads-1)/work_threads);}
    Impl(const Scenario& s,const ThermalMesh& m,int tile):scenario(s),mesh(m),mass_mesh(build_mass_mesh(s,m)),
        volumes(m.volumes.size()),links(m.links.size()),materials(s.cells.size()),
        state_a(m.volumes.size()),state_b(m.volumes.size()),local_state(m.volumes.size()),edge_state(m.links.size()),edge_next(m.links.size()),macro_edges(s.interfaces.size()),
        local_heat(m.volumes.size()),source_transfer(m.volumes.size()),channel_transfer(s.gas_channels.size()),gas_reaction(s.gas_compartments.size()),
        macro_cells(s.cells.size()),offsets(m.adjacency_offsets.size()),adjacency(m.adjacency_links.size()),
        object_offsets(m.object_offsets.size()),interface_offsets(m.interface_offsets.size()),interface_ids(m.interface_links.size()),
        flux(m.links.size()),dt(1),constant_g(m.links.size()),cfl(1),result(1),profile_offsets(s.boundaries.size()+1),
        gas_a(s.gas_compartments.size()),gas_b(s.gas_compartments.size()),gas_config(s.gas_compartments.size()),
        channels(s.gas_channels.size()),channel_state(s.gas_channels.size()),channel_next(s.gas_channels.size()),gas_patches(m.gas_patches.size()),gas_flux(m.gas_patches.size()),gas_conductance(m.gas_patches.size()),channel_mass_cap(s.gas_channels.size()),work_threads(tile) {
        if(tile!=64 && tile!=128 && tile!=256)throw std::invalid_argument("Unsupported CUDA work tile size");
        if(s.gas_model.cpu_local_transport)cpu_gas=std::make_unique<LocalGasTransport>(s,m,mass_mesh);
        if(s.gas_model.cuda_kawasaki_transport)mc_gas=std::make_unique<CudaKawasakiTransport>(s,m,mass_mesh);
        cudaDeviceProp info{};int id=0;check(cudaGetDevice(&id),"CUDA device");check(cudaGetDeviceProperties(&info,id),"CUDA device properties");device=info.name;
        std::vector<DeviceVolume> device_volumes;
        for(const auto& v:m.volumes)device_volumes.push_back({v.object,v.capacity_j_per_k});
        volumes.upload(device_volumes);offsets.upload(m.adjacency_offsets);adjacency.upload(m.adjacency_links);
        object_offsets.upload(m.object_offsets);interface_offsets.upload(m.interface_offsets);interface_ids.upload(m.interface_links);
        std::vector<Material> host;
        std::vector<CurrentPoint> drive;
        for(const auto& c:s.cells)for(const auto& q:c.current_profile_a)drive.push_back({q[0],q[1]});
        current_points=std::make_unique<Buffer<CurrentPoint>>(drive.size());current_points->upload(drive);
        int current_offset=0;
        for(const auto& c:s.cells) {
            host.push_back({c.heat_capacity_j_per_k,c.runaway_energy_j,c.runaway_duration_s,c.damage_onset_k,
                c.damage_rate_at_onset_per_s,c.damage_temperature_scale_k,1+c.aging_damage_multiplier*c.aging_fraction,c.isothermal,c.is_battery,c.heat_capacity_law,c.runaway_pulse,boundary_phase(c),
                c.reaction_kinetics,c.reaction_structure,c.energy_outlet,c.conversion,s.gas_model,c.electrical,
                {current_points->p?current_points->p+current_offset:nullptr,static_cast<int>(c.current_profile_a.size())}});
            current_offset+=c.current_profile_a.size();
            if(c.isothermal && c.heat_capacity_law.factor.count)throw std::runtime_error("Variable capacity requires finite_volume transfer");
            has_isothermal|=c.isothermal;has_spatial|=!c.isothermal;
        }
        materials.upload(host);
        if(s.gas_model.split_transport) {
            gas_stage=std::make_unique<Buffer<GasState>>(s.gas_compartments.size());
            mass_bindings=std::make_unique<Buffer<MassBinding>>(mass_mesh.bindings.size());mass_bindings->upload(mass_mesh.bindings);
            mass_dt=std::make_unique<Buffer<double>>(2);
        }
        std::vector<int> conversion,legacy;
        for(std::size_t i=0;i<m.volumes.size();++i)
            (s.cells[m.volumes[i].object].conversion.enabled?conversion:legacy).push_back(static_cast<int>(i));
        conversion_ids=std::make_unique<Buffer<int>>(conversion.size());conversion_ids->upload(conversion);
        legacy_ids=std::make_unique<Buffer<int>>(legacy.size());legacy_ids->upload(legacy);
        gas_patches.upload(m.gas_patches);std::vector<DeviceGas> gas_data;
        std::vector<int> aperture_ids;
        std::vector<std::vector<int>> aperture_rows(s.gas_channels.size());
        for(std::size_t j=0;j<m.gas_patches.size();++j)if(m.gas_patches[j].channel>=0) {
            aperture_ids.push_back(static_cast<int>(j));aperture_rows[m.gas_patches[j].channel].push_back(static_cast<int>(j));
        }
        aperture_patch_ids=std::make_unique<Buffer<int>>(aperture_ids.size());aperture_patch_ids->upload(aperture_ids);
        aperture_channel_rows=std::make_unique<Incidence>(aperture_rows);
        for(std::size_t i=0;i<s.gas_compartments.size();++i) {
            const auto& g=s.gas_compartments[i];gas_data.push_back({g.volume_m3,g.reaction_rate_per_s,g.oxidizer_heat_j_per_kg,g.ignition_temperature_k,mass_mesh.bindings[i].thermal_volume});
        }
        gas_config.upload(gas_data);
        std::vector<DeviceChannel> channel_data;
        for(const auto& ch:s.gas_channels)channel_data.push_back({ch.from,ch.to,ch.shell_object,gas_channel_area(s,ch),ch.discharge_coefficient,
            ch.opening_pressure_pa,ch.outside_pressure_pa,ch.strength_reference_k,ch.strength_temperature_scale_k,ch.rupture,ch.opening,ch.shell_thermal_volume});
        channels.upload(channel_data);
        if(std::any_of(s.gas_channels.begin(),s.gas_channels.end(),[](const auto& c){return c.opening.enabled && discrete_shell_events(c.opening);}))
            shell_event_pressure_snapshot=std::make_unique<Buffer<GasState>>(s.gas_compartments.size());
        if(std::any_of(s.gas_channels.begin(),s.gas_channels.end(),[](const auto& c){return c.opening.enabled && c.opening.fracture_energy_j_per_m2>0;}))
            shell_work_power=std::make_unique<Buffer<double>>(s.gas_compartments.size());
        if(std::any_of(s.gas_channels.begin(),s.gas_channels.end(),[](const auto& c){return c.opening.enabled;})) {
            std::vector<DeviceShellPatch> patches(m.volumes.size());
            for(std::size_t j=0;j<s.gas_channels.size();++j) {
                const auto& ch=s.gas_channels[j];if(!ch.opening.enabled)continue;
                const auto& panel=s.cells[ch.shell_object];
                const auto port=std::find_if(s.ports.begin(),s.ports.end(),[&](const auto& p){return p.id==ch.port_id;});
                const int axis=port->surface[1]-'x',u=(axis+1)%3,v=(axis+2)%3;
                for(int k=m.object_offsets[ch.shell_object];k<m.object_offsets[ch.shell_object+1];++k) {
                    if(ch.shell_thermal_volume>=0 && k!=m.object_offsets[ch.shell_object]+ch.shell_thermal_volume)continue;
                    const auto& tv=m.volumes[k];const double center[3]{tv.local_center_m.x,tv.local_center_m.y,tv.local_center_m.z};
                    const double size[3]{tv.size_m.x,tv.size_m.y,tv.size_m.z};
                    patches[k]={static_cast<int>(j),shell_seam_fraction(center[u],center[v],size[u],size[v],panel.dimensions_m[u],panel.dimensions_m[v],ch.opening.seam_width_m)};
                    patches[k].area_m2=size[u]*size[v];
                    patches[k].rectangle=opening_rectangle(center[u],center[v],size[u],size[v],panel.dimensions_m[u],panel.dimensions_m[v],ch.opening.seam_width_m);
                    patches[k].weak_fraction=patches[k].seam_fraction;
                    patches[k].weak_factor=ch.opening.seam_strength_factor;
                    if(ch.opening.valve_radius_m>0) {
                        if(ch.opening.deterministic_aperture && patches[k].seam_fraction>0)throw std::runtime_error("A local patch cannot declare both a valve and a weak seam");
                        patches[k].weak_fraction=clipped_disk_area(size[u],size[v],ch.opening.valve_u_m,ch.opening.valve_v_m,ch.opening.valve_radius_m)/patches[k].area_m2;
                        patches[k].weak_factor=ch.opening.valve_pressure_factor;
                    }
                    patches[k].random_key=pixel_mix(ch.opening.rupture_seed^pixel_mix(pixel_channel_key(ch.id))^
                        pixel_mix(static_cast<std::uint64_t>(k-m.object_offsets[ch.shell_object])));
                    if(ch.opening.stochastic_pixels)patches[k].failure_hazard=pixel_failure_hazard(ch.opening.rupture_seed,
                        pixel_channel_key(ch.id),static_cast<std::uint64_t>(k-m.object_offsets[ch.shell_object]));
                }
            }
            shell_patches=std::make_unique<Buffer<DeviceShellPatch>>(patches.size());shell_patches->upload(patches);
            shell_geometry=patches;
            shell_damage=std::make_unique<Buffer<ShellDamageState>>(patches.size());
            shell_next=std::make_unique<Buffer<ShellDamageState>>(patches.size());
            shell_rates=std::make_unique<Buffer<ShellDamageState>>(patches.size());
            if(std::any_of(s.gas_channels.begin(),s.gas_channels.end(),[](const auto& c){return c.opening.deterministic_aperture;})) {
                prefix_solid=std::make_unique<Buffer<ThermalState>>(state_a.n);
                prefix_gas=std::make_unique<Buffer<GasState>>(gas_a.n);
                prefix_channels=std::make_unique<Buffer<GasChannelState>>(channel_state.n);
                prefix_shell=std::make_unique<Buffer<ShellDamageState>>(patches.size());
            }
        }
        std::vector<std::vector<int>> vp(m.volumes.size()),gp(s.gas_compartments.size()),gc(s.gas_compartments.size()),gs(s.gas_compartments.size()),sc(s.cells.size());
        for(std::size_t j=0;j<m.gas_patches.size();++j) {vp[m.gas_patches[j].volume].push_back(j);gp[m.gas_patches[j].compartment].push_back(j);}
        for(std::size_t j=0;j<s.gas_channels.size();++j) {
            const auto& c=s.gas_channels[j];gc[c.from].push_back(j);if(c.to>=0)gc[c.to].push_back(j);
            if(c.rupture)sc[c.shell_object].push_back(j);
        }
        for(std::size_t j=0;j<m.volumes.size();++j) {
            if(mass_mesh.source_gas[j]>=0)gs[mass_mesh.source_gas[j]].push_back(j);
        }
        volume_patches=std::make_unique<Incidence>(vp);gas_patches_rows=std::make_unique<Incidence>(gp);
        gas_channels_rows=std::make_unique<Incidence>(gc);gas_sources_rows=std::make_unique<Incidence>(gs);shell_channels_rows=std::make_unique<Incidence>(sc);
        std::vector<int> starts{0};std::vector<ProfilePoint> points;
        for(const auto& b:s.boundaries) {
            for(const auto& p:b.temperature_profile)points.push_back({p[0],p[1]});
            starts.push_back(static_cast<int>(points.size()));
        }
        profile_offsets.upload(starts);profile_points=std::make_unique<Buffer<ProfilePoint>>(points.size());profile_points->upload(points);
        std::vector<double> constants;
        std::vector<DeviceLink> device_links;std::vector<ConnectivityLaw> unique_laws;
        std::map<decltype(law_key(ConnectivityLaw{})),int> law_ids;
        std::array<std::vector<int>,4> edge_groups;
        for(const auto& e:m.links) {
            const auto& l=e.law;
            bool fixed=l.kind==TransitionKind::Constant && l.damage_rate_at_onset_per_s==0 && l.endpoint_damage_weight==0 &&
                l.active_conductance_w_per_k==0 && e.initial.damage==0 && e.initial.phase==EdgePhase::Low &&
                e.initial.l_m==l.l_low_m;
            constants.push_back(fixed?l.background_conductance_w_per_k*e.weight:-1);
            const int kind=fixed?0:l.kind==TransitionKind::DirectedGeometry?1:l.kind==TransitionKind::StateGeometry?2:3;
            edge_groups[kind].push_back(static_cast<int>(device_links.size()));
            int law_id=-1;
            if(!fixed) {
                auto [entry,inserted]=law_ids.emplace(law_key(l),static_cast<int>(unique_laws.size()));
                law_id=entry->second;if(inserted)unique_laws.push_back(l);
            }
            device_links.push_back({e.a,e.b,e.interface_index,law_id,e.weight,e.series_resistance_k_per_w,e.boundary_temperature_k,e.maximum_g,e.boundary_index,e.geometry_axis,e.series_resistance_a_k_per_w});
        }
        constant_g.upload(constants);
        links.upload(device_links);laws=std::make_unique<Buffer<ConnectivityLaw>>(unique_laws.size());laws->upload(unique_laws);
        for(int k=0;k<4;++k) {edge_ids[k]=std::make_unique<Buffer<int>>(edge_groups[k].size());edge_ids[k]->upload(edge_groups[k]);}
    }
    template<int Kind> void propose_edges(RuntimeParameters runtime,double time) {
        const auto& ids=*edge_ids[Kind];if(!ids.n)return;
        edge_kernel<Kind><<<work_blocks(ids.n),work_threads>>>(ids.n,ids.p,links.p,edge_state.p,edge_next.p,state_a.p,runtime,dt.p,flux.p,constant_g.p,laws->p,time,profile_offsets.p,profile_points->p,volumes.p,materials.p);
    }
    void propose_edges(RuntimeParameters runtime,double time) {
        propose_edges<0>(runtime,time);propose_edges<1>(runtime,time);propose_edges<2>(runtime,time);propose_edges<3>(runtime,time);
    }
    void select_solid_step(double time,double cap,double safety,const RuntimeParameters&);
    void propose_gas_transport(double time);
    void select_gas_step(double time,double safety,const RuntimeParameters&);
    void propose_processes(double time,const RuntimeParameters&);
    void propose_local_chemistry(double time,const RuntimeParameters&);
    void gather_solid_candidate(double time,const RuntimeParameters&);
    void gather_candidate(double time,const RuntimeParameters&);
    double close_and_validate_candidate(double time,const RuntimeParameters&);
    double prepare_split_step(double time,double cap,double safety,const RuntimeParameters&);
    void begin_prefix_inputs();
    void restore_prefix_inputs() noexcept;
    void apply_leading_shell(double time,const RuntimeParameters&);
    void equilibrate(ThermalState* solid,GasState* gas,double speed) {
        if(gas_a.n)mixture_equilibrium_kernel<<<work_blocks(gas_a.n),work_threads>>>(gas_a.n,mass_bindings->p,
            gas_config.p,volumes.p,materials.p,solid,gas,scenario.gas_model,speed);
    }
    void aggregate() {
        macro_kernel<<<scenario.cells.size(),128>>>(volumes.p,materials.p,state_a.p,object_offsets.p,macro_cells.p);
        if(macro_edges.n) interface_macro_kernel<<<macro_edges.n,128>>>(macro_edges.n,links.p,edge_state.p,interface_offsets.p,interface_ids.p,macro_edges.p);
        check(cudaGetLastError(),"CUDA aggregate launch");
    }
};
#include "cuda_step.cuh"
#include "leading_shell_step.cuh"
#include "mass_step.cuh"
CudaSolver::CudaSolver(const Scenario& s,const ThermalMesh& m,int tile):impl_(std::make_unique<Impl>(s,m,tile)) {reset();}
CudaSolver::~CudaSolver()=default;
const std::string& CudaSolver::device_name() const {return impl_->device;}
const ThermalState* CudaSolver::device_field() const {return impl_->state_a.p;}
void CudaSolver::reset() {
    auto& p=*impl_;p.prepared=false;p.profiling={};std::vector<ThermalState> state;
    if(p.mc_gas)p.mc_gas->reset();
    for(const auto& v:p.mesh.volumes) {
        ThermalState s;s.temperature_k=v.initial_temperature_k;
        s.damage=shell_material_damage(initial_shell_state(p.scenario,v.object,static_cast<int>(state.size())-p.mesh.object_offsets[v.object]),0.);
        s.stored_energy_j=thermal_energy(p.scenario.cells[v.object].heat_capacity_law,v.capacity_j_per_k,s.temperature_k,s.damage);
        const auto& c=p.scenario.cells[v.object];
        s.chemical_energy_remaining_j=c.runaway_energy_j*v.capacity_j_per_k/c.heat_capacity_j_per_k;
        if(s.damage>=(c.runaway_pulse.profile==RunawayProfile::Kinetic?c.reaction_kinetics.activation_damage:1.)) {
            if(p.scenario.cells[v.object].is_battery) {s.mode=CellMode::Runaway;s.runaway_started_s=0;}
            else s.mode=CellMode::Spent;
        }
        if(c.runaway_pulse.profile==RunawayProfile::Kinetic)
            refresh_reaction(c.reaction_kinetics,c.reaction_structure,s,s.chemical_energy_remaining_j,1.);
        if(c.conversion.enabled) {
            const double fraction=v.capacity_j_per_k/c.heat_capacity_j_per_k;
            const double mass=c.conversion.mass_kg*fraction;
            s.activated_kg=mass*c.conversion.initial_precursor_fraction;
            s.reactant_kg=mass-s.activated_kg;
            const double cap=conversion_capacity(c.conversion,s,v.capacity_j_per_k,fraction);
            s.stored_energy_j=v.capacity_j_per_k*p.scenario.gas_model.reference_temperature_k+
                cap*(s.temperature_k-p.scenario.gas_model.reference_temperature_k);
            refresh_conversion(c.conversion,s,v.capacity_j_per_k/c.heat_capacity_j_per_k,1.);
        }
        state.push_back(s);
    }
    p.state_a.upload(state);
    std::vector<GasState> gas;
    for(const auto& c:p.scenario.gas_compartments) {
        GasState g;g.mass_kg=c.initial_mass_kg;g.temperature_k=c.initial_temperature_k;
        g.chemical_j=c.initial_chemical_j;g.oxidizer_kg=c.initial_mass_kg*c.initial_oxidizer_mass_fraction;
        g.energy_j=g.mass_kg*p.scenario.gas_model.cv_j_per_kg_k*(g.temperature_k-p.scenario.gas_model.reference_temperature_k);
        g.free_volume_m3=c.volume_m3;
        g.pressure_pa=g.mass_kg*p.scenario.gas_model.specific_r_j_per_kg_k*g.temperature_k/c.volume_m3;gas.push_back(g);
    }
    p.gas_a.upload(gas);p.channel_state.upload(std::vector<GasChannelState>(p.channels.n));
    if(p.scenario.gas_model.split_transport)p.equilibrate(p.state_a.p,p.gas_a.p,1.);
    std::vector<InterfaceState> edges;for(const auto& e:p.mesh.links)edges.push_back(e.initial);
    p.edge_state.upload(edges);p.dt.upload({0});
    if(p.shell_patches) {
        std::vector<ShellDamageState> damage;for(const auto& v:p.mesh.volumes)damage.push_back(initial_shell_state(p.scenario,v.object,static_cast<int>(damage.size())-p.mesh.object_offsets[v.object]));
        p.shell_damage->upload(damage);
        shell_aperture_kernel<<<p.work_blocks(p.channels.n),p.work_threads>>>(p.channels.n,p.channels.p,p.object_offsets.p,
            p.shell_patches->p,p.shell_damage->p,p.channel_state.p,0.,p.dt.p);
    }
    p.propose_edges(RuntimeParameters{},0);
    std::swap(p.edge_state.p,p.edge_next.p);
    check(cudaGetLastError(),"CUDA reset");p.aggregate();
}
void CudaSolver::ignite(std::size_t object,int local_volume,double time) {
    discard_step();
    if(!impl_->scenario.cells.at(object).is_battery)throw std::runtime_error("Cannot ignite a passive material");
    auto& p=*impl_;int begin=p.mesh.object_offsets.at(object),end=p.mesh.object_offsets.at(object+1);
    if(local_volume>=0) {begin+=local_volume;end=begin+1;}
    ignition_kernel<<<p.work_blocks(end-begin),p.work_threads>>>(p.state_a.p,p.volumes.p,p.materials.p,begin,end,time);
    check(cudaGetLastError(),"CUDA ignition");p.aggregate();
}
double CudaSolver::prepare_step(double time,double cap,double safety,const RuntimeParameters& runtime) {
    auto& p=*impl_;p.prepared=false;
    struct Timer {double& total;std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
        ~Timer(){total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}} timer{p.profiling.prepare_s};
    if(!std::isfinite(time)||!std::isfinite(cap)||cap<=0||!std::isfinite(safety)||safety<=0||safety>1)
        throw std::invalid_argument("Invalid CUDA substep controls");
    for(double x:{runtime.interface_conductance_scale,runtime.ambient_conductance_scale,runtime.damage_rate_scale,runtime.runaway_power_scale})
        if(!std::isfinite(x)||x<0)throw std::invalid_argument("Invalid CUDA runtime multiplier");
    if(p.scenario.gas_model.split_transport) {
        const double accepted=p.prepare_split_step(time,cap,safety,runtime);p.prepared=true;return accepted;
    }
    p.propose_gas_transport(time);
    p.select_solid_step(time,cap,safety,runtime);
    p.select_gas_step(time,safety,runtime);
    p.propose_processes(time,runtime);
    p.gather_candidate(time,runtime);
    const double dt=p.close_and_validate_candidate(time,runtime);
    p.prepared=true;return dt;
}
void CudaSolver::discard_step() {impl_->prepared=false;}
SolverProfile CudaSolver::profile() const {
    auto result=impl_->profiling;
    if(impl_->mc_gas)impl_->mc_gas->statistics(result);
    return result;
}
void CudaSolver::commit_step() {
    auto& p=*impl_;if(!p.prepared)throw std::logic_error("No prepared CUDA substep");
    p.prepared=false;
    std::swap(p.state_a.p,p.state_b.p);std::swap(p.gas_a.p,p.gas_b.p);
    std::swap(p.edge_state.p,p.edge_next.p);std::swap(p.channel_state.p,p.channel_next.p);
    if(p.shell_patches)std::swap(p.shell_damage->p,p.shell_next->p);
    if(p.mc_gas)p.mc_gas->commit();
    p.aggregate();
}
double CudaSolver::step(double time,double cap,double safety,const RuntimeParameters& runtime) {
    const double dt=prepare_step(time,cap,safety,runtime);commit_step();return dt;
}
void CudaSolver::download_macros(std::vector<CellState>& cells,std::vector<InterfaceState>& edges) {
    impl_->macro_cells.download(cells);impl_->macro_edges.download(edges);
}
void CudaSolver::download_field(std::vector<ThermalState>& state) const {impl_->state_a.download(state);}
void CudaSolver::download_shell_states(std::vector<ShellDamageState>& state) const {
    if(impl_->shell_damage)impl_->shell_damage->download(state);else state.clear();
}
void CudaSolver::download_link_states(std::vector<InterfaceState>& state) const {impl_->edge_state.download(state);}
ProcessDiagnostics CudaSolver::process_diagnostics() const {
    const auto& p=*impl_;ProcessDiagnostics out;p.gas_a.download(out.gas);p.channel_state.download(out.channels);
    for(std::size_t i=0;i<out.channels.size();++i) {
        const auto& c=p.scenario.gas_channels[i];if(!c.opening.deterministic_aperture)continue;
        const auto& patch=p.shell_geometry[p.mesh.object_offsets[c.shell_object]+c.shell_thermal_volume];
        auto& d=out.channels[i];
        if(patch.seam_fraction>0)opened_strips(patch.rectangle,d.opening_fraction,d.aperture_strips_m.data());
        else d.aperture_radius_m=clipped_disk_radius(patch.rectangle.width,patch.rectangle.height,
            c.opening.valve_u_m,c.opening.valve_v_m,d.opening_fraction);
    }
    std::vector<ThermalState> field;p.state_a.download(field);
    double initial_thermal=0,current_thermal=0,released=0,boundary=0,vent=0,chemical=0,initial_chemical=0,initial_mass=0,current_mass=0,legacy_ejected=0;
    double initial_oxidizer=0,accounted_oxidizer=0;
    for(std::size_t i=0;i<field.size();++i) {
        const auto& v=p.mesh.volumes[i];const auto& c=p.scenario.cells[v.object];const auto& st=field[i];
        const double fraction=v.capacity_j_per_k/c.heat_capacity_j_per_k;
        double initial_node_energy=thermal_energy(c.heat_capacity_law,v.capacity_j_per_k,v.initial_temperature_k,
            shell_material_damage(initial_shell_state(p.scenario,v.object,static_cast<int>(i)-p.mesh.object_offsets[v.object]),0.));
        if(c.conversion.enabled)
            initial_node_energy+=c.conversion.mass_kg*fraction*c.conversion.initial_precursor_fraction*
                (c.conversion.activated_cp-c.conversion.reactant_cp)*(v.initial_temperature_k-p.scenario.gas_model.reference_temperature_k);
        initial_thermal+=initial_node_energy;
        current_thermal+=st.stored_energy_j;released+=st.released_energy_j;boundary+=st.boundary_energy_out_j;vent+=st.vent_heat_out_j;
        out.joule_energy_j+=st.joule_energy_j;
        chemical+=st.chemical_energy_remaining_j;legacy_ejected+=st.ejected_chemical_energy_j;
        initial_chemical+=fraction*(c.conversion.enabled?c.conversion.mass_kg*c.conversion.net_heat_j_per_kg:c.runaway_energy_j);
        if(c.conversion.enabled) {
            initial_mass+=c.conversion.mass_kg*fraction;current_mass+=st.reactant_kg+st.activated_kg+st.residue_kg;
            if(st.reactant_kg<0 || st.activated_kg<0 || st.residue_kg<0)throw std::runtime_error("Negative reacting inventory");
        }
    }
    for(std::size_t i=0;i<out.gas.size();++i) {
        const auto& g=out.gas[i];const auto& c=p.scenario.gas_compartments[i];
        if(!std::isfinite(g.temperature_k)||g.temperature_k<=0 || !std::isfinite(g.pressure_pa)||g.mass_kg<=0 || g.chemical_j< -1e-10 ||
            !std::isfinite(g.chemical_j)||!std::isfinite(g.oxidizer_kg)||g.oxidizer_kg< -1e-14||g.oxidizer_kg>g.mass_kg+1e-14)
            throw std::runtime_error("Nonphysical gas state");
        initial_mass+=c.initial_mass_kg;current_mass+=g.mass_kg;chemical+=g.chemical_j;current_thermal+=g.energy_j;
        initial_chemical+=c.initial_chemical_j;released+=g.released_energy_j;
        initial_oxidizer+=c.initial_mass_kg*c.initial_oxidizer_mass_fraction;accounted_oxidizer+=g.oxidizer_kg+g.consumed_oxidizer_kg;
        initial_thermal+=c.initial_mass_kg*p.scenario.gas_model.cv_j_per_kg_k*(c.initial_temperature_k-p.scenario.gas_model.reference_temperature_k);
    }
    for(std::size_t i=0;i<out.channels.size();++i)if(p.scenario.gas_channels[i].to<0) {
        const auto& ch=out.channels[i];out.escaped_mass_kg+=ch.transferred_kg;out.escaped_energy_j+=ch.transferred_energy_j;out.escaped_chemical_j+=ch.transferred_chemical_j;
        out.escaped_oxidizer_kg+=ch.transferred_oxidizer_kg;
    }
    out.mass_residual_kg=current_mass+out.escaped_mass_kg-initial_mass;
    out.oxidizer_residual_kg=accounted_oxidizer+out.escaped_oxidizer_kg-initial_oxidizer;
    for(const auto& channel:out.channels)out.fracture_work_j+=channel.fracture_work_j;
    out.thermal_residual_j=current_thermal+out.fracture_work_j-initial_thermal-(released+out.joule_energy_j-boundary-vent-out.escaped_energy_j);
    out.chemical_residual_j=chemical+released+legacy_ejected+out.escaped_chemical_j-initial_chemical;
    return out;
}
void CudaSolver::download_gas_states(std::vector<GasState>& gas,std::vector<GasChannelState>& channels) const {
    impl_->gas_a.download(gas);impl_->channel_state.download(channels);
}
} // namespace battery

#include "battery_native/kawasaki_transport.hpp"
#include "battery_native/kawasaki_gas_math.hpp"
#include "battery_native/conversion.hpp"
#include "batched_gas.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace battery {
namespace {
namespace kg=kawasaki;
void checked(cudaError_t error) {if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
template<class T> struct DeviceArray {
    T* p=nullptr;std::size_t n;
    explicit DeviceArray(std::size_t count):n(count){if(n)checked(cudaMalloc(&p,n*sizeof(T)));}
    ~DeviceArray(){if(p)cudaFree(p);}
    void upload(const std::vector<T>& v){if(v.size()!=n)throw std::logic_error("MC upload size");if(n)checked(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
};
struct Binding {
    int solid=-1;
    double capacity=0,fraction=0;
    ConversionLaw conversion;
};
__global__ void import_mixture(int count,const Binding* bindings,const GasState* gas,
    const ThermalState* solid,kg::Node* nodes,GasModel model) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& b=bindings[i];const auto& g=gas[i];double cs=0;
    if(b.solid>=0)cs=b.conversion.enabled?conversion_capacity(b.conversion,solid[b.solid],b.capacity,b.fraction):b.capacity;
    auto n=kg::Node{g.mass_kg,(cs+model.cv_j_per_kg_k*g.mass_kg)*g.temperature_k,
        g.free_volume_m3,cs,g.chemical_j,g.oxidizer_kg};
    nodes[i]=n;
}
// A single cooperative block owns this small network. Barriers between matchings
// avoid hundreds of kernel launches for the common chamber hub. No floating
// atomics, no simultaneous writes to shared endpoints, no host state roundtrip.
__global__ void network_step(int colors,const int* offsets,const int* ids,const kg::Edge* topology,
    kg::Node* nodes,kg::Clock* clocks,GasChannelState* channels,kg::Settings settings,
    GasModel model,double duration,int* error,int node_count,GasCombustionView combustion) {
    __shared__ int failed;
    if(threadIdx.x==0)failed=0;
    __syncthreads();
    double elapsed=0;
    while(elapsed<duration) {
        const double h=fmin(duration-elapsed,fmin(model.local_exchange_interval_s,
            combustion_interval(combustion.maximum_rate_per_s,model.max_fraction_per_step)));
        if(elapsed+h==elapsed){if(threadIdx.x==0)*error=5;return;}
        if(combustion.laws)for(int i=threadIdx.x;i<node_count;i+=blockDim.x)
            burn_gas(nodes[i],combustion.laws[i],settings,h/2);
        __syncthreads();
        for(int pass=0;pass<2;++pass)for(int step=0;step<colors;++step) {
            const int color=pass?colors-1-step:step;
            for(int j=offsets[color]+threadIdx.x;j<offsets[color+1];j+=blockDim.x) {
                const int id=ids[j];auto edge=topology[id];auto& channel=channels[id];
                edge.area=channel.open?channel.effective_area_m2:0.;
                const auto before=nodes[edge.a];
                const int code=kg::evolve(nodes[edge.a],nodes[edge.b],edge,clocks[id],settings,h/2,kg::mix(model.kawasaki_seed),id);
                if(code)atomicMax(&failed,code);
                const auto& after=nodes[edge.a];
                const double dm=before.mass-after.mass;
                // Convert absolute mixture energy packets back to gas-reference convention.
                channel.transferred_kg+=dm;
                channel.transferred_energy_j+=before.energy-after.energy-dm*model.cv_j_per_kg_k*model.reference_temperature_k;
                channel.transferred_chemical_j+=before.chemical-after.chemical;
                channel.transferred_oxidizer_kg+=before.oxidizer-after.oxidizer;
                channel.mass_flow_kg_s+=dm/duration;
            }
            __syncthreads();
            if(failed){if(threadIdx.x==0)*error=failed;return;}
        }
        if(combustion.laws)for(int i=threadIdx.x;i<node_count;i+=blockDim.x)
            burn_gas(nodes[i],combustion.laws[i],settings,h/2);
        __syncthreads();
        elapsed+=h;
    }
}
__global__ void clear_rates(int count,GasChannelState* channels) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)channels[i].mass_flow_kg_s=0;
}
__global__ void export_mixture(int count,const Binding* bindings,const kg::Node* nodes,
    const GasState* input,GasState* output,ThermalState* solid,GasModel model) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& b=bindings[i];const auto& old=input[i];auto n=nodes[i];auto g=old;
    g.released_energy_j+=n.reaction_released_j;
    g.consumed_oxidizer_kg+=n.reaction_oxidizer_kg;
    const double t=n.energy/(n.solid_capacity+model.cv_j_per_kg_k*n.mass);
    g.mass_kg=n.mass;g.chemical_j=n.chemical;g.oxidizer_kg=n.oxidizer;
    g.temperature_k=t;g.energy_j=n.mass*model.cv_j_per_kg_k*(t-model.reference_temperature_k);
    g.pressure_pa=n.mass*model.specific_r_j_per_kg_k*t/n.volume;
    if(b.solid>=0) {
        auto& s=solid[b.solid];
        s.stored_energy_j+=n.solid_capacity*(t-s.temperature_k);
        s.temperature_k=t;s.mobile_capacity_j_per_k=n.mass*model.cv_j_per_kg_k;
    }
    output[i]=g;
}
}
struct CudaKawasakiTransport::Impl {
    GasModel model;kg::Settings settings;int colors=0;
    DeviceArray<Binding> bindings;DeviceArray<kg::Node> nodes;
    DeviceArray<GasCombustionLaw> combustion;
    double maximum_reaction_rate=0.;
    DeviceArray<kg::Edge> edges;DeviceArray<kg::Clock> committed,candidate;
    std::unique_ptr<DeviceArray<int>> ids,offsets;
    DeviceArray<int> error{1};
    std::unique_ptr<DeviceArray<int>> adjacency,adjacency_offsets;
    std::unique_ptr<DeviceArray<kg::Node>> scratch;
    std::unique_ptr<DeviceArray<batched_gas::Proposal>> proposals;
    std::unique_ptr<DeviceArray<batched_gas::Budget>> budgets;
    DeviceArray<batched_gas::Control> batch_control{1};
    Impl(const Scenario& s,const ThermalMesh& mesh,const MassMesh& mass):model(s.gas_model),
        bindings(s.gas_compartments.size()),nodes(bindings.n),combustion(bindings.n),edges(s.gas_channels.size()),committed(edges.n),candidate(edges.n) {
        settings.cv=model.cv_j_per_kg_k;settings.R=model.specific_r_j_per_kg_k;
        settings.packet_fraction=model.kawasaki_packet_fraction;settings.tail_atol_pa=model.kawasaki_tail_atol_pa;settings.tail_rtol=model.kawasaki_tail_rtol;
        kg::Network topology;topology.settings=settings;
        std::vector<Binding> binding;
        std::vector<GasCombustionLaw> chemistry;
        for(std::size_t i=0;i<bindings.n;++i) {
            const auto& g=s.gas_compartments[i];Binding b;b.solid=mass.bindings[i].thermal_volume;
            chemistry.push_back({g.reaction_rate_per_s,g.ignition_temperature_k,g.oxidizer_heat_j_per_kg});
            maximum_reaction_rate=std::max(maximum_reaction_rate,g.reaction_rate_per_s);
            if(b.solid>=0) {
                const auto& v=mesh.volumes[b.solid];const auto& c=s.cells[v.object];
                if(c.heat_capacity_law.factor.count)throw std::runtime_error("CUDA Kawasaki adapter requires temperature-independent material capacity; refusing a silent approximation");
                b.capacity=v.capacity_j_per_k;b.fraction=b.capacity/c.heat_capacity_j_per_k;b.conversion=c.conversion;
            }
            binding.push_back(b);
            topology.initial.push_back({g.initial_mass_kg,g.initial_mass_kg*settings.cv*g.initial_temperature_k,g.volume_m3,0,g.initial_chemical_j,g.initial_mass_kg*g.initial_oxidizer_mass_fraction});
        }
        for(const auto& c:s.gas_channels) {
            if(c.to<0)throw std::runtime_error("CUDA Kawasaki adapter requires finite receiving volumes, not an external sink");
            topology.edges.push_back({c.from,c.to,0,c.discharge_coefficient});
        }
        kg::schedule(topology);bindings.upload(binding);combustion.upload(chemistry);edges.upload(topology.edges);
        colors=topology.colors.size();std::vector<int> order,starts{0};
        for(const auto& color:topology.colors){order.insert(order.end(),color.begin(),color.end());starts.push_back(order.size());}
        ids=std::make_unique<DeviceArray<int>>(order.size());ids->upload(order);
        offsets=std::make_unique<DeviceArray<int>>(starts.size());offsets->upload(starts);
        if(model.cuda_batched_transport) {
            std::vector<std::vector<int>> neighbors(nodes.n);
            for(int e=0;e<int(topology.edges.size());++e) {
                neighbors[topology.edges[e].a].push_back(e);neighbors[topology.edges[e].b].push_back(e);
            }
            std::vector<int> flat,offset{0};
            for(const auto& row:neighbors){flat.insert(flat.end(),row.begin(),row.end());offset.push_back(flat.size());}
            adjacency=std::make_unique<DeviceArray<int>>(flat.size());adjacency->upload(flat);
            adjacency_offsets=std::make_unique<DeviceArray<int>>(offset.size());adjacency_offsets->upload(offset);
            scratch=std::make_unique<DeviceArray<kg::Node>>(nodes.n);
            proposals=std::make_unique<DeviceArray<batched_gas::Proposal>>(edges.n);
            budgets=std::make_unique<DeviceArray<batched_gas::Budget>>(nodes.n);
        }
    }
};
CudaKawasakiTransport::CudaKawasakiTransport(const Scenario& s,const ThermalMesh& m,const MassMesh& b):impl_(std::make_unique<Impl>(s,m,b)){reset();}
CudaKawasakiTransport::~CudaKawasakiTransport()=default;
void CudaKawasakiTransport::reset(){auto& p=*impl_;p.committed.upload(std::vector<kg::Clock>(p.committed.n));p.candidate.upload(std::vector<kg::Clock>(p.candidate.n));}
void CudaKawasakiTransport::prepare(const GasState* input,GasState* output,ThermalState* solid,GasChannelState* channels,double duration) {
    auto& p=*impl_;if(!(duration>0 && std::isfinite(duration)))throw std::invalid_argument("Invalid MC gas duration");
    if(p.committed.n)checked(cudaMemcpyAsync(p.candidate.p,p.committed.p,p.committed.n*sizeof(kg::Clock),cudaMemcpyDeviceToDevice));
    checked(cudaMemsetAsync(p.error.p,0,sizeof(int)));
    const GasCombustionView chemistry=p.maximum_reaction_rate>0.?
        GasCombustionView{p.combustion.p,p.maximum_reaction_rate}:GasCombustionView{};
    if(p.nodes.n)import_mixture<<<(p.nodes.n+127)/128,128>>>(p.nodes.n,p.bindings.p,input,solid,p.nodes.p,p.model);
    if(p.edges.n)clear_rates<<<(p.edges.n+127)/128,128>>>(p.edges.n,channels);
    if(p.model.cuda_batched_transport)
        checked(batched_gas::launch(p.nodes.n,p.edges.n,p.adjacency_offsets->p,p.adjacency->p,p.edges.p,
            p.nodes.p,p.scratch->p,p.proposals->p,p.budgets->p,p.candidate.p,channels,p.settings,p.model,duration,p.error.p,p.batch_control.p,
            256,0,chemistry));
    else network_step<<<1,256>>>(p.colors,p.offsets->p,p.ids->p,p.edges.p,p.nodes.p,p.candidate.p,channels,p.settings,p.model,duration,p.error.p,p.nodes.n,
        chemistry);
    if(p.nodes.n)export_mixture<<<(p.nodes.n+127)/128,128>>>(p.nodes.n,p.bindings.p,p.nodes.p,input,output,solid,p.model);
    int error=0;checked(cudaGetLastError());checked(cudaMemcpy(&error,p.error.p,sizeof(int),cudaMemcpyDeviceToHost));
    if(error)throw std::runtime_error("CUDA Kawasaki pressure step failed: "+std::to_string(error));
}
void CudaKawasakiTransport::commit(){std::swap(impl_->committed.p,impl_->candidate.p);}
void CudaKawasakiTransport::statistics(SolverProfile& profile) const {
    profile.kawasaki_events=profile.kawasaki_tail_updates=profile.gas_batches=profile.gas_limited_batches=0;
    const auto& p=*impl_;std::vector<kg::Clock> clocks(p.committed.n);
    if(!clocks.empty())checked(cudaMemcpy(clocks.data(),p.committed.p,clocks.size()*sizeof(kg::Clock),cudaMemcpyDeviceToHost));
    for(const auto& c:clocks){profile.kawasaki_events+=c.events;profile.kawasaki_tail_updates+=c.tails;
        profile.gas_batches+=c.batches;profile.gas_limited_batches+=c.limited_batches;}
    if(p.model.cuda_deterministic_transport || p.model.cuda_network_transport) {
        batched_gas::Control control{};
        checked(cudaMemcpy(&control,p.batch_control.p,sizeof(control),cudaMemcpyDeviceToHost));
        profile.deterministic_pressure_gap_pa=control.last_max_pressure_gap_pa;
    }
}
}

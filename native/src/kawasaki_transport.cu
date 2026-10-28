#include "battery_native/kawasaki_transport.hpp"
#include "battery_native/checkpoint_io.hpp"
#include "battery_native/kawasaki_gas_math.hpp"
#include "battery_native/conversion.hpp"
#include "battery_native/gas_ambient.hpp"
#include "batched_gas.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

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
    n.vapor=g.vapor_kg;n.latent=g.latent_j;
    nodes[i]=n;
}
// A single cooperative block owns this small network. Barriers between matchings
// avoid hundreds of kernel launches for the common chamber hub. No floating
// atomics, no simultaneous writes to shared endpoints, no host state roundtrip.
__global__ void network_step(int colors,const int* offsets,const int* ids,const kg::Edge* topology,
    kg::Node* nodes,kg::Clock* clocks,GasChannelState* channels,kg::Settings settings,
    GasModel model,double duration,int* error,int node_count,GasCombustionView combustion) {
    __shared__ int failed;
    __shared__ double limits[256];
    if(threadIdx.x==0)failed=0;
    __syncthreads();
    double elapsed=0;
    while(elapsed<duration) {
        double limit=1e300;
        if(combustion.enabled)for(int i=threadIdx.x;i<node_count;i+=blockDim.x)
            limit=fmin(limit,combustion_step_limit(nodes[i],combustion.laws[i],settings,
                model.max_fraction_per_step,model.max_temperature_step_k));
        limits[threadIdx.x]=limit;__syncthreads();
        for(int width=blockDim.x/2;width;width/=2) {
            if(threadIdx.x<width)limits[threadIdx.x]=fmin(limits[threadIdx.x],limits[threadIdx.x+width]);
            __syncthreads();
        }
        const double h=fmin(duration-elapsed,fmin(model.local_exchange_interval_s,limits[0]));
        if(elapsed+h==elapsed){if(threadIdx.x==0)*error=5;return;}
        if(combustion.laws)for(int i=threadIdx.x;i<node_count;i+=blockDim.x)
            if(!burn_gas(nodes[i],combustion.laws[i],settings,h/2,
                model.max_fraction_per_step,model.max_temperature_step_k))atomicMax(&failed,19);
        __syncthreads();
        if(failed){if(threadIdx.x==0)*error=failed;return;}
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
            if(!burn_gas(nodes[i],combustion.laws[i],settings,h/2,
                model.max_fraction_per_step,model.max_temperature_step_k))atomicMax(&failed,19);
        __syncthreads();
        if(failed){if(threadIdx.x==0)*error=failed;return;}
        elapsed+=h;
    }
}
__global__ void clear_rates(int count,GasChannelState* channels) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)channels[i].mass_flow_kg_s=0;
}
__global__ void export_mixture(int count,const Binding* bindings,const kg::Node* nodes,
    const GasState* input,GasState* output,ThermalState* solid,GasModel model,
    const GasAmbientBoundary* ambient,double duration,int* error) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& b=bindings[i];const auto& old=input[i];auto n=nodes[i];auto g=old;
    g.released_energy_j+=n.reaction_released_j;
    g.consumed_oxidizer_kg+=n.reaction_oxidizer_kg;
    if(ambient && ambient[i].enabled){gas_ambient::Transfer q;kg::Settings settings;
        settings.cv=model.cv_j_per_kg_k;settings.R=model.specific_r_j_per_kg_k;
        if(!gas_ambient::apply(n,ambient[i],settings,model.reference_temperature_k,duration,q))atomicMax(error,32);
        g.ambient_mass_out_kg+=q.mass;g.ambient_energy_out_j+=q.energy;
        g.ambient_chemical_out_j+=q.chemical;g.ambient_oxidizer_out_kg+=q.oxidizer;
        g.ambient_gross_in_kg+=q.gross_in;g.ambient_gross_out_kg+=q.gross_out;g.ambient_entropy_j_k+=q.entropy;
    }
    const double t=n.energy/(n.solid_capacity+model.cv_j_per_kg_k*n.mass);
    g.mass_kg=n.mass;g.chemical_j=n.chemical;g.oxidizer_kg=n.oxidizer;
    g.vapor_kg=n.vapor;g.latent_j=n.latent;
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
#include "gas_film_adapter.cuh"
struct CudaKawasakiTransport::Impl {
    std::vector<std::string> node_names,channel_names;
    GasModel model;kg::Settings settings;int colors=0;
    DeviceArray<Binding> bindings;DeviceArray<kg::Node> nodes;
    DeviceArray<GasCombustionLaw> combustion;
    std::unique_ptr<DeviceArray<GasAmbientBoundary>> ambient;
    bool has_combustion=false;
    std::unique_ptr<FilmAdapter> films;
    DeviceArray<kg::Edge> edges;DeviceArray<kg::Clock> committed,candidate;
    std::unique_ptr<DeviceArray<int>> ids,offsets;
    DeviceArray<int> error{1};
    std::unique_ptr<DeviceArray<int>> adjacency,adjacency_offsets;
    std::unique_ptr<DeviceArray<kg::Node>> scratch;
    std::unique_ptr<DeviceArray<batched_gas::Proposal>> proposals;
    std::unique_ptr<DeviceArray<batched_gas::Budget>> budgets;
    std::unique_ptr<DeviceArray<batched_gas::AverageState>> average_a,average_b;
    std::unique_ptr<DeviceArray<batched_gas::MixSolveState>> mix_work;
    DeviceArray<batched_gas::Control> batch_control{1};
    Impl(const Scenario& s,const ThermalMesh& mesh,const MassMesh& mass):model(s.gas_model),
        bindings(s.gas_compartments.size()),nodes(bindings.n),combustion(bindings.n),edges(s.gas_channels.size()),committed(edges.n),candidate(edges.n) {
        model.minimum_step_s=s.settings.minimum_dt_s;
        if(model.reactive_transit && (!model.averaged_network_transport || !model.cuda_network_transport ||
            (model.coupled_gas_films && !model.joint_gas_surface_mixing)))
            throw std::runtime_error("Reactive transit requires averaged CUDA transport and joint gas films");
        if(model.cuda_network_transport)model.cuda_batched_transport=true;
        if(model.coupled_gas_films && !model.cuda_network_transport)
            throw std::runtime_error("Coupled gas contacts require the conservative CUDA network backend");
        settings.cv=model.cv_j_per_kg_k;settings.R=model.specific_r_j_per_kg_k;
        if(s.has_gas_mixing_connectivity)settings.mixing_connectivity=s.gas_mixing_connectivity;
        settings.mixing_connectivity.endpoint_a=settings.mixing_connectivity.endpoint_b=s.gas_surface_connectivity;
        if(model.coupled_gas_films)films=std::make_unique<FilmAdapter>(film_layout(s,mesh,mass));
        settings.vapor_fuel_heat_j_kg=model.vapor_fuel_heat_j_kg;
        settings.thermal_stirring_coefficient=model.thermal_stirring_coefficient;
        settings.mixing_length_cap_m=model.mixing_length_cap_m;
        settings.thermal_stirring_length_m=model.thermal_stirring_length_m;
        settings.packet_fraction=model.kawasaki_packet_fraction;settings.tail_atol_pa=model.kawasaki_tail_atol_pa;settings.tail_rtol=model.kawasaki_tail_rtol;
        kg::Network topology;topology.settings=settings;
        std::vector<Binding> binding;
        std::vector<GasCombustionLaw> chemistry;
        for(std::size_t i=0;i<bindings.n;++i) {
            const auto& g=s.gas_compartments[i];Binding b;b.solid=mass.bindings[i].thermal_volume;
            node_names.push_back(g.id);
            chemistry.push_back(gas_combustion_law(g));
            has_combustion|=g.reaction_rate_per_s>0.;
            if(b.solid>=0) {
                const auto& v=mesh.volumes[b.solid];const auto& c=s.cells[v.object];
                if(c.heat_capacity_law.factor.count)throw std::runtime_error("CUDA Kawasaki adapter requires temperature-independent material capacity; refusing a silent approximation");
                b.capacity=v.capacity_j_per_k;b.fraction=b.capacity/c.heat_capacity_j_per_k;b.conversion=c.conversion;
            }
            binding.push_back(b);
            topology.initial.push_back({g.initial_mass_kg,g.initial_mass_kg*settings.cv*g.initial_temperature_k,g.volume_m3,0,g.initial_chemical_j,g.initial_mass_kg*g.initial_oxidizer_mass_fraction});
        }
        for(const auto& c:s.gas_channels) {
            channel_names.push_back(c.id);
            if(c.to<0)throw std::runtime_error("CUDA Kawasaki adapter requires finite receiving volumes, not an external sink");
            topology.edges.push_back({c.from,c.to,0,c.discharge_coefficient,c.mixing_length_m,c.mixing_axis});
            topology.edges.back().hydraulic_resistance_factor=c.hydraulic_resistance_factor;
            const int a=mass.bindings[c.from].thermal_volume,b=mass.bindings[c.to].thermal_volume;
            topology.edges.back().molecular_mixing=internal_porous_mixing(model.free_gas_response_only,c.rupture,
                a>=0?int(mesh.volumes[a].object):-1,b>=0?int(mesh.volumes[b].object):-1);
        }
        kg::schedule(topology);bindings.upload(binding);combustion.upload(chemistry);edges.upload(topology.edges);
        if(std::any_of(s.gas_compartments.begin(),s.gas_compartments.end(),[](const auto& g){return g.ambient.enabled;})){
            std::vector<GasAmbientBoundary> boundaries;for(const auto& g:s.gas_compartments)boundaries.push_back(g.ambient);
            ambient=std::make_unique<DeviceArray<GasAmbientBoundary>>(boundaries.size());ambient->upload(boundaries);
        }
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
            if(model.averaged_network_transport){
                average_a=std::make_unique<DeviceArray<batched_gas::AverageState>>(nodes.n);
                average_b=std::make_unique<DeviceArray<batched_gas::AverageState>>(nodes.n);
            }
            if(model.implicit_network_mixing)mix_work=std::make_unique<DeviceArray<batched_gas::MixSolveState>>(
                nodes.n+(model.joint_gas_surface_mixing?mesh.volumes.size():0));
        }
    }
};
CudaKawasakiTransport::CudaKawasakiTransport(const Scenario& s,const ThermalMesh& m,const MassMesh& b):impl_(std::make_unique<Impl>(s,m,b)){reset();}
CudaKawasakiTransport::~CudaKawasakiTransport()=default;
void CudaKawasakiTransport::reset(){auto& p=*impl_;p.committed.upload(std::vector<kg::Clock>(p.committed.n));p.candidate.upload(std::vector<kg::Clock>(p.candidate.n));p.batch_control.upload({batched_gas::Control{}});}
void CudaKawasakiTransport::prepare(const GasState* input,GasState* output,ThermalState* solid,GasChannelState* channels,double duration) {
    auto& p=*impl_;if(!(duration>0 && std::isfinite(duration)))throw std::invalid_argument("Invalid MC gas duration");
    if(p.committed.n)checked(cudaMemcpyAsync(p.candidate.p,p.committed.p,p.committed.n*sizeof(kg::Clock),cudaMemcpyDeviceToDevice));
    checked(cudaMemsetAsync(p.error.p,0,sizeof(int)));
    if(p.nodes.n)import_mixture<<<(p.nodes.n+127)/128,128>>>(p.nodes.n,p.bindings.p,input,solid,p.nodes.p,p.model);
    if(p.films)p.films->prepare(solid,p.nodes.p,p.settings);
    const GasCombustionView chemistry=p.has_combustion || p.films?
        GasCombustionView{p.combustion.p,p.has_combustion,p.films?p.films->states.p:nullptr,
            p.films?p.films->view(solid):GasFilmView{}}:GasCombustionView{};
    if(p.edges.n)clear_rates<<<(p.edges.n+127)/128,128>>>(p.edges.n,channels);
    if(p.model.cuda_batched_transport)
        checked(batched_gas::launch(p.nodes.n,p.edges.n,p.adjacency_offsets->p,p.adjacency->p,p.edges.p,
            p.nodes.p,p.scratch->p,p.proposals->p,p.budgets->p,p.candidate.p,channels,p.settings,p.model,duration,p.error.p,p.batch_control.p,
            256,0,chemistry,p.average_a?p.average_a->p:nullptr,p.average_b?p.average_b->p:nullptr,p.mix_work?p.mix_work->p:nullptr));
    else network_step<<<1,256>>>(p.colors,p.offsets->p,p.ids->p,p.edges.p,p.nodes.p,p.candidate.p,channels,p.settings,p.model,duration,p.error.p,p.nodes.n,
        chemistry);
    if(p.nodes.n)export_mixture<<<(p.nodes.n+127)/128,128>>>(p.nodes.n,p.bindings.p,p.nodes.p,input,output,solid,p.model,
        p.ambient?p.ambient->p:nullptr,duration,p.error.p);
    if(p.films)p.films->gather(solid);
    int error=0;checked(cudaGetLastError());checked(cudaMemcpy(&error,p.error.p,sizeof(int),cudaMemcpyDeviceToHost));
    if(error){
        std::ostringstream message;message<<"CUDA Kawasaki pressure step failed: "<<error;
        if(error>=29 && error<=31){batched_gas::Control c{};
            checked(cudaMemcpy(&c,p.batch_control.p,sizeof(c),cudaMemcpyDeviceToHost));
            message<<"; reactive_iterations="<<c.reactive_iterations<<" reactive_residual="<<c.reactive_residual;}
        if(error==16) {
            // Failure-only readout of BOTH alternating buffers. These are not
            // labelled before/after: the rejected cooperative stage may have
            // swapped them. No clipping or change to the transport trajectory.
            nlohmann::json snapshot={{"error_code",error},{"duration_s",duration},
                {"buffers",nlohmann::json::object()}};
            message.precision(17);
            for(int stage=0;stage<(p.scratch?2:1);++stage) {
                std::vector<kg::Node> values(p.nodes.n);
                checked(cudaMemcpy(values.data(),stage?p.scratch->p:p.nodes.p,
                    values.size()*sizeof(kg::Node),cudaMemcpyDeviceToHost));
                const char* label=stage?"adapter_scratch":"adapter_nodes";
                auto& rows=snapshot["buffers"][label];rows=nlohmann::json::array();
                int invalid=0;
                for(std::size_t i=0;i<values.size();++i) {
                    const auto& n=values[i];
                    const bool bad=!(std::isfinite(n.mass)&&n.mass>0&&std::isfinite(n.energy)&&n.energy>0&&
                        std::isfinite(n.chemical)&&n.chemical>=0&&std::isfinite(n.oxidizer)&&n.oxidizer>=0&&
                        std::isfinite(n.vapor)&&n.vapor>=0&&n.vapor<=n.mass+1e-14&&std::isfinite(n.latent)&&n.latent>=0);
                    rows.push_back({{"id",p.node_names[i]},{"invalid",bad},{"mass_kg",n.mass},
                        {"energy_j",n.energy},{"chemical_j",n.chemical},{"oxidizer_kg",n.oxidizer},
                        {"vapor_kg",n.vapor},{"latent_j",n.latent},{"volume_m3",n.volume},
                        {"solid_capacity_j_k",n.solid_capacity}});
                    if(bad && invalid++<4)message<<"; "<<label<<":"<<p.node_names[i]
                        <<" mass="<<n.mass<<" E="<<n.energy<<" chem="<<n.chemical
                        <<" O2="<<n.oxidizer<<" vapor="<<n.vapor<<" latent="<<n.latent;
                }
                message<<"; "<<label<<" invalid_count="<<invalid;
            }
            if(const char* path=std::getenv("BATTERY_GAS_FAILURE_SNAPSHOT")) {
                std::ofstream file(path);file<<snapshot.dump(2);
                message<<(file?"; snapshot=":"; snapshot write failed=")<<path;
            }
        }
        if(error==25 || error==26){batched_gas::Control c{};checked(cudaMemcpy(&c,p.batch_control.p,sizeof(c),cudaMemcpyDeviceToHost));
            message<<"; mixing_iterations="<<c.mixing_iterations<<" mixing_residual="<<c.mixing_residual;
            if(const char* path=std::getenv("BATTERY_MIXING_FAILURE_SNAPSHOT")) {
                std::vector<batched_gas::MixSolveState> work(p.mix_work->n);
                std::vector<batched_gas::Proposal> flux(p.proposals->n);
                std::vector<kg::Edge> edges(p.edges.n);
                checked(cudaMemcpy(work.data(),p.mix_work->p,work.size()*sizeof(work[0]),cudaMemcpyDeviceToHost));
                checked(cudaMemcpy(flux.data(),p.proposals->p,flux.size()*sizeof(flux[0]),cudaMemcpyDeviceToHost));
                checked(cudaMemcpy(edges.data(),p.edges.p,edges.size()*sizeof(edges[0]),cudaMemcpyDeviceToHost));
                nlohmann::json dump={{"error_code",error},{"iterations",c.mixing_iterations},
                    {"normalized_residual",c.mixing_residual},{"gas_count",p.nodes.n},
                    {"step_s",c.accepted_reaction_step},{"cv_j_kg_k",p.settings.cv},
                    {"temperature_tolerance_k",p.model.implicit_temperature_tolerance_k},
                    {"tracer_relative_tolerance",p.model.implicit_tracer_relative_tolerance},
                    {"rows",nlohmann::json::array()},{"gas_edges",nlohmann::json::array()},
                    {"film_edges",nlohmann::json::array()}};
                for(std::size_t i=0;i<work.size();++i) {
                    const auto& w=work[i];nlohmann::json row={{"mobile",w.mobile},{"thermal",w.thermal},
                        {"sum",w.sum},{"film_sum",w.film_sum},
                        {"name",i<p.node_names.size()?p.node_names[i]:"solid-volume-"+std::to_string(i-p.nodes.n)}};
                    for(int f=0;f<5;++f){row["x"].push_back(w.x[f]);row["origin"].push_back(w.origin[f]);row["r"].push_back(w.r[f]);}
                    dump["rows"].push_back(row);
                }
                for(std::size_t e=0;e<edges.size();++e)if(flux[e].mixing.mass_each_way>0.)
                    dump["gas_edges"].push_back({edges[e].a,edges[e].b,flux[e].mixing.mass_each_way});
                if(p.model.joint_gas_surface_mixing && p.films) {
                    std::vector<GasThermalPatch> patches(p.films->patches.n);
                    std::vector<double> conductance(p.films->conductance.n);
                    checked(cudaMemcpy(patches.data(),p.films->patches.p,patches.size()*sizeof(patches[0]),cudaMemcpyDeviceToHost));
                    checked(cudaMemcpy(conductance.data(),p.films->conductance.p,conductance.size()*sizeof(double),cudaMemcpyDeviceToHost));
                    for(std::size_t j=0;j<patches.size();++j)if(conductance[j]>0.)dump["film_edges"].push_back(
                        {patches[j].compartment,int(p.nodes.n)+patches[j].volume,c.accepted_reaction_step*conductance[j]/p.settings.cv});
                }
                std::ofstream file(path);file<<dump.dump();message<<(file?"; mixing_snapshot=":"; snapshot write failed=")<<path;
            }
        }
        if(error==23 || error==24 || error==33 || error==34 || error==35){
            batched_gas::Control c{};checked(cudaMemcpy(&c,p.batch_control.p,sizeof(c),cudaMemcpyDeviceToHost));
            nlohmann::json diagnostic={{"window_s",c.averaged_window_s},{"elapsed_s",c.averaged_elapsed_s},
                {"pressure_iterations_used",c.averaged_pressure_used},{"pressure_relative_residual",c.averaged_pressure_residual},
                {"attempts",nlohmann::json::array()},{"nodes",nlohmann::json::array()},
                {"rejected_nodes",nlohmann::json::array()},{"channels",nlohmann::json::array()}};
            diagnostic["error_code"]=error;
            diagnostic["nonlinear_iterations"]=c.averaged_nonlinear_used;
            diagnostic["flow_law_defect"]=c.averaged_flow_defect;
            message<<"; nonlinear_iterations="<<c.averaged_nonlinear_used<<" flow_law_defect="<<c.averaged_flow_defect;
            diagnostic["enthalpy_iterations"]=c.enthalpy_iterations;
            diagnostic["enthalpy_residual"]=c.enthalpy_residual;
            message<<"; enthalpy_iterations="<<c.enthalpy_iterations<<" enthalpy_residual="<<c.enthalpy_residual;
            message<<"; pressure_iterations="<<c.averaged_pressure_used<<" relative_residual="<<c.averaged_pressure_residual;
            for(int a=0;error==23 && a<8;++a){diagnostic["attempts"].push_back({{"scale",c.averaged_attempt_scale[a]},
                {"invalid_mask",c.averaged_invalid_mask[a]},{"first_invalid_node",c.averaged_first_invalid[a]},
                {"entropy_change_j_per_k",c.averaged_entropy[a]}});
                message<<"; attempt="<<a<<" scale="<<c.averaged_attempt_scale[a]
                    <<" invalid="<<c.averaged_invalid_mask[a]<<" node="<<c.averaged_first_invalid[a]
                    <<" dS="<<c.averaged_entropy[a];}
            if(const char* path=std::getenv("BATTERY_AVERAGED_FAILURE_SNAPSHOT")){
                std::vector<kg::Node> old(p.nodes.n),rejected(p.nodes.n);
                const auto* before=c.averaged_old_is_initial?p.nodes.p:p.scratch->p;
                const auto* after=c.averaged_old_is_initial?p.scratch->p:p.nodes.p;
                checked(cudaMemcpy(old.data(),before,old.size()*sizeof(kg::Node),cudaMemcpyDeviceToHost));
                checked(cudaMemcpy(rejected.data(),after,rejected.size()*sizeof(kg::Node),cudaMemcpyDeviceToHost));
                for(std::size_t i=0;i<old.size();++i)for(int stage=0;stage<(error==23?2:1);++stage){const auto& n=stage?rejected[i]:old[i];
                    diagnostic[stage?"rejected_nodes":"nodes"].push_back({{"id",p.node_names[i]},
                        {"mass_kg",n.mass},{"free_volume_m3",n.volume},{"heat_capacity_j_per_k",n.solid_capacity+p.settings.cv*n.mass},
                        {"temperature_k",kg::temperature(n,p.settings)},{"energy_j",n.energy},{"chemical_j",n.chemical},
                        {"oxidizer_kg",n.oxidizer},{"vapor_kg",n.vapor},{"latent_j",n.latent}});}
                std::vector<GasChannelState> channel(p.edges.n);
                checked(cudaMemcpy(channel.data(),channels,channel.size()*sizeof(GasChannelState),cudaMemcpyDeviceToHost));
                for(std::size_t e=0;e<channel.size();++e)diagnostic["channels"].push_back({{"id",p.channel_names[e]},
                    {"effective_area_m2",channel[e].open?channel[e].effective_area_m2:0.}});
                std::ofstream file(path);file<<diagnostic.dump(2);
                if(!file)message<<"; failure snapshot write failed";else message<<"; snapshot="<<path;
            }
        }
        throw std::runtime_error(message.str());
    }
}
void CudaKawasakiTransport::commit(){std::swap(impl_->committed.p,impl_->candidate.p);}
std::vector<double> CudaKawasakiTransport::last_film_heat_j() const {
    const auto& p=*impl_;if(!p.films)throw std::runtime_error("Coupled gas film trace requires film adapter");
    std::vector<double> heat(p.films->heat.n);
    if(!heat.empty())checked(cudaMemcpy(heat.data(),p.films->heat.p,heat.size()*sizeof(double),cudaMemcpyDeviceToHost));
    return heat;
}
std::string CudaKawasakiTransport::save_state() const {
    const auto& p=*impl_;std::vector<kg::Clock> clocks(p.committed.n);
    if(!clocks.empty())checked(cudaMemcpy(clocks.data(),p.committed.p,clocks.size()*sizeof(kg::Clock),cudaMemcpyDeviceToHost));
    checkpoint::Writer w;w.text("gas-clocks-v1");w.vector(clocks);return w.finish();
}
void CudaKawasakiTransport::restore_state(const std::string& bytes) {
    auto& p=*impl_;checkpoint::Reader r(bytes);
    if(r.text()!="gas-clocks-v1")throw std::runtime_error("Unsupported gas restart version");
    auto clocks=r.vector<kg::Clock>(p.committed.n);r.finish();
    p.committed.upload(clocks);p.candidate.upload(clocks);p.batch_control.upload({batched_gas::Control{}});
}
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
        profile.gas_last_reaction_intervals=control.reaction_intervals;
        profile.gas_max_local_reactor_iterations=control.max_local_reactor_iterations;
        profile.gas_last_reaction_step_s=control.accepted_reaction_step;
        profile.gas_last_pressure_iterations=control.averaged_pressure_used;
        profile.gas_pressure_pattern_attempts+=control.pressure_pattern_attempts;
        profile.gas_pressure_pattern_accepted+=control.pressure_pattern_accepted;
        profile.gas_last_mixing_iterations=control.mixing_iterations;
        profile.gas_last_mixing_residual=control.mixing_residual;
        profile.gas_last_mixing_raw_residual=control.mixing_raw_residual;
        if(p.model.averaged_network_transport){
            profile.gas_averaged_min_scale=std::min(profile.gas_averaged_min_scale,control.averaged_min_scale);
            profile.gas_averaged_windows+=control.averaged_windows;
            profile.gas_averaged_limited_windows+=control.averaged_limited_windows;
            profile.gas_averaged_entropy_backtracks+=control.averaged_entropy_backtracks;
        }
    }
}
}

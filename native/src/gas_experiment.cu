#include "battery_native/gas_experiment_math.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace battery::gas_experiment {
namespace {
void check(cudaError_t e,const char* where) {
    if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
template<class T> struct Buffer {
    T* p=nullptr;std::size_t size=0;
    Buffer()=default;
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(p)cudaFree(p);}
    void resize(std::size_t n){if(p)check(cudaFree(p),"free");p=nullptr;size=n;if(n)check(cudaMalloc(&p,n*sizeof(T)),"allocate");}
    void upload(const std::vector<T>& v){resize(v.size());if(size)check(cudaMemcpy(p,v.data(),size*sizeof(T),cudaMemcpyHostToDevice),"upload");}
    std::vector<T> download() const {std::vector<T> v(size);if(size)check(cudaMemcpy(v.data(),p,size*sizeof(T),cudaMemcpyDeviceToHost),"download");return v;}
};
int blocks(std::size_t n){return static_cast<int>((n+127)/128);}
__global__ void propose_kernel(int count,const Edge* edges,const GasState* state,const int* offsets,GasModel model,Proposal* q) {
    const int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;const auto& e=edges[j];
    q[j]=propose(state[e.from],state[e.to],offsets[e.from+1]-offsets[e.from],offsets[e.to+1]-offsets[e.to],e,model);
}
__global__ void limit_kernel(int count,const Edge* edges,const GasState* state,const int* offsets,const int* adjacent,
    const Proposal* q,GasModel model,double* limits) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;Budget b;
    for(int k=offsets[i];k<offsets[i+1];++k) {
        const int j=adjacent[k];const auto& e=edges[j];const int donor=q[j].rate>=0?e.from:e.to;
        add_budget(b,state[i],state[donor],donor==i,q[j],model);
    }
    limits[i]=limit(b,state[i],model);
}
__global__ void minimum_kernel(int count,const double* limits,double cap,double* step) {
    // Small benchmark networks; one block, fixed reduction order.
    __shared__ double values[128];double dt=cap;
    for(int i=threadIdx.x;i<count;i+=128)dt=fmin(dt,limits[i]);
    values[threadIdx.x]=dt;__syncthreads();
    for(int stride=64;stride;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]=fmin(values[threadIdx.x],values[threadIdx.x+stride]);__syncthreads();}
    if(threadIdx.x==0)*step=values[0];
}
__global__ void transfer_kernel(int count,const Edge* edges,const GasState* state,const Proposal* q,const double* dt,
    GasModel model,Packet* packet,Packet* total) {
    const int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;const auto& e=edges[j];
    packet[j]=transfer(q[j],state[e.from],state[e.to],*dt,model);accumulate(total[j],packet[j]);
}
__global__ void advance_kernel(int count,const Edge* edges,const GasState* old,GasState* next,const int* offsets,
    const int* adjacent,const Packet* packet,GasModel model,int* error) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;auto s=old[i];
    for(int k=offsets[i];k<offsets[i+1];++k){const int j=adjacent[k];apply(s,packet[j],edges[j].from==i?-1.:1.);}
    refresh(s,model);if(!valid(s))atomicExch(error,1);next[i]=s;
}
struct PatchWorkspace {
    GasState state[max_patch_nodes],next[max_patch_nodes];
    Proposal proposed[max_patch_edges];
    Packet packet[max_patch_edges],total[max_patch_edges];
    double limits[max_patch_nodes],dt,elapsed;
    unsigned long long count;
    bool done;
};
__global__ void patch_kernel(const int* ids,const Patch* patches,const Edge* edges,GasState* global,
    Packet* total,GasModel model,double duration,unsigned long long* counts,int* error) {
    const int id=ids[blockIdx.x];const auto& p=patches[id];const int lane=threadIdx.x;
    extern __shared__ __align__(16) unsigned char memory[];
    auto& w=*reinterpret_cast<PatchWorkspace*>(memory);
    if(lane<p.nodes)w.state[lane]=global[p.node[lane]];
    if(lane<p.edges)w.total[lane]={};
    if(lane==0){w.elapsed=0;w.count=0;w.done=false;}
    __syncthreads();
    while(!w.done) {
        if(lane<p.edges)w.proposed[lane]=propose(w.state[p.from[lane]],w.state[p.to[lane]],
            p.degree[p.from[lane]],p.degree[p.to[lane]],edges[p.edge[lane]],model);
        __syncthreads();
        if(lane<p.nodes) {
            Budget b;
            for(int j=0;j<p.edges;++j)if(p.from[j]==lane || p.to[j]==lane) {
                const int donor=w.proposed[j].rate>=0?p.from[j]:p.to[j];
                add_budget(b,w.state[lane],w.state[donor],donor==lane,w.proposed[j],model);
            }
            w.limits[lane]=limit(b,w.state[lane],model);
        }
        __syncthreads();
        if(lane==0) {
            w.dt=duration-w.elapsed;for(int i=0;i<p.nodes;++i)w.dt=fmin(w.dt,w.limits[i]);
            if(!(w.dt>0) || w.elapsed+w.dt==w.elapsed || ++w.count>2000000){atomicExch(error,2);w.done=true;}
        }
        __syncthreads();if(w.done)break;
        if(lane<p.edges) {
            w.packet[lane]=transfer(w.proposed[lane],w.state[p.from[lane]],w.state[p.to[lane]],w.dt,model);
            accumulate(w.total[lane],w.packet[lane]);
        }
        __syncthreads();
        if(lane<p.nodes) {
            auto s=w.state[lane];for(int j=0;j<p.edges;++j) {
                if(p.from[j]==lane)apply(s,w.packet[j],-1.);else if(p.to[j]==lane)apply(s,w.packet[j],1.);
            }
            refresh(s,model);if(!valid(s))atomicExch(error,1);w.next[lane]=s;
        }
        __syncthreads();
        if(lane<p.nodes)w.state[lane]=w.next[lane];
        if(lane==0){w.elapsed+=w.dt;w.done=w.elapsed>=duration;}
        __syncthreads();
    }
    if(lane<p.nodes)global[p.node[lane]]=w.state[lane];
    if(lane<p.edges)accumulate(total[p.edge[lane]],w.total[lane]);
    if(lane==0)counts[id]+=w.count;
}
} // namespace
struct Cuda::Impl {
    const Network& network;
    Buffer<GasState> state,next;
    Buffer<Edge> edges;
    Buffer<int> offsets,adjacent,ids,error;
    Buffer<Patch> patches;
    Buffer<Proposal> proposed;
    Buffer<Packet> packet,total;
    Buffer<double> limits,step;
    Buffer<unsigned long long> counts;
    std::vector<int> color_offsets;
    Work work;
    explicit Impl(const Network& n):network(n) {
        edges.upload(n.edges);offsets.upload(n.offsets);adjacent.upload(n.adjacent);patches.upload(n.patches);
        std::vector<int> ordered;color_offsets.push_back(0);
        for(const auto& c:n.colors){ordered.insert(ordered.end(),c.begin(),c.end());color_offsets.push_back(ordered.size());}
        ids.upload(ordered);state.resize(n.initial.size());next.resize(n.initial.size());
        proposed.resize(n.edges.size());packet.resize(n.edges.size());total.resize(n.edges.size());
        limits.resize(n.initial.size());step.resize(1);error.resize(1);counts.resize(n.patches.size());
    }
};
Cuda::Cuda(const Network& n):impl_(std::make_unique<Impl>(n)){reset();}
Cuda::~Cuda()=default;
void Cuda::reset() {
    auto& p=*impl_;check(cudaMemcpy(p.state.p,p.network.initial.data(),p.state.size*sizeof(GasState),cudaMemcpyHostToDevice),"reset gas");
    if(p.total.size)check(cudaMemset(p.total.p,0,p.total.size*sizeof(Packet)),"reset transfer");
    if(p.counts.size)check(cudaMemset(p.counts.p,0,p.counts.size*sizeof(unsigned long long)),"reset counters");
    check(cudaMemset(p.error.p,0,sizeof(int)),"reset error");p.work={};
}
double Cuda::global_step(double cap) {
    if(!(cap>0) || !std::isfinite(cap))throw std::runtime_error("Invalid CUDA step");auto& p=*impl_;
    if(p.edges.size)propose_kernel<<<blocks(p.edges.size),128>>>(p.edges.size,p.edges.p,p.state.p,p.offsets.p,p.network.model,p.proposed.p);
    limit_kernel<<<blocks(p.state.size),128>>>(p.state.size,p.edges.p,p.state.p,p.offsets.p,p.adjacent.p,p.proposed.p,p.network.model,p.limits.p);
    minimum_kernel<<<1,128>>>(p.state.size,p.limits.p,cap,p.step.p);
    if(p.edges.size)transfer_kernel<<<blocks(p.edges.size),128>>>(p.edges.size,p.edges.p,p.state.p,p.proposed.p,p.step.p,p.network.model,p.packet.p,p.total.p);
    advance_kernel<<<blocks(p.state.size),128>>>(p.state.size,p.edges.p,p.state.p,p.next.p,p.offsets.p,p.adjacent.p,p.packet.p,p.network.model,p.error.p);
    double dt;check(cudaMemcpy(&dt,p.step.p,sizeof(double),cudaMemcpyDeviceToHost),"global dt");
    if(!(dt>0) || !std::isfinite(dt))throw std::runtime_error("Invalid CUDA gas timestep");
    std::swap(p.state.p,p.next.p);++p.work.steps;p.work.edge_evaluations+=p.edges.size;p.work.launches+=p.edges.size?5:3;return dt;
}
void Cuda::local_window(double duration) {
    if(!(duration>0) || !std::isfinite(duration))throw std::runtime_error("Invalid CUDA patch interval");auto& p=*impl_;
    auto launch=[&](int c) {
        const int begin=p.color_offsets[c],count=p.color_offsets[c+1]-begin;
        patch_kernel<<<count,64,sizeof(PatchWorkspace)>>>(p.ids.p+begin,p.patches.p,p.edges.p,p.state.p,p.total.p,
            p.network.model,duration/2,p.counts.p,p.error.p);++p.work.launches;
    };
    for(int c=0;c<static_cast<int>(p.network.colors.size());++c)launch(c);
    for(int c=static_cast<int>(p.network.colors.size())-1;c>=0;--c)launch(c);
    check(cudaGetLastError(),"patch launch");++p.work.steps;
}
void Cuda::synchronize() const {
    check(cudaDeviceSynchronize(),"gas experiment synchronize");int error;
    check(cudaMemcpy(&error,impl_->error.p,sizeof(int),cudaMemcpyDeviceToHost),"gas experiment error");
    if(error)throw std::runtime_error("CUDA gas experiment failed, code "+std::to_string(error));
}
void Cuda::download(std::vector<GasState>& gas,std::vector<Packet>& packets) const {synchronize();gas=impl_->state.download();packets=impl_->total.download();}
Work Cuda::work() const {
    synchronize();auto w=impl_->work;const auto counts=impl_->counts.download();
    for(std::size_t i=0;i<counts.size();++i){w.patch_substeps+=counts[i];w.edge_evaluations+=counts[i]*impl_->network.patches[i].edges;}
    return w;
}
double Cuda::state_roundtrip_seconds(int repetitions) {
    if(repetitions<=0)throw std::runtime_error("Invalid repetitions");synchronize();auto& p=*impl_;
    std::vector<GasState> host(p.state.size);const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<repetitions;++i) {
        check(cudaMemcpy(host.data(),p.state.p,host.size()*sizeof(GasState),cudaMemcpyDeviceToHost),"state D2H");
        check(cudaMemcpy(p.state.p,host.data(),host.size()*sizeof(GasState),cudaMemcpyHostToDevice),"state H2D");
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repetitions;
}
} // namespace battery::gas_experiment

#include "battery_native/kawasaki_gas_math.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace battery::kawasaki {
namespace {
void check(cudaError_t e,const char* where) {
    if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
template<class T> struct Buffer {
    T* p=nullptr;std::size_t count=0;
    explicit Buffer(std::size_t n):count(n){if(n)check(cudaMalloc(&p,n*sizeof(T)),"allocate Kawasaki");}
    ~Buffer(){if(p)cudaFree(p);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    void upload(const std::vector<T>& v){if(v.size()!=count)throw std::runtime_error("Kawasaki buffer size");if(count)check(cudaMemcpy(p,v.data(),count*sizeof(T),cudaMemcpyHostToDevice),"upload Kawasaki");}
    std::vector<T> download()const{std::vector<T> v(count);if(count)check(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost),"download Kawasaki");return v;}
};
__global__ void pairs(int count,int replicas,int nodes,int edges,const int* ids,const Edge* topology,
    Node* state,Clock* clocks,Settings settings,double dt,std::uint64_t seed,int* error) {
    const auto tid=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid>=std::size_t(count)*replicas)return;
    const int replica=tid/count,id=ids[tid%count];const auto e=topology[id];
    const auto base=std::size_t(replica)*nodes;
    const int code=evolve(state[base+e.a],state[base+e.b],e,clocks[std::size_t(replica)*edges+id],settings,dt,mix(seed+replica),id);
    if(code)atomicMax(error,code); // diagnostics only, never state accumulation
}
Network checked(const Network& n,int replicas,int block) {
    validate(n);if(replicas<=0 || (block!=64 && block!=128 && block!=256))throw std::invalid_argument("Invalid Kawasaki launch");
    return n;
}
}
struct Cuda::Impl {
    Network network;int replicas,block;std::uint64_t seed;
    Buffer<Node> states;Buffer<Edge> edges;Buffer<Clock> clocks;Buffer<int> ids,error;
    std::vector<int> offsets;
    Impl(const Network& n,int r,std::uint64_t s,int b):network(checked(n,r,b)),replicas(r),block(b),seed(s),
        states(n.initial.size()*r),edges(n.edges.size()),clocks(n.edges.size()*r),ids(n.edges.size()),error(1) {
        edges.upload(n.edges);std::vector<int> ordered;offsets.push_back(0);
        for(const auto& color:n.colors){ordered.insert(ordered.end(),color.begin(),color.end());offsets.push_back(ordered.size());}
        ids.upload(ordered);
    }
};
Cuda::Cuda(const Network& n,int r,std::uint64_t seed,int block):impl_(std::make_unique<Impl>(n,r,seed,block)){reset();}
Cuda::~Cuda()=default;
void Cuda::reset() {
    auto& p=*impl_;std::vector<Node> state;
    for(int i=0;i<p.replicas;++i)state.insert(state.end(),p.network.initial.begin(),p.network.initial.end());
    p.states.upload(state);p.clocks.upload(std::vector<Clock>(p.clocks.count));p.error.upload({0});
}
void Cuda::advance(double dt) {
    if(!(dt>0 && std::isfinite(dt)))throw std::invalid_argument("Invalid Kawasaki interval");auto& p=*impl_;
    auto launch=[&](int c) {
        const int start=p.offsets[c],count=p.offsets[c+1]-start;
        if(!count)return;
        const auto work=std::size_t(count)*p.replicas;
        pairs<<<(work+p.block-1)/p.block,p.block>>>(count,p.replicas,p.network.initial.size(),p.network.edges.size(),
            p.ids.p+start,p.edges.p,p.states.p,p.clocks.p,p.network.settings,dt/2,p.seed,p.error.p);
    };
    for(int c=0;c<int(p.network.colors.size());++c)launch(c);
    for(int c=int(p.network.colors.size())-1;c>=0;--c)launch(c);
    check(cudaGetLastError(),"Kawasaki launch");
}
void Cuda::synchronize()const {
    check(cudaDeviceSynchronize(),"Kawasaki synchronize");const int code=impl_->error.download()[0];
    if(code)throw std::runtime_error("Kawasaki CUDA failure "+std::to_string(code));
}
void Cuda::download(std::vector<Node>& n,std::vector<Clock>& c)const{synchronize();n=impl_->states.download();c=impl_->clocks.download();}
} // namespace battery::kawasaki

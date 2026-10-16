#include "battery_native/cuda_gl_field.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <algorithm>
#include <stdexcept>

namespace battery {
namespace {
void check(cudaError_t value,const char* what) {
    if(value!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(value));
}
__device__ float4 mix(float4 a,float4 b,float t) {
    return make_float4(a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,a.z+(b.z-a.z)*t,0);
}
__global__ void samples_kernel(int count,const ThermalState* state,const int* ids,
    const float* aging,float4* samples) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& s=state[ids[i]];
    samples[i]=make_float4(static_cast<float>(s.temperature_k),
        static_cast<float>(1-(1-aging[i])*(1-s.damage)),s.mode==CellMode::Spent?1.f:0.f,0.f);
}
__global__ void colors_kernel(int count,const ThermalState* state,const int* ids,const float* aging,
    const float4* thermal,const float4* degradation,float4 destroyed,int layer,float tmin,float tmax,float opacity,float4* colors) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& s=state[ids[i]];float4 color;
    if(layer==0) {
        float value=static_cast<float>((s.temperature_k-tmin)/fmaxf(tmax-tmin,1.0f));
        float position=fminf(fmaxf(value,0),1)*255;
        int lo=static_cast<int>(position),hi=min(lo+1,255);
        color=mix(thermal[lo],thermal[hi],position-lo);
    } else if(s.mode==CellMode::Spent) color=destroyed;
    else {
        float value=static_cast<float>(1-(1-aging[i])*(1-s.damage));
        value=fminf(fmaxf(value,0),1);int upper=1;
        while(upper<4&&value>degradation[upper].w)++upper;
        auto a=degradation[upper-1],b=degradation[upper];color=mix(a,b,(value-a.w)/(b.w-a.w));
    }
    color.w=opacity;colors[i]=color;
}
}
struct CudaGlField::Impl {
    cudaGraphicsResource_t resource=nullptr;
    int* ids=nullptr;float* aging=nullptr;float4* thermal=nullptr;float4* degradation=nullptr;
    int count=0;
    ~Impl() {clear();}
    void clear() {
        if(resource)cudaGraphicsUnregisterResource(resource);
        if(ids)cudaFree(ids);if(aging)cudaFree(aging);if(thermal)cudaFree(thermal);if(degradation)cudaFree(degradation);
        resource=nullptr;ids=nullptr;aging=nullptr;thermal=degradation=nullptr;count=0;
    }
};
CudaGlField::CudaGlField():impl_(std::make_unique<Impl>()) {}
CudaGlField::~CudaGlField()=default;
void CudaGlField::detach() {impl_->clear();}
void CudaGlField::attach(unsigned int buffer,const std::vector<int>& ids,const std::vector<float>& aging) {
    detach();if(ids.size()!=aging.size())throw std::runtime_error("CUDA visual map size mismatch");
    if(ids.empty())return;auto& p=*impl_;p.count=static_cast<int>(ids.size());
    check(cudaMalloc(&p.ids,ids.size()*sizeof(int)),"Visual indices");
    check(cudaMalloc(&p.aging,aging.size()*sizeof(float)),"Visual aging");
    check(cudaMalloc(&p.thermal,256*sizeof(float4)),"Thermal LUT");
    check(cudaMalloc(&p.degradation,5*sizeof(float4)),"Degradation stops");
    check(cudaMemcpy(p.ids,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice),"Visual indices upload");
    check(cudaMemcpy(p.aging,aging.data(),aging.size()*sizeof(float),cudaMemcpyHostToDevice),"Visual aging upload");
    std::vector<float4> thermal,degradation;
    for(auto c:thermal_lut)thermal.push_back(make_float4(c[0],c[1],c[2],0));
    for(auto s:degradation_stops)degradation.push_back(make_float4(s.color[0],s.color[1],s.color[2],s.position));
    check(cudaMemcpy(p.thermal,thermal.data(),256*sizeof(float4),cudaMemcpyHostToDevice),"Thermal LUT upload");
    check(cudaMemcpy(p.degradation,degradation.data(),5*sizeof(float4),cudaMemcpyHostToDevice),"Damage LUT upload");
    check(cudaGraphicsGLRegisterBuffer(&p.resource,buffer,cudaGraphicsRegisterFlagsWriteDiscard),"CUDA/OpenGL color-buffer registration");
}
void CudaGlField::update(const ThermalState* state,FieldLayer layer,float tmin,float tmax,float opacity) {
    auto& p=*impl_;if(!p.count)return;
    check(cudaGraphicsMapResources(1,&p.resource),"CUDA/OpenGL map");
    try {
        float4* colors=nullptr;std::size_t bytes=0;
        check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&colors),&bytes,p.resource),"Mapped color buffer");
        if(bytes<static_cast<std::size_t>(p.count)*sizeof(float4))throw std::runtime_error("Mapped color buffer too small");
        colors_kernel<<<(p.count+127)/128,128>>>(p.count,state,p.ids,p.aging,p.thermal,p.degradation,
            make_float4(destroyed_rgb[0],destroyed_rgb[1],destroyed_rgb[2],0),static_cast<int>(layer),tmin,tmax,opacity,colors);
        check(cudaGetLastError(),"CUDA colors launch");
    } catch(...) {cudaGraphicsUnmapResources(1,&p.resource);throw;}
    check(cudaGraphicsUnmapResources(1,&p.resource),"CUDA/OpenGL unmap");
}
void CudaGlField::update_samples(const ThermalState* state) {
    auto& p=*impl_;if(!p.count)return;
    check(cudaGraphicsMapResources(1,&p.resource),"CUDA/OpenGL sample map");
    try {
        float4* samples=nullptr;std::size_t bytes=0;
        check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&samples),&bytes,p.resource),"Mapped sample buffer");
        if(bytes<static_cast<std::size_t>(p.count)*sizeof(float4))throw std::runtime_error("Mapped sample buffer too small");
        samples_kernel<<<(p.count+127)/128,128>>>(p.count,state,p.ids,p.aging,samples);
        check(cudaGetLastError(),"CUDA samples launch");
    } catch(...) {cudaGraphicsUnmapResources(1,&p.resource);throw;}
    check(cudaGraphicsUnmapResources(1,&p.resource),"CUDA/OpenGL sample unmap");
}
}

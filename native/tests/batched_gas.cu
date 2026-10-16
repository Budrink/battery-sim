#include "battery_native/cuda_solver.hpp"
#include "../src/batched_gas.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
namespace kg=battery::kawasaki;
namespace bg=battery::batched_gas;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
template<class T> struct Buffer {
    T* p=nullptr;size_t n;
    explicit Buffer(size_t size):n(size){if(n)checked(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){if(p)cudaFree(p);}
    void put(const std::vector<T>& v){require(v.size()==n,"buffer size");if(n)checked(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);if(n)checked(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
kg::Node node(double pressure,double temperature,double volume=1e-6,double solid=0) {
    const double mass=pressure*volume/(287*temperature);
    return {mass,(solid+1000*mass)*temperature,volume,solid,mass*1e6,mass*.15};
}
kg::Network chain(int count=6) {
    kg::Network n;n.settings.packet_fraction=.01;
    for(int i=0;i<count;++i)n.initial.push_back(node(i?1e5:3e5,i?300:900,1e-6,.001));
    for(int i=1;i<count;++i)n.edges.push_back({i-1,i,2e-8,.7});
    kg::schedule(n);return n;
}
struct Result {std::vector<kg::Node> nodes;std::vector<kg::Clock> clocks;std::vector<battery::GasChannelState> channels;double pressure_gap_pa=0;};
Result run(const kg::Network& n,double end,double interval,std::uint64_t seed=71,
           double threshold=32,double pressure_fraction=.2,int block=256,int blocks=0,int deterministic_sweeps=0,
           bool network=false,double network_fraction=.35) {
    Buffer<kg::Node> state(n.initial.size()),scratch(n.initial.size());state.put(n.initial);
    Buffer<kg::Edge> edges(n.edges.size());edges.put(n.edges);
    Buffer<kg::Clock> clocks(n.edges.size());clocks.put(std::vector<kg::Clock>(n.edges.size()));
    Buffer<bg::Proposal> proposals(n.edges.size());Buffer<bg::Budget> budgets(n.initial.size());
    Buffer<battery::GasChannelState> channels(n.edges.size());
    std::vector<battery::GasChannelState> channel(n.edges.size());
    std::vector<std::vector<int>> rows(n.initial.size());
    for(int i=0;i<int(n.edges.size());++i){rows[n.edges[i].a].push_back(i);rows[n.edges[i].b].push_back(i);
        channel[i].open=n.edges[i].area>0;channel[i].effective_area_m2=n.edges[i].area;}
    channels.put(channel);std::vector<int> flat,offset{0};
    for(auto& row:rows){flat.insert(flat.end(),row.begin(),row.end());offset.push_back(flat.size());}
    Buffer<int> ids(flat.size()),offsets(offset.size()),error(1);ids.put(flat);offsets.put(offset);error.put({0});
    battery::GasModel model;model.kawasaki_seed=seed;model.local_exchange_interval_s=interval;
    model.batch_pressure_fraction=pressure_fraction;model.batch_mean_threshold=threshold;
    if(deterministic_sweeps>0){model.cuda_deterministic_transport=true;model.deterministic_sweeps_per_step=deterministic_sweeps;model.deterministic_pressure_fraction=pressure_fraction;}
    model.cuda_network_transport=network;model.network_pressure_fraction=network_fraction;
    Buffer<bg::Control> control(1);
    checked(bg::launch(state.n,edges.n,offsets.p,ids.p,edges.p,state.p,scratch.p,
        proposals.p,budgets.p,clocks.p,channels.p,n.settings,model,end,error.p,control.p,block,blocks));
    checked(cudaGetLastError());const int code=error.get()[0];
    if(code)throw std::runtime_error("Batched test kernel error "+std::to_string(code));
    const auto controls=control.get();
    return {state.get(),clocks.get(),channels.get(),(deterministic_sweeps>0 || network)?controls[0].last_max_pressure_gap_pa:0};
}
void balances(const kg::Network& n,const Result& result) {
    double before[5]{},after[5]{};
    for(size_t i=0;i<n.initial.size();++i) {
        const auto& a=n.initial[i];const auto& b=result.nodes[i];
        require(b.mass>0 && b.energy>0 && b.chemical>=0 && b.oxidizer>=0,"negative inventory");
        const double x[]{a.mass,a.energy,a.chemical,a.oxidizer,kg::entropy(a,n.settings)};
        const double y[]{b.mass,b.energy,b.chemical,b.oxidizer,kg::entropy(b,n.settings)};
        for(int k=0;k<5;++k){before[k]+=x[k];after[k]+=y[k];}
        double dm=0,de=0,dc=0,doo=0;
        for(size_t j=0;j<n.edges.size();++j){const auto& e=n.edges[j];const auto& c=result.channels[j];
            const double sign=e.a==int(i)?-1:e.b==int(i)?1:0;
            dm+=sign*c.transferred_kg;
            de+=sign*(c.transferred_energy_j+c.transferred_kg*1000*298.15);
            dc+=sign*c.transferred_chemical_j;doo+=sign*c.transferred_oxidizer_kg;}
        require(std::abs(b.mass-a.mass-dm)<1e-11*std::max(a.mass,b.mass),"mass channel ledger");
        require(std::abs(b.energy-a.energy-de)<1e-11*std::max(a.energy,b.energy),"energy channel ledger");
        require(std::abs(b.chemical-a.chemical-dc)<1e-11*std::max(a.chemical,b.chemical),"chemical channel ledger");
        require(std::abs(b.oxidizer-a.oxidizer-doo)<1e-11*std::max(a.oxidizer,b.oxidizer),"oxidizer channel ledger");
    }
    for(int k=0;k<4;++k)require(std::abs(before[k]-after[k])<1e-11*std::abs(before[k]),"global conservation");
    require(after[4]>=before[4]-1e-10*std::max(1.,std::abs(before[4])),"global entropy");
}
double temperatures(const Result& a,const Result& b,const kg::Settings& s) {
    double error=0;for(size_t i=0;i<a.nodes.size();++i)error=std::max(error,std::abs(kg::temperature(a.nodes[i],s)-kg::temperature(b.nodes[i],s)));return error;
}
__global__ void draw_counts(unsigned* counts,int samples,double mean) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<samples)counts[i]=bg::poisson(mean,71,i,19);
}
void poisson_checks() {
    constexpr int samples=65536;Buffer<unsigned> counts(samples);
    for(double mean:{.01,.3,4.,31.9}) {
        draw_counts<<<samples/256,256>>>(counts.p,samples,mean);checked(cudaGetLastError());
        const auto values=counts.get();double average=0,square=0,zeros=0;
        for(auto count:values){average+=double(count)/samples;square+=double(count)*count/samples;zeros+=count==0?1./samples:0;}
        const double variance=square-average*average,zero=std::exp(-mean);
        require(std::abs(average-mean)<6*std::sqrt(mean/samples),"Poisson mean");
        require(std::abs(variance-mean)<6*std::sqrt((mean+2*mean*mean)/samples),"Poisson variance");
        require(std::abs(zeros-zero)<6*std::sqrt(zero*(1-zero)/samples)+1./samples,"Poisson zero-event probability");
    }
}
void topology_checks() {
    auto n=chain();const auto a=run(n,.01,.0001),b=run(n,.01,.0001,71,32,.2,64);
    balances(n,a);balances(n,b);require(temperatures(a,b,n.settings)==0,"CUDA block changes random path");
    for(size_t i=0;i<n.edges.size();++i)require(a.clocks[i].events==b.clocks[i].events,"CUDA block changes event count");
    auto faster=n;for(auto& e:faster.edges)e.area*=4;
    const auto fast=run(faster,.0025,.000025);balances(faster,fast);
    require(temperatures(a,fast,n.settings)<1e-9,"Aperture/time scaling");
    auto closed=n;for(auto& e:closed.edges)e.area=0;const auto shut=run(closed,.01,.001);balances(closed,shut);
    require(temperatures(shut,{closed.initial,{},{}},n.settings)==0,"closed channel heats");
    for(const auto& c:shut.channels)require(c.transferred_kg==0,"closed channel leaks");
    kg::Network hub;hub.settings=n.settings;hub.initial.push_back(node(1e5,300,.01));
    for(int i=1;i<=500;++i){hub.initial.push_back(node(i%2?3e5:5e4,i%2?900:400,1e-6,.001));hub.edges.push_back({i,0,1e-8,.7});}
    kg::schedule(hub);const auto mixed=run(hub,.002,.0001);balances(hub,mixed);
    const auto single=run(hub,.002,.0001,71,32,.2,256,1);balances(hub,single);
    require(temperatures(mixed,single,hub.settings)==0,"cooperative grid changes trajectory");
    require(mixed.channels[0].transferred_kg>0 && mixed.channels[1].transferred_kg<0,"hub loses bidirectional streams");
    // Two volumes exchange exactly their finite inventory, not half per step.
    auto pair=chain(2);pair.initial={node(100050,300),node(100000,300)};
    const auto near=run(pair,.1,.001);balances(pair,near);
    require(std::abs(kg::pressure(near.nodes[0],pair.settings)-kg::pressure(near.nodes[1],pair.settings))<.001,"tail fails equilibration");
}
void convergence() {
    auto n=chain();const auto coarse=run(n,.01,.001,71,0,.2);
    const auto fine=run(n,.01,.00025,71,0,.05),reference=run(n,.01,.0000625,71,0,.0125);
    balances(n,coarse);balances(n,fine);balances(n,reference);
    const double e0=temperatures(coarse,reference,n.settings),e1=temperatures(fine,reference,n.settings);
    std::cout<<"deterministic temperature refinement: "<<e0<<" -> "<<e1<<" K\n";
    require(e1<e0*.7 && e1<.5,"mean flux fails temporal convergence");
    constexpr int samples=128;kg::Cuda event_reference(n,samples,1000);
    for(int k=0;k<40;++k)event_reference.advance(.00025);
    std::vector<kg::Node> event;std::vector<kg::Clock> clocks;event_reference.download(event,clocks);
    double means[3]{},squares[3]{};
    for(int k=0;k<samples;++k) {
        const auto a=run(n,.01,.001,1000+k),b=run(n,.01,.00025,1000+k,32,.05);
        balances(n,a);balances(n,b);
        const double values[]{kg::temperature(a.nodes[1],n.settings),kg::temperature(b.nodes[1],n.settings),kg::temperature(event[k*n.initial.size()+1],n.settings)};
        for(int i=0;i<3;++i){means[i]+=values[i]/samples;squares[i]+=values[i]*values[i]/samples;}
    }
    for(int i=0;i<2;++i) {
        const double se=std::sqrt(std::max(0.,squares[i]-means[i]*means[i]+squares[2]-means[2]*means[2])/(samples-1));
        std::cout<<"ensemble "<<i<<": mean="<<means[i]<<" event reference="<<means[2]<<" combined SE="<<se<<" K\n";
        require(std::abs(means[i]-means[2])<4*se+.5,"tau-leap mean disagrees with event CUDA reference");
    }
}
void deterministic_sweeps() {
    auto n=chain();
    const auto one=run(n,.01,.001,71,32,.2,256,0,1);
    const auto two=run(n,.01,.001,71,32,.2,256,0,2);
    const auto four=run(n,.01,.001,71,32,.2,256,0,4);
    const auto eight=run(n,.01,.001,71,32,.2,256,0,8);
    balances(n,one);balances(n,two);balances(n,four);balances(n,eight);
    require(one.pressure_gap_pa>0 && two.pressure_gap_pa>0 && four.pressure_gap_pa>0 && eight.pressure_gap_pa>0,"deterministic pressure-gap diagnostic missing");
    const auto repeat=run(n,.01,.001,71,32,.2,256,0,4);
    require(temperatures(four,repeat,n.settings)==0,"deterministic sweeps are not reproducible");
    std::cout<<"deterministic sweeps max open-edge dP: 1="<<one.pressure_gap_pa<<", 2="<<two.pressure_gap_pa<<", 4="<<four.pressure_gap_pa<<", 8="<<eight.pressure_gap_pa<<" Pa\n";
}
void network_integrator() {
    auto n=chain();
    const auto coarse=run(n,.01,.001,71,32,.2,256,0,0,true,1.);
    balances(n,coarse);
    require(coarse.pressure_gap_pa>0,"network pressure diagnostic missing");
    require(std::abs(coarse.nodes[0].mass-n.initial[0].mass)>1e-14,"analytic network transported no mass");
    const auto repeat=run(n,.01,.001,71,32,.2,256,0,0,true,1.);
    require(temperatures(coarse,repeat,n.settings)==0,"analytic network is not reproducible");
    auto reversed=n;std::reverse(reversed.edges.begin(),reversed.edges.end());
    const auto permuted=run(reversed,.01,.001,71,32,.2,256,0,0,true,1.);
    balances(reversed,permuted);
    require(temperatures(coarse,permuted,n.settings)<1e-8,"network integration depends on edge ordering");
    kg::Network hub;hub.settings=n.settings;hub.initial.push_back(node(1e5,300,.01));
    for(int i=1;i<=64;++i){hub.initial.push_back(node(i%2?3e5:5e4,i%2?900:400,1e-6,.001));hub.edges.push_back({i,0,1e-8,.7});}
    const auto hub_a=run(hub,.002,.0001,71,32,.2,256,0,0,true,1.);balances(hub,hub_a);
    std::reverse(hub.edges.begin(),hub.edges.end());
    const auto hub_b=run(hub,.002,.0001,71,32,.2,256,0,0,true,1.);balances(hub,hub_b);
    require(temperatures(hub_a,hub_b,hub.settings)<1e-8,"branched network depends on edge ordering");
    std::cout<<"analytic network: max open-edge dP="<<coarse.pressure_gap_pa<<" Pa, branched hub order invariant\n";
}
int main(){try {poisson_checks();topology_checks();convergence();deterministic_sweeps();network_integrator();std::cout<<"CUDA batched: stochastic, fixed-sweep and analytic network checks pass\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

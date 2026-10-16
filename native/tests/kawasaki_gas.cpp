#include "battery_native/kawasaki_gas_math.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace kg=battery::kawasaki;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
kg::Node node(double p,double t,double volume=1e-6,double solid=0) {
    const double m=p*volume/(287*t);return {m,(solid+1000*m)*t,volume,solid,.1,m*.15};
}
kg::Network fixture(int count=2) {
    kg::Network n;n.settings.tail_atol_pa=0;
    for(int i=0;i<count;++i)n.initial.push_back(node(i?1e5:3e5,i?300:900));
    for(int i=1;i<count;++i)n.edges.push_back({i-1,i,2e-8,.7});kg::schedule(n);return n;
}
void balances(const kg::Network& n,const std::vector<kg::Node>& state,int replicas) {
    double m0=0,u0=0,c0=0,o0=0,s0=0;
    for(const auto& v:n.initial){m0+=v.mass;u0+=v.energy;c0+=v.chemical;o0+=v.oxidizer;s0+=kg::entropy(v,n.settings);}
    for(int r=0;r<replicas;++r) {
        double m=0,u=0,c=0,o=0,s=0;
        for(std::size_t i=0;i<n.initial.size();++i) {
            const auto& v=state[r*n.initial.size()+i];m+=v.mass;u+=v.energy;c+=v.chemical;o+=v.oxidizer;s+=kg::entropy(v,n.settings);
            require(v.mass>0 && v.energy>0 && v.chemical>=0 && v.oxidizer>=0,"Invalid inventory");
        }
        require(std::abs(m-m0)<1e-12*m0 && std::abs(u-u0)<1e-12*u0 &&
            std::abs(c-c0)<1e-12*c0 && std::abs(o-o0)<1e-12*o0,"Conservation failure");
        require(s>=s0-1e-11*std::max(1.,std::abs(s0)),"Negative entropy production");
    }
}
double difference(const std::vector<kg::Node>& a,const std::vector<kg::Node>& b,double scale=1.) {
    require(a.size()==b.size(),"Different arrays");double error=0;
    for(std::size_t i=0;i<a.size();++i) {
        error=std::max(error,std::abs(a[i].mass-b[i].mass/scale)/a[i].mass);
        error=std::max(error,std::abs(a[i].energy-b[i].energy/scale)/a[i].energy);
        error=std::max(error,std::abs(a[i].chemical-b[i].chemical/scale)/std::max(1e-10,a[i].chemical));
    }
    return error;
}
struct Snapshot {
    std::vector<kg::Node> states;
    std::vector<kg::Clock> clocks;
    explicit Snapshot(const kg::Cuda& gpu){gpu.download(states,clocks);}
};
void tail_checks() {
    require(kg::Settings{}.tail_atol_pa==100,"Unexpected optimized default");
    auto n=fixture();n.settings.tail_atol_pa=100;
    n.initial[0]=node(100050,300);n.initial[1]=node(100000,300);
    const double tau=2*kg::cap(n.initial[0],n.initial[1],n.settings)/kg::flow(n.initial[0],n.initial[1],n.edges[0],n.settings);
    kg::Cuda full(n),parts(n);full.advance(.4*tau);
    for(int i=0;i<100;++i)parts.advance(.004*tau);
    const Snapshot a(full),b(parts);balances(n,a.states,1);balances(n,b.states,1);
    require(a.clocks[0].tails>0 && a.clocks[0].events==0,"Tail did not replace small-dp events");
    require(difference(a.states,b.states)<1e-7,"Tail time subdivision error");
    const double gap=kg::pressure(a.states[0],n.settings)-kg::pressure(a.states[1],n.settings);
    require(gap>0 && gap<50,"Tail overshot or ignored finite time");
    auto faster=n;faster.edges[0].area*=4;kg::Cuda fast(faster);fast.advance(.1*tau);
    require(difference(a.states,Snapshot(fast).states)<1e-12,"Tail aperture/time scaling");
    auto slow=n;slow.edges[0].area=1e-22;kg::Cuda blocked(slow);blocked.advance(.4*tau);
    const Snapshot stopped(blocked);
    require(difference(n.initial,stopped.states)<1e-13,"Tiny aperture falsely equilibrated");
    auto sealed=n;sealed.edges[0].area=0;kg::Cuda closed(sealed);closed.advance(10);
    require(difference(sealed.initial,Snapshot(closed).states)==0,"Tail leaked through closed barrier");
    full.reset();full.advance(10*tau);const Snapshot settled(full);balances(n,settled.states,1);
    require(std::abs(kg::pressure(settled.states[0],n.settings)-kg::pressure(settled.states[1],n.settings))<1e-5,"Tail failed equilibration");
    auto lte=n;lte.initial[0]=node(100050,900,1e-6,.01);lte.initial[1]=node(100000,300,1e-6,.003);
    kg::Cuda material(lte);material.advance(.1);const Snapshot mixed(material);balances(lte,mixed.states,1);
    require(mixed.clocks[0].tails>0,"Material tail not covered");
    const double lte_time=.2*kg::cap(lte.initial[0],lte.initial[1],lte.settings)/kg::flow(lte.initial[0],lte.initial[1],lte.edges[0],lte.settings);
    kg::Cuda material_full(lte),material_parts(lte);material_full.advance(lte_time);
    for(int k=0;k<100;++k)material_parts.advance(lte_time/100);
    require(difference(Snapshot(material_full).states,Snapshot(material_parts).states)<1e-7,"LTE tail subdivision error");
    auto strong=fixture();auto strong_tail=strong;strong_tail.settings.tail_atol_pa=100;
    kg::Cuda original(strong,64,71),optimized(strong_tail,64,71);
    original.advance(.001);optimized.advance(.001);
    require(difference(Snapshot(original).states,Snapshot(optimized).states)==0,"Tail changed large-dp events");
    auto invalid=n;invalid.settings.tail_atol_pa=-1;
    bool rejected=false;try{kg::validate(invalid);}catch(const std::exception&){rejected=true;}
    require(rejected,"Invalid tail tolerance accepted");
}
int main() {try {
    tail_checks();
    kg::Settings s;
    auto n=fixture();constexpr int replicas=64;kg::Cuda whole(n,replicas,71),split(n,replicas,71);
    whole.advance(.04);for(int i=0;i<100;++i)split.advance(.0004);
    const Snapshot full(whole),parts(split);
    const double clock_error=difference(full.states,parts.states);
    require(clock_error<1e-10,"Pair clock depends on interval subdivision");
    for(int i=0;i<replicas;++i)require(full.clocks[i].events==parts.clocks[i].events,"Event clock reset at boundary");
    balances(n,full.states,replicas);
    kg::Cuda relaxed(n,8,71);relaxed.advance(10.);const Snapshot equilibrium(relaxed);balances(n,equilibrium.states,8);
    for(int r=0;r<8;++r)require(std::abs(kg::pressure(equilibrium.states[2*r],s)-kg::pressure(equilibrium.states[2*r+1],s))<.001,
        "Finite-time equilibration did not finish");
    auto lte=n;lte.initial[0]=node(3e5,900,1e-6,.01);lte.initial[1]=node(1e5,300,1e-6,.003);
    kg::Cuda lte_whole(lte,8,19),lte_split(lte,8,19);lte_whole.advance(.04);
    for(int i=0;i<100;++i)lte_split.advance(.0004);
    const Snapshot lte_full(lte_whole),lte_parts(lte_split);
    require(difference(lte_full.states,lte_parts.states)<1e-10,"LTE pair clock subdivision");balances(lte,lte_full.states,8);
    auto faster=n;faster.edges[0].area*=4;kg::Cuda fast(faster,replicas,71);fast.advance(.01);
    require(difference(full.states,Snapshot(fast).states)<1e-10,"Area/time scaling");
    auto larger=n;for(auto& v:larger.initial){v.mass*=3;v.energy*=3;v.volume*=3;v.chemical*=3;v.oxidizer*=3;}
    kg::Cuda large(larger,replicas,71);large.advance(.12);
    require(difference(full.states,Snapshot(large).states,3)<1e-10,"Volume/time scaling");
    whole.reset();whole.advance(.04);require(difference(full.states,Snapshot(whole).states)==0,"Reset RNG");
    auto closed=n;closed.edges[0].area=0;kg::Cuda shut(closed);shut.advance(1.);
    require(difference(closed.initial,Snapshot(shut).states)==0,"Closed barrier leaked");
    auto eq=n;eq.initial[0]=eq.initial[1];kg::Cuda steady(eq);steady.advance(1.);
    require(difference(eq.initial,Snapshot(steady).states)==0,"Equilibrium drift");
    auto near=n;near.initial[0]=node(1e5+1e-8,300);near.initial[1]=node(1e5,300);
    kg::Cuda snapped(near);snapped.advance(.1);const Snapshot closure(snapped);
    require(closure.clocks[0].snaps==1,"Near-equilibrium closure missing");balances(near,closure.states,1);
    auto graph=fixture(8);auto bad=graph;bad.colors={{0,1,2,3,4,5,6}};
    bool rejected=false;try{kg::validate(bad);}catch(const std::exception&){rejected=true;}require(rejected,"Overlapping pair map accepted");
    bad=graph;bad.colors[0].pop_back();rejected=false;try{kg::validate(bad);}catch(const std::exception&){rejected=true;}require(rejected,"Missing pair accepted");
    std::vector<kg::Node> tile_reference;
    for(int block:{64,128,256}) {
        kg::Cuda gpu(graph,replicas,91,block);for(int i=0;i<40;++i)gpu.advance(.001);
        const Snapshot result(gpu);balances(graph,result.states,replicas);
        if(tile_reference.empty())tile_reference=result.states;
        else require(difference(tile_reference,result.states)==0,"CUDA tile changed trajectory");
        gpu.reset();for(int i=0;i<40;++i)gpu.advance(.001);
        require(difference(result.states,Snapshot(gpu).states)==0,"CUDA reset nondeterminism");
    }
    std::cout<<"CUDA Kawasaki: balances, entropy, clocks, scaling, coverage, tiling, reset and finite-time tail pass; clock="<<clock_error<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "../src/batched_gas.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace kg=battery::kawasaki;
namespace bg=battery::batched_gas;
using battery::GasCombustionLaw;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void checked(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
template<class T> struct Buffer {
    T* p=nullptr;std::size_t n;
    explicit Buffer(std::size_t count):n(count){if(n)checked(cudaMalloc(&p,n*sizeof(T)));}
    ~Buffer(){if(p)cudaFree(p);}
    void put(const std::vector<T>& values){require(values.size()==n,"buffer size");if(n)checked(cudaMemcpy(p,values.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get()const{std::vector<T> values(n);if(n)checked(cudaMemcpy(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return values;}
};
struct Run {std::vector<kg::Node> nodes;std::vector<battery::GasChannelState> channels;unsigned reaction_intervals=1;};
Run run(const kg::Network& net,const std::vector<GasCombustionLaw>& laws,double duration,int steps=1,int block=128,int blocks=0) {
    require(laws.size()==net.initial.size(),"one reaction law per volume");
    Buffer<kg::Node> state(net.initial.size()),scratch(net.initial.size());state.put(net.initial);
    Buffer<kg::Edge> edges(net.edges.size());edges.put(net.edges);
    Buffer<kg::Clock> clocks(net.edges.size());clocks.put(std::vector<kg::Clock>(net.edges.size()));
    Buffer<bg::Proposal> proposals(net.edges.size());Buffer<bg::Budget> budgets(net.initial.size());
    Buffer<GasCombustionLaw> chemistry(laws.size());chemistry.put(laws);
    Buffer<battery::GasChannelState> channels(net.edges.size());std::vector<battery::GasChannelState> initial(net.edges.size());
    std::vector<std::vector<int>> adjacency(net.initial.size());
    for(int j=0;j<int(net.edges.size());++j){const auto&e=net.edges[j];adjacency[e.a].push_back(j);adjacency[e.b].push_back(j);
        initial[j].open=true;initial[j].effective_area_m2=e.area;}
    channels.put(initial);std::vector<int> flat,offset{0};
    for(const auto& row:adjacency){flat.insert(flat.end(),row.begin(),row.end());offset.push_back(flat.size());}
    Buffer<int> ids(flat.size()),offsets(offset.size()),error(1);ids.put(flat);offsets.put(offset);error.put({0});
    Buffer<bg::Control> control(1);
    battery::GasModel model;model.cuda_network_transport=true;model.network_pressure_fraction=1.;model.max_fraction_per_step=.1;
    double max_rate=0;for(const auto& law:laws)max_rate=std::max(max_rate,law.rate_per_s);
    for(int j=0;j<steps;++j) {
        checked(bg::launch(state.n,edges.n,offsets.p,ids.p,edges.p,state.p,scratch.p,proposals.p,budgets.p,clocks.p,
            channels.p,net.settings,model,duration/steps,error.p,control.p,block,blocks,{chemistry.p,max_rate}));
        checked(cudaGetLastError());require(error.get()[0]==0,"reactive transport kernel failed");
    }
    return {state.get(),channels.get(),control.get()[0].reaction_intervals};
}
void balances(const kg::Network& net,const std::vector<GasCombustionLaw>& laws,const Run& result) {
    double mass0=0,mass1=0,total0=0,total1=0;
    for(std::size_t i=0;i<net.initial.size();++i) {
        const auto&a=net.initial[i];const auto&b=result.nodes[i];
        require(b.mass>0 && b.energy>0 && b.chemical>=0 && b.oxidizer>=0,"negative reactive inventory");
        double dm=0,de=0,df=0,do2=0;
        for(std::size_t j=0;j<net.edges.size();++j){const auto&e=net.edges[j];const auto&q=result.channels[j];
            const double sign=e.a==int(i)?-1.:e.b==int(i)?1.:0.;
            dm+=sign*q.transferred_kg;
            de+=sign*(q.transferred_energy_j+q.transferred_kg*net.settings.cv*298.15);
            df+=sign*q.transferred_chemical_j;do2+=sign*q.transferred_oxidizer_kg;}
        const double scale=std::max({1.,a.energy,b.energy,a.chemical});
        require(std::abs(b.mass-a.mass-dm)<1e-11*std::max(a.mass,b.mass),"reactive mass ledger");
        require(std::abs(b.energy-a.energy-de-b.reaction_released_j)<1e-11*scale,"sensible energy ledger");
        require(std::abs(b.chemical-a.chemical-df+b.reaction_released_j)<1e-11*scale,"chemical energy ledger");
        require(std::abs(b.oxidizer-a.oxidizer-do2+b.reaction_oxidizer_kg)<1e-11*std::max({1e-9,a.oxidizer,b.oxidizer}),"oxygen ledger");
        require(std::abs(b.reaction_released_j-b.reaction_oxidizer_kg*laws[i].oxidizer_heat_j_per_kg)<1e-11*scale,"reaction stoichiometry");
        mass0+=a.mass;mass1+=b.mass;total0+=a.energy+a.chemical;total1+=b.energy+b.chemical;
    }
    require(std::abs(mass1-mass0)<1e-11*mass0,"global reactive mass");
    require(std::abs(total1-total0)<1e-11*total0,"global thermal plus chemical energy");
}
kg::Node node(double pressure,double temperature,double volume=1e-4) {
    const double mass=pressure*volume/(287*temperature);
    return {mass,1000*mass*temperature,volume,0,0,0};
}
void closed_reactor() {
    kg::Network net;net.initial={node(1e5,500)};net.initial[0].chemical=20;net.initial[0].oxidizer=100/13.1e6;
    const std::vector<GasCombustionLaw> laws{{3.,450.,13.1e6}};
    const auto one=run(net,laws,.25),split=run(net,laws,.25,17);balances(net,laws,one);balances(net,laws,split);
    const double expected=20*(-std::expm1(-3*.25));
    require(std::abs(one.nodes[0].reaction_released_j-expected)<1e-12,"closed reactor is not exponential");
    require(std::abs(one.nodes[0].energy-split.nodes[0].energy)<1e-11,"closed chemistry depends on outer step");
    auto starved=net;starved.initial[0].oxidizer=8/13.1e6;
    const auto limited=run(starved,laws,10);balances(starved,laws,limited);
    require(std::abs(limited.nodes[0].reaction_released_j-8)<1e-11,"oxygen limit ignored");
    auto cold=net;cold.initial[0].energy=1000*cold.initial[0].mass*300;
    const auto unlit=run(cold,laws,.1);balances(cold,laws,unlit);
    require(unlit.nodes[0].reaction_released_j==0,"cold reactor burned");
    require(unlit.reaction_intervals==1,"cold fuel stock forces kinetic subcycling");
    auto oxygen_free=net;oxygen_free.initial[0].oxidizer=0;
    const auto inert=run(oxygen_free,laws,1);balances(oxygen_free,laws,inert);
    require(inert.reaction_intervals==1 && inert.nodes[0].reaction_released_j==0,"oxygen-free isolated fuel forces kinetic subcycling");
}
kg::Network reacting_chain() {
    kg::Network net;net.initial={node(3e5,500),node(1e5,500),node(1e5,500)};
    net.initial[0].chemical=150;
    net.initial[1].oxidizer=net.initial[1].mass*.23;
    net.initial[2].oxidizer=net.initial[2].mass*.23;
    net.edges={{0,1,2e-7,.7},{1,2,2e-7,.7}};return net;
}
double temperature_error(const Run&a,const Run&b,const kg::Settings&s) {
    double error=0;for(std::size_t i=0;i<a.nodes.size();++i)error=std::max(error,std::abs(kg::temperature(a.nodes[i],s)-kg::temperature(b.nodes[i],s)));return error;
}
void arrival_and_convergence() {
    const auto net=reacting_chain();const std::vector<GasCombustionLaw> laws(3,{40.,450.,13.1e6});
    const auto arrival=run(net,laws,.001);balances(net,laws,arrival);
    require(arrival.channels[0].transferred_chemical_j>0,"no incoming fuel");
    require(arrival.nodes[1].reaction_released_j>0,"incoming fuel waits until next outer step");
    require(arrival.nodes[0].reaction_released_j==0,"fuel without oxygen burned");
    auto cold_receiver=net;cold_receiver.initial[1]=node(1e5,300);
    cold_receiver.initial[1].oxidizer=cold_receiver.initial[1].mass*.23;
    auto ignition_laws=laws;ignition_laws[1].ignition_temperature_k=300.1;
    const auto ignited=run(cold_receiver,ignition_laws,.001);balances(cold_receiver,ignition_laws,ignited);
    require(ignited.nodes[1].reaction_released_j>0,"hot arrivals cannot ignite current interval");
    const auto coarse=run(net,laws,.02,10),fine=run(net,laws,.02,20),reference=run(net,laws,.02,320);
    balances(net,laws,coarse);balances(net,laws,fine);balances(net,laws,reference);
    const double e0=temperature_error(coarse,reference,net.settings),e1=temperature_error(fine,reference,net.settings);
    std::cout<<"reactive transport refinement: "<<e0<<" -> "<<e1<<" K\n";
    require(e1<.8*e0,"reactive transport does not converge with time resolution");
    const auto subcycled=run(net,laws,.02);
    balances(net,laws,subcycled);
    require(subcycled.nodes[2].reaction_released_j>0,"reaction subintervals do not carry fuel downstream");
    const auto other_tile=run(net,laws,.02,1,64,2);
    balances(net,laws,other_tile);
    require(temperature_error(subcycled,other_tile,net.settings)==0,"CUDA tile changes reactive trajectory");
    auto reversed=net;std::reverse(reversed.edges.begin(),reversed.edges.end());
    const auto permuted=run(reversed,laws,.02);balances(reversed,laws,permuted);
    require(temperature_error(subcycled,permuted,net.settings)<1e-10,"edge order changes reactive trajectory");
    auto faster=net;for(auto& e:faster.edges)e.area*=4;
    auto faster_laws=laws;for(auto& law:faster_laws)law.rate_per_s*=4;
    const auto time_scaled=run(faster,faster_laws,.02/4);balances(faster,faster_laws,time_scaled);
    require(temperature_error(subcycled,time_scaled,net.settings)<1e-10,"reaction/flow physical time scaling failed");
}
int main(){try {closed_reactor();arrival_and_convergence();std::cout<<"CUDA continuous combustion: balances, arrivals, refinement and tile invariance pass\n";}
catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

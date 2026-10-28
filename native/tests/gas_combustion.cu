#include "../src/batched_gas.cuh"
#include "battery_native/simulation.hpp"
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
Run run(const kg::Network& net,const std::vector<GasCombustionLaw>& laws,double duration,int steps=1,int block=128,int blocks=0,
    double diffusivity=0.,double vapor_heat=0.,const battery::ConnectivityLaw* mixing=nullptr) {
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
    model.mixing_diffusivity_m2_s=diffusivity;
    auto settings=net.settings;settings.vapor_fuel_heat_j_kg=vapor_heat;
    if(mixing)settings.mixing_connectivity=*mixing;
    double max_rate=0;for(const auto& law:laws)max_rate=std::max(max_rate,law.rate_per_s);
    for(int j=0;j<steps;++j) {
        checked(bg::launch(state.n,edges.n,offsets.p,ids.p,edges.p,state.p,scratch.p,proposals.p,budgets.p,clocks.p,
            channels.p,settings,model,duration/steps,error.p,control.p,block,blocks,{chemistry.p,max_rate>0.}));
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
void arrhenius_reactors() {
    const GasCombustionLaw law{20.,0.,13.1e6,600.,80000.};
    require(battery::combustion_rate(law,600.)==20.,"R3 reference rate");
    require(battery::combustion_rate(law,300.)<3e-6 && battery::combustion_rate(law,900.)>4000.,"R3 has no thermal activation");
    require(std::abs(battery::combustion_rate(law,450.)/battery::combustion_rate(law,350.)-
        std::exp(80000./8.31446261815324*(1./350.-1./450.)))<1e-10,"R3 Arrhenius ratio");
    // Negligible heat stock holds T without a synthetic thermostat. Fuel energy
    // sets the budget; it must not be interpreted as already-present temperature.
    for(double t:{300.,600.}) {
        kg::Network fixed;fixed.initial={node(1e5,t,.01)};
        fixed.initial[0].chemical=1e-8;fixed.initial[0].oxidizer=.001;
        const auto result=run(fixed,{law},.1);balances(fixed,{law},result);
        const double expected=1e-8*(-std::expm1(-battery::combustion_rate(law,t)*.1));
        require(std::abs(result.nodes[0].reaction_released_j-expected)<1e-15,"Frozen-temperature R3 is not analytic");
    }
    kg::Network reactor;reactor.initial={node(1e5,500.,.001)};
    reactor.initial[0].chemical=50.;reactor.initial[0].oxidizer=100./13.1e6;
    const auto coarse=run(reactor,{law},.25),fine=run(reactor,{law},.25,64),reference=run(reactor,{law},.25,1024);
    balances(reactor,{law},coarse);balances(reactor,{law},fine);balances(reactor,{law},reference);
    const double e0=temperature_error(coarse,reference,reactor.settings),e1=temperature_error(fine,reference,reactor.settings);
    std::cout<<"Arrhenius self-heating refinement: "<<e0<<" -> "<<e1<<" K\n";
    require(e1<.8*e0 && e1<.5,"R3 nonlinear self-heating does not converge");
    require(coarse.reaction_intervals>1,"Self-heating does not refine the coupled interval");
    auto inert=reactor;inert.initial[0].oxidizer=0.;const auto unburned=run(inert,{law},1.);
    balances(inert,{law},unburned);require(unburned.nodes[0].reaction_released_j==0.,"Arrhenius burns without oxygen");
    auto trace=inert;trace.initial[0].oxidizer=1e-30;
    const auto tiny=run(trace,{law},1.);balances(trace,{law},tiny);
    require(tiny.reaction_intervals==1 && tiny.nodes[0].reaction_released_j>0.,"Roundoff oxygen throttles transport or is discarded");
    const auto tile=run(reactor,{law},.25,1,64,2);
    require(temperature_error(coarse,tile,reactor.settings)==0.,"Arrhenius adaptive interval depends on CUDA tile");
    auto incoming=reacting_chain();incoming.initial[0]=node(3e5,800.);incoming.initial[0].chemical=150.;
    incoming.initial[1]=node(1e5,300.);incoming.initial[1].oxidizer=incoming.initial[1].mass*.23;
    const std::vector<GasCombustionLaw> laws(3,law);const auto arrival=run(incoming,laws,.001);
    balances(incoming,laws,arrival);
    require(arrival.nodes[1].reaction_released_j>0. && arrival.nodes[0].reaction_released_j==0.,"Hot arrivals/oxygen gating lost with Arrhenius");
}
void full_solver_arrhenius() {
    for(bool network:{false,true})for(double temperature:{300.,600.}) {
        battery::Scenario scene;scene.settings={.01,.01,.01,.4};
        battery::CellConfig material;material.id="inert";material.is_battery=false;
        material.dimensions_m={.01,.01,.01};material.heat_capacity_j_per_k=1.;scene.cells={material};
        auto& model=scene.gas_model;model.split_transport=network;model.cuda_network_transport=network;
        model.cuda_batched_transport=network;model.cuda_kawasaki_transport=network;
        battery::GasCompartment gas;gas.id="reactor";gas.initial_mass_kg=.001;gas.volume_m3=.001;
        gas.initial_temperature_k=temperature;gas.initial_chemical_j=1e-6;gas.initial_oxidizer_mass_fraction=.232;
        gas.reaction_rate_per_s=20.;gas.reaction_reference_k=600.;gas.reaction_activation_j_per_mol=80000.;
        scene.gas_compartments={gas};battery::Simulation sim(scene);sim.run_to_end();
        const auto result=sim.process_diagnostics();
        const double rate=battery::combustion_rate(battery::gas_combustion_law(gas),temperature);
        const double expected=gas.initial_chemical_j*(-std::expm1(-rate*.01));
        require(std::abs(result.gas[0].released_energy_j-expected)<1e-12,"Full solver ignores per-compartment R3 Arrhenius");
        require(std::abs(result.thermal_residual_j)<1e-9 && std::abs(result.chemical_residual_j)<1e-9 &&
            std::abs(result.oxidizer_residual_kg)<1e-12,"Full solver R3 balance");
        auto invalid=scene;invalid.gas_compartments[0].reaction_activation_j_per_mol=-1.;
        bool rejected=false;try{battery::build_thermal_mesh(invalid);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"Negative R3 activation accepted");
    }
}
void finite_arrival_limits() {
    const GasCombustionLaw law{20.,0.,13.1e6,600.,80000.};
    kg::Network net;net.initial={node(1e5+1e-3,900.),node(1e5,300.)};
    net.initial[0].chemical=50.;net.initial[1].chemical=50.;net.initial[1].oxidizer=net.initial[1].mass*.23;
    net.edges={{0,1,2e-7,.7}};
    const auto trace=run(net,{law,law},.01);balances(net,{law,law},trace);
    require(trace.reaction_intervals<=2,"Negligible hot arrival throttles entire graph");
    require(trace.nodes[1].reaction_released_j>0.,"Finite-arrival limiter suppresses R3");
    kg::Network oxygen;oxygen.initial={node(1e5,900.)};oxygen.initial[0].chemical=50.;
    oxygen.initial[0].oxidizer=1e-5/13.1e6;
    const auto tiny=run(oxygen,{law},.01);balances(oxygen,{law},tiny);
    require(tiny.reaction_intervals==1,"Thermally negligible oxygen stock imposes graph substeps");
    require(tiny.nodes[0].reaction_released_j>0.,"Small oxygen inventory was discarded");
    // Absolute inlet energy / old receiver capacity is NOT a useful bound for
    // an almost empty volume. Its incoming mass brings its own heat capacity.
    auto empty=node(1.,300.);empty.chemical=1.;empty.oxidizer=empty.mass*.23;
    battery::ReactionArrivals fill;fill.energy=100.;fill.chemical=50.;fill.oxygen_heat=50.;
    fill.max_temperature=500.;fill.max_specific_fuel=1e6;fill.max_specific_oxygen_heat=3e6;
    fill.fuel_rate=fill.oxygen_heat_rate=1e4;
    const double filling_step=battery::arrival_reaction_step(empty,law,net.settings,fill,.01,.1,1.);
    require(filling_step>1e-5,"Nearly empty receiver forces artificial ultra-hot kinetics");
    // A very fast oxygen-starved flame burns oxygen AS IT ARRIVES. The new
    // stock is supply limited; its microscopic lifetime is not a network dt.
    auto fed=node(1e5,900.);fed.chemical=50.;
    battery::ReactionArrivals supply;supply.oxygen_heat=.001;supply.oxygen_heat_rate=.1;
    supply.fuel_rate=1e6; // enormous EXCESS fuel must not set the oxygen time scale
    supply.max_temperature=300.;supply.max_specific_oxygen_heat=3e6;
    const double fed_step=battery::arrival_reaction_step(fed,law,net.settings,supply,.01,.1,1.);
    require(fed_step>=.004,"Fed flame throttled by lifetime of not-yet-arrived oxygen");
    auto fuel_starved=fed;fuel_starved.chemical=0.;fuel_starved.oxidizer=50./law.oxidizer_heat_j_per_kg;
    auto fuel_supply=supply;fuel_supply.chemical=.001;fuel_supply.oxygen_heat=0.;
    fuel_supply.fuel_rate=.1;fuel_supply.oxygen_heat_rate=1e6;
    const double fuel_step=battery::arrival_reaction_step(fuel_starved,law,net.settings,fuel_supply,.01,.1,1.);
    require(fuel_step>=.004,"Excess oxygen throttles a fuel-limited flame");
    // A finite resident stock between half and one thermal budget must use
    // its heat-release bound, NOT a second global chemical-lifetime clock.
    auto finite=fed;
    const double cmin=.5*net.settings.cv*finite.mass;
    finite.oxidizer=.75*cmin/law.oxidizer_heat_j_per_kg;
    auto finite_supply=supply;finite_supply.oxygen_heat=2*cmin;
    const double finite_step=battery::arrival_reaction_step(finite,law,net.settings,finite_supply,.01,.02,1.);
    const double tmax=kg::temperature(finite,net.settings)+
        battery::reaction_rise_bound(finite,law,net.settings,.01,1.)+1.;
    const double heat_bound=.5*cmin/(battery::combustion_rate(law,tmax)*.75*cmin);
    require(finite_step>=.99*heat_bound,"Local chemical clock duplicates global thermal budget");
    // The arrival envelope must bound REAL conservative packets, including
    // compressed hot arrivals and an immobile matrix sharing their temperature.
    for(double td:{300.,600.,1200.})for(double tr:{250.,450.,900.})for(double dp:{1e-5,1e3,2e5})
    for(double matrix:{0.,.1}) {
        auto a=node(1e5+dp,td),b=node(1e5,tr);b.solid_capacity=matrix;
        b.energy=(1000*b.mass+matrix)*tr;a.chemical=50.;a.oxidizer=a.mass*.1;
        battery::ReactionArrivals bound;GasCombustionLaw inert;
        battery::bound_reaction_arrival(bound,a,b,inert,law,net.edges[0],net.settings,.01,1.,.02);
        const double pa=kg::pressure(a,net.settings),pb=kg::pressure(b,net.settings);
        const double roundoff=net.settings.equilibrium_atol_pa+net.settings.equilibrium_rtol*pa;
        const double dm=pa-pb<=roundoff?0.:std::min({.49*a.mass,.9*kg::cap(a,b,net.settings),.9*kg::flow(a,b,net.edges[0],net.settings)*.01});
        if(bound.energy+1e-12<kg::packet_energy(a,dm,net.settings))
            std::cerr<<"arrival bound: Td="<<td<<", Tr="<<tr<<", dp="<<dp<<", matrix="<<matrix<<", dm="<<dm<<", bound="<<bound.energy<<", packet="<<kg::packet_energy(a,dm,net.settings)<<'\n';
        require(bound.energy+1e-12>=kg::packet_energy(a,dm,net.settings),"Arrival enthalpy envelope is not conservative");
        require(bound.chemical+1e-12>=dm/a.mass*a.chemical,"Arrival chemical envelope is not conservative");
        require(bound.oxygen_heat+1e-12>=dm/a.mass*a.oxidizer*law.oxidizer_heat_j_per_kg,"Arrival oxygen envelope is not conservative");
    }
}
void pressure_independent_mixing() {
    kg::Network n;n.initial={node(1e5,500.),node(1e5,500.)};n.edges={{0,1,1e-4,.7,.01}};
    n.initial[0].chemical=10.;n.initial[0].vapor=n.initial[0].mass*.2;n.initial[0].latent=5.;
    n.initial[1].oxidizer=n.initial[1].mass*.23;
    const std::vector<GasCombustionLaw> inert(2);
    const auto absent=run(n,inert,.1);require(absent.nodes[1].chemical==0.,"Control unexpectedly mixes at equal pressure");
    const auto a=run(n,inert,.1,1,128,0,.1),fine=run(n,inert,.1,4,64,0,.1);
    balances(n,inert,a);balances(n,inert,fine);
    require(a.nodes[1].chemical>0. && a.nodes[0].oxidizer>0.,"Equal-pressure gas cannot exchange fuel/oxygen");
    const double request=battery::mixing_request(n.initial[0],n.initial[1],n.edges[0],n.settings,.1,0.,0.,.1);
    require(std::abs(a.nodes[1].chemical-10.*request/n.initial[0].mass)<1e-12,"Analytical pair mixing is wrong");
    require(std::abs(a.nodes[1].chemical-fine.nodes[1].chemical)<1e-12,"Frozen pair mixing depends on timestep");
    require(a.nodes[0].mass==n.initial[0].mass && a.channels[0].transferred_kg==0.,"Counterexchange creates net mass flow");
    require(a.channels[0].mixed_mass_kg>0. && a.nodes[1].vapor>0. && a.nodes[1].latent>0.,"Mixing loses phase payload");
    require(std::abs(a.nodes[0].vapor+a.nodes[1].vapor-n.initial[0].vapor)<1e-15,"Vapor mixing mass loss");
    require(std::abs(a.nodes[0].latent+a.nodes[1].latent-5.)<1e-12,"Latent mixing energy loss");
    auto closed=n;closed.edges[0].area=0.;const auto shut=run(closed,inert,.1,1,128,0,.1);
    require(shut.nodes[1].chemical==0. && shut.channels[0].mixed_mass_kg==0.,"Mixing bypasses closed barrier");
    const auto tile=run(n,inert,.1,1,256,0,.1);
    require(a.nodes[1].chemical==tile.nodes[1].chemical,"Mixing depends on CUDA tiling");
    const std::vector<GasCombustionLaw> burning(2,{10.,0.,13.1e6});
    const auto flame=run(n,burning,.01,2,128,0,.1);balances(n,burning,flame);
    require(flame.nodes[0].reaction_released_j>0. && flame.nodes[1].reaction_released_j>0.,"Arriving mixed reactants wait until next thermal step");
    std::cout<<"equal-pressure mixing: analytical time equivalence, closed barrier, phase payload, CUDA tile invariance pass\n";
}
void unburned_oxygen_is_advected() {
    kg::Network net;net.initial={node(3e5,500.),node(1e5,500.),node(.8e5,500.)};
    net.edges={{0,1,2e-7,.7},{1,2,2e-7,.7}};
    net.initial[0].oxidizer=.232*net.initial[0].mass;
    for(auto& n:net.initial)n.chemical=10.;
    const double oxygen0=net.initial[0].oxidizer;
    const std::vector<GasCombustionLaw> slow(3,{.05,0.,13.1e6,500.,60000.});
    const auto result=run(net,slow,.02,10);balances(net,slow,result);
    double consumed=0.;for(const auto& n:result.nodes)consumed+=n.reaction_oxidizer_kg;
    require(consumed>0. && consumed<.01*oxygen0,"Slow R3 consumed oxygen instantaneously");
    require(result.nodes[2].oxidizer>0. && result.channels[1].transferred_oxidizer_kg>0.,
        "Unburned oxygen cannot travel through multiple volumes");
    require(result.channels[0].transferred_oxidizer_kg>consumed,
        "Oxygen was burned instead of being carried by the mixture");
    std::cout<<"oxygen: finite-rate consumption and downstream advection of unburned mixture pass\n";
}
void temperature_dependent_mixing() {
    kg::Network n;n.initial={node(1e5,900.),node(1e5,300.)};n.edges={{0,1,1e-4,.7,.01}};
    n.initial[0].chemical=10.;n.initial[1].oxidizer=.23*n.initial[1].mass;
    const std::vector<GasCombustionLaw> inert(2);
    auto law=battery::reference_directed_connectivity(1.);
    law.background_conductance_w_per_k=.01;law.active_conductance_w_per_k=29.99;
    law.geometry_reference_logit=std::log(.99/29.);
    law.endpoint_a.temperature_log_response.count=law.endpoint_b.temperature_log_response.count=4;
    for(int i=0;i<4;++i) {
        const double tk[4]={298.15,350.,450.,900.},response[4]={0.,0.,0.,2.08};
        law.endpoint_a.temperature_log_response.temperature_k[i]=law.endpoint_b.temperature_log_response.temperature_k[i]=tk[i];
        law.endpoint_a.temperature_log_response.value[i]=law.endpoint_b.temperature_log_response.value[i]=response[i];
    }
    const auto base=run(n,inert,.001,1,128,0,.001),hot=run(n,inert,.001,1,128,0,.001,0.,&law);
    balances(n,inert,base);balances(n,inert,hot);
    require(hot.channels[0].transferred_kg==0.,"Thermal mixing creates pressure mass flow at initially equal P");
    require(hot.channels[0].mixed_mass_kg>5.*base.channels[0].mixed_mass_kg,"Hot S does not accelerate gas exchange at equal pressure");
    require(kg::temperature(hot.nodes[1],n.settings)>kg::temperature(base.nodes[1],n.settings),"Temperature branch does not heat cold gas");
    require(hot.nodes[0].oxidizer>base.nodes[0].oxidizer && hot.nodes[1].chemical>base.nodes[1].chemical,"Thermal stirring does not carry oxygen/fuel");
    auto settings=n.settings;settings.mixing_connectivity=law;
    const double dm=battery::mixing_request(n.initial[0],n.initial[1],n.edges[0],settings,.001,0.,0.,.001);
    require(std::abs(hot.channels[0].mixed_mass_kg-dm)<1e-15,"State-dependent pair exchange disagrees with exponential integral");
    const auto tiled=run(n,inert,.001,1,64,2,.001,0.,&law);
    require(temperature_error(hot,tiled,n.settings)==0.,"Thermal S depends on CUDA tiling");
    auto shut=n;shut.edges[0].area=0.;const auto blocked=run(shut,inert,.001,1,128,0,.001,0.,&law);
    require(blocked.channels[0].mixed_mass_kg==0. && blocked.nodes[1].chemical==0.,"Hot mixing bypasses a closed barrier");
    auto opposite=n;opposite.edges[0].a=1;opposite.edges[0].b=0;
    const auto reversed=run(opposite,inert,.001,1,128,0,.001,0.,&law);balances(opposite,inert,reversed);
    require(temperature_error(hot,reversed,n.settings)==0.,"Edge orientation changes physical hot-to-cold exchange");
    const auto coarse=run(n,inert,.001,8,128,0,.001,0.,&law);
    const auto fine=run(n,inert,.001,16,128,0,.001,0.,&law);
    const auto reference=run(n,inert,.001,256,128,0,.001,0.,&law);
    balances(n,inert,coarse);balances(n,inert,fine);balances(n,inert,reference);
    const double e0=temperature_error(coarse,reference,n.settings),e1=temperature_error(fine,reference,n.settings);
    require(e1<e0,"Thermal mixing does not converge with timestep refinement");
    // The source law is NOT averaged with its reverse. Changing the cold
    // endpoint response cannot alter a fixed hot-a snapshot's selected S.
    auto directed=settings;directed.mixing_connectivity.endpoint_b.linear={100.,100.,100.};
    directed.mixing_connectivity.endpoint_b.temperature_log_response.count=0;
    const double g=battery::mixing_conductance(n.initial[0],n.initial[1],n.edges[0],settings,.001,0.,0.);
    require(g==battery::mixing_conductance(n.initial[0],n.initial[1],n.edges[0],directed,.001,0.,0.),"Sij and Sji were averaged");
    require(battery::mixing_conductance(n.initial[0],n.initial[1],n.edges[0],settings,0.,.25,1e-5)==
        battery::mixing_conductance(n.initial[0],n.initial[1],n.edges[0],n.settings,0.,.25,1e-5),
        "S multiplies the separate pressure entrainment coefficient");
    auto anisotropic=settings;anisotropic.mixing_connectivity.endpoint_a.temperature_log_response.count=0;
    anisotropic.mixing_connectivity.endpoint_b.temperature_log_response.count=0;
    anisotropic.mixing_connectivity.endpoint_a.linear=anisotropic.mixing_connectivity.endpoint_b.linear={0.,1.,2.};
    auto transverse=n.edges[0];transverse.geometry_axis=2;
    require(battery::mixing_conductance(n.initial[0],n.initial[1],transverse,anisotropic,.001,0.,0.)>
        battery::mixing_conductance(n.initial[0],n.initial[1],n.edges[0],anisotropic,.001,0.,0.),"Gas tensor ignores edge direction");
    const std::vector<GasCombustionLaw> finite(2,{.05,0.,13.1e6,600.,60000.});
    const auto flame=run(n,finite,.001,4,128,0,.001,0.,&law);balances(n,finite,flame);
    double consumed=0.;for(const auto& v:flame.nodes)consumed+=v.reaction_oxidizer_kg;
    require(consumed>0. && consumed<.01*n.initial[1].oxidizer,"Thermally mixed oxygen burns instantaneously");
    auto shifted=n;shifted.initial[0]=node(2e5,900.);
    const auto pressure_base=run(shifted,inert,.001,1,128,0,.001);
    const auto pressure_hot=run(shifted,inert,.001,1,128,0,.001,0.,&law);
    require(pressure_base.channels[0].transferred_kg==pressure_hot.channels[0].transferred_kg,"Temperature mixing changes the mechanical orifice law");
    balances(shifted,inert,pressure_hot);
    std::cout<<"gas S(T): equal-pressure thermal exchange, O2 transport, closed barrier, directed source, timestep and CUDA invariance pass\n";
}
void porous_mixture_is_not_a_free_eddy() {
    auto a=node(3e5,900.),b=node(3e5,500.);kg::Edge edge{0,1,1e-4,.7,.02};kg::Settings settings;
    settings.mixing_connectivity.background_conductance_w_per_k=.01;
    settings.mixing_connectivity.active_conductance_w_per_k=999.99;
    settings.mixing_connectivity.geometry_reference_logit=std::log(.99/999.);
    settings.mixing_connectivity.endpoint_a.linear=settings.mixing_connectivity.endpoint_b.linear={10.,10.,10.};
    const double free=battery::mixing_conductance(a,b,edge,settings,2e-5,.25,1e-7);
    edge.molecular_mixing=true;
    const double porous=battery::mixing_conductance(a,b,edge,settings,2e-5,.25,1e-7);
    const double reference=battery::reference_mixing_rate(a,b,edge,2e-5,.25,1e-7);
    std::cout<<"gas media rates: free="<<free<<" porous="<<porous<<" molecular="<<reference<<'\n';
    require(std::abs(porous-reference)<1e-15,"Free-gas calibration changes porous-cell heat transport");
    require(free>100*porous,"Free-gas and porous geometry responses were conflated");
    std::cout<<"gas media: molecular porous mixture and free exterior eddy have separate S profiles\n";
}
int main(){try {porous_mixture_is_not_a_free_eddy();temperature_dependent_mixing();unburned_oxygen_is_advected();closed_reactor();arrival_and_convergence();arrhenius_reactors();full_solver_arrhenius();finite_arrival_limits();pressure_independent_mixing();std::cout<<"CUDA continuous combustion: Arrhenius, balances, arrivals, refinement and tile invariance pass\n";}
catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

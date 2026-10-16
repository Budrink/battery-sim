#include "battery_native/gas_experiment.hpp"
#include "battery_native/local_gas_transport.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
namespace gx=battery::gas_experiment;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
Scenario fixture(int count=12) {
    Scenario s;s.gas_model.split_transport=true;s.gas_model.reference_temperature_k=300.;
    CellConfig c;c.id="inert";c.is_battery=false;c.dimensions_m={.1,.1,.1};c.heat_capacity_j_per_k=10;
    c.thermal_divisions={1,1,1};c.runaway_energy_j=0;s.cells={c};
    for(int i=0;i<count;++i) {
        GasCompartment g;g.id="v"+std::to_string(i);g.volume_m3=1e-6;g.initial_temperature_k=300.+i;
        g.initial_mass_kg=(i==0?110000.:100000.)*g.volume_m3/(287.*g.initial_temperature_k);
        g.initial_chemical_j=.1*(i+1);g.initial_oxidizer_mass_fraction=.1;s.gas_compartments.push_back(g);
        if(i) {GasChannel e;e.id="e"+std::to_string(i);e.from=i-1;e.to=i;e.passage=true;e.area_m2=1e-6;s.gas_channels.push_back(e);}
    }
    return s;
}
void balance(const gx::Network& n,const std::vector<GasState>& states) {
    double mass=0,energy=0,chemical=0,oxidizer=0;
    for(std::size_t i=0;i<states.size();++i) {
        mass+=states[i].mass_kg-n.initial[i].mass_kg;energy+=states[i].energy_j-n.initial[i].energy_j;
        chemical+=states[i].chemical_j-n.initial[i].chemical_j;oxidizer+=states[i].oxidizer_kg-n.initial[i].oxidizer_kg;
        require(states[i].mass_kg>0 && states[i].temperature_k>0,"Negative inventory/temperature");
    }
    require(std::abs(mass)<1e-17 && std::abs(energy)<1e-11 && std::abs(chemical)<1e-11 && std::abs(oxidizer)<1e-17,"Gas ledgers");
}
double difference(const std::vector<GasState>& a,const std::vector<GasState>& b) {
    double d=0;for(std::size_t i=0;i<a.size();++i)d=std::max(d,std::abs(a[i].pressure_pa-b[i].pressure_pa));return d;
}
int main(int argc,char**) {try {
    const bool gpu=argc>1;auto scene=fixture();const auto n=gx::make_network(scene,6);
    require(n.patches.size()>1 && n.colors.size()>1,"Test must cross overlapping groups");
    // The production adapter must reproduce the benchmark's mass operator
    // when no LTE material or chemical reaction is present.
    {
        auto full=scene;full.gas_model.cpu_local_transport=true;full.gas_model.local_exchange_interval_s=1e-7;
        const auto topology=build_thermal_mesh(full);const auto mass=build_mass_mesh(full,topology);
        const auto bench=gx::make_network(full,9);gx::Cpu expected(bench);
        LocalGasTransport production(full,topology,mass);auto gas=bench.initial;
        std::vector<ThermalState> solid(topology.volumes.size());
        std::vector<GasChannelState> channels(full.gas_channels.size());
        for(std::size_t j=0;j<channels.size();++j){channels[j].open=true;channels[j].effective_area_m2=full.gas_channels[j].area_m2;}
        for(int step=0;step<20;++step){expected.local_window(1e-7);production.advance(gas,solid,channels,1e-7);}
        balance(bench,gas);
        require(difference(gas,expected.states())<1e-5,"Production/benchmark multi-group mismatch");
        for(std::size_t j=0;j<channels.size();++j)
            require(std::abs(channels[j].transferred_energy_j-expected.transfers()[j].energy)<1e-9,"Production packet accounting");
    }
    auto bad=n;bad.patches[0].degree[0]++;
    bool rejected=false;try{gx::validate_partition(bad);}catch(const std::exception&){rejected=true;}
    require(rejected,"Bad patch degree accepted");
    bad=n;bad.colors.push_back(bad.colors[0]);rejected=false;
    try{gx::validate_partition(bad);}catch(const std::exception&){rejected=true;}
    require(rejected,"Duplicate patch schedule accepted");
    auto unsupported=scene;unsupported.gas_compartments[0].reaction_rate_per_s=1;rejected=false;
    try{gx::make_network(unsupported);}catch(const std::exception&){rejected=true;}
    require(rejected,"Gas experiment silently accepts chemistry");
    gx::Cpu reference(n);const double duration=2e-5;
    for(double t=0;t<duration;)t+=reference.global_step(std::min(1e-9,duration-t));
    balance(n,reference.states());double errors[2];
    for(int level=0;level<2;++level) {
        gx::Cpu local(n);const double dt=level?1e-7:1e-6;
        for(double t=0;t<duration;) {const double h=std::min(dt,duration-t);local.local_window(h);t+=h;}
        balance(n,local.states());errors[level]=difference(local.states(),reference.states());
    }
    require(errors[1]<errors[0] && errors[1]<2.,"Patch interval refinement failed");
    std::cout<<"patch pressure errors="<<errors[0]<<" / "<<errors[1]<<" Pa\n";
    gx::Cpu one(n);double dt=one.global_step(1e-5);const auto saved=one.states();one.reset();
    require(dt==one.global_step(1e-5) && difference(saved,one.states())==0,"CPU reset nondeterminism");
    // Single connected patch equals the same gas operator with two half windows.
    const auto small=gx::make_network(fixture(3),9);gx::Cpu global(small),local(small);
    for(int half=0;half<2;++half)for(double t=0;t<1e-5;)t+=global.global_step(1e-5-t);
    local.local_window(2e-5);require(difference(global.states(),local.states())<1e-8,"Single patch differs from global operator");
    if(gpu) {
        for(bool patches:{false,true}) {
            gx::Cpu cpu(n);gx::Cuda cuda(n);
            for(int i=0;i<20;++i) {
                if(patches){cpu.local_window(1e-7);cuda.local_window(1e-7);}
                else {const double a=cpu.global_step(1e-8),b=cuda.global_step(1e-8);require(std::abs(a-b)<1e-15,"CPU/CUDA dt");}
            }
            std::vector<GasState> state;std::vector<gx::Packet> packets;cuda.download(state,packets);balance(n,state);
            require(difference(state,cpu.states())<1e-5,"CPU/CUDA state mismatch");
            for(std::size_t j=0;j<packets.size();++j)require(std::abs(packets[j].mass-cpu.transfers()[j].mass)<1e-15,"CPU/CUDA transfer mismatch");
            const auto before=state;cuda.reset();
            for(int i=0;i<20;++i){if(patches)cuda.local_window(1e-7);else cuda.global_step(1e-8);}
            cuda.download(state,packets);require(difference(before,state)==0,"CUDA reset nondeterminism");
        }
    }
    std::cout<<"gas experiment: partition, conservative fluxes, refinement, reset"<<(gpu?", CPU/CUDA parity":"")<<" pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

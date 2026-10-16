#include "battery_native/local_gas_transport.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Scenario fixture(int count,bool lte,bool outside=false) {
    Scenario s;s.gas_model.split_transport=s.gas_model.cpu_local_transport=true;
    s.gas_model.reference_temperature_k=300;
    CellConfig c;c.id="matrix";c.is_battery=false;c.runaway_energy_j=0;c.thermal_divisions={count,1,1};
    c.dimensions_m={.1*count,.1,.1};c.heat_capacity_j_per_k=count*.03;c.initial_temperature_k=300;
    s.cells={c};
    for(int i=0;i<count;++i) {
        GasCompartment g;g.id="g"+std::to_string(i);g.volume_m3=1e-6;g.initial_temperature_k=300;
        g.initial_mass_kg=(i==0?200000.:100000.)*1e-6/(287*300.);
        g.initial_chemical_j=i==0?.1:0.;g.initial_oxidizer_mass_fraction=i==0?.2:.1;
        if(lte){g.thermal_cell=0;g.thermal_volume=i;}
        s.gas_compartments.push_back(g);
        if(i) {
            GasChannel e;e.id="e"+std::to_string(i);e.from=i-1;e.to=i;e.area_m2=1e-6;e.passage=true;s.gas_channels.push_back(e);
        }
    }
    if(outside) {
        s.ports.push_back({"vent","matrix",0,"vent","+z",.01,{0,0,.05},{0,0,1}});
        GasChannel e;e.id="outside";e.from=count-1;e.to=-1;e.area_m2=1e-7;e.outside_pressure_pa=0.;e.shell_object=0;e.port_id="vent";
        s.gas_channels.push_back(e);
    }
    return s;
}
struct Run {
    std::vector<GasState> gas;std::vector<ThermalState> solid;std::vector<GasChannelState> channels;
    GasTransportStats stats;double seconds=0;
};
Run solve(const Scenario& s,double duration,double h) {
    const auto mesh=build_thermal_mesh(s);const auto mass=build_mass_mesh(s,mesh);LocalGasTransport solver(s,mesh,mass);
    Run out;
    for(const auto& c:s.gas_compartments) {
        GasState g;g.mass_kg=c.initial_mass_kg;g.temperature_k=c.initial_temperature_k;g.free_volume_m3=c.volume_m3;
        g.pressure_pa=g.mass_kg*s.gas_model.specific_r_j_per_kg_k*g.temperature_k/g.free_volume_m3;
        g.energy_j=g.mass_kg*s.gas_model.cv_j_per_kg_k*(g.temperature_k-s.gas_model.reference_temperature_k);
        g.chemical_j=c.initial_chemical_j;g.oxidizer_kg=g.mass_kg*c.initial_oxidizer_mass_fraction;out.gas.push_back(g);
    }
    for(const auto& v:mesh.volumes){ThermalState t;t.temperature_k=300;t.stored_energy_j=v.capacity_j_per_k*300;out.solid.push_back(t);}
    for(const auto& c:s.gas_channels){GasChannelState e;e.open=true;e.opening_fraction=1.;e.effective_area_m2=c.area_m2;out.channels.push_back(e);}
    const auto start=std::chrono::steady_clock::now();
    for(double time=0;time<duration;){const double dt=std::min(h,duration-time);solver.advance(out.gas,out.solid,out.channels,dt,&out.stats);time+=dt;}
    out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    double dm=0,de=0,dc=0,dox=0;
    for(std::size_t i=0;i<out.gas.size();++i) {
        const auto& g=out.gas[i];const auto& before=s.gas_compartments[i];
        require(g.mass_kg>0 && g.temperature_k>0 && g.chemical_j>=-1e-12 && g.oxidizer_kg>=-1e-14,"negative implicit inventory");
        dm+=g.mass_kg-before.initial_mass_kg;
        de+=g.energy_j-before.initial_mass_kg*s.gas_model.cv_j_per_kg_k*
            (before.initial_temperature_k-s.gas_model.reference_temperature_k);
        dc+=g.chemical_j-before.initial_chemical_j;
        dox+=g.oxidizer_kg-before.initial_mass_kg*before.initial_oxidizer_mass_fraction;
        if(before.thermal_cell>=0)require(std::abs(g.temperature_k-out.solid[before.thermal_volume].temperature_k)<1e-12,"LTE mismatch");
    }
    for(std::size_t i=0;i<out.solid.size();++i)de+=out.solid[i].stored_energy_j-mesh.volumes[i].capacity_j_per_k*300;
    for(std::size_t j=0;j<out.channels.size();++j)if(s.gas_channels[j].to<0) {
        const auto& c=out.channels[j];dm+=c.transferred_kg;de+=c.transferred_energy_j;dc+=c.transferred_chemical_j;dox+=c.transferred_oxidizer_kg;
    }
    require(std::abs(dm)<1e-16 && std::abs(de)<1e-9 && std::abs(dc)<1e-12 && std::abs(dox)<1e-16,"implicit conservation");
    return out;
}
double difference(const Run& a,const Run& b,bool temperature=false) {
    double result=0;for(std::size_t i=0;i<a.gas.size();++i)result=std::max(result,std::abs(temperature?
        a.gas[i].temperature_k-b.gas[i].temperature_k:a.gas[i].pressure_pa-b.gas[i].pressure_pa));return result;
}
int main(){try {
    for(bool lte:{false,true})for(int count:{2,9,12}) {
        auto s=fixture(count,lte,true);s.gas_model.local_exchange_interval_s=2e-6;
        const double duration=2e-5;
        auto ref=solve(s,duration,1e-9);
        s.gas_model.implicit_local_transport=true;
        const auto coarse=solve(s,duration,2e-6);
        s.gas_model.pressure_absolute_tolerance_pa*=.25;s.gas_model.pressure_relative_tolerance*=.25;
        s.gas_model.implicit_temperature_tolerance_k*=.25;s.gas_model.implicit_tracer_relative_tolerance*=.25;
        s.gas_model.local_exchange_interval_s*=.25;
        const auto fine=solve(s,duration,5e-7);
        const double e0=difference(ref,coarse),e1=difference(ref,fine),t=difference(ref,fine,true);
        std::cout<<"nodes="<<count<<" LTE="<<lte<<" pressure="<<e0<<" -> "<<e1<<" Pa, T="<<t
                 <<" K, times="<<ref.seconds<<" / "<<coarse.seconds<<" / "<<fine.seconds
                 <<", implicit steps="<<fine.stats.implicit_steps<<", rejects="<<fine.stats.rejections<<std::endl;
        require(e1<e0 && e1<150 && t<.1,"implicit refinement accuracy");
    }
    for(bool lte:{false,true}) {
        auto s=fixture(9,lte);
        auto ring=s.gas_channels[0];ring.id="ring";ring.from=0;ring.to=8;s.gas_channels.push_back(ring);
        for(auto& e:s.gas_channels)e.area_m2*=1000;
        s.gas_model.local_exchange_interval_s=1e-5;
        const auto reference=solve(s,2e-5,1e-9);
        const auto explicit_adaptive=solve(s,2e-5,1e-5);
        s.gas_model.implicit_local_transport=true;
        const auto implicit=solve(s,2e-5,1e-5);
        const double p=difference(reference,implicit),t=difference(reference,implicit,true);
        std::cout<<"stiff cycle LTE="<<lte<<" P="<<p<<" T="<<t<<" explicit="<<explicit_adaptive.seconds
            <<" implicit="<<implicit.seconds<<" steps="<<implicit.stats.implicit_steps<<" rejects="<<implicit.stats.rejections<<std::endl;
        require(p<25 && t<.2,"stiff cycle accuracy");
    }
    {
        auto s=fixture(12,false);
        for(int i=0;i<12;++i) {
            auto& g=s.gas_compartments[i];
            g.initial_temperature_k=i%3==0?900.:300.;
            g.initial_mass_kg=(i%3==0?300000.:100000.)*g.volume_m3/
                (s.gas_model.specific_r_j_per_kg_k*g.initial_temperature_k);
            if(i>1) {auto e=s.gas_channels[0];e.id="cross"+std::to_string(i);e.from=0;e.to=i;s.gas_channels.push_back(e);}
        }
        for(auto& e:s.gas_channels)e.area_m2*=1000;
        s.gas_model.local_exchange_interval_s=1e-5;
        const auto reference=solve(s,2e-5,1e-9);
        const auto coarse_explicit=solve(s,2e-5,1e-5);
        s.gas_model.implicit_local_transport=true;
        const auto coarse=solve(s,2e-5,1e-5);
        s.gas_model.local_exchange_interval_s=1e-6;
        s.gas_model.implicit_temperature_tolerance_k*=.1;
        s.gas_model.pressure_absolute_tolerance_pa*=.1;s.gas_model.pressure_relative_tolerance*=.1;
        const auto fine=solve(s,2e-5,1e-6);
        const double t0=difference(reference,coarse,true),t1=difference(reference,fine,true);
        s.gas_model.local_exchange_interval_s=1e-8;
        const auto resolved=solve(s,2e-5,1e-8);
        const double t2=difference(reference,resolved,true);
        std::cout<<"hot multigroup T="<<t0<<" -> "<<t1<<" -> "<<t2
            <<" K, coarse explicit="<<difference(reference,coarse_explicit,true)
            <<", P="<<difference(reference,resolved)<<std::endl;
        require(t1<t0 && t2<t1 && t2<1.,"hot multigroup refinement accuracy");
    }
    std::cout<<"implicit gas: finite permeability, outside discharge, LTE, tracer conservation and refinement pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

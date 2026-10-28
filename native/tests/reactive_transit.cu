#include "battery_native/reactive_transit.hpp"
#include "battery_native/simulation.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace rt=battery::reactive_transit;
void require(bool a,const char* why){if(!a)throw std::runtime_error(why);}
void close(double a,double b,double tolerance,const char* why){require(std::abs(a-b)<=tolerance,why);}
struct Input{double f,o,fi,oi,h,s;};
__global__ void evaluate(const Input* in,rt::Pair* out,int count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& p=in[i];out[i]=rt::integrate(p.f,p.o,p.fi,p.oi,p.h,p.s);
}
void closure(const Input& p,const rt::Pair& r){
    require(r.fuel>=-1e-12 && r.oxygen>=-1e-12 && r.heat>=0. && r.fuel_out>=-1e-12 && r.oxygen_out>=-1e-12,"Negative transit constituent");
    close(r.fuel+r.fuel_out+r.heat,p.f+p.fi,2e-10*std::max(1.,p.f+p.fi),"Transit chemical energy not conserved");
    close(r.oxygen+r.oxygen_out+r.heat,p.o+p.oi,2e-10*std::max(1.,p.o+p.oi),"Transit oxygen not conserved");
}
void subdivide(const Input& p){
    const auto whole=rt::integrate(p.f,p.o,p.fi,p.oi,p.h,p.s);
    const auto a=rt::integrate(p.f,p.o,p.fi/2,p.oi/2,p.h/2,p.s/2);
    const auto b=rt::integrate(a.fuel,a.oxygen,p.fi/2,p.oi/2,p.h/2,p.s/2);
    close(whole.fuel,b.fuel,2e-10,"Transit fuel depends on split step");
    close(whole.oxygen,b.oxygen,2e-10,"Transit oxygen depends on split step");
    close(whole.heat,a.heat+b.heat,2e-10,"Transit reaction heat depends on split step");
    close(whole.fuel_out,a.fuel_out+b.fuel_out,2e-10,"Transit exit fuel depends on split step");
}
void closed_full_solver(){
    using namespace battery;
    Scenario input;input.settings={.1,.01,.01,.4};input.settings.minimum_dt_s=.01;
    input.gas_model.split_transport=input.gas_model.cuda_kawasaki_transport=true;
    input.gas_model.cuda_network_transport=input.gas_model.averaged_network_transport=true;
    input.gas_model.cuda_batched_transport=input.gas_model.reactive_transit=true;
    input.gas_model.coupled_gas_films=input.gas_model.joint_gas_surface_mixing=true;
    input.gas_model.implicit_network_mixing=true;
    CellConfig solid;solid.id="inert-holder";solid.is_battery=false;solid.dimensions_m={.01,.01,.01};
    solid.thermal_divisions={1,1,1};solid.heat_capacity_j_per_k=1.;input.cells.push_back(solid);
    GasCompartment gas;gas.id="closed";gas.volume_m3=1e-5;gas.initial_mass_kg=1e-5;
    gas.initial_temperature_k=600.;gas.initial_chemical_j=.1;gas.initial_oxidizer_mass_fraction=.01;
    gas.reaction_rate_per_s=20.;gas.reaction_activation_j_per_mol=0.;input.gas_compartments.push_back(gas);
    Simulation sim(input);sim.run_to_end();const auto d=sim.process_diagnostics();
    close(d.gas[0].chemical_j,.1*std::exp(-2.),1e-10,"Full CUDA solver lost exact closed reaction");
    close(d.chemical_residual_j,0.,1e-9,"Full closed chemical balance");
    close(d.thermal_residual_j,0.,1e-9,"Full closed sensible balance");
}
void blocked_passage_full_solver(){
    using namespace battery;
    Scenario s;s.settings={.02,.01,.01,.4};s.settings.minimum_dt_s=.01;
    s.gas_model.split_transport=s.gas_model.cuda_kawasaki_transport=true;
    s.gas_model.cuda_network_transport=s.gas_model.averaged_network_transport=true;
    s.gas_model.cuda_batched_transport=s.gas_model.reactive_transit=true;
    s.gas_model.coupled_gas_films=s.gas_model.joint_gas_surface_mixing=true;
    s.gas_model.implicit_network_mixing=true;s.gas_model.coupled_transport_interval_s=.01;
    CellConfig solid;solid.id="holder";solid.is_battery=false;solid.dimensions_m={.01,.01,.01};
    solid.thermal_divisions={1,1,1};solid.heat_capacity_j_per_k=1.;s.cells.push_back(solid);
    for(int i=0;i<2;++i){GasCompartment gas;gas.id="blocked-"+std::to_string(i);
        gas.volume_m3=1e-5;gas.initial_mass_kg=i?1e-5:5e-5;
        gas.initial_temperature_k=i?300.:1000.;gas.initial_chemical_j=i?0.:1.;
        gas.initial_oxidizer_mass_fraction=i?.23:0.;gas.reaction_rate_per_s=0.;s.gas_compartments.push_back(gas);}
    GasChannel c;c.id="sealed";c.from=0;c.to=1;c.passage=true;c.area_m2=0.;s.gas_channels.push_back(c);
    Simulation sim(s);sim.run_to_end();const auto d=sim.process_diagnostics();
    for(int i=0;i<2;++i){close(d.gas[i].mass_kg,s.gas_compartments[i].initial_mass_kg,1e-16,"Blocked passage leaked mass");
        close(d.gas[i].temperature_k,s.gas_compartments[i].initial_temperature_k,1e-9,"Blocked passage exchanged heat");
        close(d.gas[i].chemical_j,s.gas_compartments[i].initial_chemical_j,1e-12,"Blocked passage exchanged fuel");}
    close(d.channels[0].transferred_kg,0.,0.,"Blocked pressure flow is not exactly zero");
    close(d.channels[0].mixed_mass_kg,0.,0.,"Blocked molecular mixing is not exactly zero");
    close(d.oxidizer_residual_kg,0.,1e-15,"Blocked passage oxygen balance");
}
double flowing_full_solver(double step,bool corrected){
    using namespace battery;
    Scenario s;s.settings={.05,step,step,.4};s.settings.minimum_dt_s=step;
    s.gas_model.split_transport=s.gas_model.cuda_kawasaki_transport=true;
    s.gas_model.cuda_network_transport=s.gas_model.averaged_network_transport=true;
    s.gas_model.cuda_batched_transport=true;s.gas_model.reactive_transit=corrected;
    s.gas_model.coupled_gas_films=s.gas_model.joint_gas_surface_mixing=true;
    s.gas_model.implicit_network_mixing=true;s.gas_model.coupled_transport_interval_s=step;
    s.gas_model.averaged_payload_iterations=32;
    CellConfig solid;solid.id="inert-holder";solid.is_battery=false;solid.dimensions_m={.01,.01,.01};
    solid.thermal_divisions={1,1,1};solid.heat_capacity_j_per_k=1.;s.cells.push_back(solid);
    for(int i=0;i<3;++i){GasCompartment gas;gas.id="reactive-path-"+std::to_string(i);
        gas.volume_m3=i==1?1e-8:1.;gas.initial_temperature_k=800.;
        gas.initial_mass_kg=(3-i)*1e5*gas.volume_m3/(287.*800.);
        gas.initial_chemical_j=i==0?gas.initial_mass_kg:0.;gas.initial_oxidizer_mass_fraction=.01;
        gas.reaction_rate_per_s=i==1?30000.:0.;gas.reaction_activation_j_per_mol=0.;
        s.gas_compartments.push_back(gas);}
    for(int i=0;i<2;++i){GasChannel c;c.id="path-"+std::to_string(i);c.from=i;c.to=i+1;
        c.passage=true;c.area_m2=1e-6;c.discharge_coefficient=.7;s.gas_channels.push_back(c);}
    Simulation sim(s);sim.run_to_end();const auto d=sim.process_diagnostics();
    close(d.mass_residual_kg,0.,1e-12,"Flowing reactor mass balance");
    close(d.chemical_residual_j,0.,1e-9,"Flowing reactor chemical balance");
    close(d.thermal_residual_j,0.,1e-6,"Flowing reactor energy balance");
    close(d.oxidizer_residual_kg,0.,1e-12,"Flowing reactor oxygen balance");
    const double fraction=d.gas[1].released_energy_j/d.channels[0].transferred_chemical_j;
    std::cout<<"Actual CUDA path: step="<<step<<" corrected="<<corrected<<" burn fraction="<<fraction<<'\n';
    return fraction;
}
int main(){try{
    std::vector<Input> cases;
    for(double f:{0.,.1,1.,10.})for(double o:{0.,.1,1.,10.})
    for(double fi:{0.,1.,10.})for(double oi:{0.,1.,10.})
    for(double h:{0.,1e-8,.1,1.,20.,2000.})for(double s:{0.,1e-8,.1,1.,20.,2000.}){
        const Input p{f,o,fi,oi,h,s};const auto r=rt::integrate(f,o,fi,oi,h,s);
        closure(p,r);subdivide(p);cases.push_back(p);
    }
    for(double m:{1e-9,1e-3,1.,1e3})for(double incoming:{0.,1e-8,.01,1.,1e3})
    for(double fraction:{1e-6,.01,.2,.8,.99}){
        const double out=(m+incoming)*fraction;double h;
        require(rt::exposure(m,incoming,out,h),"Carrier exposure did not converge");
        close(rt::carrier_out(m,incoming,h),out,2e-11*(m+incoming),"Carrier turnover does not match accepted mass");
    }
    // Same counterexample as fsri30_uncertainty: dt=.01, tau=.0005, k=2000.
    // Exact flow-through integral burns ~half the arriving fuel, not ~5%.
    auto stock=rt::integrate(0.,10.,.01,200.,20.,20.);
    close(stock.heat,.004875,1e-12,"Known analytical throughflow heat");
    Input* in=nullptr;rt::Pair* out=nullptr;
    require(cudaMalloc(&in,cases.size()*sizeof(Input))==cudaSuccess,"CUDA input allocation");
    require(cudaMalloc(&out,cases.size()*sizeof(rt::Pair))==cudaSuccess,"CUDA output allocation");
    cudaMemcpy(in,cases.data(),cases.size()*sizeof(Input),cudaMemcpyHostToDevice);
    evaluate<<<(cases.size()+127)/128,128>>>(in,out,cases.size());
    std::vector<rt::Pair> gpu(cases.size());
    require(cudaMemcpy(gpu.data(),out,gpu.size()*sizeof(rt::Pair),cudaMemcpyDeviceToHost)==cudaSuccess,"CUDA reactive transit");
    for(std::size_t i=0;i<cases.size();++i){const auto& p=cases[i];closure(p,gpu[i]);
        const auto host=rt::integrate(p.f,p.o,p.fi,p.oi,p.h,p.s);
        close(gpu[i].heat,host.heat,2e-10,"Host/device transit heat mismatch");}
    cudaFree(in);cudaFree(out);
    closed_full_solver();blocked_passage_full_solver();
    const double coarse=flowing_full_solver(.01,true),fine=flowing_full_solver(.005,true);
    const double split=flowing_full_solver(.01,false);
    require(coarse>.1 && coarse>5*split,"Reactive transit did not recover passing-fuel reaction");
    require(std::abs(coarse-fine)<.05*std::max(coarse,fine),"Reactive path is not time-consistent within5%");
    std::cout<<"Reactive transit: "<<cases.size()<<" analytical cases, limiting-species switches, exact step subdivision, mass/energy/oxygen and CUDA agreement pass\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

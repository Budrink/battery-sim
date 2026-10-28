#include "battery_native/gas_ambient.hpp"
#include "battery_native/simulation.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace ga = battery::gas_ambient;
namespace kg = battery::kawasaki;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void close(double a, double b, double eps, const char* why) { require(std::abs(a-b)<=eps,why); }
struct Case { kg::Node n; battery::GasAmbientBoundary b; double dt; };
struct Result { kg::Node n; ga::Transfer q; bool ok; };
Result evaluate(const Case& c) {
    Result r{c.n,{},false}; kg::Settings s; s.cv=1000.; s.R=287.;
    r.ok=ga::apply(r.n,c.b,s,298.15,c.dt,r.q); return r;
}
__global__ void device_evaluate(const Case* cases,Result* results,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    auto& r=results[i];r.n=cases[i].n;kg::Settings s;s.cv=1000.;s.R=287.;
    r.ok=ga::apply(r.n,cases[i].b,s,298.15,cases[i].dt,r.q);
}
void analytical() {
    kg::Settings s;s.cv=1000.;s.R=287.;
    Case c;c.n={.01,7000.,.01,0.,12.,.001};c.n.vapor=.001;c.n.latent=20.;
    c.b.enabled=true;c.b.renewal_flow_m3_s=.001;c.dt=2.;
    const auto r=evaluate(c);require(r.ok,"Analytic renewal rejected");
    const double exposure=c.dt*c.b.renewal_flow_m3_s*c.b.pressure_pa/(s.R*c.b.temperature_k*c.n.mass);
    close(r.n.energy,c.n.mass*s.cv*c.b.temperature_k+(c.n.energy-c.n.mass*s.cv*c.b.temperature_k)*std::exp(-1.287*exposure),1e-10,"Renewal enthalpy integral");
    close(r.n.chemical,12.*std::exp(-exposure),1e-12,"Renewal fuel integral");
    close(r.n.oxidizer,c.n.mass*.232+(.001-c.n.mass*.232)*std::exp(-exposure),1e-14,"Renewal oxygen integral");
    c.dt=1.;auto first=evaluate(c);c.n=first.n;auto second=evaluate(c);
    close(r.n.energy,second.n.energy,1e-10,"Renewal time subdivision");
    close(r.q.energy,first.q.energy+second.q.energy,1e-10,"Renewal energy ledger subdivision");
    close(r.q.chemical,first.q.chemical+second.q.chemical,1e-12,"Renewal chemical ledger subdivision");
    c.b.enabled=false;auto off=evaluate(c);close(off.n.energy,c.n.energy,0.,"Disabled boundary changed energy");
    close(off.n.mass,c.n.mass,0.,"Disabled boundary changed mass");
    c.b.enabled=true;c.b.renewal_flow_m3_s=0.;auto zero=evaluate(c);
    close(zero.n.energy,c.n.energy,0.,"Zero boundary changed energy");
    for(double pressure:{1000.,10000.,1e6,1e8}) {
        c.b.area_m2=1.;c.dt=100.;c.n.mass=pressure*.01/(s.R*700.);c.n.energy=c.n.mass*s.cv*700.;
        c.n.chemical=10.;c.n.oxidizer=.1*c.n.mass;c.n.vapor=.1*c.n.mass;c.n.latent=0.;
        auto p=evaluate(c);require(p.ok,"Pressure relaxation rejected");
        close(kg::pressure(p.n,s),c.b.pressure_pa,.01,"Open pressure did not approach ambient");
        if(pressure>c.b.pressure_pa) {
            close(kg::temperature(p.n,s),700.*std::pow(p.n.mass/c.n.mass,s.R/s.cv),1e-8,"Adiabatic exhaust temperature");
            close(p.n.chemical/c.n.chemical,p.n.mass/c.n.mass,1e-12,"Pressure outflow constituent");
        } else {
            close(p.n.energy-c.n.energy,(s.cv+s.R)*c.b.temperature_k*(p.n.mass-c.n.mass),1e-8,"Inlet ambient enthalpy");
        }
    }
}
void full_solver() {
    using namespace battery;
    Scenario s;s.settings={1.,.01,.01,.4};s.settings.minimum_dt_s=.01;
    auto& m=s.gas_model;m.split_transport=m.cuda_kawasaki_transport=true;
    m.cuda_network_transport=m.averaged_network_transport=m.cuda_batched_transport=m.reactive_transit=true;
    m.coupled_gas_films=m.joint_gas_surface_mixing=m.implicit_network_mixing=true;
    CellConfig holder;holder.id="holder";holder.is_battery=false;holder.dimensions_m={.01,.01,.01};
    holder.thermal_divisions={1,1,1};holder.heat_capacity_j_per_k=1.;s.cells.push_back(holder);
    GasCompartment gas;gas.id="open-air";gas.volume_m3=.001;gas.initial_temperature_k=700.;
    gas.initial_mass_kg=2e5*gas.volume_m3/(287.*700.);gas.initial_chemical_j=10.;
    gas.initial_oxidizer_mass_fraction=.1;gas.reaction_rate_per_s=0.;
    gas.ambient.enabled=true;gas.ambient.area_m2=1e-5;gas.ambient.renewal_flow_m3_s=.001;
    s.gas_compartments.push_back(gas);
    Simulation sim(s);for(int k=0;k<30;++k)sim.step();
    const auto checkpoint=sim.save_checkpoint("ambient-test-v2");
    Simulation restarted(s);restarted.restore_checkpoint(checkpoint,"ambient-test-v2");
    sim.run_to_end();restarted.run_to_end();const auto d=sim.process_diagnostics();
    const auto rd=restarted.process_diagnostics();
    close(d.mass_residual_kg,0.,1e-13,"Open solver mass balance");
    close(d.thermal_residual_j,0.,1e-8,"Open solver energy balance");
    close(d.chemical_residual_j,0.,1e-9,"Open solver chemical balance");
    close(d.oxidizer_residual_kg,0.,1e-13,"Open solver oxygen balance");
    require(d.escaped_chemical_j>0. && d.gas[0].ambient_gross_in_kg>0.,"No actual outgoing fuel / incoming air");
    require(d.gas[0].temperature_k<700.,"Open gas did not cool");
    close(d.escaped_energy_j,rd.escaped_energy_j,0.,"Restart lost boundary ledger");
    close(d.gas[0].temperature_k,rd.gas[0].temperature_k,0.,"Restart changed open trajectory");
}
// Coupled pressure path -> open far field. This exercises operator splitting,
// not merely the isolated exact CSTR formula. Never shrink the physical floor.
std::vector<double> flowing_solver(double step) {
    using namespace battery;
    Scenario s;s.settings={2.,step,step,.4};s.settings.minimum_dt_s=.01;
    auto& m=s.gas_model;m.split_transport=m.cuda_kawasaki_transport=true;
    m.cuda_network_transport=m.averaged_network_transport=m.cuda_batched_transport=m.reactive_transit=true;
    m.coupled_gas_films=m.joint_gas_surface_mixing=m.implicit_network_mixing=true;
    m.coupled_transport_interval_s=.025;
    CellConfig holder;holder.id="holder";holder.is_battery=false;holder.dimensions_m={.01,.01,.01};
    holder.thermal_divisions={1,1,1};holder.heat_capacity_j_per_k=1.;s.cells.push_back(holder);
    for(int i=0;i<2;++i) {
        GasCompartment gas;gas.id=i?"far-field":"hot-source";gas.volume_m3=.01;
        gas.initial_temperature_k=i?298.15:800.;
        gas.initial_mass_kg=(i?101325.:200000.)*gas.volume_m3/(287.*gas.initial_temperature_k);
        gas.initial_chemical_j=i?0.:10.;gas.initial_oxidizer_mass_fraction=.1;
        if(i){gas.ambient.enabled=true;gas.ambient.area_m2=.05;gas.ambient.renewal_flow_m3_s=.0001;}
        s.gas_compartments.push_back(gas);
    }
    GasChannel pipe;pipe.id="open-pipe";pipe.from=0;pipe.to=1;pipe.passage=true;pipe.area_m2=1e-6;
    s.gas_channels.push_back(pipe);Simulation sim(s);sim.run_to_end();const auto d=sim.process_diagnostics();
    close(d.mass_residual_kg,0.,1e-13,"Open network mass balance");
    close(d.thermal_residual_j,0.,1e-7,"Open network thermal balance");
    close(d.chemical_residual_j,0.,1e-10,"Open network chemical balance");
    close(d.oxidizer_residual_kg,0.,1e-13,"Open network oxygen balance");
    return {d.gas[0].temperature_k,d.gas[1].temperature_k,d.escaped_energy_j};
}
int main() { try {
    analytical();std::vector<Case> cases;
    for(double pressure:{1000.,1e5,101325.,1e6,1e8})for(double t:{100.,298.15,700.,3000.})
    for(double dt:{.01,.1,1.,100.})for(double area:{0.,1e-6,.01})for(double flow:{0.,.01}) {
        Case c;c.b.enabled=true;c.b.area_m2=area;c.b.renewal_flow_m3_s=flow;c.dt=dt;
        const double mass=pressure*.001/(287.*t);c.n={mass,mass*1000.*t,.001,0.,10.,.1*mass};
        c.n.vapor=.05*mass;c.n.latent=1.;cases.push_back(c);
    }
    Case* in=nullptr;Result* out=nullptr;cudaMalloc(&in,cases.size()*sizeof(Case));cudaMalloc(&out,cases.size()*sizeof(Result));
    cudaMemcpy(in,cases.data(),cases.size()*sizeof(Case),cudaMemcpyHostToDevice);
    device_evaluate<<<(cases.size()+127)/128,128>>>(in,out,cases.size());std::vector<Result> gpu(cases.size());
    require(cudaMemcpy(gpu.data(),out,gpu.size()*sizeof(Result),cudaMemcpyDeviceToHost)==cudaSuccess,"Boundary CUDA failure");
    for(std::size_t i=0;i<cases.size();++i) {
        const auto host=evaluate(cases[i]);require(host.ok && gpu[i].ok,"Boundary positivity/entropy failure");
        close(host.n.energy,gpu[i].n.energy,1e-9*std::max(1.,host.n.energy),"Boundary CPU/CUDA energy");
        close(host.n.mass,gpu[i].n.mass,1e-12*std::max(1.,host.n.mass),"Boundary CPU/CUDA mass");
        close(host.q.chemical,gpu[i].q.chemical,1e-10,"Boundary CPU/CUDA fuel");
    }
    cudaFree(in);cudaFree(out);full_solver();
    const auto coarse=flowing_solver(.1),medium=flowing_solver(.05),fine=flowing_solver(.01);
    for(std::size_t i=0;i<fine.size();++i) {
        std::cout<<"Open network quantity "<<i<<": .1="<<coarse[i]<<" .05="<<medium[i]<<" .01="<<fine[i]<<'\n';
        require(std::abs(coarse[i]-fine[i])<=.05*std::max(1.,std::abs(fine[i])),"Open pressure path exceeds 5% step sensitivity");
        // Coupled gas clocks partition .1/.05/.01 differently; tiny errors
        // need not decrease monotonically. Bound BOTH coarser trajectories.
        require(std::abs(medium[i]-fine[i])<=.05*std::max(1.,std::abs(fine[i])),"Open pressure path exceeds 5% medium-step sensitivity");
    }
    std::cout<<"Ambient boundary: "<<cases.size()<<" cases, analytic renewal, pressure enthalpy, positivity/entropy, CUDA balances and restart pass\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; } }

#include "battery_native/gas_local_reactor.hpp"
#include "battery_native/simulation.hpp"
#include "battery_native/gas_thermal_contact.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
void close(double a,double b,double tol,const char* why){require(std::abs(a-b)<=tol,why);}
kawasaki::Node initial() {return {4e-7,.0004*600.,4e-7,0.,.1,.1/13.1e6};}
struct Pair {kawasaki::Node n;GasFilmState f;};
__global__ void advance(Pair* p,GasCombustionLaw law,double dt) {
    if(!advance_gas_reactor(p->n,law,{},dt,p->f))p->n.energy=-1.;
}
void analytic(double rate,double conductance) {
    GasCombustionLaw law{rate,0.,13.1e6,600.,0.};
    Pair a{initial(),{conductance,300.,2.}},b=a;const double dt=.03;
    const double capacity=.0004,alpha=conductance/capacity,diff=std::abs(alpha-rate);
    const double convolution=std::exp(-std::min(alpha,rate)*dt)*(diff>0?-std::expm1(-diff*dt)/diff:dt);
    const double expected=300.+300.*std::exp(-alpha*dt)+rate*.1/capacity*convolution;
    require(advance_gas_reactor(b.n,law,{},dt,b.f),"local reactor failed");
    close(kawasaki::temperature(b.n,{}),expected,1e-9,"frozen-k analytical temperature");
    close(b.n.chemical,.1*std::exp(-rate*dt),1e-12,"frozen-k fuel decay");
    close(a.n.energy+a.n.chemical,b.n.energy+b.n.chemical+b.f.heat_to_solid,1e-12,"local energy closure");
    require(b.n.mass==a.n.mass && b.n.oxidizer>=0 && b.n.chemical>=0,"local mass/positivity");
    Pair* p=nullptr;require(cudaMalloc(&p,sizeof(Pair))==cudaSuccess,"CUDA allocate");
    cudaMemcpy(p,&a,sizeof(Pair),cudaMemcpyHostToDevice);advance<<<1,1>>>(p,law,dt);
    Pair gpu;require(cudaMemcpy(&gpu,p,sizeof(Pair),cudaMemcpyDeviceToHost)==cudaSuccess,"CUDA local reactor");cudaFree(p);
    close(gpu.n.energy,b.n.energy,1e-11,"host/device gas reactor agreement");
    close(gpu.f.heat_to_solid,b.f.heat_to_solid,1e-11,"host/device film heat agreement");
}
Scenario fixture(double step,bool reactive=true) {
    Scenario s;s.settings={.1,step,.1,.4};
    s.gas_model.split_transport=s.gas_model.cuda_kawasaki_transport=true;
    s.gas_model.cuda_batched_transport=s.gas_model.cuda_network_transport=true;
    s.gas_model.coupled_gas_films=true;s.gas_model.max_temperature_step_k=step/.02;
    s.gas_model.coupled_transport_interval_s=step/2;
    CellConfig c;c.id="surface";c.is_battery=false;c.dimensions_m={.1,.1,.1};c.thermal_divisions={1,1,1};
    c.heat_capacity_j_per_k=2.;c.initial_temperature_k=300.;c.damage_onset_k=300.01;
    c.damage_rate_at_onset_per_s=.01;c.damage_temperature_scale_k=100.;s.cells.push_back(c);
    s.ports.push_back({"film","surface",0,"contact","+z",.01,{0,0,.05},{0,0,1}});
    GasCompartment g;g.id="gas";g.volume_m3=4e-7;g.initial_mass_kg=4e-7;g.initial_temperature_k=600.;
    g.initial_chemical_j=reactive?.1:0.;g.initial_oxidizer_mass_fraction=.1;
    g.reaction_rate_per_s=reactive?20.:0.;g.reaction_reference_k=600.;g.reaction_activation_j_per_mol=40000.;
    s.gas_compartments.push_back(g);s.gas_contacts.push_back({"film",0,10.});
    return s;
}
struct Result {double gas,solid,damage;};
Scenario flow_fixture(double step) {
    auto s=fixture(step);s.cells[0].initial_temperature_k=600.;
    auto c=s.cells[0];c.id="receiver";c.position_m.z=.2;c.initial_temperature_k=300.;s.cells.push_back(c);
    s.ports.push_back({"receiver-film","receiver",1,"contact","+z",.01,{0,0,.25},{0,0,1}});
    auto& a=s.gas_compartments[0];a.initial_mass_kg=3e5*a.volume_m3/(287*600.);a.initial_oxidizer_mass_fraction=0.;
    auto b=a;b.id="receiver-gas";b.initial_temperature_k=300.;b.initial_mass_kg=1e5*b.volume_m3/(287*300.);
    b.initial_chemical_j=0.;b.initial_oxidizer_mass_fraction=.232;s.gas_compartments.push_back(b);
    s.gas_contacts.push_back({"receiver-film",1,10.});
    GasChannel path;path.id="pressure-path";path.from=0;path.to=1;path.passage=true;path.area_m2=1e-9;
    s.gas_channels.push_back(path);return s;
}
std::array<double,3> flow(double step) {
    Simulation s(flow_fixture(step));s.run_to_end();const auto d=s.process_diagnostics();
    require(std::abs(d.thermal_residual_j)<1e-8 && std::abs(d.mass_residual_kg)<1e-14 &&
        std::abs(d.chemical_residual_j)<1e-9 && std::abs(d.oxidizer_residual_kg)<1e-14,"flowing reactive film balance");
    require(d.channels[0].transferred_kg>0. && d.gas[1].chemical_j>0.,"actual arriving fuel lost");
    require(d.gas[1].released_energy_j>0.,"arrivals did not react in same interval");
    return {s.cells()[1].temperature_k,d.gas[1].pressure_pa,d.gas[1].released_energy_j};
}
void tiling_and_retry() {
    auto s=fixture(.02);s.cells[0].thermal_divisions={5,5,5};
    s.cells[0].surface_connectivity.linear={.2,.3,.4};
    s.gas_contacts[0].gas_surface_connectivity.linear={.3,.4,.5};
    const auto mesh=build_thermal_mesh(s);
    CudaSolver a(s,mesh,64),b(s,mesh,256);RuntimeParameters rt;
    const double da=a.prepare_step(0.,.02,.4,rt);a.discard_step();
    close(da,a.prepare_step(0.,.02,.4,rt),0.,"retry changes coupled interval");
    close(da,b.prepare_step(0.,.02,.4,rt),0.,"tiling changes coupled interval");
    a.commit_step();b.commit_step();std::vector<ThermalState> fa,fb;a.download_field(fa);b.download_field(fb);
    for(std::size_t i=0;i<fa.size();++i) {
        require(fa[i].temperature_k==fb[i].temperature_k && fa[i].stored_energy_j==fb[i].stored_energy_j &&
            fa[i].damage==fb[i].damage,"coupled scatter depends on CUDA tiling");
    }
    const auto ga=a.process_diagnostics(),gb=b.process_diagnostics();
    require(ga.gas[0].energy_j==gb.gas[0].energy_j && ga.gas[0].chemical_j==gb.gas[0].chemical_j,
        "coupled gas depends on tiling/retry");
}
void coarse_mode() {
    for(double floor:{.005,.01}) {
        auto input=fixture(.02);input.settings.minimum_dt_s=floor;
        input.gas_model.max_temperature_step_k=.0001;
        input.gas_model.coupled_transport_interval_s=.00001;
        input.gas_compartments[0].reaction_rate_per_s=1e6;
        input.gas_compartments[0].initial_chemical_j=.5;
        auto source=input.cells[0];source.id="stiff-source";source.position_m.x=.3;
        source.is_battery=true;source.initial_damage=1.;
        source.runaway_energy_j=1.;source.runaway_duration_s=1e-7;input.cells.push_back(source);
        Simulation s(input);
        while(!s.finished()) {
            s.step();require(s.last_dt_s()>=floor*(1.-1e-10),"coarse outer step fell below floor");
        }
        const auto d=s.process_diagnostics();const auto profile=s.profile();
        require(std::abs(d.thermal_residual_j)<1e-9 && std::abs(d.chemical_residual_j)<1e-9,
            "coarse reaction/film lost energy");
        require(std::abs(d.mass_residual_kg)<1e-14 && std::abs(d.oxidizer_residual_kg)<1e-14,
            "coarse reaction lost mass/oxygen");
        require(profile.gas_max_local_reactor_iterations==1,"coarse mode used hidden adaptive reactor microsteps");
        require(profile.gas_last_reaction_step_s>=floor*(1.-1e-10),"coarse gas graph fell below floor");
        require(profile.minimum_regular_step_s>=floor*(1.-1e-10),"regular step diagnostics violate floor");
        require(profile.coarse_floor_activations>0,"stiff source never exercised the outer floor");
        std::cout<<"coarse floor "<<floor<<" s: "<<s.step_count()<<" steps, energy/mass closed\n";
    }
}
Result simulate(double step,bool reactive=true) {
    Simulation s(fixture(step,reactive));s.run_to_end();const auto d=s.process_diagnostics();
    require(std::abs(d.thermal_residual_j)<1e-9 && std::abs(d.chemical_residual_j)<1e-9,"coupled full-model energy closure");
    require(std::abs(d.mass_residual_kg)<1e-14 && std::abs(d.oxidizer_residual_kg)<1e-14,"coupled full-model mass closure");
    require(s.step_count()<10000,"coupled film still forces global microsteps");
    return {d.gas[0].temperature_k,s.cells()[0].temperature_k,s.cells()[0].damage};
}
void directed_surface_exchange() {
    auto passive=fixture(.005,false);passive.cells[0].thermal_divisions={2,1,1};
    passive.cells[0].surface_connectivity.linear={1.,0.,0.};
    passive.boundaries.push_back({"ambient","film",0,300.,1.});
    const auto passive_mesh=build_thermal_mesh(passive);
    for(const auto& e:passive_mesh.links) {
        require(e.law.kind==TransitionKind::DirectedGeometry,"Default passive bulk/boundary bypasses shared law");
        require(e.law.endpoint_a.linear.x==1.,"Default passive contact dropped node geometry");
    }
    GasThermalPatch p;p.law=reference_gas_connectivity(2.);p.geometry_axis=2;
    p.law.endpoint_a.temperature_scale_k=p.law.endpoint_b.temperature_scale_k=100.;
    p.law.endpoint_a.reference_temperature_k=p.law.endpoint_b.reference_temperature_k=300.;
    p.law.endpoint_a.linear={0.,0.,1.};p.law.endpoint_b.linear={0.,0.,-1.};
    const double ab=gas_contact_conductance(p,600.,300.,0.);
    const double ba=gas_contact_conductance(p,300.,600.,0.);
    require(ab>3.7 && ba<.3,"Gas law averaged directed endpoint responses");
    p.geometry_axis=0;close(gas_contact_conductance(p,600.,300.,0.),2.,1e-14,"Gas contact ignores geometry axis");
    p.geometry_axis=2;p.law.endpoint_b.damage.z=4.;
    require(gas_contact_conductance(p,300.,600.,1.)>ba*4.,"Gas contact ignores surface damage");
    auto exchange=[](bool hot_gas,double gas_slope,double solid_slope,double damage) {
        auto s=fixture(.005,false);s.settings.duration_s=.005;s.settings.minimum_dt_s=.005;
        s.gas_model.coupled_transport_interval_s=.005;
        auto& c=s.cells[0];c.initial_temperature_k=hot_gas?300.:600.;c.initial_damage=damage;
        c.damage_rate_at_onset_per_s=0.;c.surface_connectivity.reference_temperature_k=300.;
        c.surface_connectivity.temperature_scale_k=100.;c.surface_connectivity.linear.z=solid_slope;
        c.surface_connectivity.damage.z=2.;
        auto& contact=s.gas_contacts[0];contact.gas_surface_connectivity.reference_temperature_k=300.;
        contact.gas_surface_connectivity.temperature_scale_k=100.;contact.gas_surface_connectivity.linear.z=gas_slope;
        s.gas_compartments[0].initial_temperature_k=hot_gas?600.:300.;
        Simulation sim(s);sim.run_to_end();double heat=0.;for(double q:sim.last_gas_film_heat_j())heat+=q;
        require(std::abs(sim.process_diagnostics().thermal_residual_j)<1e-9,"Directed gas heat lost energy");
        return std::abs(heat);
    };
    const double base=exchange(true,0.,0.,0.);
    require(exchange(true,1.,0.,0.)>1.1*base,"CUDA gas donor profile has no effect");
    close(exchange(true,0.,-5.,0.),base,1e-12,"CUDA used cold recipient profile as donor");
    const double reverse=exchange(false,0.,0.,0.);
    require(exchange(false,0.,1.,0.)>1.1*reverse,"CUDA solid donor profile has no effect");
    close(exchange(false,-5.,0.,0.),reverse,1e-12,"CUDA symmetrized gas/solid profiles");
    require(exchange(false,0.,0.,.8)>1.1*reverse,"CUDA gas exchange ignores solid damage");
    std::cout<<"directed gas/surface: donor reversal, anisotropy, damage, CUDA heat and energy closure pass\n";
}
int main(){try {
    directed_surface_exchange();
    analytic(3.,.1);analytic(250.,.1);analytic(0.,1000.);
    const auto coarse=simulate(.02),fine=simulate(.01),ref=simulate(.0005);
    const double a=std::abs(coarse.solid-ref.solid),b=std::abs(fine.solid-ref.solid);
    std::cout<<"surface refinement: "<<coarse.solid<<" / "<<fine.solid<<" / "<<ref.solid
             <<"; errors "<<a<<" / "<<b<<" K\n";
    require(b<a && a<.01,"coupled surface temperature fails refinement");
    require(std::abs(fine.damage-ref.damage)<1e-4,"surface aging not reconciled after gas heating");
    const auto fc=flow(.02),ff=flow(.01),fr=flow(.0005);
    std::cout<<"flow pressure errors: "<<std::abs(fc[1]-fr[1])/fr[1]<<" -> "<<std::abs(ff[1]-fr[1])/fr[1]<<'\n';
    require(std::abs(fc[0]-fr[0])<.01 && std::abs(ff[0]-fr[0])<.01,"reactive flow surface accuracy");
    require(std::abs(fc[1]-fr[1])/fr[1]<.01 && std::abs(ff[1]-fr[1])<std::abs(fc[1]-fr[1]),"reactive flow pressure convergence");
    const auto cold=simulate(.02,false);
    const double equilibrium=(2.*300.+.0004*600.)/2.0004;
    close(cold.solid,equilibrium,.001,"finite film heat did not reach surface");
    auto s=fixture(.02);auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh,64);RuntimeParameters rt;
    const double first=solver.prepare_step(0.,.02,.4,rt);solver.discard_step();
    close(first,solver.prepare_step(0.,.02,.4,rt),0.,"coupled retry changes dt");solver.commit_step();
    require(std::abs(solver.process_diagnostics().thermal_residual_j)<1e-9,"coupled retry double spends film heat");
    tiling_and_retry();coarse_mode();
    std::cout<<"coupled gas films: analytical/resonance, CUDA, energy/mass, aging, retry, refinement "<<a<<" -> "<<b<<" K pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

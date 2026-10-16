#include "battery_native/simulation.hpp"
#include "battery_native/constitutive.hpp"
#include "battery_native/capped_budget.hpp"
#include "battery_native/gas_timestep.hpp"
#include "battery_native/conversion.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <random>
using namespace battery;
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
void budget_test() {
    std::mt19937 rng(716);std::uniform_real_distribution<double> logscale(-5,5);
    for(int trial=0;trial<2000;++trial) {
        std::vector<CappedDemand> demands;const int count=trial%8;
        for(int j=0;j<count;++j)demands.push_back({std::pow(10.,logscale(rng)),std::pow(10.,logscale(rng))});
        const double linear=trial%3?std::pow(10.,logscale(rng)):0.;
        const double budget=std::pow(10.,logscale(rng));
        auto demand=[&](int j){return demands[j];};
        auto spent=[&](double dt){double q=linear*dt;for(auto d:demands)q+=std::min(d.rate*dt,d.cap);return q;};
        const double dt=capped_budget_dt(budget,linear,count,demand);
        require(spent(dt)<=budget*(1+1e-10),"Capped budget overdrew shared stock");
        if(dt<1e299)require(std::abs(spent(dt)-budget)<1e-9*budget,"Capped budget unnecessarily restricts dt");
        else require(linear==0 && spent(1e100)<=budget*(1+1e-10),"Incorrect unbounded budget");
    }
}
Scenario gas_fixture() {
    Scenario s;s.gas_model.reference_temperature_k=300.;
    CellConfig c;c.id="wall";c.is_battery=false;c.dimensions_m={.1,.1,.1};c.thermal_divisions={1,1,1};
    c.heat_capacity_j_per_k=100;c.initial_temperature_k=300;s.cells={c};
    s.ports.push_back({"out","wall",0,"vent","+z",.01,{0,0,.05},{0,0,1}});
    for(int i=0;i<3;++i) {
        GasCompartment g;g.id="gas"+std::to_string(i);g.volume_m3=i==2?.001:1e-5;g.initial_temperature_k=300;
        g.initial_mass_kg=(i==0?100001.:100000.)*g.volume_m3/(287*300);
        if(i==2) {g.initial_mass_kg=.001;g.initial_chemical_j=100;g.initial_oxidizer_mass_fraction=.232;g.reaction_rate_per_s=1000;}
        s.gas_compartments.push_back(g);
    }
    GasChannel ch;ch.id="AB";ch.from=0;ch.to=1;ch.shell_object=0;ch.port_id="out";ch.area_m2=1e-4;s.gas_channels={ch};return s;
}
void gas_cap_test() {
    auto s=gas_fixture();auto mesh=build_thermal_mesh(s);RuntimeParameters runtime;double dt[2],mass[2];
    for(int k=0;k<2;++k) {
        CudaSolver solver(s,mesh);dt[k]=solver.step(0,k?.001:.1,.4,runtime);auto d=solver.process_diagnostics();mass[k]=d.channels[0].transferred_kg;
        require(std::abs(d.mass_residual_kg)<1e-12 && std::abs(d.thermal_residual_j)<1e-8,"Gas ledger failed");
    }
    const double equilibrium_mass=1./((1+287./1000)*287*300*(1/1e-5+1/1e-5));
    const double raw=gas_orifice_flow(100001.,100000.,300.,1e-4,.7,s.gas_model);
    const auto& g=s.gas_model;
    const double expected_dt=std::min(5e-6,gas_pressure_dt(100001.,100000.,raw,equilibrium_mass,
        g.pressure_relaxation_fraction,g.pressure_absolute_tolerance_pa,g.pressure_relative_tolerance));
    require(dt[0]==dt[1] && std::abs(dt[0]-expected_dt)<1e-14*expected_dt,"Pressure/reactor must impose identical accepted dt");
    require(std::abs(mass[0]-mass[1])<1e-12*std::max(mass[0],mass[1]),"Gas transfer depends on rejected preliminary dt");
    require(std::abs(mass[0]-std::min(raw*expected_dt,equilibrium_mass))<1e-20,"Accepted pressure-limited transfer is wrong");
}
void pressure_timestep_test() {
    const auto dt=[](double flow,double cap,double fraction=.2) {
        return gas_pressure_dt(2e5,1e5,flow,cap,fraction,1.,1e-5);
    };
    require(std::abs(dt(.1,.01)-.02)<1e-15,"Pressure relaxation timescale");
    require(dt(.2,.01)==dt(.1,.01)/2,"Wider channel must shorten dt");
    require(dt(.1,.02)==2*dt(.1,.01),"Larger pressure inventory must lengthen dt");
    require(dt(.1,.01,.1)==dt(.1,.01)/2,"Numerical refinement control");
    require(gas_pressure_dt(1e5,2e5,-.1,.01,.2,1.,1e-5)==dt(.1,.01),"Direction dependence of time bound");
    require(dt(0.,0.)==1e300,"Closed path constrains dt");
    require(gas_pressure_dt(1e5,1e5,0.,0.,.2,1.,1e-5)==1e300,"Equilibrium constrains dt");
    require(gas_pressure_dt(100000.001,100000.,1e-5,1e-10,.2,1.,0.)>1e-3,"Absolute tolerance must relax near-equilibrium stiffness");
    auto s=gas_fixture();
    for(int i=0;i<4;++i) {
        auto bad=s;
        if(i==0)bad.gas_model.pressure_relaxation_fraction=0;
        if(i==1)bad.gas_model.pressure_relaxation_fraction=.6;
        if(i==2)bad.gas_model.pressure_absolute_tolerance_pa=0;
        if(i==3)bad.gas_model.pressure_relative_tolerance=-1;
        bool rejected=false;try{build_thermal_mesh(bad);}catch(const std::exception&){rejected=true;}
        require(rejected,"Invalid pressure controls accepted");
    }
}
void reservoir_degree_test() {
    auto s=gas_fixture();s.gas_model.split_transport=true;s.gas_compartments.resize(2);
    auto& a=s.gas_compartments[0];auto& b=s.gas_compartments[1];
    a.volume_m3=1e-6;a.initial_mass_kg=100100.*a.volume_m3/(287.*300.);
    b.volume_m3=.1;b.initial_mass_kg=100000.*b.volume_m3/(287.*300.);
    const auto advance=[](const Scenario& scene) {
        const auto mesh=build_thermal_mesh(scene);CudaSolver solver(scene,mesh);RuntimeParameters runtime;
        const double dt=solver.step(0,1e-3,.4,runtime);const auto d=solver.process_diagnostics();
        require(std::abs(d.mass_residual_kg)<1e-12 && std::abs(d.thermal_residual_j)<1e-8,"Hub transfer ledger");
        return dt;
    };
    const double first=advance(s);
    const auto chamber=b;
    for(int i=0;i<100;++i) {
        auto extra=chamber;extra.id="side"+std::to_string(i);s.gas_compartments.push_back(extra);
        auto edge=s.gas_channels[0];edge.id=extra.id;edge.from=1;edge.to=i+2;s.gas_channels.push_back(edge);
    }
    const double second=advance(s);
    // The chamber pressure responds 10^5 times less than the source pressure.
    // Its extra ports must not impose a 100-fold smaller source timestep.
    require(std::abs(second/first-1)<.002,"Reservoir degree spuriously throttles source pore");
}
void rupture_capacity_test() {
    Scenario s;CellConfig c;c.id="cell";c.dimensions_m={.02,.02,.02};c.thermal_divisions={1,1,1};
    c.heat_capacity_j_per_k=100;c.initial_temperature_k=400;s.cells={c};
    ShellConfig sh;sh.contents_id=c.id;sh.material_id="case";sh.thickness_m=.001;sh.density_kg_m3=2000;
    sh.specific_heat_j_kg_k=1000;sh.conductivity_w_per_m_k={1,1,1};sh.contents_contact_h=10;sh.seam_contact_h=10;add_physical_shell(s,sh);
    auto& law=s.cells[6].heat_capacity_law;law.reference_temperature_k=300;law.factor.count=2;
    law.factor.temperature_k[0]=200;law.factor.temperature_k[1]=1000;law.factor.value[0]=law.factor.value[1]=1;law.damage_factor_at_one=.5;
    GasCompartment g;g.id="inside";g.owner_cell=0;g.volume_m3=1e-6;g.initial_temperature_k=400;
    g.initial_mass_kg=3e5*g.volume_m3/(287*400);s.gas_compartments={g};
    GasChannel ch;ch.id="rupture";ch.from=0;ch.to=-1;ch.shell_object=6;ch.port_id="cell:case:+z:outer";
    ch.area_m2=1e-8;ch.opening_pressure_pa=1000;ch.rupture=true;s.gas_channels={ch};
    auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh);RuntimeParameters runtime;
    std::vector<ThermalState> before,nodes;solver.download_field(before);solver.step(0,1e-6,.4,runtime);solver.download_field(nodes);
    const int i=mesh.object_offsets[6];const auto& n=nodes[i];
    require(n.damage==1 && n.mode==CellMode::Spent,"Rupture fixture did not open");
    require(n.stored_energy_j==before[i].stored_energy_j,"Rupture created heat instead of changing capacity");
    const double expected=thermal_temperature(law,mesh.volumes[i].capacity_j_per_k,n.stored_energy_j,n.damage);
    require(std::abs(expected-n.temperature_k)<1e-10 && std::abs(expected-500)<1e-10,"Rupture left stale temperature after changing D");
    require(std::abs(solver.process_diagnostics().thermal_residual_j)<1e-8,"Rupture violated global energy balance");
}
int main(){int failures=0;for(auto test:{budget_test,gas_cap_test,pressure_timestep_test,reservoir_degree_test,rupture_capacity_test})try{test();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';++failures;}
    if(!failures)std::cout<<"Gas accepted-step independence and rupture constitutive consistency pass\n";return failures?1:0;}

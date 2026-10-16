#include "battery_native/simulation.hpp"
#include "battery_native/mass_mesh.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
bool cpu_backend=false;
bool implicit_backend=false;
bool mc_backend=false;
bool batched_backend=false;
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
void balance(const Simulation& s) {
    const auto d=s.process_diagnostics();
    require(std::abs(d.mass_residual_kg)<1e-11,"split mass balance");
    require(std::abs(d.thermal_residual_j)<1e-7,"split energy balance");
    require(std::abs(d.chemical_residual_j)<1e-7,"split chemical balance");
    require(std::abs(d.oxidizer_residual_kg)<1e-11,"split oxidizer balance");
    const auto& states=s.thermal_states();const auto& mesh=s.thermal_mesh();
    for(std::size_t i=0;i<d.gas.size();++i) {
        const auto& g=s.scenario().gas_compartments[i];if(g.thermal_cell<0)continue;
        const auto& solid=states[mesh.object_offsets[g.thermal_cell]+g.thermal_volume];
        require(solid.temperature_k==d.gas[i].temperature_k,"local phases not in equilibrium");
        require(solid.mobile_capacity_j_per_k==d.gas[i].mass_kg*s.scenario().gas_model.cv_j_per_kg_k,"stale mobile capacity");
    }
}
Scenario base(double temperature=300) {
    Scenario s;s.settings={1.,.001,.1,.4};s.gas_model.split_transport=true;s.gas_model.reference_temperature_k=300;
    s.gas_model.cpu_local_transport=cpu_backend;
    s.gas_model.cuda_kawasaki_transport=mc_backend;
    s.gas_model.cuda_batched_transport=batched_backend;
    s.gas_model.kawasaki_packet_fraction=.0078125;
    s.gas_model.implicit_local_transport=implicit_backend;
    s.gas_model.local_exchange_interval_s=.001;
    CellConfig c;c.id="matrix";c.is_battery=false;c.dimensions_m={.1,.1,.1};c.thermal_divisions={1,1,1};
    c.heat_capacity_j_per_k=10;c.initial_temperature_k=temperature;s.cells={c};
    GasCompartment g;g.id="pore";g.volume_m3=1e-4;g.initial_mass_kg=.001;g.initial_temperature_k=temperature;g.thermal_cell=0;
    s.gas_compartments={g};s.ports.push_back({"out","matrix",0,"vent","+z",.01,{0,0,.05},{0,0,1}});
    return s;
}
GasChannel vent() {
    GasChannel c;c.id="path";c.from=0;c.to=-1;c.shell_object=0;c.port_id="out";c.area_m2=5e-7;c.outside_pressure_pa=0;return c;
}
void analytic() {
    auto s=base();auto& g=s.gas_compartments[0];g.initial_chemical_j=100;g.initial_oxidizer_mass_fraction=.5;g.reaction_rate_per_s=2;
    Simulation sim(s);sim.run_to_end();balance(sim);
    const double heat=100*(-std::expm1(-2.));
    require(std::abs(sim.cells()[0].temperature_k-(300+heat/11))<1e-8,"LTE finite-fuel analytic temperature");
    require(std::abs(sim.cells()[0].heat_capacity_j_per_k-11)<1e-12,"macro mixture capacity");
    if(mc_backend) {
        auto unsupported=s;unsupported.gas_channels={vent()};
        bool rejected=false;try{Simulation bad(unsupported);}catch(const std::exception&){rejected=true;}
        require(rejected,"MC silently accepted external sink");
        unsupported=s;unsupported.cells[0].heat_capacity_law.factor.count=2;
        auto& table=unsupported.cells[0].heat_capacity_law.factor;
        table.temperature_k[0]=300;table.temperature_k[1]=600;table.value[0]=1;table.value[1]=2;
        rejected=false;try{Simulation bad(unsupported);}catch(const std::exception&){rejected=true;}
        require(rejected,"MC silently approximated nonlinear capacity");
        return;
    }
    auto b=base(500);b.settings.duration_s=.2;b.gas_channels={vent()};Simulation blow(b);blow.run_to_end();balance(blow);
    const auto d=blow.process_diagnostics();
    const double exact=500*std::pow((10+1000*d.gas[0].mass_kg)/11.,.287);
    require(d.escaped_mass_kg>0 && blow.cells()[0].temperature_k<500,"LTE blowdown must cool the matrix");
    require(std::abs(blow.cells()[0].temperature_k-exact)<.02,"LTE enthalpy blowdown analytic solution");
    require(blow.cells()[0].heat_capacity_j_per_k<11,"lost mass must reduce capacity");
    std::cout<<"blowdown: T="<<d.gas[0].temperature_k<<" analytic="<<exact<<" mass="<<d.gas[0].mass_kg<<'\n';
    auto heating=base();heating.settings={.1,.1,.1,.4};
    heating.cells[0].conductivity_w_per_m_k={1e12,1e12,1e12};
    heating.cells[0].damage_onset_k=304.7;heating.cells[0].damage_rate_at_onset_per_s=1;
    heating.boundaries.push_back({"heater","out",0,400,10});
    Simulation warm(heating);const double heater_g=warm.thermal_link_states().at(0).conductance_w_per_k;
    warm.run_to_end();balance(warm);
    require(warm.step_count()==1,"analytic heating fixture requires one explicit substep");
    require(std::abs(warm.cells()[0].temperature_k-(300+heater_g*100*.1/11))<1e-10,"heat operator/LTE energy sharing");
    require(warm.cells()[0].damage==0,"damage evaluated at provisional solid-only temperature");
    auto nonlinear=s;auto& law=nonlinear.cells[0].heat_capacity_law;
    law.reference_temperature_k=300;law.factor.count=2;
    law.factor.temperature_k[0]=300;law.factor.temperature_k[1]=600;law.factor.value[0]=1;law.factor.value[1]=2;
    Simulation tabulated(nonlinear);tabulated.run_to_end();balance(tabulated);
    const double rise=2*heat/(11+std::sqrt(121+heat/15));
    require(std::abs(tabulated.cells()[0].temperature_k-(300+rise))<1e-8,"nonlinear LTE energy inversion");
}
Scenario receiver() {
    auto s=base(500);s.settings.duration_s=.1;s.settings.max_dt_s=.0005;
    auto c=s.cells[0];c.id="membrane";c.position_m.z=.2;c.initial_temperature_k=300;c.heat_capacity_j_per_k=.5;
    c.damage_onset_k=301;c.damage_rate_at_onset_per_s=.1;c.damage_temperature_scale_k=100;s.cells.push_back(c);
    auto g=s.gas_compartments[0];g.id="membrane-pore";g.thermal_cell=1;g.initial_temperature_k=300;g.initial_mass_kg=.0001;
    g.initial_oxidizer_mass_fraction=.232;g.reaction_rate_per_s=2;s.gas_compartments.push_back(g);
    s.gas_compartments[0].initial_chemical_j=100;
    auto ch=vent();ch.to=1;s.gas_channels={ch};return s;
}
void independent_geometry() {
    auto s=receiver();
    // A pathwise same-seed comparison is not a stochastic convergence test.
    // Isolate integration error here; ensemble checks cover tau-leap noise.
    if(batched_backend)s.gas_model.batch_mean_threshold=0;
    Simulation open(s);open.run_to_end();balance(open);auto d=open.process_diagnostics();
    require(open.thermal_mesh().links.empty(),"Fixture accidentally conducts heat");
    require(open.cells()[1].temperature_k>301 && open.cells()[1].damage>0,"advected energy must heat and damage receiving matrix");
    require(d.gas[1].chemical_j>0 && d.gas[1].released_energy_j>0,"advected fuel does not react in receiver");
    require(d.gas[1].mass_kg>s.gas_compartments[1].initial_mass_kg,"incoming gas mass disappeared in membrane");
    auto refined=s;refined.settings.max_dt_s*=.5;refined.gas_model.max_fraction_per_step*=.5;
    refined.gas_model.max_temperature_step_k*=.5;Simulation fine(refined);fine.run_to_end();balance(fine);
    require(std::abs(fine.cells()[1].temperature_k-open.cells()[1].temperature_k)<.15,"split step refinement");
    auto closed=s;closed.gas_channels[0].opening_pressure_pa=1e12;Simulation blocked(closed);blocked.run_to_end();balance(blocked);
    require(blocked.cells()[1].temperature_k==300 && blocked.process_diagnostics().channels[0].transferred_kg==0,"impermeable barrier leaks");
    // A pressure gradient can carry matter toward hotter material. No temperature
    // gradient is used as a gate, and an opened internal path is bidirectional.
    auto reverse=s;reverse.gas_compartments[1].initial_mass_kg=.004;Simulation back(reverse);back.step();balance(back);
    require(back.process_diagnostics().channels[0].mass_flow_kg_s<0,"pressure-driven backflow into hotter receiver blocked");
    auto equal=s;equal.gas_compartments[1].initial_mass_kg=.001*500/300.;
    equal.gas_compartments[0].initial_chemical_j=0;Simulation still(equal);still.step();balance(still);
    require(std::abs(still.process_diagnostics().channels[0].mass_flow_kg_s)<1e-12,"temperature difference drives spurious mass flow at equal pressure");
    auto near=s;near.cells[0].initial_temperature_k=near.gas_compartments[0].initial_temperature_k=300;
    near.cells[1].initial_temperature_k=near.gas_compartments[1].initial_temperature_k=1000;
    near.gas_compartments[0].initial_mass_kg=.001*(1+1./861000);
    near.gas_compartments[1].initial_mass_kg=.0003;near.gas_compartments[0].initial_chemical_j=0;
    Simulation near_balance(near);near_balance.step();balance(near_balance);
    const auto pressures=near_balance.process_diagnostics();
    require(pressures.gas[0].pressure_pa>=pressures.gas[1].pressure_pa-1e-4,"gas-only cap overshoots pressure equilibrium in hot matrix");
}
void local_production() {
    auto s=base(500);s.settings.duration_s=.1;s.cells[0].is_battery=true;s.cells[0].thermal_divisions={2,1,1};
    auto& p=s.cells[0].conversion;p.enabled=true;p.mass_kg=.004;p.gas_compartment=0;
    p.degradation_rate_per_s=3;p.degradation_reference_k=500;p.reaction_rate_per_s=2;p.reaction_activation_j_per_mol=80000;
    p.net_heat_j_per_kg=200000;p.gas_yield=p.gas_yield_max=.1;p.gas_chemical_fraction=.25;
    auto& g=s.gas_compartments[0];g.owner_cell=0;g.reactant_density_kg_m3=1000;g.residue_density_kg_m3=1500;
    auto other=g;other.id="second-pore";other.owner_cell=-1;other.thermal_volume=1;s.gas_compartments.push_back(other);
    Simulation sim(s);sim.run_to_end();balance(sim);const auto d=sim.process_diagnostics();const auto& nodes=sim.thermal_states();
    for(int i=0;i<2;++i) {
        const double produced=.1*(.002-nodes[i].reactant_kg-nodes[i].activated_kg);
        require(std::abs(d.gas[i].mass_kg-(.001+produced))<1e-12,"production not delivered exactly once to its local volume");
        require(d.gas[i].free_volume_m3>1e-4,"condensed conversion does not free pore volume");
    }
    require(std::abs(d.gas[0].mass_kg-d.gas[1].mass_kg)<1e-12,"uniform source broken by local mapping");
}
void invalid_geometry() {
    const auto reject=[](Scenario s) {bool failed=false;try{build_mass_mesh(s,build_thermal_mesh(s));}catch(const std::exception&){failed=true;}
        require(failed,"invalid mixture geometry accepted");};
    auto s=base();s.gas_model.split_transport=false;reject(s);
    s=base();s.gas_compartments.push_back(s.gas_compartments[0]);reject(s);
    s=base();s.gas_compartments[0].thermal_volume=1;reject(s);
    s=base();s.gas_compartments[0].volume_m3=.01;reject(s);
    s=base();s.gas_compartments[0].initial_temperature_k=301;reject(s);
    s=base();s.cells[0].isothermal=true;reject(s);
    s=base();s.gas_compartments[0].reactant_density_kg_m3=1000;reject(s);
    s=base();auto c=vent();c.surface_link_h_w_per_m2_k=1;s.gas_channels={c};reject(s);
}
// Nearly empty receiving pore (10^-9 of donor mass), not an exact vacuum:
// positive seed mass is required by the current gas energy representation.
// Internal heat conduction equalizes final T; pressure-only advection by itself
// can stop with different temperatures and need not produce equal masses.
Scenario equalization_fixture(double area,double cap=.02) {
    auto s=base(300);s.settings={40.,cap,.1,.4};
    s.cells[0].thermal_divisions={2,1,1};s.cells[0].conductivity_w_per_m_k={10,10,10};
    s.gas_compartments[0].initial_mass_kg=.001;
    auto g=s.gas_compartments[0];g.id="empty";g.thermal_volume=1;g.initial_mass_kg=1e-12;
    s.gas_compartments.push_back(g);
    GasChannel path;path.id="permeable";path.from=0;path.to=1;path.shell_object=0;
    path.internal=true;path.area_m2=area;s.gas_channels={path};return s;
}
void equalization() {
    double early[2]{},late[2]{};
    for(int k=0;k<2;++k) {
        Simulation sim(equalization_fixture(k?4e-7:1e-7));
        while(sim.time_s()<.2)sim.step();
        early[k]=sim.process_diagnostics().channels[0].transferred_kg;
        sim.run_to_end();balance(sim);const auto d=sim.process_diagnostics();
        late[k]=d.channels[0].transferred_kg;
        require(std::abs(d.gas[0].temperature_k-d.gas[1].temperature_k)<1e-5,"Final gas temperatures differ");
        require(std::abs(d.gas[0].pressure_pa-d.gas[1].pressure_pa)<.05,"Open path stops before equal pressure");
        require(std::abs(late[k]-(.001-1e-12)/2)<1e-10,"Equal volumes do not exchange half of initial mass difference");
    }
    require(early[1]>early[0]*1.5,"Permeability does not control equalization speed");
    Simulation fine(equalization_fixture(1e-7,.01));fine.run_to_end();balance(fine);
    require(std::abs(fine.process_diagnostics().channels[0].transferred_kg-late[0])<1e-10,"Equilibrium depends on timestep");
    std::cout<<"equalization: early transfer="<<early[0]<<" / "<<early[1]
             <<" kg, final transfer="<<late[0]<<" / "<<late[1]<<" kg (target 0.0005)\n";
}
void mc_transactions() {
    auto s=receiver();const auto mesh=build_thermal_mesh(s);
    CudaSolver direct(s,mesh,64),retried(s,mesh,256);RuntimeParameters r;double t=0;
    for(int k=0;k<20;++k) {
        const double dt=direct.step(t,.0005,.4,r);
        const auto before=retried.process_diagnostics();
        retried.prepare_step(t,.0005,.4,r);retried.discard_step();
        retried.prepare_step(t,.0001,.4,r);retried.discard_step();
        const auto untouched=retried.process_diagnostics();
        require(before.gas[0].mass_kg==untouched.gas[0].mass_kg,"MC prepare published mass");
        require(dt==retried.prepare_step(t,.0005,.4,r),"MC retry changed outer time");
        retried.commit_step();t+=dt;
    }
    const auto a=direct.process_diagnostics(),b=retried.process_diagnostics();
    for(std::size_t i=0;i<a.gas.size();++i) {
        require(a.gas[i].mass_kg==b.gas[i].mass_kg && a.gas[i].energy_j==b.gas[i].energy_j &&
            a.gas[i].chemical_j==b.gas[i].chemical_j,"MC retry or tile changed stochastic trajectory");
    }
    require(a.channels[0].transferred_kg>0 && a.channels[0].transferred_kg==b.channels[0].transferred_kg,"MC active-channel ledger rollback");
    require(direct.profile().kawasaki_events>0 && direct.profile().kawasaki_events==retried.profile().kawasaki_events,"MC event clock rollback");
    direct.reset();retried.reset();direct.step(0,.0005,.4,r);retried.step(0,.0005,.4,r);
    require(direct.process_diagnostics().gas[0].energy_j==retried.process_diagnostics().gas[0].energy_j,"MC clock reset");
}
int main(int argc,char** argv){try {
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        cpu_backend|=arg=="--cpu-local";implicit_backend|=arg=="--implicit";mc_backend|=arg=="--cuda-kawasaki";
        batched_backend|=arg=="--cuda-batched";mc_backend|=batched_backend;
    }
    analytic();independent_geometry();local_production();invalid_geometry();equalization();
    if(mc_backend)mc_transactions();
    std::cout<<"split transport: analytic LTE/blowdown, separate geometry, backflow, local production, pore volume, ledgers and refinement pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

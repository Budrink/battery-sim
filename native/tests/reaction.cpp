#include "battery_native/simulation.hpp"
#include "battery_native/reaction.hpp"
#include "battery_native/failure.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool x,const char* why) {if(!x)throw std::runtime_error(why);}
void balanced(const Simulation& sim) {
    require(std::abs(sim.energy_balance_residual_j())<1e-7,"Thermal energy ledger mismatch");
    require(std::abs(sim.chemical_balance_residual_j())<1e-7,"Chemical reserve ledger mismatch");
    for(const auto& c:sim.cells())require(c.chemical_energy_remaining_j>=0 && c.vent_heat_out_j<=c.released_energy_j+1e-9,"Negative fuel or double vent heat");
}
Scenario scene() {
    Scenario s;s.name="Kinetic test";s.settings={4.,.2,1.,.4};
    CellConfig c;c.id="C";c.dimensions_m={.03,.03,.03};c.thermal_divisions={3,3,3};
    c.heat_capacity_j_per_k=27;c.conductivity_w_per_m_k={.01,.01,.01};c.initial_temperature_k=500;
    c.runaway_energy_j=2700;c.runaway_pulse.profile=RunawayProfile::Kinetic;
    auto& k=c.reaction_kinetics;k.reference_temperature_k=500;k.rate_at_reference_per_s=.1;
    k.activation_energy_j_per_mol=0;k.damage_sensitivity=0;
    s.cells.push_back(c);s.primary_failures={{0,"C",0.,-1}};return s;
}
void add_outlet(Scenario& s) {
    auto& c=s.cells[0];c.energy_outlet.enabled=true;c.energy_outlet.heat_fraction=.3;
    c.energy_outlet.ejected_energy_per_released_energy=.4;c.energy_outlet_port="vent";
    PortConfig p;p.id="vent";p.cell_id="C";p.kind="vent";p.surface="+z";p.area_m2=.0009;p.centroid_m={0,0,.015};p.normal={0,0,1};s.ports.push_back(p);
}
void add_cooler(Scenario& s,double g) {
    PortConfig p;p.id="cool";p.cell_id="C";p.kind="cooling";p.surface="+x";p.area_m2=.0009;p.centroid_m={.015,0,0};p.normal={1,0,0};s.ports.push_back(p);
    AmbientBoundary b;b.id="bath";b.port_id=p.id;b.temperature_k=300;b.conductance_w_per_k=g;s.boundaries.push_back(b);
}
int main(){try {
    // Independent analytic oracle: constant rate, spatial CUDA, variable T.
    auto a=scene();Simulation exact(a);
    require(exact.cells()[0].damage==.2 && exact.cells()[0].released_energy_j==0,"Ignition is an instantaneous destruction/energy jump");
    while(!exact.finished()) {
        exact.step();const double released=2700*(-std::expm1(-.1*exact.time_s()));
        require(std::abs(exact.released_energy_j()-released)<1e-8,"Reaction differs from analytic finite-reserve solution");
        require(std::abs(exact.cells()[0].temperature_k-(500+released/27))<1e-8,"Source heat is wrong");balanced(exact);
    }
    const double final=exact.cells()[0].temperature_k;exact.reset();exact.run_to_end();
    require(exact.cells()[0].temperature_k==final,"Reset retains reaction history");
    auto inactive=scene();inactive.primary_failures.clear();Simulation waiting(inactive);waiting.run_to_end();
    require(waiting.released_energy_j()==0 && waiting.cells()[0].temperature_k==500,"Unactivated chemistry burns");
    auto automatic=inactive;automatic.cells[0].damage_onset_k=500;automatic.cells[0].damage_rate_at_onset_per_s=.2;
    Simulation trigger(automatic);trigger.run_to_end();require(trigger.cells()[0].runaway_started_s>0 && trigger.released_energy_j()>0,"Damage threshold does not activate chemistry");
    auto zero=scene();Simulation paused(zero);paused.runtime().runaway_power_scale=0;paused.step();
    require(paused.released_energy_j()==0 && paused.cells()[0].mode==CellMode::Quenched,"Paused reaction consumes fuel");
    paused.runtime().runaway_power_scale=1;paused.step();require(paused.released_energy_j()>0,"Quenched reaction cannot resume");balanced(paused);
    // Cooling suppresses the reaction with fuel left; subsequent reheat restarts it.
    auto cold=scene();cold.settings={14.,.02,1.,.4};cold.cells[0].thermal_divisions={1,1,1};
    cold.cells[0].conductivity_w_per_m_k={1000,1000,1000};cold.cells[0].reaction_kinetics.activation_energy_j_per_mol=80000;
    add_cooler(cold,300.);cold.boundaries[0].temperature_profile={{0,300},{5,300},{6,600},{14,600}};
    Simulation quench(cold);while(quench.time_s()<4-1e-10)quench.step(4-quench.time_s());
    const double reserve=quench.chemical_energy_remaining_j();
    require(quench.cells()[0].mode==CellMode::Quenched && reserve>2600 && !quench.all_batteries_destroyed(),"Cooling did not quench with fuel remaining");
    const double cold_power=quench.cells()[0].chemical_power_w;bool resumed=false;
    while(!quench.finished()) {quench.step();resumed|=quench.cells()[0].mode==CellMode::Runaway;balanced(quench);}
    require(resumed && quench.chemical_energy_remaining_j()<reserve*.1,"Reheating does not consume remaining fuel");
    std::cout<<"quenched reserve_j="<<reserve<<" cold_power_w="<<cold_power<<" reheated_remaining_j="<<quench.chemical_energy_remaining_j()<<'\n';
    // Effective discharge competes for the SAME reserve, with separate ledgers.
    auto vented=scene();vented.settings.duration_s=100.;add_outlet(vented);
    Simulation discharge(vented);discharge.run_to_end();balanced(discharge);
    require(discharge.vent_heat_out_j()>100 && discharge.ejected_chemical_energy_j()>100,"Outlet does not remove energy");
    require(std::abs(discharge.cells()[0].temperature_k-500-(discharge.released_energy_j()-discharge.vent_heat_out_j())/27)<1e-8,"Ejected chemical potential counted as heat");
    // Independent exact competing-rates check at an already open outlet.
    ThermalState node;node.mode=CellMode::Runaway;node.temperature_k=500;node.damage=.2;node.chemical_energy_remaining_j=100;node.released_energy_j=100;
    auto outlet=vented.cells[0].energy_outlet;outlet.ejected_energy_per_released_energy=1;outlet.heat_fraction=.25;
    const double heat=advance_reaction(a.cells[0].reaction_kinetics,outlet,node,200.,1.,1.);
    const double generated=50*(-std::expm1(-.2));
    require(std::abs(heat-.75*generated)<1e-12 && std::abs(node.ejected_chemical_energy_j-generated)<1e-12,"Competing release/export integral is wrong");
    outlet.closure_start_conversion=.5;outlet.closure_end_conversion=.8;
    require(outlet_opening(outlet,.3)==1 && outlet_opening(outlet,.9)==0,"Outlet cannot open and close with conversion");
    // State-based anisotropy, with one activated hot core volume.
    auto spatial=scene();auto& sc=spatial.cells[0];sc.reaction_structure={true,{12.,3.,.5},{.02,.1,.2},.1};
    spatial.primary_failures[0].thermal_volume=13;sc.initial_temperature_field_k.assign(27,300.);sc.initial_temperature_field_k[13]=500.;
    Simulation tensor(spatial);const auto old=tensor.thermal_states();tensor.step();const auto links=tensor.thermal_link_states();
    for(std::size_t j=0;j<links.size();++j) {
        const auto& e=tensor.thermal_mesh().links[j];
        const double factor=reaction_bulk_factor(old[e.a],old[e.b],e.geometry_axis);
        const double expected=e.law.background_conductance_w_per_k*factor;
        require(std::abs(links[j].conductance_w_per_k-expected)<1e-12,"Internal factor averaged, isotropized or ignored");
        require(links[j].conductance_w_per_k<=e.maximum_g+1e-12,"Reaction structure violates CFL envelope");
    }
    require(reaction_bulk_factor(old[13],old[14],0,1)!=reaction_bulk_factor(old[13],old[14],0,-1),"Opposite structural directions averaged");
    tensor.run_to_end();balanced(tensor);
    // Same state at different ages must have the same conductivity: no timer.
    ThermalState state=old[13],later=state;later.runaway_started_s=1234;
    refresh_reaction(sc.reaction_kinetics,sc.reaction_structure,state,100,1);
    refresh_reaction(sc.reaction_kinetics,sc.reaction_structure,later,100,1);
    require(state.conductivity_factor[0]==later.conductivity_factor[0],"Structure depends on wall-clock age");
    state.chemical_energy_remaining_j=0;state.mode=CellMode::Spent;
    refresh_reaction(sc.reaction_kinetics,sc.reaction_structure,state,100,1);
    require(std::abs(state.conductivity_factor[0]-.02)<1e-14,"Spent material does not reach configured residue");
    // Boundary half-volume and CFL must include exactly the same local factor.
    auto boundary=scene();boundary.cells[0].thermal_divisions={1,1,1};boundary.cells[0].reaction_structure={true,{100.,1.,1.},{.1,1.,1.},.1};
    boundary.cells[0].reaction_kinetics.rate_at_reference_per_s=1e-9;boundary.cells[0].reaction_structure.activity_rate_per_s=1e-9;
    boundary.settings={1000,1000,1000,.4};add_cooler(boundary,100.);
    Simulation bound(boundary);const auto e=bound.thermal_mesh().links[0];
    const double g=1/(1/(e.law.background_conductance_w_per_k*e.weight)+e.series_resistance_k_per_w/100.);
    bound.step();require(std::abs(bound.last_dt_s()-.4*27/g)<1e-8,"Boundary CFL ignores reaction peak");
    require(std::abs(bound.thermal_link_states()[0].conductance_w_per_k-g)<1e-12,"Boundary uses intact half-volume");balanced(bound);
    // A small isothermal neighbor must also see the reacting material's high-k
    // half-volume in its OWN CFL bound, not only in the source volume's bound.
    auto mixed=boundary;mixed.ports.clear();mixed.boundaries.clear();
    CellConfig sink;sink.id="sink";sink.is_battery=false;sink.isothermal=true;sink.dimensions_m={.03,.03,.03};
    sink.thermal_divisions={1,1,1};sink.position_m={.03,0,0};sink.heat_capacity_j_per_k=.01;sink.initial_temperature_k=300;
    mixed.cells.push_back(sink);
    mixed.ports={{"a","C",0,"contact","+x",.0009,{.015,0,0},{1,0,0}},
                 {"b","sink",1,"contact","-x",.0009,{.015,0,0},{-1,0,0}}};
    InterfaceConfig contact;contact.id="C-sink";contact.cell_b=1;contact.port_a="a";contact.port_b="b";
    contact.law.background_conductance_w_per_k=100;contact.law.destroyed_background_conductance_w_per_k=100;mixed.interfaces.push_back(contact);
    Simulation neighbor(mixed);const auto face=neighbor.thermal_mesh().links[0];
    const double contact_g=1/(1/100.+face.series_resistance_k_per_w/100.);
    neighbor.step();require(std::abs(neighbor.last_dt_s()-.4*.01/contact_g)<1e-10,"Isothermal neighbor CFL ignores reacting half-volume");
    require(std::abs(neighbor.cells()[1].temperature_k-380)<1e-9,"Isothermal neighbor overshoots conservative exchange");balanced(neighbor);
    // Temperature feedback refinement (not just an exactly integrated source).
    auto refine=scene();refine.cells[0].thermal_divisions={1,1,1};refine.settings.duration_s=2;
    refine.cells[0].reaction_kinetics.activation_energy_j_per_mol=60000;
    double temperatures[3]{};
    for(int i=0;i<3;++i) {auto r=refine;r.settings.max_dt_s=.02/std::pow(2.,i);r.cells[0].reaction_kinetics.max_temperature_rise_per_step_k=.2/std::pow(2.,i);
        Simulation sim(r);sim.run_to_end();balanced(sim);temperatures[i]=sim.cells()[0].temperature_k;}
    require(std::abs(temperatures[1]-temperatures[2])<std::abs(temperatures[0]-temperatures[1]) && std::abs(temperatures[1]-temperatures[2])<.2,"Kinetic feedback fails step refinement");
    std::cout<<"refinement_T_k="<<temperatures[0]<<','<<temperatures[1]<<','<<temperatures[2]<<'\n';
    auto variable=scene();auto& cp=variable.cells[0].heat_capacity_law;cp.factor.count=2;cp.factor.temperature_k[0]=300;cp.factor.temperature_k[1]=600;cp.factor.value[0]=1;cp.factor.value[1]=2;cp.damage_factor_at_one=.8;
    Simulation capacity(variable);capacity.run_to_end();balanced(capacity);
    auto invalid=scene();invalid.cells[0].energy_outlet.enabled=true;
    bool rejected=false;try {Simulation bad(invalid);}catch(const std::exception&){rejected=true;}
    require(rejected,"Invented/missing vent accepted");
    invalid=scene();invalid.cells[0].isothermal=true;rejected=false;
    try {Simulation bad(invalid);}catch(const std::exception&){rejected=true;}require(rejected,"Unsupported isothermal kinetics accepted");
    std::cout<<"CUDA reaction, quench/reheat, anisotropy, outlet, CFL, refinement and both energy ledgers PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

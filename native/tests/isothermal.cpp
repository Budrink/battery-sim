#include "battery_native/simulation.hpp"
#include "battery_native/playback.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
Scenario pair_fixture() {
    Scenario s;s.name="isothermal pair";s.settings={0.5,0.001,0.1,0.2};
    CellConfig c;c.id="hot";c.dimensions_m={2,1,1};c.thermal_divisions={8,3,2};
    c.isothermal=true;c.heat_capacity_j_per_k=2;c.initial_temperature_k=400;c.position_m={-1.01,0,0};
    s.cells.push_back(c);c.id="cold";c.initial_temperature_k=300;c.position_m={1.01,0,0};s.cells.push_back(c);
    s.ports={{"a","hot",0,"contact","+x",1,{-0.01,0,0},{1,0,0}},
             {"b","cold",1,"contact","-x",1,{0.01,0,0},{-1,0,0}}};
    InterfaceConfig e;e.id="contact";e.port_a="a";e.port_b="b";e.cell_b=1;e.law.background_conductance_w_per_k=2;
    s.interfaces.push_back(e);return s;
}
void uniform_objects(const Simulation& sim) {
    for(const auto& c:sim.cells())require(c.min_temperature_k==c.max_temperature_k,"Isothermal field is exactly uniform");
}
int main(int argc,char**argv) {try {
    require(argc==2 || (argc==3 && std::string(argv[2])=="--quick"),"Expected Narada scenario path [--quick]");
    auto pair=pair_fixture();Simulation analytic(pair);analytic.run_to_end();uniform_objects(analytic);
    require(analytic.thermal_mesh().volumes.size()==96,"Isothermal mode retains the full volume mesh");
    require(std::abs(analytic.cells()[0].temperature_k-(350+50*std::exp(-1.0)))<0.025,"Analytical isothermal pair exchange");
    require(std::abs(analytic.interfaces()[0].conductance_w_per_k-2)<1e-12,"No finite internal resistance hidden in isothermal ports");
    require(std::abs(analytic.energy_balance_residual_j())<1e-9,"Isothermal pair conserves energy");
    analytic.reset();uniform_objects(analytic);require(std::abs(analytic.cells()[0].temperature_k-400)<1e-12,"Reset restores isothermal field");
    pair.settings={1,1,1,0.2};Simulation cfl(pair);cfl.step();
    require(std::abs(cfl.last_dt_s()-0.2)<1e-12,"Isothermal CFL uses whole capacity");
    for(auto&c:pair.cells)c.thermal_divisions={16,6,4};Simulation fine_cfl(pair);fine_cfl.step();
    require(std::abs(fine_cfl.last_dt_s()-cfl.last_dt_s())<1e-12,"Isothermal CFL does not depend on mesh resolution");

    auto mixed=pair_fixture();mixed.cells[1].isothermal=false;Simulation mixed_sim(mixed);
    require(std::abs(mixed_sim.interfaces()[0].conductance_w_per_k-1.6)<1e-12,"Mixed contact retains only spatial-side half resistance");
    mixed_sim.run_to_end();
    require(mixed_sim.cells()[0].min_temperature_k==mixed_sim.cells()[0].max_temperature_k,"Mixed system retains isothermal constraint");
    require(mixed_sim.cells()[1].max_temperature_k>mixed_sim.cells()[1].min_temperature_k,"Mixed system retains spatial gradients");
    require(std::abs(mixed_sim.energy_balance_residual_j())<1e-9,"Mixed interface conserves energy");

    Scenario heating;heating.name="mix before ignition";heating.settings={1,1,1,0.2};
    CellConfig c;c.id="solid";c.dimensions_m={1,1,1};c.thermal_divisions={8,8,8};c.isothermal=true;
    c.heat_capacity_j_per_k=8;c.initial_temperature_k=300;c.damage_onset_k=500;c.damage_rate_at_onset_per_s=1e9;c.runaway_energy_j=100;
    heating.cells={c};heating.ports={{"face","solid",0,"contact","+x",1,{0.5,0,0},{1,0,0}}};
    heating.boundaries={{"heater","face",0,800,1}};
    Simulation heat(heating);heat.run_to_end();uniform_objects(heat);
    require(std::abs(heat.cells()[0].temperature_k-362.5)<1e-12,"Surface heat is shared across the whole capacity");
    require(heat.cells()[0].mode==CellMode::Healthy,"Damage is evaluated after thermal equilibration, not at artificial surface hotspots");
    require(std::abs(heat.energy_balance_residual_j())<1e-9,"Boundary heating energy ledger");
    auto invalid=heating;invalid.cells[0].initial_temperature_field_k.assign(512,300);invalid.cells[0].initial_temperature_field_k[0]=400;
    bool rejected=false;try{Simulation bad(invalid);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Nonuniform isothermal initial field must not be silently discarded");
    invalid=heating;invalid.cells[0].has_bulk_connectivity=true;rejected=false;
    try{Simulation bad(invalid);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Finite bulk connectivity conflicts explicitly with the isothermal constraint");
    auto local=heating;local.boundaries.clear();local.cells[0].damage_rate_at_onset_per_s=0;
    local.cells[0].runaway_duration_s=0.1;local.primary_failures={{0,"solid",0,0}};
    Simulation local_source(local);local_source.run_to_end();uniform_objects(local_source);
    require(std::abs(local_source.released_energy_j()-100.0/512)<1e-12,"Local source retains its finite volume budget under mixing");
    require(local_source.cells()[0].mode==CellMode::PartiallySpent,"Mixing heat does not silently ignite all sources");
    require(std::abs(local_source.energy_balance_residual_j())<1e-9,"Local source conserves energy under mixing");
    if(argc==3){std::cout<<"CUDA isothermal quick fixtures: PASS\n";return 0;}

    // Historical single-temperature limit is a numerical regression, not an
    // experimental calibration target. Keep its equations and timing explicit.
    auto scenario=load_scenario(argv[1]);auto lumped=scenario;
    for(auto& c:lumped.cells){c.thermal_divisions={1,1,1};c.conductivity_w_per_m_k={1e12,1e12,1e12};}
    Simulation reference(lumped);reference.run_to_end();
    require(reference.runaway_order().size()==52,"Historical reference propagates to all 52 objects");
    require(std::abs(reference.cells()[19].runaway_started_s-5)<1e-8,"Primary schedule unchanged");
    for(int scale:{1,2}) {
        auto test=scenario;for(auto& c:test.cells){c.isothermal=true;for(auto&n:c.thermal_divisions)n*=scale;}
        Simulation spatial(test);double all_spent=-1;
        while(spatial.step())if(all_spent<0 && all_cells_destroyed(spatial))all_spent=spatial.time_s();
        uniform_objects(spatial);
        require(spatial.runaway_order()==reference.runaway_order(),"Full mesh reproduces historical propagation order");
        for(size_t i=0;i<spatial.cells().size();++i) {
            require(std::abs(spatial.cells()[i].runaway_started_s-reference.cells()[i].runaway_started_s)<1e-7,"Every ignition time matches the historical lumped limit");
            require(std::abs(spatial.cells()[i].temperature_k-reference.cells()[i].temperature_k)<1e-7,"Whole-object temperatures match the historical limit");
            require(spatial.cells()[i].mode==CellMode::Spent,"All 52 objects fully destroyed");
        }
        require(std::abs(all_spent-1190.2)<1e-7,"All-destroyed transition remains at 1190.2 s, independent of mesh");
        require(std::abs(spatial.released_energy_j()-52*4e6)<1e-3,"Full reaction energy preserved");
        require(std::abs(spatial.energy_balance_residual_j())<1e-5,"52-object isothermal energy ledger");
        std::cout<<"mesh="<<spatial.thermal_mesh().volumes.size()<<" all_destroyed_s="<<all_spent<<" residual_j="<<spatial.energy_balance_residual_j()<<'\n';
    }
    std::cout<<"CUDA isothermal constraint, mixed coupling, source budget, mesh-independent timing: PASS\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

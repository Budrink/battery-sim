#include "battery_native/playback.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Scenario fixture() {
    Scenario s;s.name="coarse battery / fine pad";s.settings={0.2,0.2,0.2,0.2,0.2};
    CellConfig b;b.id="battery";b.dimensions_m={1,1,1};b.position_m={-0.5,0,0};
    b.thermal_divisions={2,2,2};b.heat_capacity_j_per_k=100;b.initial_temperature_k=400;b.conductivity_w_per_m_k={20,20,20};
    CellConfig p;p.id="pad";p.is_battery=false;p.material_id="test-pad";p.dimensions_m={0.1,1,1};p.position_m={0.05,0,0};
    p.thermal_divisions={4,4,4};p.heat_capacity_j_per_k=1;p.initial_temperature_k=300;p.conductivity_w_per_m_k={0.2,0.2,0.2};
    s.cells={b,p};s.ports={{"b","battery",0,"contact","+x",1,{0,0,0},{1,0,0}},
                        {"p","pad",1,"contact","-x",1,{0,0,0},{-1,0,0}}};
    InterfaceConfig e;e.id="battery-pad";e.port_a="b";e.port_b="p";e.cell_b=1;e.law.background_conductance_w_per_k=10;s.interfaces={e};
    return s;
}
int main(int argc,char**argv){try{
    auto s=fixture();Simulation sim(s),manual(s);
    require(sim.thermal_mesh().volumes.size()==72,"Independent coarse/fine volume counts");
    require(sim.thermal_mesh().interface_links.size()==16,"One coarse face couples to multiple fine patches");
    double area_fraction=0;for(int i:sim.thermal_mesh().interface_links)area_fraction+=sim.thermal_mesh().links[i].weight;
    require(std::abs(area_fraction-1)<1e-12,"Contact area accounted exactly once");
    const double expected_g=1/(0.1+0.5/(2*20)+0.025/(2*0.2));
    require(std::abs(sim.interfaces()[0].conductance_w_per_k-expected_g)<1e-12,"Nonconforming contact uses area and both half resistances");
    sim.advance();require(sim.last_substep_count()>1 && std::abs(sim.time_s()-0.2)<1e-10,"One outer step contains stable physical substeps");
    while(!manual.finished())manual.step(0.2-manual.time_s());
    const auto&a=sim.thermal_states();const auto&b=manual.thermal_states();
    for(size_t i=0;i<a.size();++i){
        require(std::abs(a[i].temperature_k-b[i].temperature_k)<1e-12,"Outer stepping equals explicit substep sequence");
        require(a[i].temperature_k>=300 && a[i].temperature_k<=400,"CFL bounds all temperatures on nonconforming grid");
    }
    require(std::abs(sim.energy_balance_residual_j())<1e-8,"Coarse/fine energy balance");
    require(sim.cells()[0].stored_energy_j<40000 && sim.cells()[1].stored_energy_j>300,"Donor and receiver exchange energy");
    auto temporal=s;temporal.settings.max_dt_s=0.002;Simulation coarse_time(temporal);coarse_time.run_to_end();
    temporal.settings.max_dt_s=0.001;Simulation fine_time(temporal);fine_time.run_to_end();
    temporal.settings.max_dt_s=0.0005;Simulation reference_time(temporal);reference_time.run_to_end();
    double coarse_error=0,fine_error=0;
    for(size_t i=0;i<a.size();++i) {
        const double reference=reference_time.thermal_states()[i].temperature_k;
        coarse_error+=std::pow(coarse_time.thermal_states()[i].temperature_k-reference,2);
        fine_error+=std::pow(fine_time.thermal_states()[i].temperature_k-reference,2);
    }
    require(fine_error<coarse_error*0.3,"Nonconforming transient converges under timestep refinement");
    std::cout<<"Temporal RMS differences: "<<std::sqrt(coarse_error/a.size())<<" -> "<<std::sqrt(fine_error/a.size())<<" K\n";
    sim.reset();require(sim.outer_step_count()==0 && sim.last_substep_count()==0,"Reset clears outer-step accounting");
    auto refined=s;refine_environment_mesh(refined,2);
    require(refined.cells[0].thermal_divisions==s.cells[0].thermal_divisions,"Refinement preserves battery zone model");
    require(refined.cells[1].thermal_divisions==std::array<int,3>{8,8,8},"Refinement applies to passive material");

    auto coupling=fixture();coupling.cells[0].initial_damage=0.4;coupling.cells[1].initial_damage=0.8;
    auto&law=coupling.interfaces[0].law;law.endpoint_damage_weight=1;law.damage_conductance_factor_at_one=0.25;
    Simulation damaged(coupling);
    const double damaged_g=1/(1/(10*std::pow(0.25,0.6))+0.5/(2*20)+0.025/(2*0.2));
    require(std::abs(damaged.interfaces()[0].damage-0.6)<1e-12,"Connectivity sees adjacent accumulated material state");
    require(std::abs(damaged.interfaces()[0].conductance_w_per_k-damaged_g)<1e-12,"Material damage changes instantaneous flow conductance");

    auto passive=fixture();passive.cells[1].initial_damage=1;Simulation passive_sim(passive);
    require(passive_sim.cells()[1].mode==CellMode::Spent && passive_sim.runaway_order().empty(),"Destroyed passive material never reports battery runaway");
    require(!all_cells_destroyed(passive_sim),"Destroyed pad does not destroy the surviving battery");
    passive.cells[0].runaway_energy_j=100;passive.cells[0].runaway_duration_s=0.05;passive.primary_failures={{0,"battery",0,-1}};
    passive.cells[1].initial_damage=0;passive.settings.step_interval_s=0.2;
    Simulation playback(passive);
    require(advance_playback(playback,1,AutoPauseMode::AllCellsDestroyed)==PauseReason::AllCellsDestroyed,"Outer step observes all-battery destruction inside interval");
    require(std::abs(playback.time_s()-0.05)<1e-9 && playback.time_s()<0.2,"Pause does not overshoot within outer step");
    require(playback.cells()[1].mode==CellMode::Healthy,"Healthy pad does not prevent all-battery pause");

    auto scheduled=fixture();scheduled.settings={0.2,0.2,0.04,0.2,0.2};
    scheduled.cells[0].runaway_energy_j=100;scheduled.cells[0].runaway_duration_s=0.05;scheduled.primary_failures={{0,"battery",0.07,-1}};
    Simulation events(scheduled);events.advance();
    require(std::abs(events.cells()[0].runaway_started_s-0.07)<1e-12,"Outer interval respects scheduled ignition");
    require(events.history().time_s.size()==6,"Output boundaries are retained inside an outer interval");
    require(std::abs(events.released_energy_j()-100)<1e-8,"Reaction source exhaustion respected during subcycling");
    require(std::abs(events.energy_balance_residual_j())<1e-8,"Source and nonconforming conduction conserve energy");
    auto invalid=fixture();invalid.primary_failures={{1,"pad",0,-1}};bool rejected=false;
    try{Simulation bad(invalid);}catch(const std::runtime_error&){rejected=true;}require(rejected,"Passive ignition rejected");

    if(argc==2) {
        auto module=load_scenario(argv[1]);module.settings.duration_s=10;
        Simulation m(module);m.run_to_end();
        require(m.battery_count()==52 && module.cells.size()==139,"52 batteries plus 87 heat-storing interlayers");
        require(m.thermal_mesh().volumes.size()==34832,"Battery zones remain 416; detail belongs to pads");
        require(m.runaway_order().size()==1,"Early module phase has only prescribed primary event");
        require(m.step_count()>m.outer_step_count(),"Fine material causes physical subcycling");
        require(std::abs(m.energy_balance_residual_j())<1e-5,"Multiscale module energy ledger");
    }
    std::cout<<"CUDA multiscale contacts, passive damage, substeps, events and energy: PASS\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

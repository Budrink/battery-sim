#include "battery_native/simulation.hpp"
#include "battery_native/failure.hpp"
#include "battery_native/state_geometry.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
Scenario pulse_scene(bool isothermal=false) {
    Scenario s;s.name="Finite chemical reserve";s.settings={15.,.7,1.,.4};
    CellConfig c;c.id="C";c.dimensions_m={.03,.03,.03};c.thermal_divisions={3,3,3};
    c.heat_capacity_j_per_k=27.;c.conductivity_w_per_m_k={.01,.01,.01};c.initial_temperature_k=300.;c.isothermal=isothermal;
    c.runaway_energy_j=2700.;c.runaway_duration_s=.4;c.runaway_pulse={RunawayProfile::BurstTail,.7,2.};
    s.cells.push_back(c);s.primary_failures.push_back({0,"C",.5,-1});return s;
}
Scenario structure(double factor) {
    auto s=pulse_scene();s.settings={.8,.02,.1,.4};s.primary_failures={{0,"C",.1,-1}};
    auto& c=s.cells[0];c.runaway_energy_j=0.;c.has_bulk_connectivity=true;c.has_bulk_response=true;
    auto& l=c.bulk_connectivity;l.kind=TransitionKind::DirectedGeometry;l.background_conductance_w_per_k=1.;
    l.endpoint_damage_weight=1.;l.destroyed_background_conductance_w_per_k=factor;l.destroyed_active_conductance_w_per_k=0.;
    l.l0_m=1e-4;l.destroyed_l0_m=2e-4;l.destroyed_l_m=2e-4;c.bulk_destruction_on_boundary=true;
    c.initial_temperature_field_k.assign(27,300.);c.initial_temperature_field_k[13]=350.;
    PortConfig p;p.id="C:x";p.cell_id="C";p.surface="+x";p.area_m2=.0009;p.centroid_m={.015,0,0};p.normal={1,0,0};s.ports.push_back(p);
    AmbientBoundary b;b.id="environment";b.port_id=p.id;b.temperature_k=300.;b.conductance_w_per_k=10.;s.boundaries.push_back(b);
    return s;
}
int main(){try {
    RunawayPulse p{RunawayProfile::BurstTail,.7,2.};
    require(pulse_fraction(p,.4,0)==0 && pulse_fraction(p,.4,12)==1,"Pulse endpoints");
    for(double f:{0.,.2,1.}) {
        p.fast_fraction=f;double sum=0,previous=0;
        for(int i=1;i<=2000;++i){double age=i*.01;double v=pulse_fraction(p,.4,age);require(v>=previous,"Pulse monotonicity");sum+=v-previous;previous=v;}
        require(std::abs(sum-1)<1e-14,"Pulse normalization");
    }
    for(bool iso:{false,true}) {
        auto s=pulse_scene(iso);Simulation sim(s);
        while(!sim.finished()) {
            sim.step();double target=2700*pulse_fraction(s.cells[0].runaway_pulse,.4,std::max(0.,sim.time_s()-.5));
            require(std::abs(sim.released_energy_j()-target)<1e-8,"CUDA pulse integral differs from analytic reserve");
            require(std::abs(sim.energy_balance_residual_j())<1e-8,"Source creates unaccounted energy");
        }
        require(sim.cells()[0].mode==CellMode::Spent,"Pulse never completes");
        require(std::abs(sim.cells()[0].temperature_k-400)<1e-8,"Wrong full reserve heat");
        sim.reset();sim.run_to_end();require(std::abs(sim.released_energy_j()-2700)<1e-8,"Reset retains pulse progress");
    }
    auto s=pulse_scene();s.primary_failures[0].time_s=0.;Simulation paused(s);paused.step();
    const double released=paused.released_energy_j(),age=paused.thermal_states()[0].reaction_progress_s;
    paused.runtime().runaway_power_scale=0;for(int i=0;i<4;++i)paused.step();
    require(paused.released_energy_j()==released && paused.thermal_states()[0].reaction_progress_s==age,"Paused pulse keeps burning");
    paused.runtime().runaway_power_scale=2;paused.run_to_end();require(std::abs(paused.released_energy_j()-2700)<1e-8,"Changing playback speed changes energy reserve");
    auto automatic=pulse_scene();automatic.primary_failures.clear();automatic.cells[0].damage_onset_k=299.;automatic.cells[0].damage_rate_at_onset_per_s=10.;
    Simulation secondary(automatic);secondary.run_to_end();require(secondary.cells()[0].runaway_started_s>0 && secondary.cells()[0].mode==CellMode::Spent,"Thermal ignition did not start pulse");
    for(double factor:{.01,30.}) {
        auto scene=structure(factor);Simulation sim(scene);const auto& mesh=sim.thermal_mesh();
        while(!sim.finished()) {
            const auto before=sim.thermal_states();const double time=sim.time_s();sim.step();const auto links=sim.thermal_link_states();
            for(std::size_t k=0;k<mesh.links.size();++k) {
                const auto& e=mesh.links[k];const auto& state=links[k];
                if(e.b>=0) {
                    require((state.phase==EdgePhase::Destroyed)==(time>=.1-1e-9),"Bulk destruction missed or healed");
                    if(time>=.1-1e-9)require(state.l0_m==2e-4,"Bulk l0 did not switch");
                } else {
                    const double r=phase_resistance(e.series_resistance_k_per_w,e.series_resistance_a_k_per_w,before[e.a].damage,0,boundary_phase(scene.cells[0]),{});
                    const double g=1/(1/(e.law.background_conductance_w_per_k*e.weight)+r);
                    require(std::abs(g-state.conductance_w_per_k)<1e-12,"Boundary half-volume still intact");
                }
            }
            require(std::abs(sim.energy_balance_residual_j())<1e-8,"Changing structure creates energy");
            for(auto& v:sim.thermal_states())require(v.temperature_k>=300-1e-9 && v.temperature_k<=350+1e-9,"Phase switch breaks maximum principle/CFL");
        }
    }
    // Endpoint history is irreversible even when a diagnostic re-evaluates it cold.
    // A rupture can briefly INCREASE transport, then leave an insulating residue.
    auto transient=structure(.01);transient.settings={4.,.4,4.,.4};transient.primary_failures[0].time_s=.1;
    transient.cells[0].bulk_failure_transient={30.,.4};
    Simulation evolving(transient);bool saw_peak=false,saw_residue=false;
    while(!evolving.finished()) {
        const auto before=evolving.thermal_states();const double time=evolving.time_s();evolving.step();
        const auto links=evolving.thermal_link_states();const auto& mesh=evolving.thermal_mesh();
        const auto phase=boundary_phase(transient.cells[0]);
        for(std::size_t j=0;j<mesh.links.size();++j) {
            const auto& e=mesh.links[j];const auto& actual=links[j];
            if(e.b>=0 && time>=.1-1e-9) {
                const double f=phase_factor(phase,1.,time,.1);
                const double reference=e.law.destroyed_background_conductance_w_per_k/.01;
                require(std::abs(actual.conductance_w_per_k-reference*f)<1e-12,"Wrong transient internal conductance");
                require(actual.conductance_w_per_k<=e.maximum_g+1e-12,"Transient peak exceeds CFL conductance bound");
                saw_peak|=f>20.;saw_residue|=f==.01;
            } else if(e.b<0) {
                const double r=phase_resistance(e.series_resistance_k_per_w,e.series_resistance_a_k_per_w,
                    before[e.a].damage,0,phase,{},time,before[e.a].runaway_started_s,-1);
                require(std::abs(actual.conductance_w_per_k-1/(1/(e.law.background_conductance_w_per_k*e.weight)+r))<1e-12,"Transient boundary disagrees with internal medium");
            }
        }
        for(auto& v:evolving.thermal_states())require(v.temperature_k>=300-1e-9 && v.temperature_k<=350+1e-9,"Transient maximum principle");
        require(std::abs(evolving.energy_balance_residual_j())<1e-8 && evolving.released_energy_j()==0,"Structural transient creates heat");
    }
    require(saw_peak && saw_residue,"Transient did not evolve after source became spent");
    auto phase=boundary_phase(transient.cells[0]);
    require(phase_factor(phase,0,100,0)==1.,"Healthy node affected by rupture clock");
    const auto ab=bulk_failure_multiplier(phase,310,300,1,1,0,.8,1,1);
    const auto ba=bulk_failure_multiplier(phase,310,300,1,1,0,.8,1,-1);
    require(ab!=ba,"Structural ages were averaged across directed branches");
    // Isolate boundary CFL: a large peak must reduce dt even if residual k is tiny.
    auto stiff=structure(.01);stiff.settings={1000,1000,1000,.4};stiff.primary_failures[0].time_s=0;
    auto& sc=stiff.cells[0];sc.thermal_divisions={1,1,1};sc.initial_temperature_field_k={350};sc.bulk_failure_transient={100.,1e6};
    sc.runaway_pulse.profile=RunawayProfile::ConstantPower; // no independent burst-resolution cap
    Simulation bounded(stiff);const auto& edge=bounded.thermal_mesh().links[0];
    const double maxg=1/(1/(edge.law.background_conductance_w_per_k*edge.weight)+edge.series_resistance_k_per_w/100.);
    bounded.step();require(std::abs(bounded.time_s()-.4*27/maxg)<1e-9,"CFL ignores high transient boundary conductivity");
    require(std::abs(bounded.cells()[0].temperature_k-330)<1e-9,"Peak CFL heat balance");
    auto law=structure(.1).cells[0].bulk_connectivity;InterfaceState state;
    state=evaluate_directed_geometry(law,state,0,300,300,1,0,0,1,0);
    state=evaluate_directed_geometry(law,state,0,290,290,0,0,0,1,0);
    require(state.phase==EdgePhase::Destroyed,"Destroyed edge healed");
    auto invalid=structure(1.);invalid.cells[0].bulk_connectivity.endpoint_damage_weight=0;
    bool rejected=false;try{Simulation bad(invalid);}catch(const std::exception&){rejected=true;}
    require(rejected,"Inconsistent boundary/bulk switch accepted");
    invalid=structure(1.);invalid.cells[0].bulk_failure_transient={3.,-1.};rejected=false;
    try{Simulation bad(invalid);}catch(const std::exception&){rejected=true;}require(rejected,"Negative structural relaxation accepted");
    invalid=pulse_scene();invalid.cells[0].runaway_pulse.fast_fraction=1.1;rejected=false;
    try{Simulation bad(invalid);}catch(const std::exception&){rejected=true;}require(rejected,"Invalid pulse fraction accepted");
    std::cout<<"CUDA chemical reserve, pulse clock, phase switch, boundary resistance, CFL and conservation PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

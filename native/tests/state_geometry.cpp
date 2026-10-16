#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool b,const char* msg) {if(!b)throw std::runtime_error(msg);}
void close(double a,double b,const char* msg,double eps=1e-10) {require(std::abs(a-b)<eps,msg);}
ConnectivityLaw law() {
    ConnectivityLaw l;l.kind=TransitionKind::StateGeometry;l.l0_m=1e-4;
    l.background_conductance_w_per_k=.1;l.active_conductance_w_per_k=9.9;
    l.endpoint_damage_weight=1;l.destroyed_l0_m=2e-4;l.destroyed_l_m=2e-4;
    l.destroyed_background_conductance_w_per_k=.01;
    l.endpoint_a.linear={2,-2,0};l.endpoint_b.linear={1,-1,0};
    l.endpoint_a.quadratic=l.endpoint_b.quadratic={-2,2,0};
    l.endpoint_a.damage=l.endpoint_b.damage={-3,-3,-3};return l;
}
InterfaceState at(const ConnectivityLaw& l,double a,double b,int axis=0,double da=0,double db=0,InterfaceState prev={}) {
    return evaluate_state_geometry(l,prev,axis,a,b,da,db,0,1,0);
}
Scenario fixture(int ny=2) {
    Scenario s;s.name="state geometry fixture";s.settings={2,.02,.1,.4};
    CellConfig c;c.id="a";c.is_battery=false;c.dimensions_m={1,1,1};
    c.thermal_divisions={1,ny,1};c.heat_capacity_j_per_k=100;
    c.conductivity_w_per_m_k={1,1,1};c.position_m={-.5,0,0};
    c.surface_connectivity=law().endpoint_a;
    for(int i=0;i<ny;++i)c.initial_temperature_field_k.push_back(i<ny/2?400:330);
    s.cells.push_back(c);c.id="b";c.position_m={.5,0,0};c.initial_temperature_field_k.assign(ny,300);
    c.surface_connectivity=law().endpoint_b;s.cells.push_back(c);
    s.ports={{"a+x","a",0,"contact","+x",1,{0,0,0},{1,0,0}},
             {"b-x","b",1,"contact","-x",1,{0,0,0},{-1,0,0}}};
    InterfaceConfig e;e.id="ab";e.port_a="a+x";e.port_b="b-x";e.cell_b=1;e.law=law();
    s.interfaces.push_back(e);return s;
}
int main() {try {
    auto l=law();const auto x=at(l,360,320),y=at(l,360,320,1);
    require(x.conductance_w_per_k!=y.conductance_w_per_k,"Directional local metric ignored");
    auto swapped=l;std::swap(swapped.endpoint_a,swapped.endpoint_b);
    close(x.conductance_w_per_k,at(swapped,320,360).conductance_w_per_k,"Endpoint reversal changed G");
    require(at(l,360,320).conductance_w_per_k!=at(l,340,340).conductance_w_per_k,"Endpoint states collapsed to average T before nonlinear response");
    require(at(l,338.15,338.15).conductance_w_per_k>at(l,298.15,298.15).conductance_w_per_k &&
            at(l,338.15,338.15).conductance_w_per_k>at(l,498.15,498.15).conductance_w_per_k,"Nonmonotonic response lost");
    require(at(l,360,320,0,.5,0).conductance_w_per_k<x.conductance_w_per_k,"Accumulated endpoint damage ignored");
    for(double t:{1.,298.15,800.,1e5}) {
        auto q=at(l,t,t);require(q.l_m>=q.l0_m && q.conductance_w_per_k>=.1 && q.conductance_w_per_k<=10,"Positivity/floor/ceiling");
    }
    auto dead=at(l,360,320,0,1,0);
    require(dead.phase==EdgePhase::Destroyed,"One destroyed endpoint must suffice, not mean damage");
    close(dead.conductance_w_per_k,.01,"Destruction did not close contact sharply");
    auto cooled=at(l,298,298,0,0,0,dead);
    require(cooled.phase==EdgePhase::Destroyed,"Destruction healed after cooling");
    close(cooled.l0_m,2e-4,"Destroyed structural floor not selected");
    l.destroyed_background_conductance_w_per_k=30;
    close(at(l,360,320,0,1,0).conductance_w_per_k,30,"Destruction cannot open a new high-conductance state");
    l=law();l.damage_onset_k=300;l.damage_rate_at_onset_per_s=.1;
    auto history=evaluate_state_geometry(l,{},0,350,350,0,0,.1,1,0);
    const auto observed=evaluate_state_geometry(l,history,0,350,350,0,0,0,1,0);
    close(observed.damage,history.damage,"Observation advanced damage history");

    auto s=fixture();Simulation sim(s);const auto mesh=sim.thermal_mesh();
    auto edges=sim.thermal_link_states();double first=-1;
    for(int index:mesh.interface_links) {
        const auto& e=mesh.links[index];const auto& field=sim.thermal_states();
        auto expected=evaluate_state_geometry(e.law,{},e.geometry_axis,field[e.a].temperature_k,field[e.b].temperature_k,0,0,0,e.weight,e.series_resistance_k_per_w);
        close(edges[index].conductance_w_per_k,expected.conductance_w_per_k,"CPU/CUDA law disagree at reset");
        if(first<0)first=edges[index].conductance_w_per_k;
        else require(std::abs(first-edges[index].conductance_w_per_k)>1e-4,"Local patches forced to a macro-average state");
    }
    sim.step();auto new_edges=sim.thermal_link_states();
    for(int index:mesh.interface_links)close(edges[index].conductance_w_per_k,new_edges[index].conductance_w_per_k,"First step did not use old-state snapshot");
    sim.run_to_end();require(std::abs(sim.energy_balance_residual_j())<1e-8,"Dynamic pair violates energy balance");
    for(const auto& v:sim.thermal_states())require(v.temperature_k>=300 && v.temperature_k<=400,"Dynamic CFL/max principle failed");
    require(std::abs(sim.thermal_link_states()[mesh.interface_links[0]].conductance_w_per_k-first)>1e-6,"Conductance stayed frozen");

    auto uniform=fixture();for(auto& c:uniform.cells)c.initial_temperature_field_k.assign(2,350);
    Simulation eq(uniform);eq.run_to_end();for(auto& v:eq.thermal_states())close(v.temperature_k,350,"Equal temperatures generated heat");
    auto damaged=fixture();damaged.cells[0].initial_damage=1;
    Simulation failed(damaged);for(int i:failed.thermal_mesh().interface_links)require(failed.thermal_link_states()[i].phase==EdgePhase::Destroyed,"CUDA destruction branch not selected");
    auto event=fixture();event.cells[0].is_battery=true;event.primary_failures={{0,"a",.5,-1}};
    Simulation jump(event);while(jump.time_s()<.4)jump.step();
    require(jump.interfaces()[0].phase!=EdgePhase::Destroyed,"Failure happened before scheduled state change");
    while(jump.time_s()<.6)jump.step();
    require(jump.interfaces()[0].phase==EdgePhase::Destroyed,"Scheduled failure did not switch CUDA contact");
    jump.run_to_end();require(jump.interfaces()[0].phase==EdgePhase::Destroyed,"CUDA failure branch reverted");
    require(std::abs(jump.energy_balance_residual_j())<1e-8,"Jump generated parasitic heat");

    // Rotate geometry, ports and material frames together: projection invariant.
    auto rotated=fixture();const double half=std::sqrt(.5);
    for(auto& c:rotated.cells) {c.orientation_xyzw={0,0,half,half};c.position_m={-c.position_m.y,c.position_m.x,c.position_m.z};}
    for(auto& p:rotated.ports) {p.centroid_m={-p.centroid_m.y,p.centroid_m.x,p.centroid_m.z};p.normal={-p.normal.y,p.normal.x,p.normal.z};}
    Simulation rotation(rotated);rotation.run_to_end();
    for(std::size_t i=0;i<sim.thermal_states().size();++i)close(sim.thermal_states()[i].temperature_k,rotation.thermal_states()[i].temperature_k,"Rigid rotation changed thermal evolution");

    auto driven=fixture();driven.interfaces.clear();driven.ports.resize(1);driven.cells.resize(1);
    AmbientBoundary boundary;boundary.id="dynamic-source";boundary.port_id="a+x";boundary.temperature_k=500;
    boundary.has_connectivity=true;boundary.connectivity=law();boundary.remote_surface_connectivity=law().endpoint_b;
    driven.boundaries.push_back(boundary);Simulation boundary_sim(driven);boundary_sim.run_to_end();
    require(boundary_sim.boundary_energy_out_j()<0 && std::abs(boundary_sim.energy_balance_residual_j())<1e-8,"Dynamic boundary energy ledger");
    const auto& bm=boundary_sim.thermal_mesh();const auto states=boundary_sim.thermal_link_states();
    for(std::size_t i=0;i<bm.links.size();++i)if(bm.links[i].b<0)require(states[i].conductance_w_per_k>0,"Dynamic boundary ignored");
    auto half_dt=s;half_dt.settings.max_dt_s*=.5;Simulation finer(half_dt);finer.run_to_end();
    double error=0;for(std::size_t i=0;i<sim.thermal_states().size();++i)error=std::max(error,std::abs(sim.thermal_states()[i].temperature_k-finer.thermal_states()[i].temperature_k));
    auto quarter_dt=half_dt;quarter_dt.settings.max_dt_s*=.5;Simulation finest(quarter_dt);finest.run_to_end();
    double refined_error=0;for(std::size_t i=0;i<sim.thermal_states().size();++i)refined_error=std::max(refined_error,std::abs(finer.thermal_states()[i].temperature_k-finest.thermal_states()[i].temperature_k));
    require(error<.01 && refined_error<error*.55,"Dynamic timestep refinement failed to converge at first order");
    std::cout<<"State geometry: reciprocity, direction, local states, nonlinear response, damage, irreversible jump, CPU/CUDA agreement, conservation, CFL, rotation, boundary, timestep: PASS; refinement differences "<<error<<" -> "<<refined_error<<" K\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool b,const char* message) {if(!b)throw std::runtime_error(message);}
void close(double a,double b,const char* message,double eps=1e-10) {require(std::abs(a-b)<eps,message);}
ConnectivityLaw law() {
    ConnectivityLaw l;l.kind=TransitionKind::DirectedGeometry;l.l0_m=1e-4;
    l.background_conductance_w_per_k=.1;l.active_conductance_w_per_k=9.9;
    l.endpoint_damage_weight=1;l.destroyed_l0_m=l.destroyed_l_m=2e-4;
    l.destroyed_background_conductance_w_per_k=.01;
    l.endpoint_a.temperature_contrast={3,-3,0};l.endpoint_b.temperature_contrast={-2,2,0};
    l.endpoint_a.damage_contrast={2,2,2};l.endpoint_b.damage_contrast={-1,-1,-1};return l;
}
InterfaceState at(const ConnectivityLaw& l,double a,double b,int direction=0,int axis=0,double da=0,double db=0,InterfaceState history={}) {
    return evaluate_directed_geometry(l,history,axis,a,b,da,db,0,1,0,1,direction);
}
Scenario fixture() {
    Scenario s;s.name="Directed geometry fixture";s.settings={2,.01,.1,.4};
    CellConfig c;c.id="a";c.is_battery=false;c.dimensions_m={1,1,1};c.thermal_divisions={1,1,1};
    c.heat_capacity_j_per_k=100;c.isothermal=true;c.position_m={-.5,0,0};c.initial_temperature_k=400;
    c.surface_connectivity=law().endpoint_a;s.cells.push_back(c);
    c.id="b";c.position_m={.5,0,0};c.initial_temperature_k=300;c.surface_connectivity=law().endpoint_b;
    c.heat_capacity_j_per_k=80;s.cells.push_back(c);
    s.ports={{"a+x","a",0,"contact","+x",1,{0,0,0},{1,0,0}},
             {"b-x","b",1,"contact","-x",1,{0,0,0},{-1,0,0}}};
    InterfaceConfig e;e.id="ab";e.port_a="a+x";e.port_b="b-x";e.cell_b=1;e.law=law();s.interfaces.push_back(e);return s;
}
int main(){try {
    auto l=law();const auto ab=at(l,400,300,1),ba=at(l,400,300,-1);
    require(std::abs(ab.conductance_w_per_k-ba.conductance_w_per_k)>.1,"Forward and reverse laws were symmetrized");
    close(at(l,400,300).conductance_w_per_k,ab.conductance_w_per_k,"Hot donor a not selected");
    close(at(l,300,400).conductance_w_per_k,at(l,300,400,-1).conductance_w_per_k,"Hot donor b not selected");
    require(std::abs(at(l,400,300).conductance_w_per_k-at(l,300,400).conductance_w_per_k)>1,"Physical gradient reversal did not rectify heterogeneous contact");
    require(at(l,400,300,1,1).conductance_w_per_k!=ab.conductance_w_per_k,"Direction axis ignored");
    require(at(l,400,300,1,0,.5,0).conductance_w_per_k!=ab.conductance_w_per_k,"Signed damage contrast ignored");
    auto swapped=l;std::swap(swapped.endpoint_a,swapped.endpoint_b);
    close(at(swapped,300,400,-1).conductance_w_per_k,ab.conductance_w_per_k,"Storage endpoint permutation changed physical a->b law");
    close(at(swapped,300,400,1).conductance_w_per_k,ba.conductance_w_per_k,"Storage permutation did not swap directional records");
    auto symmetric=l;symmetric.endpoint_a=symmetric.endpoint_b={};
    close(at(symmetric,400,300,1).conductance_w_per_k,at(symmetric,400,300,-1).conductance_w_per_k,"Symmetric special case broken");
    auto nonlinear=l;nonlinear.endpoint_a.linear.x=2;nonlinear.endpoint_a.quadratic.x=-3;
    nonlinear.endpoint_a.temperature_contrast={};
    require(at(nonlinear,330,300,1).conductance_w_per_k>at(nonlinear,500,300,1).conductance_w_per_k,"Nonmonotonic sender response lost");
    for(double t:{1.,298.15,1000.,1e5})for(int dir:{-1,1}) {
        auto e=at(l,t,300,dir);require(e.l_m>=e.l0_m && e.conductance_w_per_k>=.1 && e.conductance_w_per_k<=10,"Directed law escaped CFL bounds");
    }
    auto failed=at(l,400,300,0,0,1,0);require(failed.phase==EdgePhase::Destroyed,"One failed endpoint not sufficient");
    close(failed.conductance_w_per_k,.01,"Destruction did not sharply close contact");
    require(at(l,300,400,0,0,0,0,failed).phase==EdgePhase::Destroyed,"Reversed gradient healed destruction");
    l.destroyed_background_conductance_w_per_k=30;
    close(at(l,400,300,0,0,0,1).conductance_w_per_k,30,"Destruction cannot increase connectivity");

    auto s=fixture();Simulation sim(s);const int index=sim.thermal_mesh().interface_links[0];
    close(sim.thermal_link_states()[index].conductance_w_per_k,ab.conductance_w_per_k,"CUDA and host directed laws differ");
    sim.step();const double q=sim.last_dt_s()*ab.conductance_w_per_k*100;
    close(sim.cells()[0].temperature_k,400-q/100,"GPU used averaged conductance instead of selected direction");
    close(sim.cells()[1].temperature_k,300+q/80,"Opposite energy credit/capacity wrong");
    sim.run_to_end();require(std::abs(sim.energy_balance_residual_j())<1e-8,"Directed network created heat");
    for(const auto& v:sim.thermal_states())require(v.temperature_k>=300 && v.temperature_k<=400,"Maximum principle failed");
    Simulation traced(s);double cpu_a=400,cpu_b=300;
    while(!traced.finished()) {
        const double g=at(law(),cpu_a,cpu_b).conductance_w_per_k;
        traced.step();const double transfer=traced.last_dt_s()*g*(cpu_a-cpu_b);
        cpu_a-=transfer/100;cpu_b+=transfer/80;
        close(cpu_a,traced.cells()[0].temperature_k,"Step-by-step CPU/CUDA donor mismatch",1e-8);
        close(cpu_b,traced.cells()[1].temperature_k,"Step-by-step CPU/CUDA receiver mismatch",1e-8);
    }

    // One coarse face maps to several finer receiver patches. Temperatures and
    // donor identities vary locally; no macro-interface temperature is used.
    auto spatial=s;for(auto& c:spatial.cells)c.isothermal=false;
    spatial.cells[0].thermal_divisions={1,2,1};spatial.cells[0].initial_temperature_field_k={400,300};
    spatial.cells[1].thermal_divisions={1,4,1};spatial.cells[1].initial_temperature_field_k={300,300,400,400};
    Simulation local(spatial);const auto& mesh=local.thermal_mesh();const auto start=local.thermal_states();
    const auto initial_edges=local.thermal_link_states();bool saw_forward=false,saw_reverse=false;
    for(int idx:mesh.interface_links) {
        const auto& edge=mesh.links[idx];const double ta=start[edge.a].temperature_k,tb=start[edge.b].temperature_k;
        const auto expected=evaluate_directed_geometry(edge.law,{},edge.geometry_axis,ta,tb,0,0,0,edge.weight,edge.series_resistance_k_per_w);
        close(initial_edges[idx].conductance_w_per_k,expected.conductance_w_per_k,"Patch law replaced by macro average");
        saw_forward|=ta>tb;saw_reverse|=tb>ta;
    }
    require(saw_forward && saw_reverse,"Fixture must exercise opposite local donors on one macro contact");
    local.run_to_end();require(std::abs(local.energy_balance_residual_j())<1e-8,"Nonconforming directed contact lost energy");
    for(auto& v:local.thermal_states())require(v.temperature_k>=300 && v.temperature_k<=400,"Spatial directed CFL failed");

    auto relabelled=s;auto& e=relabelled.interfaces[0];std::swap(e.port_a,e.port_b);std::swap(e.cell_a,e.cell_b);
    Simulation same(relabelled);same.run_to_end();
    for(int i=0;i<2;++i)close(same.cells()[i].temperature_k,sim.cells()[i].temperature_k,"Storage endpoint order affected trajectory");
    auto reversed=s;std::swap(reversed.cells[0].initial_temperature_k,reversed.cells[1].initial_temperature_k);
    Simulation reverse(reversed);reverse.step();
    close(reverse.cells()[0].temperature_k,300+reverse.last_dt_s()*at(law(),300,400).conductance_w_per_k,"Reverse CUDA donor branch wrong");
    reverse.run_to_end();require(std::abs(reverse.energy_balance_residual_j())<1e-8,"Reverse heat balance failed");

    auto uniform=s;for(auto& c:uniform.cells)c.initial_temperature_k=350;
    uniform.cells[0].surface_connectivity.linear={2,2,2}; // unequal directional G at equal T
    Simulation equilibrium(uniform);equilibrium.run_to_end();
    for(auto& v:equilibrium.thermal_states())close(v.temperature_k,350,"Unequal links generated heat at equilibrium");

    auto scheduled=s;scheduled.cells[0].is_battery=true;scheduled.primary_failures={{0,"a",.5,-1}};
    Simulation rupture(scheduled);while(rupture.time_s()<.4)rupture.step();require(rupture.interfaces()[0].phase!=EdgePhase::Destroyed,"Premature rupture");
    while(rupture.time_s()<.6)rupture.step();require(rupture.interfaces()[0].phase==EdgePhase::Destroyed,"In-run failure did not switch contact");
    rupture.run_to_end();require(std::abs(rupture.energy_balance_residual_j())<1e-8,"Failure generated parasitic heat");

    auto boundary=s;boundary.interfaces.clear();boundary.cells.resize(1);boundary.ports.resize(1);
    boundary.cells[0].initial_temperature_k=350;AmbientBoundary b;b.id="measured";b.port_id="a+x";
    b.temperature_k=400;b.temperature_profile={{0,400},{.5,400},{1,300},{2,300}};
    b.has_connectivity=true;b.connectivity=law();b.remote_surface_connectivity=law().endpoint_b;boundary.boundaries.push_back(b);
    Simulation driven(boundary);driven.step();
    close(driven.cells()[0].temperature_k,350+driven.last_dt_s()*at(law(),350,400).conductance_w_per_k*.5,"Boundary hot donor was not remote material");
    driven.run_to_end();require(std::abs(driven.energy_balance_residual_j())<1e-8,"Boundary reversal energy balance");
    auto history=driven.thermal_link_states()[0];const auto field=driven.thermal_states()[0];
    close(history.conductance_w_per_k,at(law(),field.temperature_k,300).conductance_w_per_k,"Boundary did not switch back to local sender",.01);

    auto half=s;half.settings.max_dt_s*=.5;Simulation fine(half);fine.run_to_end();
    auto quarter=half;quarter.settings.max_dt_s*=.5;Simulation finest(quarter);finest.run_to_end();
    const double err=std::abs(sim.cells()[0].temperature_k-fine.cells()[0].temperature_k);
    const double err2=std::abs(fine.cells()[0].temperature_k-finest.cells()[0].temperature_k);
    require(err<.02 && err2<.55*err,"Directed timestep convergence failed");
    std::cout<<"Directed geometry: asymmetry, no averaging, gradient/storage reversal distinction, CPU/CUDA, boundary reversal, conservation, equilibrium, failure, timestep PASS ("<<err<<" -> "<<err2<<" K)\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

// A cooling wall is an ordinary passive solid, not a prescribed temperature.
#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
Scenario fixture(double sink) {
    Scenario s;s.settings={.2,.001,.1,.4};
    CellConfig source;source.id="source";source.is_battery=false;
    source.dimensions_m={.02,.02,.02};source.thermal_divisions={2,2,2};
    source.heat_capacity_j_per_k=10.;source.conductivity_w_per_m_k={10,10,10};source.initial_temperature_k=800.;
    source.surface_connectivity.linear={.3,.1,0};
    auto wall=source;wall.id="wall";wall.position_m={.02,0,0};wall.initial_temperature_k=300.;
    wall.heat_capacity_j_per_k=5.;wall.conductivity_w_per_m_k={235,235,235};
    wall.damage_rate_at_onset_per_s=0.;wall.initial_damage=0.;wall.surface_connectivity.linear={-.8,.2,0};
    wall.has_bulk_connectivity=true;wall.has_bulk_response=true;
    wall.bulk_connectivity.kind=TransitionKind::DirectedGeometry;
    wall.bulk_connectivity.background_conductance_w_per_k=.1;wall.bulk_connectivity.active_conductance_w_per_k=1.8;
    wall.bulk_response.linear={-.3,.2,0};
    s.cells={source,wall};
    s.ports.push_back({"source-out","source",0,"contact","+x",.0004,{.01,0,0},{1,0,0}});
    s.ports.push_back({"wall-in","wall",1,"contact","-x",.0004,{.01,0,0},{-1,0,0}});
    s.ports.push_back({"wall-out","wall",1,"contact","+x",.0004,{.03,0,0},{1,0,0}});
    InterfaceConfig e;e.id="source-wall";e.port_a="source-out";e.port_b="wall-in";e.cell_a=0;e.cell_b=1;
    e.law.kind=TransitionKind::DirectedGeometry;e.law.background_conductance_w_per_k=.1;e.law.active_conductance_w_per_k=2.;
    s.interfaces.push_back(e);s.boundaries.push_back({"water","wall-out",1,300.,sink});return s;
}
int main(){try {
    Simulation closed(fixture(0)),cooled(fixture(10));closed.run_to_end();cooled.run_to_end();
    require(closed.cells()[1].temperature_k>300.1,"Wall did not store heat");
    require(cooled.cells()[1].temperature_k>300. && cooled.cells()[1].temperature_k<closed.cells()[1].temperature_k,
            "Sink must cool a finite-capacity wall, not clamp its temperature");
    require(cooled.boundary_energy_out_j()>0,"No heat removed by water boundary");
    for(const auto* sim:{&closed,&cooled}) {
        require(std::abs(sim->energy_balance_residual_j())<1e-8,"Wall energy ledger does not close");
        for(const auto& v:sim->thermal_states()) {
            require(v.damage==0.,"Indestructible wall accumulated damage");
            require(v.temperature_k>=300 && v.temperature_k<=800,"Temperature left source/sink bounds");
        }
    }
    // Same interface law in both directions; no symmetrization of responses.
    const auto m=build_thermal_mesh(fixture(10));const auto& edge=m.links[m.interface_links[0]];
    auto a=evaluate_directed_geometry(edge.law,edge.initial,0,300,350,0,0,0,edge.weight,edge.series_resistance_k_per_w);
    auto b=evaluate_directed_geometry(edge.law,edge.initial,0,300,450,0,0,0,edge.weight,edge.series_resistance_k_per_w);
    require(std::abs(a.conductance_w_per_k-b.conductance_w_per_k)>1e-5,"Warm wall has no state-dependent connection");
    // One-volume passive wall: exact constant-G Newton cooling including its
    // half-volume resistance. No fluid state or custom wall kernel is involved.
    auto isolated=fixture(10);isolated.cells={isolated.cells[1]};auto& w=isolated.cells[0];
    w.position_m={0,0,0};w.thermal_divisions={1,1,1};w.initial_temperature_k=400.;
    w.surface_connectivity={}; // Explicit neutral-metric analytical limit.
    isolated.interfaces.clear();isolated.ports={{"sink","wall",0,"contact","+x",.0004,{.01,0,0},{1,0,0}}};
    isolated.boundaries={{"water","sink",0,300.,10.}};
    Simulation exact(isolated);exact.run_to_end();
    const double g=1./(.1+.01/(235*.0004));
    const double temperature=300+100*std::exp(-g*.2/5);
    require(std::abs(exact.cells()[0].temperature_k-temperature)<.02,"Constant sink does not follow finite-capacity cooling");
    // Transactional proposals and different SIMD tile sizes preserve fields.
    const auto s=fixture(10);CudaSolver single(s,m,64),retry(s,m,256);RuntimeParameters p;
    for(int i=0;i<10;++i) {
        retry.prepare_step(i*.001,.001,.4,p);retry.discard_step();
        single.step(i*.001,.001,.4,p);retry.step(i*.001,.001,.4,p);
    }
    std::vector<ThermalState> x,y;single.download_field(x);retry.download_field(y);
    for(std::size_t i=0;i<x.size();++i)require(x[i].temperature_k==y[i].temperature_k &&
        x[i].stored_energy_j==y[i].stored_energy_j && x[i].boundary_energy_out_j==y[i].boundary_energy_out_j,
        "Wall retry/tile-size changed the committed state");
    std::cout<<"Passive cooling wall: storage, sink, nonlinear contacts, no damage, conservation and retry OK\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

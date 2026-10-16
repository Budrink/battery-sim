#include "battery_native/simulation.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace battery;
void require(bool b,const char* message) {if(!b)throw std::runtime_error(message);}
Scenario fixture(int nx=2) {
    Scenario s;s.name="CUDA numerical fixture";s.settings={0.5,0.001,0.1,0.2};
    CellConfig c;c.id="solid";c.dimensions_m={2,1,1};c.thermal_divisions={nx,1,1};
    c.heat_capacity_j_per_k=2;c.conductivity_w_per_m_k={1,1,1};c.initial_temperature_k=300;
    s.cells.push_back(c);return s;
}
double eigenmode_error(int n) {
    auto s=fixture(n);s.settings={0.1,1,0.1,0.2};s.cells[0].dimensions_m={1,1,1};s.cells[0].heat_capacity_j_per_k=1;
    for(int i=0;i<n;++i)s.cells[0].initial_temperature_field_k.push_back(300+10*std::cos(3.141592653589793*(i+0.5)/n));
    Simulation sim(s);sim.run_to_end();double error=0;
    for(int i=0;i<n;++i) {
        double exact=300+10*std::cos(3.141592653589793*(i+0.5)/n)*std::exp(-3.141592653589793*3.141592653589793*0.1);
        error+=std::pow(sim.thermal_states()[i].temperature_k-exact,2);
    }
    require(std::abs(sim.energy_balance_residual_j())<1e-9,"Closed slab energy conservation");
    return std::sqrt(error/n);
}
int main() {try {
    auto s=fixture();s.cells[0].initial_temperature_field_k={400,300};
    Simulation sim(s);std::cout<<"CUDA device: "<<sim.device_name()<<'\n';sim.run_to_end();
    const auto& state=sim.thermal_states();
    const double exact=350+50*std::exp(-1.0);
    require(std::abs(state[0].temperature_k-exact)<0.025,"Two-volume analytic exchange");
    require(std::abs(state[0].temperature_k+state[1].temperature_k-700)<1e-9,"Pairwise conservation");
    require(std::abs(sim.cells()[0].temperature_k-350)<1e-10,"Capacity-weighted macro temperature");
    require(sim.cells()[0].max_temperature_k==state[0].temperature_k,"Macro maximum comes from local field");
    require(sim.thermal_mesh().links.size()==1,"No empty-space links");
    sim.reset();require(sim.thermal_states()[0].temperature_k==400,"Reset restores device field");

    Simulation uniform(fixture(8));uniform.run_to_end();
    for(const auto& v:uniform.thermal_states())require(v.temperature_k==300,"Uniform equilibrium remains exact");
    const double e8=eigenmode_error(8),e16=eigenmode_error(16);
    require(e16<e8*0.4,"Mesh refinement converges toward analytical insulated-slab solution");
    std::cout<<"Slab RMS error: "<<e8<<" -> "<<e16<<" K\n";

    auto hot=fixture(4);hot.settings={1,0.01,0.1,0.2};
    hot.cells[0].runaway_energy_j=100;hot.cells[0].runaway_duration_s=0.2;
    hot.primary_failures={{0,"solid",0,0}};
    Simulation ignition(hot);ignition.runtime().runaway_power_scale=2;ignition.run_to_end();
    require(std::abs(ignition.released_energy_j()-25)<1e-9,"Local ignition only consumes its own finite energy budget");
    require(std::abs(ignition.cells()[0].spent_fraction-0.25)<1e-12,"GPU macro destroyed-volume fraction");
    require(ignition.cells()[0].mode==CellMode::PartiallySpent,"Partial destruction is not reported as healthy/full destruction");
    require(std::abs(ignition.energy_balance_residual_j())<1e-8,"Source energy conservation");
    require(ignition.thermal_states()[3].mode==CellMode::Healthy,"Ignition does not teleport to the whole battery");

    auto cooling=fixture(4);cooling.cells[0].initial_temperature_k=400;
    cooling.ports.push_back({"face","solid",0,"cooling","+x",1,{1,0,0},{1,0,0}});
    cooling.boundaries.push_back({"ambient","face",0,300,2});
    Simulation cool(cooling);cool.runtime().max_dt_s=10;cool.run_to_end();
    require(cool.cells()[0].min_temperature_k>=300 && cool.cells()[0].max_temperature_k<=400,"CFL positivity");
    require(cool.thermal_states().front().temperature_k>cool.thermal_states().back().temperature_k,"Cooling is applied at the surface, not uniformly");
    require(cool.boundary_energy_out_j()>0 && std::abs(cool.energy_balance_residual_j())<1e-8,"Boundary energy ledger");

    auto ramp=fixture(1);ramp.settings={1,0.001,0.1,0.2};
    ramp.cells[0].heat_capacity_j_per_k=1;ramp.cells[0].isothermal=true;
    ramp.ports.push_back({"face","solid",0,"cooling","+x",1,{1,0,0},{1,0,0}});
    AmbientBoundary measured{"measured","face",0,300,1};
    measured.temperature_profile={{0,300},{0.5,300.5},{1,301}};
    ramp.boundaries.push_back(measured);
    Simulation driven(ramp);driven.run_to_end();
    require(std::abs(driven.cells()[0].temperature_k-(300+std::exp(-1.)))<0.0003,"Measured ramp follows analytical forced solution");
    require(std::abs(driven.energy_balance_residual_j())<1e-8,"Measured boundary heat is in the energy ledger");
    driven.reset();driven.run_to_end();
    require(std::abs(driven.cells()[0].temperature_k-(300+std::exp(-1.)))<0.0003,"Reset rewinds measured forcing");
    for(int test=0;test<4;++test) {
        auto invalid=ramp;
        if(test==0)invalid.boundaries[0].temperature_profile.back()[0]=0.9;
        if(test==1)invalid.boundaries[0].temperature_profile[1][0]=0;
        if(test==2)invalid.boundaries[0].temperature_profile[1][1]=-10;
        if(test==3)invalid.boundaries[0].temperature_profile.front()[0]=0.1;
        bool threw=false;try{build_thermal_mesh(invalid);}catch(const std::exception&){threw=true;}
        require(threw,"Invalid measured forcing must be rejected before CUDA launch");
    }

    auto contact=fixture(2);contact.cells[0].position_m={-1.01,0,0};contact.cells[0].initial_temperature_k=400;
    contact.cells.push_back(contact.cells[0]);contact.cells[1].id="cold";contact.cells[1].position_m={1.01,0,0};contact.cells[1].initial_temperature_k=300;
    contact.ports={{"a","solid",0,"contact","+x",1,{-0.01,0,0},{1,0,0}},
                   {"b","cold",1,"contact","-x",1,{0.01,0,0},{-1,0,0}}};
    InterfaceConfig e;e.id="contact";e.port_a="a";e.port_b="b";e.cell_b=1;e.law.background_conductance_w_per_k=2;
    e.law.destroyed_background_conductance_w_per_k=0.2;e.law.damage_onset_k=300;e.law.damage_rate_at_onset_per_s=2;
    e.law.damage_temperature_scale_k=100;e.law.destroyed_l0_m=0.001;e.law.destroyed_l_m=0.0001;
    contact.interfaces.push_back(e);
    Simulation coupled(contact);coupled.run_to_end();
    require(coupled.interfaces()[0].phase==EdgePhase::Destroyed,"Local contact destruction reduced to macro interface");
    require(std::abs(coupled.interfaces()[0].l0_m-0.001)<1e-12,"Destruction changes local l0");
    require(std::abs(coupled.energy_balance_residual_j())<1e-8,"Contact pair conservation");

    auto changing=contact;changing.settings={5,5,5,0.2};
    changing.interfaces[0].law.damage_rate_at_onset_per_s=0;
    Simulation scaled(changing);scaled.runtime().interface_conductance_scale=0.01;scaled.step();
    double slow_bound=scaled.last_dt_s();scaled.runtime().interface_conductance_scale=1000;scaled.step();
    require(scaled.last_dt_s()<slow_bound*0.7,"Changing runtime G invalidates cached CFL bound");
    require(std::abs(scaled.energy_balance_residual_j())<1e-8,"Runtime scaling keeps conservation");

    auto bulk=fixture();bulk.settings={1,1,1,0.2};bulk.cells[0].initial_temperature_field_k={400,300};
    bulk.cells[0].has_bulk_connectivity=true;auto& law=bulk.cells[0].bulk_connectivity;
    law.kind=TransitionKind::FirstOrder;law.transition_temperature_k=340;
    law.l0_m=0.0005;law.l_high_m=0.001;law.activation_width_m=0.00001;
    law.background_conductance_w_per_k=0.1;law.active_conductance_w_per_k=10;
    Simulation switched_bulk(bulk);switched_bulk.step();
    require(std::abs(switched_bulk.last_dt_s()-0.2/10.1)<1e-12,"CFL includes the high-conductivity phase before transfer");
    require(std::abs(switched_bulk.thermal_states()[0].temperature_k-380)<1e-8,"Bulk connectivity changes face flow locally");
    require(std::abs(switched_bulk.energy_balance_residual_j())<1e-8,"Bulk phase transition conserves heat");

    bool rejected=false;auto bad=fixture();bad.cells[0].thermal_divisions[0]=0;
    try{Simulation invalid(bad);}catch(const std::exception&){rejected=true;}
    require(rejected,"Invalid mesh must fail before launch");
    std::cout<<"CUDA field, macro reductions, local ignition, energy, CFL and convergence: PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

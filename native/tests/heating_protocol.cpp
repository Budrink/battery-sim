#include "battery_native/simulation.hpp"
#include "battery_native/gas_mixing.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Scenario fixture() {
    Scenario s;s.settings.duration_s=10.;s.settings.max_dt_s=.05;s.settings.output_interval_s=.5;
    CellConfig c;c.id="heater";c.is_battery=false;c.dimensions_m={.01,.01,.01};
    c.thermal_divisions={1,1,1};c.initial_temperature_k=300.;c.heat_capacity_j_per_k=45.;
    c.electrical.enabled=true;c.electrical.resistance_ohm=1.;
    c.electrical.min_resistance_ohm=c.electrical.max_resistance_ohm=1.;s.cells={c};
    HeatingProtocol p;p.id="ramp";p.heater_id=p.sensor_id="heater";p.heater_object=p.sensor_object=0;
    p.initial_temperature_k=300.;p.ramp_k_per_s=.1;p.feedforward_w=4.5;p.proportional_w_per_k=2.;
    s.heating_protocols={p};return s;
}
int main(){try {
    auto s=fixture();Simulation sim(s);sim.run_to_end();
    require(std::abs(sim.cells()[0].temperature_k-301.)<1e-10,"Ramp does not match analytic RC solution");
    require(std::abs(sim.joule_energy_j()-45.)<1e-9 && std::abs(sim.energy_balance_residual_j())<1e-8,"Heater bypassed UI/I2R ledger");
    const auto e=sim.electrical_states()[0];require(e.power_relative_residual<1e-10,"Controlled drive violates Kirchhoff power");
    Simulation prefix(s);while(prefix.time_s()<2.5-1e-9)prefix.step(2.5-prefix.time_s());
    const auto bytes=prefix.save_checkpoint("heater-test");Simulation restored(s);restored.restore_checkpoint(bytes,"heater-test");
    require(restored.save_checkpoint("heater-test")==bytes,"Heating checkpoint is not byte-stable");
    prefix.run_to_end();restored.run_to_end();
    require(prefix.joule_energy_j()==restored.joule_energy_j() && prefix.cells()[0].temperature_k==restored.cells()[0].temperature_k,"Heating restart diverged");
    s.heating_protocols[0].cutoff_temperature_k=300.5;s.heating_protocols[0].cutoff_rate_k_per_s=.05;
    Simulation stop(s);stop.run_to_end();const auto state=stop.heating_protocol_states()[0];
    require(state.stopped && state.commanded_power_w==0. && state.cutoff_time_s>=4.99 && state.cutoff_time_s<=5.06,"Autonomous heater cutoff failed");
    require(stop.cells()[0].temperature_k<300.506,"Heater continued after cutoff");
    HeatingProtocol p;HeatingProtocolState before;before.previous_temperature_k=298.15;
    const auto off=heating_command(p,before,1.,400.);
    require(off.commanded_power_w==0. && !off.stopped,"Controller is a temperature clamp/cooler");
    const auto a=heating_command(p,before,1.,300.),b=heating_command(p,before,1.,300.);
    require(a.commanded_power_w==b.commanded_power_w && before.previous_time_s==0.,"Prepare mutates controller state");
    kawasaki::Settings gas;gas.thermal_stirring_coefficient=.05;
    kawasaki::Node hot{.001,1000.,.001,0.},cold{.001,300.,.001,0.};kawasaki::Edge edge{0,1,.0001,.7,.02,0};
    const double d=gas_effective_diffusivity(hot,cold,edge,gas,2e-5);
    require(d>2e-5 && d==gas_effective_diffusivity(cold,hot,edge,gas,2e-5),"Free-gas stirring ignores thermal contrast");
    require(gas_effective_diffusivity(cold,cold,edge,gas,2e-5)==2e-5,"Uniform gas invents circulation");
    edge.molecular_mixing=true;require(gas_effective_diffusivity(hot,cold,edge,gas,2e-5)==2e-5,"Stirring changed internal electrolyte pores");
    edge.molecular_mixing=false;const double flow=kawasaki::flow(hot,cold,edge,gas);
    double physical_diffusivity=0.;
    for(double h:{.04,.02,.01,.005}) {
        edge.mixing_length=h;
        hot.energy=hot.mass*gas.cv*(500.+1000.*h/2.);
        cold.energy=cold.mass*gas.cv*(500.-1000.*h/2.);
        const double value=gas_effective_diffusivity(hot,cold,edge,gas,2e-5);
        if(physical_diffusivity==0.)physical_diffusivity=value;
        require(std::abs(value-physical_diffusivity)<1e-15,"Grid refinement changed physical stirring for a linear field");
    }
    edge.mixing_length=.02;hot.energy=1000.;cold.energy=300.;
    edge.hydraulic_resistance_factor=4.;require(std::abs(kawasaki::flow(hot,cold,edge,gas)-flow/2)<1e-14,"Hydraulic resistance law");
    require(gas_effective_diffusivity(hot,cold,edge,gas,2e-5)==d,"Hydraulic resistance changed diffusion geometry");
    std::cout<<"Heating protocol: analytic ramp, strict UI, cutoff, restart, immutable prepare; stirring and passage scaling passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

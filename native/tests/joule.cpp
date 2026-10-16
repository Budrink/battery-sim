#include "battery_native/simulation.hpp"
#include "battery_native/joule.hpp"
#include "battery_native/conversion.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Scenario fixture() {
    Scenario s;s.settings={2.,.2,.5,.4};CellConfig c;c.id="battery";c.dimensions_m={.02,.02,.02};
    c.thermal_divisions={3,3,3};c.heat_capacity_j_per_k=10.;c.initial_temperature_k=300.;
    c.electrical.enabled=true;c.electrical.current_a=35.;c.electrical.resistance_ohm=.02;
    s.cells={c};return s;
}
void analytic_and_mesh() {
    for(int n:{1,3,4})for(bool uniform:{false,true})for(double sign:{-1.,1.}) {
        auto scene=fixture();scene.cells[0].thermal_divisions={n,n,n};scene.cells[0].isothermal=uniform;
        scene.cells[0].electrical.current_a*=sign;Simulation sim(scene);sim.run_to_end();
        const double heat=35.*35.*.02*2.;
        require(std::abs(sim.joule_energy_j()-heat)<1e-10,"Whole-cell Joule heat multiplied by voxel count");
        require(std::abs(sim.cells()[0].temperature_k-(300.+heat/10.))<1e-10,"I^2 R temperature differs from analytic solution");
        require(sim.released_energy_j()==0 && std::abs(sim.chemical_balance_residual_j())<1e-12,"Joule heat is charged to chemical fuel");
        require(std::abs(sim.energy_balance_residual_j())<1e-8,"Joule heat missing from global thermal balance");
    }
}
void profile_and_resistance() {
    ElectricalLaw p;CurrentPoint points[]{{0.,-2.},{1.,2.},{2.,0.}};CurrentView drive{points,3};
    require(std::abs(squared_current_integral(p,drive,0.,2.)-8./3.)<1e-14,"Piecewise linear I^2 integral");
    require(std::abs(squared_current_integral(p,drive,.3,1.4)-
        squared_current_integral(p,drive,.3,.2)-squared_current_integral(p,drive,.5,1.2))<1e-14,"Current knots change energy with time partition");
    auto scene=fixture();scene.cells[0].electrical.current_a=0.;
    scene.cells[0].current_profile_a={{{0.,-2.}},{{1.,2.}},{{2.,0.}}};
    scene.settings.max_dt_s=2.;Simulation one(scene);one.run_to_end();
    scene.settings.max_dt_s=.037;Simulation split(scene);split.run_to_end();
    require(std::abs(one.joule_energy_j()-.02*8./3.)<1e-14 && std::abs(split.joule_energy_j()-one.joule_energy_j())<1e-14,"CUDA integration of ramped/reversed current");
    p.temperature_log_slope_per_k=-.01;p.damage_log_slope=2.;
    require(electrical_resistance(p,400.,0.)<electrical_resistance(p,300.,0.),"Temperature resistance law ignored");
    require(electrical_resistance(p,300.,1.)>electrical_resistance(p,300.,0.),"Defect resistance law ignored");
    require(electrical_resistance(p,1e6,0.)==p.min_resistance_ohm,"Resistance lower bound");
    p.damage_log_slope=1000.;require(electrical_resistance(p,300.,1.)==p.max_resistance_ohm,"Resistance upper bound");
}
void mixed_energy_and_tiles() {
    auto scene=fixture();scene.gas_model.reference_temperature_k=300.;scene.gas_model.split_transport=true;scene.gas_model.cuda_network_transport=true;
    auto& p=scene.cells[0].conversion;p.enabled=true;p.mass_kg=.004;p.net_heat_j_per_kg=200000.;
    p.degradation_rate_per_s=.02;p.degradation_reference_k=300.;p.degradation_activation_j_per_mol=20000.;
    p.reaction_rate_per_s=.1;p.reaction_reference_k=300.;p.reaction_activation_j_per_mol=100000.;
    p.initial_precursor_fraction=.1;p.gas_compartment=0;
    GasCompartment g;g.id="internal";g.owner_cell=0;g.volume_m3=1e-6;g.initial_temperature_k=300.;g.initial_mass_kg=1e-6;
    scene.gas_compartments={g};scene.settings.duration_s=.1;Simulation mixed(scene);mixed.run_to_end();
    const auto d=mixed.process_diagnostics();
    require(mixed.released_energy_j()>0 && mixed.joule_energy_j()>0,"Chemical and Joule sources not both active");
    require(std::abs(d.chemical_residual_j)<1e-9 && std::abs(d.thermal_residual_j)<1e-8,"Mixed source ledger");
    const auto mesh=build_thermal_mesh(scene);std::vector<ThermalState> baseline;
    for(int tile:{64,128,256}) {
        CudaSolver solver(scene,mesh,tile);RuntimeParameters runtime;for(int i=0;i<10;++i)solver.step(i*.0001,.0001,.4,runtime);
        std::vector<ThermalState> state;solver.download_field(state);
        if(baseline.empty())baseline=state;
        else for(std::size_t i=0;i<state.size();++i)require(state[i].stored_energy_j==baseline[i].stored_energy_j &&
            state[i].joule_energy_j==baseline[i].joule_energy_j,"Joule source depends on CUDA block");
    }
}
void heat_neutral_overcharge() {
    ConversionLaw c;c.mass_kg=.004;c.net_heat_j_per_kg=200000.;
    c.degradation_rate_per_s=1e-8;c.degradation_reference_k=300.;c.degradation_activation_j_per_mol=40000.;
    c.reaction_rate_per_s=0.;c.overcharge_conversion_per_charge=2.;
    ElectricalLaw e;e.enabled=true;e.overcharge_fraction=1.;e.nominal_capacity_coulomb=35.*3600.;e.reference_temperature_k=300.;
    const double extra=overcharge_r1_rate(c,e,300.,35.);
    ThermalState s;s.temperature_k=300.;s.reactant_kg=c.mass_kg;GasModel g;
    const double q0=conversion_chemical(c,s);
    const double heat=advance_conversion(c,g,s,1.,10.,1.,0.,extra);
    require(heat==0. && s.generated_gas_kg==0. && s.released_energy_j==0.,"Electrical R1 creates a fourth heat source");
    require(std::abs(conversion_chemical(c,s)-q0)<1e-12,"Electrical acceleration creates/loses chemical reserve");
    require(std::abs(s.reactant_kg-c.mass_kg*std::exp(-(1e-8+extra)*10.))<1e-16,"Overcharge R1 does not integrate physical time");
    require(overcharge_r1_rate(c,e,300.,-35.)==0. && overcharge_r1_rate(c,e,300.,0.)==0.,"Discharge/zero current treated as overcharge");
    e.overcharge_fraction=0.;require(overcharge_r1_rate(c,e,300.,35.)==0.,"Ordinary charging treated as overcharge");
    CurrentPoint points[]{{0.,-2.},{1.,2.},{2.,0.}};
    require(std::abs(positive_current_integral(e,{points,3},0.,2.)-1.5)<1e-15,"Positive charge integral at reversal");
    auto scene=fixture();scene.settings.duration_s=.1;
    auto& law=scene.cells[0].conversion;law.enabled=true;law.mass_kg=.004;law.degradation_rate_per_s=1e-8;
    law.reaction_activation_j_per_mol=100000.;law.overcharge_conversion_per_charge=2.;law.gas_compartment=0;
    scene.cells[0].electrical.overcharge_fraction=1.;scene.cells[0].electrical.nominal_capacity_coulomb=35.*3600.;
    GasCompartment gas;gas.id="pore";gas.owner_cell=0;gas.volume_m3=1e-6;gas.initial_mass_kg=1e-6;gas.initial_temperature_k=300.;
    scene.gas_compartments={gas};Simulation charging(scene);charging.run_to_end();
    scene.cells[0].electrical.current_a=-35.;Simulation discharging(scene);discharging.run_to_end();
    require(charging.cells()[0].damage>discharging.cells()[0].damage+1e-6,"CUDA source omits electrical R1 acceleration");
    require(std::abs(charging.energy_balance_residual_j())<1e-8 && charging.released_energy_j()==0.,"Charge-assisted R1 breaks heat neutrality");

    // With frozen Arrhenius, integrated positive charge, not number of steps,
    // determines the surviving virgin stock. Negligible Joule heating holds T.
    scene.cells[0].electrical.resistance_ohm=1e-12;scene.cells[0].electrical.min_resistance_ohm=1e-12;
    scene.cells[0].current_profile_a={{{0.,-2.}},{{1.,2.}},{{2.,0.}}};
    scene.cells[0].conversion.degradation_reference_k=300.;
    scene.cells[0].electrical.reference_temperature_k=300.;
    scene.cells[0].conversion.reaction_rate_per_s=1e-30;
    scene.settings.duration_s=2.;scene.settings.max_dt_s=2.;Simulation one(scene);one.run_to_end();
    scene.settings.max_dt_s=.037;Simulation split(scene);split.run_to_end();
    const double expected=-std::expm1(-2e-8-2.*1.5/(35.*3600.));
    require(std::abs(one.cells()[0].damage-expected)<1e-13 &&
        std::abs(split.cells()[0].damage-expected)<1e-13,"Electrical R1 depends on time partition/current reversal");
}
int main(){try{analytic_and_mesh();profile_and_resistance();mixed_energy_and_tiles();heat_neutral_overcharge();
    std::cout<<"Joule: I^2R, exact current profiles, mesh/block invariance and independent thermal/chemical ledgers pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

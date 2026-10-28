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
void passive_resistive_heater() {
    auto s=fixture();s.cells[0].is_battery=false;s.cells[0].id="heater";
    Simulation sim(s);sim.run_to_end();
    require(sim.battery_count()==0 && sim.events().empty(),"Resistive heater is reported as a battery failure");
    require(std::abs(sim.joule_energy_j()-49.)<1e-10 &&
        std::abs(sim.cells()[0].temperature_k-304.9)<1e-10,"Passive heater I2R balance");
    require(std::abs(sim.energy_balance_residual_j())<1e-8,"Passive heater heat is missing from ledger");
    s.cells[0].electrical.charge_acceptance_fraction=.5;
    bool rejected=false;try {build_thermal_mesh(s);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Passive heater accepts battery charge chemistry");
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
    p.temperature_log_slope_per_k=-.01;
    require(electrical_resistance(p,400.)<electrical_resistance(p,300.),"Temperature resistance law ignored");
    require(electrical_resistance(p,1e6)==p.min_resistance_ohm,"Resistance lower bound");
    p.temperature_log_slope_per_k=1000.;require(electrical_resistance(p,400.)==p.max_resistance_ohm,"Resistance upper bound");
    ConversionLaw chemistry;chemistry.enabled=true;p.product_conductivity_ratio=.2;
    require(electrical_mixture_factor(p,chemistry,0.,0.,1.,1.)==.2,"Condensed residue became an electrical insulator");
    require(electrical_mixture_factor(p,chemistry,.5,0.,.5,1.)==.6,"Electrical mixture law ignored material inventory");
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
void kirchhoff_axes_and_paths() {
    for(int axis=0;axis<3;++axis) {
        auto scene=fixture();auto& c=scene.cells[0];c.dimensions_m={.011,.087,.187};
        c.thermal_divisions={2,3,4};c.electrical.terminal_axis=axis;
        Simulation sim(scene);sim.step(1e-5);const auto e=sim.electrical_states()[0];
        require(e.error==0 && std::abs(e.resistance_ohm-.02)<1e-12,"Reference whole-cell R depends on mesh/axis/aspect ratio");
        require(std::abs(e.current_a-e.outlet_current_a)<1e-9,"Kirchhoff loses terminal current");
        require(e.kcl_relative_residual<1e-10 && e.power_relative_residual<1e-10,"Electrical KCL/power residual");
    }
    for(bool parallel:{false,true}) {
        auto scene=fixture();auto& c=scene.cells[0];
        c.thermal_divisions=parallel?std::array<int,3>{1,2,1}:std::array<int,3>{2,1,1};
        c.initial_temperature_field_k={300.,310.};c.heat_capacity_j_per_k=1000.;
        c.electrical.terminal_axis=0;c.electrical.reference_temperature_k=300.;
        c.electrical.temperature_log_slope_per_k=std::log(4.)/10.;
        Simulation sim(scene);sim.step(1e-5);const auto e=sim.electrical_states()[0];const auto& nodes=sim.thermal_states();
        const double expected=.02*(parallel?1.6:2.5);
        require(std::abs(e.resistance_ohm-expected)<1e-12,"Kirchhoff differs from analytic series/parallel equivalent");
        const double ratio=nodes[1].joule_energy_j/nodes[0].joule_energy_j;
        require(std::abs(ratio-(parallel?.25:4.))<1e-10,"Series/parallel local Joule heat does not follow redistributed currents");
        require(std::abs(sim.joule_energy_j()-35.*35.*expected*sim.time_s())<1e-12,"Sum local Joule heat != I*ohmic_drop*dt");
    }
    // Multiple voxels per thread, nonuniform coefficients and a nontrivial CG
    // solution: eight layers in series, each containing 64 parallel paths.
    auto scene=fixture();auto& c=scene.cells[0];c.thermal_divisions={8,8,8};c.heat_capacity_j_per_k=1000.;
    c.electrical.reference_temperature_k=300.;c.electrical.temperature_log_slope_per_k=std::log(4.)/10.;
    double expected=0.;
    for(int z=0;z<8;++z) {
        const double temperature=300.+10.*z/7.;expected+=.02*std::pow(4.,z/7.)/8.;
        for(int i=0;i<64;++i)c.initial_temperature_field_k.push_back(temperature);
    }
    Simulation large(scene);large.step(1e-5);const auto e=large.electrical_states()[0];
    require(std::abs(e.resistance_ohm-expected)<1e-10 && e.iterations>0,"512-node Kirchhoff solve differs from layered series solution");
}
void finite_product_conductivity() {
    auto scene=fixture();scene.settings.duration_s=.1;scene.settings.max_dt_s=.01;
    auto& c=scene.cells[0];c.electrical.current_a=.01;c.electrical.product_conductivity_ratio=.2;c.heat_capacity_j_per_k=1000.;
    auto& law=c.conversion;law.enabled=true;law.mass_kg=.004;law.precursor_yield=0.;law.gas_compartment=0;
    law.degradation_rate_per_s=3.;law.degradation_reference_k=300.;law.degradation_activation_j_per_mol=20000.;
    law.reaction_activation_j_per_mol=100000.;law.net_heat_j_per_kg=200000.;
    GasCompartment gas;gas.id="pore";gas.owner_cell=0;gas.volume_m3=1e-6;gas.initial_mass_kg=1e-6;gas.initial_temperature_k=300.;
    scene.gas_compartments={gas};Simulation sim(scene);sim.run_to_end();const auto e=sim.electrical_states()[0];
    const double virgin=std::exp(-3.*e.time_s),mix=virgin+.2*(1.-virgin);
    require(std::abs(e.resistance_ohm-.02/mix)<1e-9,"R1 product mixture does not modify Kirchhoff resistance");
    require(e.resistance_ohm>.02 && sim.joule_energy_j()>0. && sim.released_energy_j()==0.,"Inert residue either isolates current or invents reaction heat");
    require(std::abs(sim.energy_balance_residual_j())<1e-7 && std::abs(sim.chemical_balance_residual_j())<1e-8,"R1 conductivity change breaks energy/chemical ledger");
    for(double bad:{0.,-1.,1.1}) {
        auto invalid=scene;invalid.cells[0].electrical.product_conductivity_ratio=bad;bool rejected=false;
        try{Simulation test(invalid);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"Invalid product conductivity silently accepted");
    }
}
void voltage_work_bound() {
    auto s=fixture();s.cells[0].electrical.reversible_voltage_floor_v=3.65;
    s.cells[0].voltage_profile_v={{{0.,4.4}},{{2.,4.4}}};
    Simulation valid(s);valid.step(.001);
    s.cells[0].voltage_profile_v={{{0.,4.2}},{{2.,4.2}}};
    bool rejected=false;try{Simulation invalid(s);invalid.step(.001);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Kirchhoff heating exceeded measured electrical work minus declared reversible-energy prior");
    ElectricalLaw e;CurrentPoint i[]{{0.,0.},{10.,2.}},u[]{{0.,4.},{10.,6.}};
    const double work=electrical_work_integral(e,{i,2},{u,2},0.,10.);
    require(std::abs(work-160./3.)<1e-12,"I*U integral must use both linear drive profiles");
    require(std::abs(work-electrical_work_integral(e,{i,2},{u,2},0.,3.1)-electrical_work_integral(e,{i,2},{u,2},3.1,6.9))<1e-12,
        "Electrical work changes when a step is partitioned");
    // The direct solver API may request a cap larger than runtime.max_dt_s.
    // A still-dormant drive inside that interval must not suppress I²R.
    auto delayed=fixture();delayed.cells[0].electrical.current_a=0.;
    delayed.cells[0].current_profile_a={{{0.,0.}},{{1.,0.}},{{1.000001,2.}},{{2.,2.}}};
    const auto mesh=build_thermal_mesh(delayed);CudaSolver solver(delayed,mesh);RuntimeParameters runtime;
    const double dt=solver.step(0.,2.,.4,runtime);
    require(dt>1.1,"Dormant-drive fixture did not cross its switch-on time");
    std::vector<CellState> cells;std::vector<InterfaceState> links;solver.download_macros(cells,links);
    CurrentPoint knots[]{{0.,0.},{1.,0.},{1.000001,2.},{2.,2.}};
    const double expected=.02*squared_current_integral(delayed.cells[0].electrical,{knots,4},0.,dt);
    require(std::abs(cells[0].joule_energy_j-expected)<1e-12 && expected>0.,"Delayed current start skipped the Kirchhoff/Joule solve");
}
void completed_drive_profiles() {
    ElectricalLaw law;
    CurrentPoint current[]{{0.,2.},{2.,2.}};
    CurrentPoint voltage[]{{0.,4.},{1.,6.}};
    require(std::abs(electrical_work_integral(law,{current,2},{voltage,2},0.,2.)-22.)<1e-13,
        "Electrical work cannot cross the shorter voltage profile's last knot");
    require(std::abs(electrical_work_integral(law,{current,2},{voltage,2},1.1,.4)-4.8)<1e-13,
        "Completed voltage profile must hold its last value");
    CurrentPoint off[]{{0.,2.},{1.,0.}},single[]{{0.,4.}};
    require(electrical_work_integral(law,{off,2},{voltage,2},1.1,.4)==0.,
        "Source OFF/compliance tail has nonzero work");
    require(std::abs(electrical_work_integral(law,{current,2},{single,1},0.,2.)-16.)<1e-13,
        "Single-point voltage profile cannot be integrated");
    // Actual CUDA regression: the physical replay continues after measured
    // electrical current ends. Validation still evaluates the zero-work tail.
    auto scene=fixture();scene.settings.duration_s=2.;scene.settings.max_dt_s=.1;
    auto& cell=scene.cells[0];cell.electrical.reversible_voltage_floor_v=3.65;
    cell.current_profile_a={{{0.,35.}},{{.49,35.}},{{.5,0.}}};
    cell.voltage_profile_v={{{0.,4.5}},{{.5,60.}}};
    Simulation sim(scene);sim.run_to_end();
    CurrentPoint drive[]{{0.,35.},{.49,35.},{.5,0.}};
    const double expected=.02*squared_current_integral(cell.electrical,{drive,3},0.,2.);
    require(std::abs(sim.time_s()-2.)<1e-12 && std::abs(sim.joule_energy_j()-expected)<1e-10,
        "CUDA replay fails to continue through the source-OFF tail");
}
void heat_neutral_charging() {
    ConversionLaw c;c.mass_kg=.004;c.net_heat_j_per_kg=200000.;
    c.degradation_rate_per_s=1e-8;c.degradation_reference_k=300.;c.degradation_activation_j_per_mol=40000.;
    c.reaction_rate_per_s=0.;c.charge_conversion_per_nominal_charge=2.;
    ElectricalLaw e;e.enabled=true;e.charge_acceptance_fraction=1.;e.nominal_capacity_coulomb=35.*3600.;e.reference_temperature_k=300.;
    const double extra=charging_precursor_rate(c,e,35.);
    ThermalState s;s.temperature_k=300.;s.reactant_kg=c.mass_kg;GasModel g;
    const double q0=conversion_chemical(c,s);
    const double heat=advance_conversion(c,g,s,1.,10.,1.,0.,extra);
    require(heat==0. && s.generated_gas_kg==0. && s.released_energy_j==0.,"Charging creates a fourth heat source");
    require(std::abs(conversion_chemical(c,s)-q0-s.electrical_chemical_input_j)<1e-12,"Charging input not stored in chemical reserve");
    require(s.electrical_chemical_input_j>0. && s.precursor_from_charge_kg>0. && s.precursor_from_degradation_kg>0.,
        "The two formation sources are not accounted separately");
    require(std::abs(s.reactant_kg-c.mass_kg*std::exp(-(1e-8+extra)*10.))<1e-16,"Charging does not integrate physical time");
    c.degradation_activation_j_per_mol=180000.;
    require(charging_precursor_rate(c,e,35.)==extra,"Charge formation inherits the thermal R1 Arrhenius");
    require(charging_precursor_rate(c,e,-35.)==0. && charging_precursor_rate(c,e,0.)==0.,"Discharge/zero current formed charging precursor");
    e.charge_acceptance_fraction=0.;require(charging_precursor_rate(c,e,35.)==0.,"Disabled charging acceptance ignored");
    CurrentPoint points[]{{0.,-2.},{1.,2.},{2.,0.}};
    require(std::abs(positive_current_integral(e,{points,3},0.,2.)-1.5)<1e-15,"Positive charge integral at reversal");
    auto scene=fixture();scene.settings.duration_s=.1;
    auto& law=scene.cells[0].conversion;law.enabled=true;law.mass_kg=.004;law.degradation_rate_per_s=1e-8;
    law.reaction_activation_j_per_mol=100000.;law.charge_conversion_per_nominal_charge=2.;law.gas_compartment=0;
    scene.cells[0].electrical.charge_acceptance_fraction=1.;scene.cells[0].electrical.nominal_capacity_coulomb=35.*3600.;
    GasCompartment gas;gas.id="pore";gas.owner_cell=0;gas.volume_m3=1e-6;gas.initial_mass_kg=1e-6;gas.initial_temperature_k=300.;
    scene.gas_compartments={gas};Simulation charging(scene);charging.run_to_end();
    scene.cells[0].electrical.current_a=-35.;Simulation discharging(scene);discharging.run_to_end();
    require(std::abs(charging.cells()[0].damage-discharging.cells()[0].damage)<1e-12,"Charging spuriously accelerates R1 damage");
    double charged_stock=0.;for(const auto& node:charging.thermal_states())charged_stock+=node.healthy_charged_kg;
    require(charged_stock>1e-9,"CUDA source omits healthy charging stock");
    require(std::abs(charging.energy_balance_residual_j())<1e-8 && charging.released_energy_j()==0.,"Charging breaks heat neutrality");

    // With frozen Arrhenius, integrated positive charge, not number of steps,
    // determines the surviving virgin stock. Negligible Joule heating holds T.
    scene.cells[0].electrical.resistance_ohm=1e-12;scene.cells[0].electrical.min_resistance_ohm=1e-12;
    scene.cells[0].current_profile_a={{{0.,-2.}},{{1.,2.}},{{2.,0.}}};
    scene.cells[0].conversion.degradation_reference_k=300.;
    scene.cells[0].electrical.reference_temperature_k=300.;
    scene.cells[0].conversion.reaction_rate_per_s=1e-30;
    scene.settings.duration_s=2.;scene.settings.max_dt_s=2.;Simulation one(scene);one.run_to_end();
    scene.settings.max_dt_s=.037;Simulation split(scene);split.run_to_end();
    const double expected=-std::expm1(-2e-8);
    require(std::abs(one.cells()[0].damage-expected)<1e-13 &&
        std::abs(split.cells()[0].damage-expected)<1e-13,"R1 damage inherits charge/current reversal");
}
void charge_resistance_and_storage_budget() {
    ElectricalLaw p;p.resistance_ohm=.02;p.nominal_capacity_coulomb=10.;p.charge_log_slope=std::log(4.);
    CurrentPoint drive[]{{0.,1.},{10.,1.}};
    require(positive_current_integral(p,{drive,2},0.,10.)==10.,"Passed charge integral");
    require(electrical_charge_polarization_fraction(0.,0.)==0. &&
        electrical_charge_polarization_fraction(1.,0.)==0. &&
        electrical_charge_polarization_fraction(.75,.25)==.25,
        "Charging polarization must use actual healthy charged stock");
    require(electrical_charge_polarization_fraction(.75,.10)<electrical_charge_polarization_fraction(.75,.25),
        "Consumed charging precursor did not reduce polarization");
    ConversionLaw reaction;reaction.degradation_rate_per_s=0.;reaction.reaction_rate_per_s=1.;
    reaction.reaction_reference_k=300.;reaction.reaction_activation_j_per_mol=0.;
    ThermalState inventory;inventory.temperature_k=300.;inventory.reactant_kg=.003;
    inventory.activated_kg=inventory.healthy_charged_kg=.001;
    const double before_polarization=electrical_charge_polarization_fraction(inventory.reactant_kg,inventory.healthy_charged_kg);
    advance_material_inventory(reaction,inventory,1.,1.,0.,.5);
    const double after_polarization=electrical_charge_polarization_fraction(inventory.reactant_kg,inventory.healthy_charged_kg);
    require(after_polarization<before_polarization &&
        electrical_resistance(p,300.,after_polarization)<electrical_resistance(p,300.,before_polarization),
        "R2 consumption does not feed back into charging polarization");
    require(std::abs(electrical_resistance(p,p.reference_temperature_k,1.)-.08)<1e-12,
        "Charge log-slope does not scale the intact ohmic R");
    p.charge_log_slope=0.;
    require(electrical_resistance(p,p.reference_temperature_k,1.)==.02,"Zero charge slope must recover R0");

    auto scene=fixture();scene.settings.duration_s=1.;scene.settings.max_dt_s=.05;
    auto& cell=scene.cells[0];cell.electrical.current_a=1.;cell.electrical.nominal_capacity_coulomb=1.;
    cell.electrical.charge_acceptance_fraction=1.;
    cell.conversion.enabled=true;cell.conversion.mass_kg=.004;cell.conversion.net_heat_j_per_kg=0.;
    cell.conversion.degradation_rate_per_s=1e-30;cell.conversion.reaction_rate_per_s=1e-30;
    cell.conversion.charge_conversion_per_nominal_charge=2.;cell.conversion.gas_compartment=0;
    GasCompartment pore;pore.id="pore";pore.owner_cell=0;pore.volume_m3=1e-6;
    pore.initial_mass_kg=1e-6;pore.initial_temperature_k=300.;scene.gas_compartments={pore};
    cell.electrical.charge_log_slope=std::log(4.);
    Simulation grown(scene);grown.run_to_end();
    cell.electrical.charge_log_slope=0.;
    Simulation flat(scene);flat.run_to_end();
    require(grown.joule_energy_j()>flat.joule_energy_j()*1.2,"Charge-dependent R does not increase I²R");
    require(std::abs(flat.joule_energy_j()-.02)<1e-10,"Zero charge slope changed the ohmic baseline");
    require(std::abs(grown.energy_balance_residual_j())<1e-8 && std::abs(flat.energy_balance_residual_j())<1e-8,
        "Charge-dependent R breaks the thermal ledger");
    cell.electrical.charge_log_slope=std::log(4.);cell.electrical.charge_acceptance_fraction=0.;
    Simulation unaccepted(scene);unaccepted.run_to_end();
    require(std::abs(unaccepted.joule_energy_j()-flat.joule_energy_j())<1e-10,
        "Unaccepted charging current increased polarization without stored precursor");

    auto invalid=fixture();invalid.cells[0].electrical.charge_log_slope=1.;
    bool rejected=false;try{Simulation bad(invalid);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Charge-dependent R without a nominal capacity was accepted");

    auto pay=fixture();pay.settings.duration_s=.2;pay.settings.max_dt_s=.01;pay.gas_model.reference_temperature_k=300.;
    auto& c=pay.cells[0];c.initial_temperature_k=300.;c.electrical.current_a=35.;c.electrical.resistance_ohm=1e-4;
    c.electrical.min_resistance_ohm=1e-6;c.electrical.reference_temperature_k=300.;
    c.electrical.reversible_voltage_floor_v=3.65;c.electrical.charge_acceptance_fraction=1.;
    c.electrical.nominal_capacity_coulomb=35.*3600.;
    c.voltage_profile_v={{{0.,4.2}},{{1.,4.2}}};
    auto& law=c.conversion;law.enabled=true;law.mass_kg=.004;law.net_heat_j_per_kg=200000.;
    law.degradation_rate_per_s=1e-12;law.degradation_reference_k=300.;law.degradation_activation_j_per_mol=40000.;
    law.reaction_rate_per_s=1e-30;law.reaction_reference_k=300.;law.reaction_activation_j_per_mol=100000.;
    law.charge_conversion_per_nominal_charge=10000.;law.gas_compartment=0;
    GasCompartment gas;gas.id="pore";gas.owner_cell=0;gas.volume_m3=1e-6;gas.initial_mass_kg=1e-6;gas.initial_temperature_k=300.;
    pay.gas_compartments={gas};
    Simulation capped(pay);capped.run_to_end();
    auto high=pay;high.cells[0].voltage_profile_v={{{0.,42.}},{{1.,42.}}};
    Simulation supplied(high);supplied.run_to_end();
    CurrentPoint cur[]{{0.,35.},{1.,35.}},volt[]{{0.,4.2},{1.,4.2}};
    const double budget=electrical_work_integral(c.electrical,{cur,2},{volt,2},0.,capped.time_s());
    const auto d=capped.process_diagnostics();
    require(supplied.process_diagnostics().electrical_chemical_input_j>d.electrical_chemical_input_j*1.05,
        "Electrical work does not constrain precursor storage");
    require(capped.joule_energy_j()+d.electrical_chemical_input_j<=budget+1e-6,
        "Joule heat plus chemical storage exceeded terminal electrical work");
    require(d.electrical_chemical_input_j>electrical_dissipative_budget(c.electrical,{cur,2},{volt,2},0.,capped.time_s()),
        "EMF work was incorrectly excluded from chemical storage");
    require(std::abs(d.thermal_residual_j)<1e-8 && std::abs(d.chemical_residual_j)<1e-9 &&
        capped.released_energy_j()<1e-12,"Paid charging invents heat or breaks ledgers");
    auto aging=pay;aging.cells[0].electrical.charge_acceptance_fraction=0.;aging.cells[0].electrical.current_a=0.;
    aging.cells[0].conversion.degradation_rate_per_s=.1;
    Simulation degraded(aging);degraded.run_to_end();
    require(degraded.thermal_states()[0].precursor_from_degradation_kg>0. &&
        degraded.process_diagnostics().electrical_chemical_input_j==0.,"Thermal degradation was electrically charged");
    pay.cells[0].voltage_profile_v={};pay.cells[0].electrical.reversible_voltage_floor_v=0.;
    bool missing_voltage=false;try{Simulation invalid(pay);}catch(const std::runtime_error&){missing_voltage=true;}
    require(missing_voltage,"Charge precursor appeared without a measurable electrical energy source");
}
void stored_charge_energy_reaction() {
    ConversionLaw p;p.mass_kg=.004;p.net_heat_j_per_kg=200000.;p.degradation_rate_per_s=.03;
    p.degradation_reference_k=300.;p.reaction_rate_per_s=.2;p.reaction_reference_k=300.;
    p.precursor_yield=.7;p.gas_chemical_fraction=.3;
    GasModel g;ThermalState initial;initial.temperature_k=300.;initial.reactant_kg=.004;
    auto whole=initial,split=initial;
    advance_conversion(p,g,whole,1.,2.,1.,0.,.08);
    for(int j=0;j<20;++j)advance_conversion(p,g,split,1.,.1,1.,j*.1,.08);
    require(std::abs(whole.activated_kg-split.activated_kg)<1e-15 &&
        std::abs(whole.precursor_charge_energy_j-split.precursor_charge_energy_j)<1e-11 &&
        std::abs(whole.electrical_chemical_input_j-split.electrical_chemical_input_j)<1e-11,
        "Exact mixed-source chemistry depends on time partition");
    // No gas mass in this fixture; count R3 reserve analytically from R2 heat.
    const double released=whole.released_energy_j/(1.-p.gas_chemical_fraction);
    require(std::abs(conversion_chemical(p,whole)+whole.sterilized_chemical_j+released-
        conversion_chemical(p,initial)-whole.electrical_chemical_input_j)<1e-10,"Mixed-source full chemical balance");
    const double input=whole.electrical_chemical_input_j;
    p.degradation_rate_per_s=0.;p.reaction_rate_per_s=10.;
    advance_conversion(p,g,whole,1.,5.,1.,2.); // source OFF, stored B keeps reacting
    require(whole.electrical_chemical_input_j==input && whole.precursor_charge_energy_j<1e-12 &&
        whole.released_energy_j>split.released_energy_j,"Stored electrical energy does not release after source OFF");
}
void cuda_charged_precursor_releases_after_off() {
    auto scene=fixture();scene.settings={1.5,.02,.1,.4};
    scene.gas_model.reference_temperature_k=300.;scene.gas_model.split_transport=true;
    scene.gas_model.cuda_network_transport=true;
    auto& cell=scene.cells[0];cell.current_profile_a={{{0.,35.}},{{.49,35.}},{{.5,0.}},{{1.5,0.}}};
    cell.voltage_profile_v={{{0.,5.}},{{1.5,5.}}};
    cell.electrical.charge_acceptance_fraction=1.;cell.electrical.nominal_capacity_coulomb=126000.;
    auto& c=cell.conversion;c.enabled=true;c.mass_kg=.004;c.net_heat_j_per_kg=200000.;
    c.degradation_rate_per_s=.01;c.degradation_reference_k=300.;c.degradation_activation_j_per_mol=40000.;
    c.reaction_rate_per_s=.4;c.reaction_reference_k=300.;c.reaction_activation_j_per_mol=100000.;
    c.charge_conversion_per_nominal_charge=200.;c.gas_yield=.1;c.gas_chemical_fraction=.3;c.gas_compartment=0;
    GasCompartment g;g.id="inside";g.owner_cell=0;g.volume_m3=1e-6;
    g.initial_mass_kg=1e-6;g.initial_temperature_k=300.;scene.gas_compartments={g};
    Simulation sim(scene);while(sim.time_s()<.5-1e-12)sim.step(.5-sim.time_s());
    const auto before=sim.process_diagnostics();const double released=sim.released_energy_j();
    sim.run_to_end();const auto after=sim.process_diagnostics();
    require(before.electrical_chemical_input_j>0. && after.electrical_chemical_input_j==before.electrical_chemical_input_j,
        "CUDA chemical storage was lost/created after source OFF");
    require(sim.released_energy_j()>released && after.gas[0].chemical_j>before.gas[0].chemical_j,
        "CUDA stored precursor stopped reacting or lost R3 reserve after OFF");
    require(std::abs(after.thermal_residual_j)<1e-7 && std::abs(after.chemical_residual_j)<1e-7 &&
        std::abs(after.mass_residual_kg)<1e-12,"CUDA charging/R2/gas full ledger");
    const auto mesh=build_thermal_mesh(scene);std::vector<ThermalState> reference;
    for(int tile:{64,128,256}) {
        CudaSolver solver(scene,mesh,tile);RuntimeParameters runtime;
        for(int j=0;j<10;++j)solver.step(j*.001,.001,.4,runtime);
        std::vector<ThermalState> field;solver.download_field(field);
        if(reference.empty())reference=field;
        for(std::size_t j=0;j<field.size();++j)require(field[j].stored_energy_j==reference[j].stored_energy_j &&
            field[j].precursor_charge_energy_j==reference[j].precursor_charge_energy_j &&
            field[j].electrical_chemical_input_j==reference[j].electrical_chemical_input_j &&
            field[j].precursor_from_charge_kg==reference[j].precursor_from_charge_kg,
            "CUDA charge chemistry depends on work tile size");
        for(std::size_t j=0;j<field.size();++j)require(field[j].healthy_charged_kg==reference[j].healthy_charged_kg &&
            field[j].degraded_mass_kg==reference[j].degraded_mass_kg,"CUDA separated health/charge depends on tile size");
    }
}
void bounded_temperature_resistance() {
    auto scene=fixture();auto& c=scene.cells[0];c.initial_temperature_k=500.;
    c.voltage_profile_v={{{0.,2.}},{{2.,2.}}};
    auto& law=c.electrical;law.reference_temperature_k=300.;
    law.temperature_log_response.count=3;
    for(int i=0;i<3;++i) {
        law.temperature_log_response.temperature_k[i]=300.+i*100.;
        law.temperature_log_response.value[i]=i==0?0.:1.;
    }
    require(electrical_resistance(law,500.)==electrical_resistance(law,2000.),"Electrical table extrapolated a hot exponential");
    law.charge_log_slope=2.;
    require(std::abs(electrical_resistance(law,500.,.2)/electrical_resistance(law,500.,0.)-std::exp(.4))<1e-14,
        "Bounded thermal response erased stored-charge polarization");
    law.charge_log_slope=0.;
    Simulation bounded(scene);bounded.run_to_end();
    const double expected=35.*35.*.02*std::exp(1.)*2.;
    require(std::abs(bounded.joule_energy_j()-expected)<1e-9,"Bounded R law does not produce actual Kirchhoff I2R");
    require(bounded.joule_energy_j()<140. && std::abs(bounded.energy_balance_residual_j())<1e-8,
        "Bounded R lost electrical or thermal balance");
    auto unbounded=scene;unbounded.cells[0].electrical.temperature_log_response={};
    unbounded.cells[0].electrical.temperature_log_slope_per_k=.01;
    bool rejected=false;
    try {Simulation bad(unbounded);bad.step(.01);}catch(const std::runtime_error& e) {
        rejected=std::string(e.what()).find("flags=256")!=std::string::npos;
    }
    require(rejected,"Strict UI was relaxed instead of changing the constitutive R law");
    auto invalid=scene;invalid.cells[0].electrical.temperature_log_response.value[0]=.1;
    rejected=false;try {Simulation bad(invalid);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"Electrical table changed the reference resistance silently");
    std::cout<<"bounded electrical temperature law: actual I2R, stored-charge factor and strict UI pass\n";
}
void products_do_not_store_polarization() {
    ElectricalLaw e;e.resistance_ohm=.02;e.charge_log_slope=8.;e.product_conductivity_ratio=.6;
    ConversionLaw c;c.enabled=true;
    const double residue=electrical_mixture_conductivity(e,c,298.15,0.,0.,1.,1.,0.);
    require(std::abs(residue-.6/.02)<1e-13,"Products acquired a healthy charge polarization");
    // Vanishing charged traces cannot impose their SOC on the finite residue.
    const double trace=electrical_mixture_conductivity(e,c,298.15,0.,1e-20,1.,1.,1e-20);
    require(std::abs(trace-residue)<1e-13,"Tiny remaining Bc polarizes all condensed residue");
    const double mixed=electrical_mixture_conductivity(e,c,298.15,.3,.2,.5,1.,.2);
    const double expected=.5/(.02*std::exp(8.*.4))+.6*.5/.02;
    require(std::abs(mixed-expected)<1e-13,"Healthy and product conductivities do not mix in parallel");
    e.charge_log_slope=0.;
    require(std::abs(electrical_mixture_conductivity(e,c,298.15,.3,.2,.5,1.,.2)-
        electrical_mixture_factor(e,c,.3,.2,.5,1.,.2)/.02)<1e-13,"Unpolarized mixture changed");
    e.charge_log_slope=8.;
    require(std::abs(electrical_mixture_conductivity(e,c,298.15,.8,.2,0.,1.,.2)-
        1./electrical_resistance(e,298.15,.2))<1e-13,"Healthy-only charging law changed");
}
void voltage_compliance() {
    ElectricalLaw law;law.current_a=2.;
    const CurrentPoint voltage[]{{0.,1.},{2.,3.}};
    const auto whole=compliant_interval(law,{}, {voltage,2},1.,0.,2.);
    const auto a=compliant_interval(law,{}, {voltage,2},1.,0.,.7);
    const auto b=compliant_interval(law,{}, {voltage,2},1.,.7,1.3);
    require(std::abs(whole.charge_c-3.5)<1e-14 && std::abs(whole.i2_a2_s-19./3.)<1e-14 &&
        std::abs(whole.work_j-22./3.)<1e-14 && whole.limited_time_s==1.,"Analytic compliance crossing");
    require(std::abs(a.charge_c+b.charge_c-whole.charge_c)<1e-14 &&
        std::abs(a.i2_a2_s+b.i2_a2_s-whole.i2_a2_s)<1e-14 &&
        std::abs(a.work_j+b.work_j-whole.work_j)<1e-14,"Compliance depends on time partition");
    law.current_a=1.;law.reversible_voltage_floor_v=1.;
    const CurrentPoint low[]{{0.,0.},{2.,2.}};
    const auto floor=compliant_interval(law,{}, {low,2},1.,0.,2.);
    require(std::abs(floor.charge_c-.5)<1e-14 && std::abs(floor.i2_a2_s-1./3.)<1e-14 &&
        std::abs(floor.work_j-5./6.)<1e-14,"Compliance below reversible floor");
    for(int count:{1,32}) {
        auto scene=fixture();auto c=scene.cells[0];c.electrical.voltage_limited_current=true;
        c.electrical.current_a=2.;c.electrical.resistance_ohm=1.;c.electrical.max_resistance_ohm=2.;
        c.voltage_profile_v={{{0.,1.}},{{2.,3.}}};c.electrical.nominal_capacity_coulomb=10.;
        scene.cells.clear();for(int j=0;j<count;++j){c.id="battery-"+std::to_string(j);c.position_m.x=j*.03;scene.cells.push_back(c);}
        Simulation sim(scene);sim.run_to_end();
        require(std::abs(sim.joule_energy_j()-count*19./3.)<1e-8,"Compliant Kirchhoff Joule integral or batch size");
        for(const auto& e:sim.electrical_states())require(std::abs(e.positive_charge_c-3.5)<1e-11 &&
            std::abs(e.terminal_work_j-22./3.)<1e-11 && std::abs(e.voltage_limited_time_s-1.)<1e-11,
            "CUDA actual charge/work ledger uses commanded current");
        require(std::abs(sim.energy_balance_residual_j())<1e-8,"Compliant current thermal balance");
    }
    auto scene=fixture();auto& c=scene.cells[0];c.electrical.voltage_limited_current=true;
    c.electrical.current_a=2.;c.electrical.resistance_ohm=1.;c.electrical.max_resistance_ohm=10.;
    c.electrical.nominal_capacity_coulomb=1.;
    c.voltage_profile_v={{{0.,1.}},{{2.,1.}}};Simulation sim(scene);
    for(int j=0;j<20;++j){const auto before=sim.electrical_states()[0];sim.step(.01);
        const auto e=sim.electrical_states()[0];
        require(std::abs(e.resistance_ohm-1.)<1e-11 && e.positive_charge_c>before.positive_charge_c,
            "Passed charge changed resistance without precursor chemistry");}
    std::cout<<"voltage compliance: analytic branches, actual Q/work, 32 independent batteries pass\n";
}
int main(){try{passive_resistive_heater();products_do_not_store_polarization();bounded_temperature_resistance();voltage_compliance();cuda_charged_precursor_releases_after_off();stored_charge_energy_reaction();completed_drive_profiles();analytic_and_mesh();profile_and_resistance();mixed_energy_and_tiles();heat_neutral_charging();kirchhoff_axes_and_paths();finite_product_conductivity();voltage_work_bound();charge_resistance_and_storage_budget();
    std::cout<<"Kirchhoff/Joule: KCL, series/parallel local heat, finite products, exact current profiles, mesh/block invariance and ledgers pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "battery_native/simulation.hpp"
#include "battery_native/conversion.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
void balance(const Simulation& s) {
    auto p=s.process_diagnostics();
    require(std::abs(p.mass_residual_kg)<1e-11,"Mass balance failed");
    require(std::abs(p.thermal_residual_j)<1e-7,"Solid+gas+outflow thermal balance failed");
    require(std::abs(p.chemical_residual_j)<1e-7,"Chemical/activation reserve double counted");
    require(std::abs(p.oxidizer_residual_kg)<1e-11,"Oxidizer transport/reaction balance failed");
}
Scenario base() {
    Scenario s;s.settings={2.,.01,.1,.4};s.gas_model.reference_temperature_k=300.;
    CellConfig c;c.id="B";c.dimensions_m={.02,.02,.02};c.thermal_divisions={1,1,1};c.heat_capacity_j_per_k=10.;c.initial_temperature_k=500.;
    auto& p=c.conversion;p.enabled=true;p.mass_kg=.004;p.reactant_cp=p.activated_cp=p.residue_cp=1000.;
    p.degradation_rate_per_s=3.;p.degradation_reference_k=500.;p.reaction_rate_per_s=2.;p.reaction_activation_j_per_mol=80000.;
    p.gas_compartment=0;p.net_heat_j_per_kg=200000;p.gas_yield=p.gas_yield_max=.1;p.gas_chemical_fraction=.25;
    p.residue_conductivity[0]=.2;p.residue_conductivity[1]=.3;p.residue_conductivity[2]=.4;
    s.cells={c};GasCompartment g;g.id="inside";g.owner_cell=0;g.volume_m3=1e-6;g.initial_temperature_k=300.;g.initial_mass_kg=101325*g.volume_m3/(287*300);s.gas_compartments={g};return s;
}
void casing(Scenario& s) {
    ShellConfig sh;sh.contents_id="B";sh.material_id="demonstration-polymer";sh.thickness_m=.001;sh.density_kg_m3=1200;sh.specific_heat_j_kg_k=1000;
    sh.conductivity_w_per_m_k={.2,.2,.2};sh.contents_contact_h=2;sh.seam_contact_h=5;add_physical_shell(s,sh);
    for(std::size_t i=1;i<s.cells.size();++i)s.cells[i].initial_temperature_k=300.;
}
void channels(Scenario& s) {
    GasCompartment cavity;cavity.id="cavity";cavity.volume_m3=.001;cavity.initial_mass_kg=101325*.001/(287*300);cavity.initial_temperature_k=300.;s.gas_compartments.push_back(cavity);
    // +z shell panel, with a separately declared narrow channel area.
    GasChannel ch;ch.id="rupture";ch.from=0;ch.to=1;ch.shell_object=6;ch.port_id="B:case:+z:outer";ch.area_m2=1e-7;ch.opening_pressure_pa=10000.;ch.rupture=true;
    s.gas_channels.push_back(ch);
    CellConfig pad;pad.id="pad";pad.is_battery=false;pad.material_id="sacrificial-pad";pad.dimensions_m={.02,.02,.002};pad.position_m={0,0,.03};pad.thermal_divisions={1,1,1};
    pad.heat_capacity_j_per_k=.02;pad.initial_temperature_k=300;pad.damage_onset_k=301.;pad.damage_rate_at_onset_per_s=1.;pad.damage_temperature_scale_k=10;
    const int pi=static_cast<int>(s.cells.size());s.cells.push_back(pad);
    s.ports.push_back({"exposure","pad",static_cast<std::size_t>(pi),"gas","-z",.0004,{0,0,.029},{0,0,-1}});
    s.gas_contacts.push_back({"exposure",1,200.});
}
int main(){try {
    // Independent finite-fuel gas reaction: analytic exponential, exact energy
    // conversion, no mass creation, and finite-oxidizer quench.
    auto burn=base();burn.cells[0].conversion.enabled=false;burn.gas_compartments[0].owner_cell=-1;
    auto& bg=burn.gas_compartments[0];bg.volume_m3=.001;bg.initial_mass_kg=.001;
    bg.initial_chemical_j=100.;bg.initial_oxidizer_mass_fraction=.5;bg.reaction_rate_per_s=2.;
    burn.settings.duration_s=1.;Simulation flame(burn);flame.run_to_end();balance(flame);
    auto bd=flame.process_diagnostics();const double expected_heat=100*(-std::expm1(-2.));
    require(std::abs(bd.gas[0].released_energy_j-expected_heat)<1e-9,"Gas reaction differs from analytic finite reserve");
    require(std::abs(bd.gas[0].temperature_k-(300+expected_heat))<1e-8,"Gas heat release missing/double counted");
    auto limited=burn;limited.gas_compartments[0].initial_oxidizer_mass_fraction=.001;limited.settings.duration_s=20.;
    Simulation oxygen(limited);oxygen.run_to_end();balance(oxygen);auto od=oxygen.process_diagnostics();
    require(std::abs(od.gas[0].released_energy_j-13.1)<1e-8 && od.gas[0].chemical_j>86.,"Gas burns after oxidizer exhaustion");
    auto inert=burn;inert.gas_compartments[0].initial_oxidizer_mass_fraction=0.;Simulation no_oxygen(inert);no_oxygen.run_to_end();balance(no_oxygen);
    require(no_oxygen.process_diagnostics().gas[0].released_energy_j==0,"Fuel reacts without oxidizer");
    inert=burn;inert.gas_compartments[0].ignition_temperature_k=400.;Simulation no_ignition(inert);no_ignition.run_to_end();balance(no_ignition);
    require(no_ignition.process_diagnostics().gas[0].released_energy_j==0,"Gas ignition threshold ignored");
    // No chemistry needed to verify independent adiabatic orifice blowdown.
    auto b=base();b.cells[0].conversion.enabled=false;b.cells[0].initial_temperature_k=300.;b.gas_compartments[0].owner_cell=-1;
    b.gas_compartments[0].initial_mass_kg=.001;b.gas_compartments[0].initial_temperature_k=500.;b.gas_compartments[0].volume_m3=.0005;
    b.gas_compartments[0].initial_oxidizer_mass_fraction=.232;
    b.ports.push_back({"out","B",0,"vent","+z",.0004,{0,0,.01},{0,0,1}});
    GasChannel ch;ch.id="vent";ch.from=0;ch.to=-1;ch.port_id="out";ch.shell_object=0;ch.area_m2=5e-7;ch.outside_pressure_pa=0;b.gas_channels={ch};b.settings.duration_s=.2;
    Simulation blow(b);blow.run_to_end();balance(blow);auto pd=blow.process_diagnostics();
    const double expected=500*std::pow(pd.gas[0].mass_kg/.001,.287);
    require(std::abs(pd.gas[0].temperature_k-expected)<.05,"Outflow omitted enthalpy/flow work");
    require(pd.escaped_mass_kg>0 && pd.escaped_energy_j>0,"Gas cannot escape without ongoing chemistry");
    require(pd.escaped_oxidizer_kg>0,"Outflow lost oxidizer tracer");
    // Once opened, an internal hole is not a check valve. Heating the receiver
    // must reverse flow; donor enthalpy/chemical concentration must reverse too.
    auto reverse=b;reverse.settings.duration_s=3;reverse.cells[0].initial_temperature_k=1000;
    GasCompartment receiver;receiver.id="heated-receiver";receiver.volume_m3=.0005;receiver.initial_mass_kg=.001;receiver.initial_temperature_k=300;
    reverse.gas_compartments.push_back(receiver);reverse.gas_channels[0].to=1;
    reverse.gas_contacts.push_back({"out",1,2000.});Simulation reversal(reverse);bool reverse_seen=false;
    while(!reversal.finished()) {reversal.step();balance(reversal);reverse_seen|=reversal.process_diagnostics().channels[0].mass_flow_kg_s<0;}
    require(reverse_seen,"An opened internal hole prevents physical backflow");
    // Independent two-stage constant-T analytic solution and step convergence.
    auto a=base();auto& ap=a.cells[0].conversion;ap.net_heat_j_per_kg=ap.gas_yield=ap.gas_chemical_fraction=0.;
    a.settings.duration_s=1.;const double ka=degradation_rate(ap,500),kb=completion_rate(ap,500);
    const double exact_a=.004*ka/(kb-ka)*(std::exp(-ka)-std::exp(-kb));double errors[2]{};
    for(int level=0;level<2;++level) {
        a.settings.max_dt_s=.002/std::pow(2.,level);Simulation s(a);s.run_to_end();balance(s);
        const auto& n=s.thermal_states()[0];require(std::abs(n.reactant_kg-.004*std::exp(-ka))<1e-12,"First stage does not integrate finite reserve");
        errors[level]=std::abs(n.activated_kg-exact_a);require(std::abs(s.cells()[0].temperature_k-500)<1e-9,"Composition creates heat with equal cp and zero reaction energy");
    }
    require(errors[0]<1e-12 && errors[1]<1e-12,"Frozen-temperature coupled chain differs from analytic solution");
    auto stiff=a;stiff.cells[0].conversion.reaction_rate_per_s=1e8;stiff.settings.max_dt_s=.05;
    Simulation stiff_run(stiff);stiff_run.run_to_end();balance(stiff_run);
    require(std::abs(stiff_run.thermal_states()[0].reactant_kg-.004*std::exp(-ka))<1e-12,"Stiff intermediate corrupts reactant depletion");
    require(stiff_run.step_count()<1000,"Tiny intermediate lifetime still throttles all transport");
    // Mesh-independent material length and homogeneous evolution.
    a.cells[0].thermal_divisions={3,3,3};Simulation fine(a);fine.run_to_end();balance(fine);
    require(std::abs(fine.thermal_states()[0].reactant_kg*27-.004*std::exp(-ka))<1e-12,"Material threshold depends on voxel size");
    // Cold aging remains continuous, but is not reported as runaway.
    auto cold=base();cold.cells[0].initial_temperature_k=300.;Simulation waiting(cold);waiting.run_to_end();balance(waiting);
    require(waiting.cells()[0].damage>0 && waiting.cells()[0].runaway_started_s<0,"Cold R1 aging is absent or reported as runaway");
    // A material sample is externally cooled/reheated: activation stops in the
    // cold, existing intermediates can finish, original material never heals.
    auto law=base().cells[0].conversion;law.reaction_activation_j_per_mol=80000;
    GasModel gm;gm.reference_temperature_k=300;
    ThermalState sample;sample.reactant_kg=law.mass_kg;sample.temperature_k=500;
    advance_conversion(law,gm,sample,1,.01,1,0);refresh_conversion(law,sample,1,1);
    const double damage=sample.damage,hot_power=sample.chemical_power_w,mr=sample.reactant_kg;
    sample.temperature_k=300;refresh_conversion(law,sample,1,1);
    require(sample.chemical_power_w<hot_power*1e-4,"Cooling does not suppress completion kinetics");
    advance_conversion(law,gm,sample,1,1,1,.01);refresh_conversion(law,sample,1,1);
    require(sample.reactant_kg<mr && sample.damage>damage,"Cold R1 aging was incorrectly stopped");
    sample.temperature_k=500;advance_conversion(law,gm,sample,1,.01,1,1.01);refresh_conversion(law,sample,1,1);
    require(sample.damage>damage,"Reheating cannot resume irreversible conversion");
    auto source=base();source.settings.duration_s=4.;casing(source);channels(source);Simulation s(source);
    bool early=false;double irreversible=0.;double first_open=-1.;
    while(!s.finished()) {s.step();balance(s);auto p=s.process_diagnostics();
        require(s.cells()[0].damage+1e-12>=irreversible,"Chemical conversion healed");irreversible=s.cells()[0].damage;
        if(p.channels[0].open && first_open<0) {first_open=p.channels[0].opened_s;early=s.thermal_states()[0].reactant_kg+s.thermal_states()[0].activated_kg>.002;}
    }
    pd=s.process_diagnostics();require(early,"Shell did not rupture before the fuel was exhausted");
    require(pd.channels[0].transferred_kg>0 && pd.gas[1].chemical_j>0,"Receiver lost mass or unspent chemical potential");
    require(s.cells()[6].damage==1 && s.cells()[6].mode==CellMode::Spent,"Pressure did not damage the physical casing");
    require(s.cells().back().temperature_k>301 && s.cells().back().damage>0,"Discharge does not heat/damage the receiving material");
    auto reactive_receiver=source;reactive_receiver.gas_compartments[1].initial_oxidizer_mass_fraction=.232;
    reactive_receiver.gas_compartments[1].reaction_rate_per_s=5.;
    double gas_temperatures[2];
    for(int level=0;level<2;++level) {
        auto refined=reactive_receiver;refined.settings.max_dt_s*=std::pow(.5,level);refined.settings.stability_factor*=std::pow(.5,level);
        refined.gas_model.max_fraction_per_step*=std::pow(.5,level);refined.gas_model.max_temperature_step_k*=std::pow(.5,level);
        refined.cells[0].conversion.max_temperature_step_k*=std::pow(.5,level);refined.cells[0].conversion.max_fraction_per_step*=std::pow(.5,level);
        Simulation coupled(refined);coupled.run_to_end();balance(coupled);auto d=coupled.process_diagnostics();
        require(d.gas[1].released_energy_j>0,"Transferred fuel does not react in receiving gas");
        require(coupled.cells().back().temperature_k>s.cells().back().temperature_k,"Gas reaction heat does not reach receiving material");
        gas_temperatures[level]=d.gas[1].temperature_k;
    }
    require(std::abs(gas_temperatures[0]-gas_temperatures[1])<.5,"Gas reaction/transport/contact refinement fails");
    const double final=s.cells().back().temperature_k,mass=pd.channels[0].transferred_kg;
    s.reset();while(!s.finished())s.step();require(s.cells().back().temperature_k==final && s.process_diagnostics().channels[0].transferred_kg==mass,"Reset is not deterministic");
    // Several outlet channels share ONE upstream mass inventory.
    auto multi=source;multi.gas_channels[0].to=-1;auto second=multi.gas_channels[0];second.id="second";second.area_m2=2e-7;multi.gas_channels.push_back(second);
    Simulation many(multi);many.run_to_end();balance(many);require(many.process_diagnostics().escaped_chemical_j>0,"Exported chemical potential lost");
    auto invalid=source;invalid.gas_channels[0].shell_object=0;bool rejected=false;try{Simulation bad(invalid);}catch(...){rejected=true;}
    require(rejected,"Pressure channel allowed to rupture battery contents instead of casing");
    invalid=source;invalid.cells[0].runaway_energy_j=1;rejected=false;try{Simulation bad(invalid);}catch(...){rejected=true;}
    require(rejected,"Conversion and legacy chemical budget overlap");
    std::cout<<"network: rupture_s="<<first_open<<" receiver_K="<<final<<" transferred_kg="<<mass<<" all balances and analytical/refinement checks pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

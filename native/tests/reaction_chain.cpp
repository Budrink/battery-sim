#include "battery_native/simulation.hpp"
#include "battery_native/conversion.hpp"
#include "battery_native/state_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void balance(const ProcessDiagnostics& d) {
    require(std::abs(d.mass_residual_kg)<1e-11,"R123 mass balance");
    require(std::abs(d.thermal_residual_j)<1e-7,"R123 sensible energy balance");
    require(std::abs(d.chemical_residual_j)<1e-7,"R123 chemical reserve balance");
    require(std::abs(d.oxidizer_residual_kg)<1e-11,"R123 oxygen balance");
}
Scenario fixture(double temperature=500.) {
    Scenario s;s.settings={2.,.01,.1,.4};s.gas_model.reference_temperature_k=300.;
    s.gas_model.split_transport=true;s.gas_model.cuda_network_transport=true;
    CellConfig c;c.id="B";c.dimensions_m={.02,.02,.02};c.thermal_divisions={1,1,1};
    c.heat_capacity_j_per_k=10.;c.initial_temperature_k=temperature;
    auto& p=c.conversion;p.enabled=true;p.mass_kg=.004;
    p.reactant_cp=p.activated_cp=p.residue_cp=1000.;
    // Synthetic fast-aging kinetics for tests, NOT fitted battery constants.
    p.degradation_rate_per_s=.02;p.degradation_reference_k=300.;p.degradation_activation_j_per_mol=20000.;
    p.reaction_rate_per_s=3.;p.reaction_reference_k=500.;p.reaction_activation_j_per_mol=120000.;
    p.net_heat_j_per_kg=200000.;p.initial_precursor_fraction=.4;
    p.gas_yield=.2;p.gas_yield_max=.6;p.gas_yield_temperature_per_k=.001;p.gas_yield_density_exponent=.3;
    p.gas_chemical_fraction=.25;p.gas_compartment=0;
    s.cells={c};GasCompartment g;g.id="inside";g.owner_cell=0;g.volume_m3=1e-6;
    g.initial_temperature_k=temperature;g.initial_mass_kg=101325*g.volume_m3/(287*temperature);
    g.reaction_rate_per_s=5.; // Even enabled R3 cannot burn without oxygen.
    s.gas_compartments={g};return s;
}
void exact_neutral_r1() {
    auto p=fixture().cells[0].conversion;GasModel g;g.reference_temperature_k=300.;
    p.reaction_rate_per_s=0.;ThermalState state;state.temperature_k=300.;state.reactant_kg=p.mass_kg;
    const double chemical=conversion_chemical(p,state),mass=state.reactant_kg;
    const double heat=advance_conversion(p,g,state,1.,100.,1.,0.);refresh_conversion(p,state,1.,1.);
    require(heat==0 && state.released_energy_j==0 && state.generated_gas_kg==0,"R1 creates heat or gas");
    require(std::abs(state.reactant_kg-mass*std::exp(-2.))<1e-16,"R1 is not continuous Arrhenius aging");
    require(std::abs(state.reactant_kg+state.activated_kg-mass)<1e-16,"R1 loses precursor mass");
    require(std::abs(conversion_chemical(p,state)-chemical)<1e-12,"R1 loses stored chemical potential");
    require(state.runaway_started_s<0 && state.mode==CellMode::Healthy,"Slow aging is reported as runaway");
    auto split=ThermalState{};split.temperature_k=300.;split.reactant_kg=mass;
    for(int j=0;j<100;++j)advance_conversion(p,g,split,1.,1.,1.,j);
    require(std::abs(split.activated_kg-state.activated_kg)<1e-16,"R1 depends on step partition");
    require(completion_rate(fixture().cells[0].conversion,500.)/
        completion_rate(fixture().cells[0].conversion,300.)>1e8,"R2 does not accelerate strongly with temperature");
}
void gas_yield_and_mesh() {
    const auto p=fixture().cells[0].conversion;ThermalState s;s.temperature_k=500.;s.reactant_kg=p.mass_kg;
    require(std::abs(effective_gas_yield(p,s,1.)-.2)<1e-15,"Gas yield reference");
    s.temperature_k=700.;const double hot=effective_gas_yield(p,s,1.);
    require(hot>.2 && hot<=p.gas_yield_max,"Gas yield temperature response");
    s.reactant_kg*=.5;require(effective_gas_yield(p,s,1.)<hot,"Gas yield ignores density");
    const double coarse=effective_gas_yield(p,s,1.);s.reactant_kg/=27.;
    require(std::abs(effective_gas_yield(p,s,1./27.)-coarse)<1e-15,"Gas yield depends on voxel mass");
    s.temperature_k=1e6;require(effective_gas_yield(p,s,1./27.)==p.gas_yield_max,"Gas yield exceeds physical bounds");
    auto scene=fixture(300.);auto& law=scene.cells[0].conversion;
    law.net_heat_j_per_kg=law.gas_yield=law.gas_chemical_fraction=0.;law.initial_precursor_fraction=.2;
    const double ka=degradation_rate(law,300.),kb=completion_rate(law,300.);
    const double virgin=law.mass_kg*.8*std::exp(-ka*scene.settings.duration_s);
    const double precursor=law.mass_kg*.2*std::exp(-kb*scene.settings.duration_s)+
        law.mass_kg*.8*ka/(kb-ka)*(std::exp(-ka*scene.settings.duration_s)-std::exp(-kb*scene.settings.duration_s));
    for(int n:{1,3}) {
        scene.cells[0].thermal_divisions={n,n,n};Simulation sim(scene);sim.run_to_end();balance(sim.process_diagnostics());
        double a=0.,b=0.;for(const auto& node:sim.thermal_states()){a+=node.reactant_kg;b+=node.activated_kg;}
        require(std::abs(a-virgin)<1e-14 && std::abs(b-precursor)<1e-14,"CUDA R1/R2 differs from analytic chain");
        require(std::abs(sim.cells()[0].temperature_k-300.)<1e-10,"Neutral chain creates heat");
    }
}
void rupture_and_oxygen() {
    auto scene=fixture();ShellConfig sh;sh.contents_id="B";sh.material_id="test-polymer";
    sh.thickness_m=.001;sh.density_kg_m3=1200.;sh.specific_heat_j_kg_k=1000.;
    sh.conductivity_w_per_m_k={.2,.2,.2};sh.contents_contact_h=2.;sh.seam_contact_h=5.;add_physical_shell(scene,sh);
    GasCompartment outer;outer.id="air";outer.volume_m3=.001;outer.initial_temperature_k=300.;
    outer.initial_mass_kg=101325*.001/(287*300.);outer.initial_oxidizer_mass_fraction=.232;outer.reaction_rate_per_s=5.;
    scene.gas_compartments.push_back(outer);
    GasChannel ch;ch.id="rupture";ch.from=0;ch.to=1;ch.shell_object=6;ch.port_id="B:case:+z:outer";
    ch.area_m2=1e-7;ch.opening_pressure_pa=1e12;ch.rupture=true;scene.gas_channels={ch};
    Simulation sealed(scene);sealed.run_to_end();auto d=sealed.process_diagnostics();balance(d);
    require(sealed.cells()[0].released_energy_j>0 && d.gas[0].mass_kg>scene.gas_compartments[0].initial_mass_kg,"Sealed R2 produces no heat/gas");
    require(d.gas[0].chemical_j>0 && d.gas[0].released_energy_j==0,"R3 burned inside oxygen-free cell");
    require(d.gas[1].released_energy_j==0 && !d.channels[0].open,"External flame crossed intact casing");
    scene.gas_channels[0].opening_pressure_pa=10000.;double temperature[2]{};
    for(int level=0;level<2;++level) {
        auto refined=scene;const double scale=level?.5:1.;refined.settings.max_dt_s*=scale;
        refined.cells[0].conversion.max_fraction_per_step*=scale;refined.cells[0].conversion.max_temperature_step_k*=scale;
        refined.gas_model.max_fraction_per_step*=scale;refined.gas_model.max_temperature_step_k*=scale;
        Simulation opened(refined);while(!opened.finished()){opened.step();balance(opened.process_diagnostics());}
        d=opened.process_diagnostics();require(d.channels[0].transferred_kg>0 && d.gas[1].released_energy_j>0,"Released gas cannot burn outside");
        require(d.gas[0].released_energy_j==0,"Internal R3 found nonexistent oxygen");
        temperature[level]=opened.cells()[0].temperature_k;
    }
    require(std::abs(temperature[1]-temperature[0])<.5,"R123 coupled trajectory fails refinement");
    std::cout<<"R123 rupture refinement: "<<std::abs(temperature[1]-temperature[0])<<" K\n";
}
void initial_capacity_and_tiles() {
    auto scene=fixture();scene.cells[0].thermal_divisions={3,3,3};scene.cells[0].conversion.activated_cp=1500.;
    auto mesh=build_thermal_mesh(scene);std::vector<ThermalState> reference;
    for(int tile:{64,128,256}) {
        CudaSolver solver(scene,mesh,tile);balance(solver.process_diagnostics());
        RuntimeParameters runtime;for(int j=0;j<20;++j)solver.step(j*.0001,.0001,.4,runtime);
        balance(solver.process_diagnostics());std::vector<ThermalState> state;solver.download_field(state);
        if(reference.empty())reference=state;
        else for(std::size_t j=0;j<state.size();++j)require(state[j].stored_energy_j==reference[j].stored_energy_j &&
            state[j].activated_kg==reference[j].activated_kg && state[j].reactant_kg==reference[j].reactant_kg,"R123 depends on CUDA tile");
    }
}
void r1_byproduct_branch() {
    ConversionLaw p=fixture().cells[0].conversion;GasModel g;g.reference_temperature_k=300.;
    p.reaction_rate_per_s=0.;p.precursor_yield=.25;p.precursor_yield_temperature_per_k=0.;p.precursor_yield_voltage_per_v=0.;
    ThermalState s;s.temperature_k=300.;s.reactant_kg=p.mass_kg;
    const double q0=conversion_chemical(p,s);
    const double heat=advance_conversion(p,g,s,1.,10.,1.,0.);
    const double da=p.mass_kg-s.reactant_kg;
    require(heat==0 && s.generated_gas_kg==0 && s.released_energy_j==0,"Branched R1 creates heat or gas");
    require(std::abs(s.activated_kg-.25*da)<1e-16,"R1 byproduct is not the precursor fraction");
    require(std::abs(s.residue_kg-.75*da)<1e-16,"Complement of R1 is not inert residue");
    require(std::abs(s.sterilized_chemical_j-p.net_heat_j_per_kg*.75*da)<1e-12,"Inert path is not a chemical sink");
    require(std::abs(conversion_chemical(p,s)+s.sterilized_chemical_j-q0)<1e-12,"Branching leaks chemical reserve");
    require(std::abs(precursor_branch(p,300.,3.65)-.25)<1e-15,"Reference branch");
    p.precursor_yield_temperature_per_k=.02;
    require(std::abs(precursor_branch(p,p.precursor_temperature_ref_k+std::log(2.)/.02,p.precursor_voltage_ref_v)-.5)<1e-12,"Heat does not raise R1→B yield");
    p.precursor_yield_temperature_per_k=0.;p.precursor_yield_voltage_per_v=1.;
    require(std::abs(precursor_branch(p,300.,3.65+1.)-.25*std::exp(1.))<1e-12,"Voltage does not raise R1→B yield");
    p.precursor_yield=2.;require(precursor_branch(p,300.,3.65)==1.,"Branch exceeds 1");
}
void tensor_without_length() {
    ConnectivityLaw law;law.kind=TransitionKind::DirectedGeometry;
    law.background_conductance_w_per_k=1.;law.active_conductance_w_per_k=9.;
    law.destroyed_background_conductance_w_per_k=.1;law.destroyed_active_conductance_w_per_k=2.;
    law.destroyed_geometry_reference_logit=-2.;law.endpoint_a.linear={1.,0.,0.};law.endpoint_b.linear={-1.,0.,0.};
    auto a=evaluate_directed_geometry(law,{},0,500.,300.,0.,0.,0.,1.,0.,1.,1);
    auto b=evaluate_directed_geometry(law,{},0,500.,300.,0.,0.,0.,1.,0.,1.,-1);
    require(a.conductance_w_per_k!=b.conductance_w_per_k,"Tensor contact symmetrizes directions");
    law.l0_m=100.;law.destroyed_l_m=1e9;law.activation_width_m=0.;
    auto unchanged=evaluate_directed_geometry(law,{},0,500.,300.,0.,0.,0.,1.,0.,1.,1);
    require(a.conductance_w_per_k==unchanged.conductance_w_per_k && unchanged.l_m==0.,"Tensor contact uses an independent length");
}
int main(){try {
    exact_neutral_r1();gas_yield_and_mesh();rupture_and_oxygen();initial_capacity_and_tiles();r1_byproduct_branch();tensor_without_length();
    std::cout<<"R1/R2/R3: analytic chain, balances, oxygen/casing, density, refinement and CUDA tiles pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "battery_native/simulation.hpp"
#include "battery_native/shell_opening.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace battery;
void require(bool x,const char* message) {if(!x)throw std::runtime_error(message);}
// Compare fields, not struct padding. Exact comparisons are intentional: retry
// is required to reproduce the same operation sequence on this device/build.
std::vector<double> snapshot(CudaSolver& solver) {
    std::vector<double> out;
    const auto add=[&](std::initializer_list<double> values){out.insert(out.end(),values);};
    std::vector<ThermalState> nodes;solver.download_field(nodes);
    std::vector<ShellDamageState> shell;solver.download_shell_states(shell);
    for(const auto& d:shell)add({d.bulk,d.seam,d.thermal,d.failure_hazard});
    for(const auto& s:nodes)add({s.liquid_kg,s.evaporated_kg,s.condensed_kg,s.r2_generated_gas_kg,s.r2_reacted_kg,
        s.temperature_k,s.stored_energy_j,s.damage,s.released_energy_j,
        s.joule_energy_j,s.precursor_charge_energy_j,s.electrical_chemical_input_j,
        s.precursor_from_charge_kg,s.precursor_from_degradation_kg,
        s.healthy_charged_kg,s.degraded_mass_kg,
        s.boundary_energy_out_j,s.runaway_started_s,s.reaction_progress_s,s.chemical_energy_remaining_j,
        s.ejected_chemical_energy_j,s.vent_heat_out_j,s.chemical_power_w,s.conductivity_factor[0],
        s.conductivity_factor[1],s.conductivity_factor[2],double(s.mode),s.reactant_kg,s.activated_kg,s.residue_kg,
        s.gross_reaction_energy_j,s.sterilized_chemical_j,s.generated_gas_kg,s.generated_gas_energy_j,s.generated_gas_chemical_j,s.mobile_capacity_j_per_k});
    std::vector<InterfaceState> links,macro_links;solver.download_link_states(links);
    std::vector<CellState> cells;solver.download_macros(cells,macro_links);
    std::vector<ElectricalState> electrical;solver.download_electrical_states(electrical);
    for(const auto& e:electrical)add({e.time_s,e.resistance_ohm,e.current_a,e.outlet_current_a,e.ohmic_drop_v,e.power_w,
        e.kcl_relative_residual,e.power_relative_residual,double(e.iterations),double(e.error),
        e.positive_charge_c,e.terminal_work_j,e.voltage_limited_time_s,e.step_charge_c,e.step_i2_a2_s,e.step_work_j});
    links.insert(links.end(),macro_links.begin(),macro_links.end());
    for(const auto& s:links)add({s.l0_m,s.l_m,s.damage,double(s.phase),s.conductance_w_per_k,s.destroyed_fraction});
    for(const auto& s:cells)add({s.temperature_k,s.damage,double(s.mode),s.runaway_started_s,s.released_energy_j,
        s.joule_energy_j,
        s.min_temperature_k,s.max_temperature_k,s.max_damage,s.spent_fraction,s.runaway_fraction,
        s.stored_energy_j,s.boundary_energy_out_j,s.heat_capacity_j_per_k,s.chemical_energy_remaining_j,
        s.ejected_chemical_energy_j,s.vent_heat_out_j,s.chemical_power_w,s.quenched_fraction});
    const auto d=solver.process_diagnostics();
    for(const auto& s:d.gas)add({s.vapor_kg,s.latent_j,s.mass_kg,s.energy_j,s.chemical_j,s.temperature_k,s.pressure_pa,
        s.oxidizer_kg,s.consumed_oxidizer_kg,s.released_energy_j,s.free_volume_m3});
    for(const auto& s:d.channels)add({s.mixed_mass_kg,s.transferred_vapor_kg,s.transferred_latent_j,
        double(s.open),s.opened_s,s.mass_flow_kg_s,s.proposed_mass_flow_kg_s,s.transferred_kg,
        s.transferred_energy_j,s.transferred_chemical_j,s.transferred_oxidizer_kg,s.surface_energy_j,s.fracture_work_j,
        s.opening_fraction,s.effective_area_m2,s.max_bulk_damage,s.max_seam_damage,s.max_thermal_damage});
    add({d.mass_residual_kg,d.thermal_residual_j,d.chemical_residual_j,d.escaped_mass_kg,
        d.escaped_energy_j,d.escaped_chemical_j,d.oxidizer_residual_kg,d.escaped_oxidizer_kg,d.fracture_work_j});
    return out;
}
Scenario fixture() {
    Scenario s;s.settings={1.,.01,.1,.4};s.gas_model.reference_temperature_k=300;
    CellConfig c;c.id="contents";c.dimensions_m={.02,.02,.02};c.thermal_divisions={3,3,3};
    c.heat_capacity_j_per_k=10.;c.initial_temperature_k=500.;
    auto& p=c.conversion;p.enabled=true;p.mass_kg=.004;p.gas_compartment=0;
    p.degradation_rate_per_s=3;p.degradation_reference_k=500;p.reaction_rate_per_s=2;p.reaction_activation_j_per_mol=80000;
    p.net_heat_j_per_kg=200000;p.gas_yield=p.gas_yield_max=.1;p.gas_chemical_fraction=.25;
    c.electrical.enabled=true;c.electrical.current_a=35.;c.electrical.resistance_ohm=.02;
    c.electrical.charge_acceptance_fraction=1.;c.electrical.nominal_capacity_coulomb=35.*3600.;
    c.electrical.reference_temperature_k=500.;p.charge_conversion_per_nominal_charge=3600.;
    c.voltage_profile_v={{{0.,5.}},{{1.,5.}}};
    s.cells.push_back(c);
    ShellConfig shell;shell.contents_id=c.id;shell.material_id="test-shell";shell.thickness_m=.001;
    shell.density_kg_m3=1200;shell.specific_heat_j_kg_k=1000;shell.conductivity_w_per_m_k={.2,.2,.2};
    shell.contents_contact_h=2;shell.seam_contact_h=5;add_physical_shell(s,shell);
    for(std::size_t i=1;i<s.cells.size();++i)s.cells[i].initial_temperature_k=300.;
    int kind=0;
    for(auto& e:s.interfaces) {
        e.law.kind=kind++%2?TransitionKind::Smooth:TransitionKind::StateGeometry;e.law.damage_onset_k=300;
        e.law.damage_rate_at_onset_per_s=.01;e.law.relaxation_time_s=e.law.kind==TransitionKind::Smooth?.1:0.;
    }
    GasCompartment a;a.id="inside";a.owner_cell=0;a.volume_m3=1e-6;a.initial_temperature_k=500;
    a.initial_mass_kg=3e5*a.volume_m3/(287*500);a.initial_chemical_j=.1;a.initial_oxidizer_mass_fraction=.1;a.reaction_rate_per_s=3;
    GasCompartment b;b.id="outside";b.volume_m3=.001;b.initial_temperature_k=300;
    b.initial_mass_kg=1e5*b.volume_m3/(287*300);b.initial_oxidizer_mass_fraction=.232;b.reaction_rate_per_s=4;
    a.reaction_reference_k=b.reaction_reference_k=500.;
    a.reaction_activation_j_per_mol=b.reaction_activation_j_per_mol=40000.;
    s.gas_compartments={a,b};
    GasChannel ch;ch.id="hole";ch.from=0;ch.to=1;ch.shell_object=6;ch.port_id="contents:case:+z:outer";
    ch.area_m2=1e-7;ch.opening_pressure_pa=1e4;ch.rupture=true;s.gas_channels.push_back(ch);
    ch.id="outflow";ch.to=-1;ch.area_m2=2e-7;ch.outside_pressure_pa=1e5;s.gas_channels.push_back(ch);
    s.gas_contacts.push_back({"contents:case:+z:outer",1,200.});
    CellConfig pad;pad.id="passive";pad.is_battery=false;pad.position_m={0,0,.1};pad.dimensions_m={.02,.02,.02};
    pad.thermal_divisions={5,5,5};pad.heat_capacity_j_per_k=2;pad.initial_temperature_k=320;
    s.cells.push_back(pad);return s;
}
void check_retries(const Scenario& s) {
    const auto mesh=build_thermal_mesh(s);CudaSolver single(s,mesh,64),retry(s,mesh,256);RuntimeParameters runtime;
    require(mesh.volumes.size()>128,"Fixture must cross a CUDA block boundary");
    const auto initial=snapshot(single);require(initial==snapshot(retry),"Initial states differ");
    double time=0.;
    for(int i=0;i<40;++i) {
        const auto before=snapshot(retry);const auto* field=retry.device_field();
        auto alternate=runtime;alternate.interface_conductance_scale=2;
        retry.prepare_step(time,.02,.3,alternate);
        require(snapshot(retry)==before && retry.device_field()==field,"Preparation changed committed state");
        retry.discard_step();
        bool rejected=false;try{retry.commit_step();}catch(const std::logic_error&){rejected=true;}
        require(rejected,"Discarded proposal was committed");
        const double dt=single.step(time,.002,.4,runtime);
        for(int j=0;j<3;++j) {
            require(retry.prepare_step(time,.002,.4,runtime)==dt,"Retry changed dt");
            require(snapshot(retry)==before,"Repeated preparation changed inventories/history");
        }
        retry.commit_step();require(snapshot(single)==snapshot(retry),"Retry differs from one accepted step");
        const auto after=snapshot(retry);rejected=false;
        try{retry.commit_step();}catch(const std::logic_error&){rejected=true;}
        require(rejected && snapshot(retry)==after,"Double commit applied sources twice");
        const auto d=retry.process_diagnostics();
        require(std::abs(d.mass_residual_kg)<1e-11 && std::abs(d.thermal_residual_j)<1e-7 &&
            std::abs(d.chemical_residual_j)<1e-7 && std::abs(d.oxidizer_residual_kg)<1e-11,"Balance after transactional step");
        time+=dt;
    }
    require(snapshot(retry)!=initial,"Fixture did not evolve");
    const auto gas=retry.process_diagnostics();
    require(gas.channels[0].open && gas.channels[0].transferred_kg>0 && gas.gas[0].released_energy_j>0,"Gas fixture did not exercise rupture/transport/reaction");
    std::vector<InterfaceState> edges;retry.download_link_states(edges);bool damage=false;
    for(const auto& e:edges)damage|=e.damage>0;require(damage,"Fixture did not exercise edge history");
    retry.prepare_step(time,.001,.4,runtime);retry.reset();require(snapshot(retry)==initial,"Reset retained staged state");
    retry.prepare_step(0,.001,.4,runtime);retry.ignite(0,0,0);single.reset();single.ignite(0,0,0);
    bool rejected=false;try{retry.commit_step();}catch(const std::logic_error&){rejected=true;}
    require(rejected && snapshot(single)==snapshot(retry),"Ignition retained obsolete proposal");
    retry.prepare_step(0,.001,.4,runtime);auto invalid=runtime;invalid.runaway_power_scale=std::numeric_limits<double>::quiet_NaN();
    rejected=false;try{retry.prepare_step(0,.001,.4,invalid);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"Nonfinite runtime accepted");
    rejected=false;try{retry.commit_step();}catch(const std::logic_error&){rejected=true;}
    require(rejected,"Failed preparation retained a stale commit");
    const auto saved=snapshot(retry);auto excessive=runtime;excessive.runaway_power_scale=std::numeric_limits<double>::max();
    rejected=false;try{retry.prepare_step(0,.001,.4,excessive);}catch(const std::runtime_error&){rejected=true;}
    require(rejected && snapshot(retry)==saved,"Invalid GPU candidate was published");
    // A valid retry still works after failed preparation.
    const auto dt=single.step(0,.001,.4,runtime);
    require(retry.prepare_step(0,.001,.4,runtime)==dt,"Failure poisoned subsequent preparation");
    retry.commit_step();require(snapshot(single)==snapshot(retry),"Recovery after invalid proposal differs");
}
void check_isothermal_retry() {
    Scenario s;CellConfig c;c.id="lumped";c.dimensions_m={.02,.02,.02};c.thermal_divisions={9,5,3};
    c.isothermal=true;c.initial_temperature_k=350;c.heat_capacity_j_per_k=10;c.initial_damage=1;
    c.runaway_energy_j=100;c.runaway_duration_s=.2;s.cells.push_back(c);
    s.ports.push_back({"cool","lumped",0,"cooling","+z",.0004,{0,0,.01},{0,0,1}});
    s.boundaries.push_back({"ambient","cool",0,300,1});
    const auto mesh=build_thermal_mesh(s);CudaSolver single(s,mesh,64),retry(s,mesh,256);RuntimeParameters runtime;
    for(int i=0;i<10;++i) {
        const auto before=snapshot(retry);const double dt=single.step(i*.001,.001,.4,runtime);
        for(int j=0;j<3;++j) {
            require(retry.prepare_step(i*.001,.001,.4,runtime)==dt,"Isothermal retry changed dt");
            require(snapshot(retry)==before,"Isothermal proposal published a source");
        }
        retry.commit_step();require(snapshot(retry)==snapshot(single),"Isothermal source/reduction retry differs");
    }
}
void check_restart(bool compliant=false) {
    auto input=fixture();
    if(compliant) {
        input.cells[0].electrical.voltage_limited_current=true;
        input.cells[0].voltage_profile_v={{{0.,.5}},{{1.,.5}}};
    }
    Simulation uninterrupted(input),resumed(input);
    for(int i=0;i<20;++i)uninterrupted.step(.002);
    const auto saved=uninterrupted.save_checkpoint("same-physics-v1");
    resumed.restore_checkpoint(saved,"same-physics-v1");
    require(saved==resumed.save_checkpoint("same-physics-v1"),"restart did not restore every field and history exactly");
    for(int i=0;i<20;++i) {
        uninterrupted.step(.002);resumed.step(.002);
        require(uninterrupted.save_checkpoint("same-physics-v1")==resumed.save_checkpoint("same-physics-v1"),
            "continued restart trajectory differs from uninterrupted CUDA trajectory");
    }
    const auto before=resumed.save_checkpoint("same-physics-v1");bool rejected=false;
    try{resumed.restore_checkpoint(saved,"different-physics");}catch(const std::runtime_error&){rejected=true;}
    require(rejected && before==resumed.save_checkpoint("same-physics-v1"),"contract mismatch corrupted committed state");
    auto broken=saved;broken[broken.size()/2]^=1;rejected=false;
    try{resumed.restore_checkpoint(broken,"same-physics-v1");}catch(const std::runtime_error&){rejected=true;}
    require(rejected && before==resumed.save_checkpoint("same-physics-v1"),"corrupt restart was accepted or mutated state");
    std::cout<<"restart: bitwise full state/continued trajectory, contract/checksum rejection pass\n";
}
void check_unvented_rebase() {
    auto input=fixture();input.gas_channels.resize(1);input.gas_channels[0].opening_pressure_pa=1e20;
    input.gas_model.split_transport=input.gas_model.cuda_kawasaki_transport=input.gas_model.cuda_network_transport=true;
    input.gas_model.coupled_gas_films=true;
    const auto mesh=build_thermal_mesh(input);CudaSolver source(input,mesh),identity(input,mesh);RuntimeParameters runtime;
    for(int i=0;i<3;++i)source.step(i*.001,.001,.4,runtime);
    identity.rebase_unvented_from(source,{{0,0,1.},{1,1,1.}});
    require(snapshot(source)==snapshot(identity),"Identity rebase changed a physical field");
    auto target=input;
    target.gas_compartments[1].volume_m3*=.25;target.gas_compartments[1].initial_mass_kg*=.25;
    auto extra=input.gas_compartments[1];extra.id="outside-b";extra.volume_m3*=.75;extra.initial_mass_kg*=.75;
    target.gas_compartments.push_back(extra);
    const auto target_mesh=build_thermal_mesh(target);CudaSolver split(target,target_mesh);
    split.rebase_unvented_from(source,{{0,0,1.},{1,1,.25},{1,2,.75}});
    const auto a=source.process_diagnostics(),b=split.process_diagnostics();
    require(std::abs(a.mass_residual_kg-b.mass_residual_kg)<1e-14 &&
        std::abs(a.thermal_residual_j-b.thermal_residual_j)<1e-8 &&
        std::abs(a.chemical_residual_j-b.chemical_residual_j)<1e-8 &&
        std::abs(a.oxidizer_residual_kg-b.oxidizer_residual_kg)<1e-14,"Split rebase lost an inventory");
    require(b.gas[1].mass_kg==a.gas[1].mass_kg*.25 && b.gas[2].chemical_j==a.gas[1].chemical_j*.75,
        "Rebase did not split extensive state by physical overlap");
    const auto saved=snapshot(split);bool rejected=false;
    try{split.rebase_unvented_from(source,{{0,0,1.},{1,1,.2},{1,2,.75}});}catch(const std::runtime_error&){rejected=true;}
    require(rejected && snapshot(split)==saved,"Invalid remap mutated target state");
    auto vented=input;vented.gas_channels[0].opening_pressure_pa=1.;
    const auto vented_mesh=build_thermal_mesh(vented);CudaSolver open(vented,vented_mesh);open.step(0,.001,.4,runtime);rejected=false;
    try{split.rebase_unvented_from(open,{{0,0,1.},{1,1,.25},{1,2,.75}});}catch(const std::runtime_error&){rejected=true;}
    require(rejected && snapshot(split)==saved,"Vented source was rebased");
    std::cout<<"unvented rebase: identity, conservative split, invalid-map/vented-source rejection pass\n";
}
int main() {try {
    check_retries(fixture());check_isothermal_retry();check_restart();check_unvented_rebase();
    auto compliant=fixture();compliant.cells[0].electrical.voltage_limited_current=true;
    compliant.cells[0].voltage_profile_v={{{0.,.5}},{{1.,.5}}};
    check_retries(compliant);check_restart(true);
    auto progressive=fixture();progressive.gas_channels.resize(1);
    progressive.gas_channels[0].opening.enabled=true;
    progressive.gas_channels[0].area_m2=0.;
    progressive.gas_channels[0].opening.damage_onset=.001;
    progressive.gas_channels[0].opening.seam_width_m=.0002;
    progressive.gas_channels[0].opening.seam_strength_factor=.5;
    progressive.gas_channels[0].opening.fracture_energy_j_per_m2=100.;
    progressive.cells[6].damage_onset_k=299.;progressive.cells[6].damage_rate_at_onset_per_s=.01;
    check_retries(progressive);
    auto split=fixture();split.gas_model.split_transport=true;
    split.gas_compartments[0].thermal_cell=0;split.gas_compartments[0].volume_m3=1e-7;
    check_retries(split);
    std::cout<<"transaction: exact retry/replace/discard, single commit, reset/ignite invalidation, all ledgers pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

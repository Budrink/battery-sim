#include "battery_native/simulation.hpp"
#include "battery_native/deterministic_aperture.hpp"
#include "battery_native/mass_mesh.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}

Scenario fixture(double pressure=200000.,double damage=0.,double cd=1e-20) {
    Scenario s;s.gas_model.split_transport=true;s.gas_model.reference_temperature_k=300.;
    CellConfig wall;wall.id="fixture-wall";wall.is_battery=false;wall.material_id="test-polymer";
    wall.dimensions_m={.001,.02,.02};wall.thermal_divisions={1,2,2};
    wall.heat_capacity_j_per_k=1.;wall.initial_temperature_k=300.;wall.initial_damage=damage;
    s.cells={wall};
    for(int sign:{-1,1}) {
        PortConfig p;p.id=sign<0?"wall:inner":"wall:outer";p.cell_id=wall.id;p.cell_index=0;
        p.kind="contact";p.surface=sign<0?"-x":"+x";p.area_m2=.0004;
        p.centroid_m={sign*.0005,0.,0.};p.normal={double(sign),0.,0.};s.ports.push_back(p);
        GasCompartment g;g.id=sign<0?"inside":"outside";g.volume_m3=1e-6;
        g.initial_temperature_k=300.;g.initial_mass_kg=(sign<0?pressure:100000.)*g.volume_m3/(287.*300.);
        const int gas=static_cast<int>(s.gas_compartments.size());s.gas_compartments.push_back(g);
        GasContact contact;contact.port_id=p.id;contact.compartment=gas;contact.thermal_volume=0;
        contact.h_w_per_m2_k=1e-20;s.gas_contacts.push_back(contact);
    }
    GasChannel ch;ch.id="wall-opening";ch.port_id="wall:outer";ch.shell_object=0;
    ch.from=0;ch.to=1;ch.rupture=true;ch.shell_thermal_volume=0;
    ch.opening_pressure_pa=50000.;ch.discharge_coefficient=cd;
    ch.opening.enabled=ch.opening.deterministic_aperture=true;
    ch.opening.area_rate_per_s=2.;ch.opening.fracture_energy_j_per_m2=100.;
    s.gas_channels={ch};return s;
}

void validation() {
    auto s=fixture();const auto m=build_thermal_mesh(s);build_mass_mesh(s,m);
    for(int variant=0;variant<10;++variant) {
        auto bad=s;
        if(variant==0)bad.cells[0].is_battery=true;
        if(variant==1)bad.gas_contacts.pop_back();
        if(variant==2)bad.gas_contacts[0].port_id="wall:outer";
        if(variant==3)bad.gas_contacts[0].thermal_volume=1;
        if(variant==4)bad.gas_channels[0].to=-1;
        if(variant==5)bad.gas_channels[0].opening.deterministic_aperture=false;
        if(variant==6)bad.gas_channels[0].shell_thermal_volume=-1;
        if(variant==7)bad.gas_contacts[0].h_w_per_m2_k=-1.;
        if(variant==8)bad.gas_contacts.push_back(bad.gas_contacts[0]);
        if(variant==9) {
            auto gas=bad.gas_compartments[0];gas.id="other-cavity";bad.gas_compartments.push_back(gas);
            auto contact=bad.gas_contacts[0];contact.compartment=2;bad.gas_contacts.push_back(contact);
        }
        bool rejected=false;try {build_thermal_mesh(bad);}catch(const std::exception&){rejected=true;}
        require(rejected,"Unsafe material-barrier topology accepted");
    }
}

ProcessDiagnostics evolve(Scenario s,double end,double cap) {
    const auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh);RuntimeParameters runtime;
    double t=0.;while(t<end-1e-14) {
        t+=solver.step(t,std::min(cap,end-t),.4,runtime);
    }
    const auto d=solver.process_diagnostics();std::vector<ShellDamageState> damage;
    solver.download_shell_states(damage);
    for(std::size_t i=1;i<damage.size();++i)require(damage[i].bulk==0.,"Opening escaped its material patch");
    require(std::abs(d.thermal_residual_j)<1e-8,"Material fracture/flow loses thermal energy");
    require(std::abs(d.mass_residual_kg)<1e-14,"Material fracture/flow loses mass");
    return d;
}

int main(int argc,char** argv) {try {
    validation();
    if(argc>2 && std::string(argv[1])=="--scenario") {
        const auto s=load_scenario(argv[2]);const auto m=build_thermal_mesh(s);build_mass_mesh(s,m);
        std::cout<<"material barrier scenario: "<<s.gas_channels.size()<<" channels, "<<m.volumes.size()<<" thermal volumes; topology checks pass\n";return 0;
    }
    if(argc>1 && std::string(argv[1])=="--geometry-only") {
        std::cout<<"material barrier: local opposite-face topology and ten rejection checks pass\n";return 0;
    }
    const auto closed=evolve(fixture(120000.),.01,.005);
    require(closed.channels[0].opening_fraction==0.,"Intact wall opened below pressure threshold");
    const auto weak=evolve(fixture(120000.,.8),.01,.005);
    require(weak.channels[0].opening_fraction>0.,"Damage did not lower the pressure barrier");
    const auto hot_no_pressure=evolve(fixture(100000.,1.),.01,.005);
    require(hot_no_pressure.channels[0].opening_fraction==0.,"Damage created an aperture without pressure");
    // Analytic constant-pressure law: fracture feedback is tested separately,
    // since spending gas energy deliberately changes pressure during a step.
    auto fixed=fixture();fixed.gas_channels[0].opening.fracture_energy_j_per_m2=0.;
    const auto a=evolve(fixed,.01,.005),b=evolve(fixed,.01,.0025);
    require(std::abs(a.channels[0].opening_fraction-b.channels[0].opening_fraction)<1e-10,
        "Time-integrated material opening changes with step partition");
    require(weak.channels[0].fracture_work_j>0.,"Material fracture costs no energy");
    const auto flow=evolve(fixture(200000.,0.,.7),.0001,.00001);
    require(flow.channels[0].transferred_kg>0.,"New material aperture carries no gas");
    std::cout<<"material barrier CUDA: pressure threshold, thermal weakening, local area, time law, fracture work, flow and ledgers pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "battery_native/simulation.hpp"
#include "battery_native/aperture_transfer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
namespace {
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
void near(double a,double b,double tol,const char* message) { require(std::abs(a-b)<=tol,message); }
void add_target(Scenario& s,const std::string& id,Vec3 position,std::vector<double> dimensions) {
    CellConfig c;c.id=id;c.is_battery=false;c.position_m=position;c.dimensions_m=std::move(dimensions);
    c.thermal_divisions={2,3,4};c.heat_capacity_j_per_k=1.;c.initial_temperature_k=300.;s.cells.push_back(c);
}
Scenario fixture(double damage=.6,double pressure=300000.) {
    Scenario s;s.gas_model.reference_temperature_k=300.;
    CellConfig c;c.id="B";c.dimensions_m={.02,.02,.02};c.thermal_divisions={3,3,3};
    c.heat_capacity_j_per_k=10.;c.initial_temperature_k=300.;s.cells={c};
    ShellConfig shell;shell.contents_id="B";shell.material_id="test-shell";shell.thickness_m=.001;
    shell.density_kg_m3=2000.;shell.specific_heat_j_kg_k=1000.;shell.contents_contact_h=1.;shell.seam_contact_h=1.;
    add_physical_shell(s,shell);
    const auto panel=std::find_if(s.cells.begin(),s.cells.end(),[](const auto& x){return x.id=="B:case:+x";});
    const int shell_id=static_cast<int>(panel-s.cells.begin());
    GasCompartment inside;inside.id="inside";inside.owner_cell=0;inside.volume_m3=1e-6;inside.initial_temperature_k=600.;
    inside.initial_mass_kg=pressure*inside.volume_m3/(287*inside.initial_temperature_k);
    GasCompartment outside;outside.id="chamber";outside.volume_m3=.01;outside.initial_temperature_k=300.;
    outside.initial_mass_kg=100000*outside.volume_m3/(287*outside.initial_temperature_k);s.gas_compartments={inside,outside};
    GasChannel ch;ch.id="opening";ch.port_id="B:case:+x:outer";ch.shell_object=shell_id;
    ch.from=0;ch.to=1;ch.rupture=true;ch.opening_pressure_pa=100000.;
    ch.opening.enabled=true;ch.opening.pressure_damage_rate_per_s=0.;ch.surface_link_h_w_per_m2_k=1e6;
    ch.opening.initial_mechanical_damage=damage;
    // Hold flow/conductance fixed to isolate energy-routing tests from the
    // separate physical-Cd pressure-feedback test in shell_opening.cpp.
    const double area=gas_channel_area(s,ch);
    ch.discharge_coefficient*=1e-7/area;ch.surface_link_h_w_per_m2_k*=1e-7/area;
    s.gas_channels={ch};
    add_target(s,"near",{.035,0,0},{.02,.022,.022});
    add_target(s,"far",{.065,0,0},{.02,.022,.022});
    return s;
}
struct Run { ProcessDiagnostics d;std::vector<ThermalState> field;double target_energy=0.; };
Run evolve(const Scenario& s,double duration=.002,double cap=.0001,int tile=128,bool retry=false) {
    const auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh,tile);RuntimeParameters r;double time=0.;
    Run result;std::vector<ThermalState> initial;solver.download_field(initial);
    while(time<duration-1e-14) {
        const double dt_cap=std::min(cap,duration-time);
        if(retry) {
            solver.prepare_step(time,dt_cap,.4,r);solver.discard_step();
            const auto d=solver.process_diagnostics();
            near(d.fracture_work_j,result.d.fracture_work_j,1e-14,"Discard changed fracture work");
        }
        time+=solver.step(time,dt_cap,.4,r);result.d=solver.process_diagnostics();
        require(std::abs(result.d.thermal_residual_j)<1e-8,"Energy not conserved with aperture/work");
        require(std::abs(result.d.mass_residual_kg)<1e-14,"Mass not conserved with aperture");
    }
    solver.download_field(result.field);
    for(std::size_t j=0;j<mesh.volumes.size();++j) {
        const auto& name=s.cells[mesh.volumes[j].object].id;
        if(name=="near" || name=="screen")result.target_energy+=result.field[j].stored_energy_j-initial[j].stored_energy_j;
        if(name=="far")near(result.field[j].temperature_k,300.,1e-12,"Heat crossed the first opaque target");
    }
    return result;
}
void geometry() {
    auto s=fixture();auto m=build_thermal_mesh(s);double fraction=0.;
    for(const auto& p:m.gas_patches)if(p.channel>=0) {
        require(s.cells[m.volumes[p.volume].object].id=="near","Not the first visible surface");fraction+=p.aperture_area_fraction;
    }
    near(fraction,1.,1e-12,"Aperture projected area is not conserved");
    add_target(s,"screen",{.02,-.0055,0},{.004,.011,.022});m=build_thermal_mesh(s);
    double screen=0,behind=0;
    for(const auto& p:m.gas_patches)if(p.channel>=0) {
        const auto& id=s.cells[m.volumes[p.volume].object].id;
        if(id=="screen")screen+=p.aperture_area_fraction;
        if(id=="near")behind+=p.aperture_area_fraction;
        require(id!="far","Geometry tunneled through two barriers");
    }
    near(screen,.5,1e-12,"Partial screen area");near(behind,.5,1e-12,"Uncovered area");
    for(auto& cell:s.cells)if(cell.id=="near")cell.thermal_divisions={3,7,2};
    m=build_thermal_mesh(s);fraction=0;for(const auto& p:m.gas_patches)if(p.channel>=0)fraction+=p.aperture_area_fraction;
    near(fraction,1.,1e-12,"Visibility depends on target grid");
    s=fixture();s.cells.erase(s.cells.end()-2,s.cells.end());m=build_thermal_mesh(s);
    require(m.gas_patches.empty(),"Empty space received a thermal link");
}
void conductance() {
    near(aperture_conductance(10,0,.5,1,1,1000),0,0,"Closed barrier conducts");
    near(aperture_conductance(10,.5,.5,1,0,1000),0,0,"Pressure equilibrium still enhances transfer");
    near(aperture_conductance(10,.5,.5,1,-1,1000),0,0,"Outward connection used reverse flow");
    near(aperture_conductance(10,.5,.5,.25,.001,1000),.25,1e-15,"Transport exceeds flow heat capacity");
}
void transfer() {
    auto s=fixture();const auto on=evolve(s);
    require(on.d.channels[0].surface_energy_j>0,"Open hot flow does not heat target");
    near(on.target_energy,on.d.channels[0].surface_energy_j,1e-10,"Solid gained heat twice or lost routed heat");
    s.gas_channels[0].surface_link_h_w_per_m2_k=0;const auto off=evolve(s);
    near(off.target_energy,0.,1e-12,"No-link target heats up");
    near(on.d.gas[0].energy_j,off.d.gas[0].energy_j,1e-10,"Surface transfer double-charged source gas");
    near(off.d.gas[1].energy_j-on.d.gas[1].energy_j,on.target_energy,1e-10,"Arriving gas did not pay for surface heat");
    near(evolve(fixture(0.)).target_energy,0.,1e-12,"Intact barrier transfers heat");
    near(evolve(fixture(.6,100000.)).target_energy,0.,1e-12,"No-pressure flow heats surface");
    near(evolve(fixture(.6,80000.)).target_energy,0.,1e-12,"Reverse flow heated forward target");
    auto cold=fixture();auto& gas=cold.gas_compartments[0];gas.initial_temperature_k=300.;
    gas.initial_mass_kg=300000*gas.volume_m3/(287*300.);
    // At the beginning equal T gives exactly zero heat despite nonzero mass flow.
    const auto mesh=build_thermal_mesh(cold);CudaSolver solver(cold,mesh);RuntimeParameters r;
    solver.step(0.,1e-8,.4,r);near(solver.process_diagnostics().channels[0].surface_energy_j,0.,1e-14,"Equal-temperature flow creates heat");
    const auto coarse=evolve(fixture(),.002,.0002),fine=evolve(fixture(),.002,.00005);
    near(coarse.target_energy,fine.target_energy,1e-5,"Aperture transfer fails time refinement");
    const auto tiled=evolve(fixture(),.002,.0001,64,true);
    near(tiled.target_energy,on.target_energy,0.,"Aperture transfer depends on tile/retry");
    std::cout<<"surface heat="<<on.target_energy<<" J, energy residual="<<on.d.thermal_residual_j<<" J\n";
}
void fracture() {
    auto already_open=fixture(.6);already_open.gas_channels[0].opening.fracture_energy_j_per_m2=1e6;
    const auto fixed=evolve(already_open);
    require(fixed.d.channels[0].transferred_kg>0 && fixed.d.fracture_work_j==0,
        "Existing aperture was charged again for mass transfer");
    auto s=fixture(.3,1000000.);auto& ch=s.gas_channels[0];const double area=gas_channel_area(s,ch);
    ch.discharge_coefficient=.7e-12/area;
    ch.opening.pressure_damage_rate_per_s=3.;ch.opening.fracture_energy_j_per_m2=1./area;
    const auto paid=evolve(s,.005,.00005);require(paid.d.fracture_work_j>0,"Fracture costs no energy");
    const double start=std::pow((.3-.2)/.8,2);
    near(paid.d.fracture_work_j,ch.opening.fracture_energy_j_per_m2*area*(paid.d.channels[0].opening_fraction-start),1e-10,"Work != fracture energy times newly opened area");
    ch.opening.fracture_energy_j_per_m2=0;const auto free=evolve(s,.005,.00005);
    require(paid.d.gas[0].temperature_k<free.d.gas[0].temperature_k,"Fracture did not cool its pressure reservoir");
    require(paid.d.channels[0].opening_fraction<free.d.channels[0].opening_fraction,"Work did not feed back into opening");
    ch.opening.fracture_energy_j_per_m2=1./area;
    const auto retry=evolve(s,.005,.00005,256,true);
    near(retry.d.fracture_work_j,paid.d.fracture_work_j,0.,"Work ledger depends on prepare/discard or tile");
    const auto fine=evolve(s,.005,.000025);
    near(fine.d.fracture_work_j,paid.d.fracture_work_j,1e-4,"Fracture work fails time refinement");
    // At equal opening increment thermal weakening lowers the work per area.
    s.cells[ch.shell_object].initial_damage=.5;
    const auto weak=evolve(s,.005,.00005);
    near(weak.d.fracture_work_j,.5*ch.opening.fracture_energy_j_per_m2*area*(weak.d.channels[0].opening_fraction-start),1e-10,
        "Thermal weakening does not reduce fracture work per new area");
    std::cout<<"fracture work="<<paid.d.fracture_work_j<<" J, gas T paid/free="<<paid.d.gas[0].temperature_k<<'/'<<free.d.gas[0].temperature_k<<" K\n";
}
void validation() {
    for(int kind=0;kind<6;++kind) {
        auto s=fixture();
        if(kind==0)s.gas_channels[0].to=-1;
        if(kind==1)s.gas_channels[0].opening.enabled=false;
        if(kind==2)s.cells.back().orientation_xyzw={0,0,.1,.99};
        if(kind==3)s.gas_channels[0].surface_link_h_w_per_m2_k=-1;
        if(kind==4)s.gas_channels[0].opening.fracture_energy_j_per_m2=-1;
        if(kind==5)s.cells.back().position_m.x=.011;
        bool rejected=false;try{build_thermal_mesh(s);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"Invalid aperture/work configuration accepted");
    }
}
void insufficient_work() {
    auto s=fixture(.3,300000.);auto& ch=s.gas_channels[0];const double area=gas_channel_area(s,ch);
    ch.discharge_coefficient=.7e-12/area;ch.opening.fracture_energy_j_per_m2=1000./area;
    ch.opening.pressure_damage_rate_per_s=1.;
    auto& gas=s.gas_compartments[0];gas.initial_temperature_k=1.001;
    gas.initial_mass_kg=300000*gas.volume_m3/(287*gas.initial_temperature_k);
    const auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh);RuntimeParameters r;
    solver.step(0.,.0001,.4,r);const auto d=solver.process_diagnostics();
    const double budget=gas.initial_mass_kg*1000*(gas.initial_temperature_k-1.);
    require(d.fracture_work_j>0 && d.fracture_work_j<=budget+1e-11,"Fracture overspent available gas energy");
    near(d.gas[0].temperature_k,1.,1e-7,"Energy-starved aperture was not limited");
    require(std::abs(d.thermal_residual_j)<1e-8,"Limited fracture loses energy");
}
void dynamic_step_bound() {
    auto closed=fixture(0.);closed.gas_channels[0].surface_link_h_w_per_m2_k=1e12;
    auto disabled=closed;disabled.gas_channels[0].surface_link_h_w_per_m2_k=0.;
    CudaSolver shut(closed,build_thermal_mesh(closed)),off(disabled,build_thermal_mesh(disabled));
    RuntimeParameters runtime;
    const double shut_dt=shut.prepare_step(0.,1.,.4,runtime);
    near(shut_dt,off.prepare_step(0.,1.,.4,runtime),0.,"Closed aperture throttles solid CFL");

    // Many shell patches feed ONE small receiving volume: sum their active
    // conductances before bounding its heat capacity, not one dt per patch.
    auto open=fixture(.6);open.gas_channels[0].surface_link_h_w_per_m2_k=1e12;
    open.gas_channels[0].discharge_coefficient=.7;
    for(auto& cell:open.cells)if(cell.id=="near") {
        cell.thermal_divisions={1,1,1};cell.heat_capacity_j_per_k=1e-5;
    }
    const auto mesh=build_thermal_mesh(open);int count=0;
    for(const auto& patch:mesh.gas_patches)if(patch.channel>=0)++count;
    require(count>1,"Many-to-one aperture test has no overlapping incidences");
    CudaSolver active(open,mesh);const double dt=active.step(0.,1.,.4,runtime);
    require(dt<shut_dt,"Active aperture omitted from solid CFL");
    std::vector<ThermalState> field;active.download_field(field);
    for(std::size_t i=0;i<field.size();++i)if(open.cells[mesh.volumes[i].object].id=="near")
        require(field[i].temperature_k>300. && field[i].temperature_k<=600.,"Aperture CFL overshoots donor temperature");
    require(std::abs(active.process_diagnostics().thermal_residual_j)<1e-8,"Dynamic CFL breaks energy balance");
    std::cout<<"aperture CFL: closed="<<shut_dt<<" s, active="<<dt<<" s\n";
}
} // namespace
int main() { try {conductance();geometry();validation();transfer();fracture();insufficient_work();dynamic_step_bound();std::cout<<"Aperture connections passed\n";}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }

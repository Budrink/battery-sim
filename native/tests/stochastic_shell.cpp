#include "battery_native/simulation.hpp"
#include "battery_native/shell_opening.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
Scenario fixture(int shell_n=12,std::uint64_t seed=17,double cd=1e-20,double pressure=200000.) {
    Scenario s;s.gas_model.split_transport=true;s.gas_model.reference_temperature_k=300.;
    CellConfig c;c.id="B";c.dimensions_m={.02,.02,.02};c.thermal_divisions={3,3,3};
    c.heat_capacity_j_per_k=100.;c.initial_temperature_k=300.;s.cells={c};
    ShellConfig sh;sh.contents_id="B";sh.material_id="test-shell";sh.thickness_m=.00015;
    sh.density_kg_m3=1600.;sh.specific_heat_j_kg_k=1000.;sh.contents_contact_h=1.;sh.seam_contact_h=1.;
    sh.surface_divisions={shell_n,shell_n,shell_n};add_physical_shell(s,sh);
    GasCompartment gas;gas.id="inside";gas.owner_cell=0;gas.volume_m3=1e-6;gas.initial_temperature_k=300.;
    gas.initial_mass_kg=pressure*gas.volume_m3/(287.*300.);s.gas_compartments={gas};
    GasChannel ch;ch.id="rupture";ch.from=0;ch.to=-1;ch.shell_object=6;ch.port_id="B:case:+z:outer";
    ch.rupture=true;ch.outside_pressure_pa=100000.;ch.opening_pressure_pa=50000.;ch.discharge_coefficient=cd;
    ch.opening.enabled=ch.opening.binary_pixels=ch.opening.stochastic_pixels=true;
    ch.opening.pixel_rate_per_s=20.;ch.opening.pixel_pressure_exponent=1.;ch.opening.rupture_seed=seed;
    s.gas_channels={ch};return s;
}
struct Result {ProcessDiagnostics diagnostics;std::vector<ShellDamageState> shell;std::vector<ThermalState> solid;};
Result run(const Scenario& s,double end,double cap,int tile=128,bool retries=false) {
    const auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh,tile);RuntimeParameters r;double time=0.;
    require(mesh.object_offsets[1]==27,"Shell refinement changed battery contents grid");
    while(time<end-1e-14) {
        const double dt=solver.prepare_step(time,std::min(cap,end-time),.4,r);
        if(retries) {
            std::vector<ShellDamageState> before,after;solver.download_shell_states(before);
            solver.discard_step();solver.prepare_step(time,std::min(cap/3,end-time),.4,r);solver.discard_step();
            require(dt==solver.prepare_step(time,std::min(cap,end-time),.4,r),"Retry changed random event time");
            solver.download_shell_states(after);
            for(std::size_t k=0;k<before.size();++k)require(before[k].failure_hazard==after[k].failure_hazard && before[k].bulk==after[k].bulk,"Rejected step advanced random clocks");
        }
        solver.commit_step();time+=dt;
    }
    Result out;out.diagnostics=solver.process_diagnostics();solver.download_shell_states(out.shell);solver.download_field(out.solid);
    require(std::abs(out.diagnostics.mass_residual_kg)<1e-13 && std::abs(out.diagnostics.thermal_residual_j)<1e-7,"Stochastic rupture breaks conservation");
    for(const auto& d:out.shell)require((d.bulk==0. || d.bulk==1.) && d.bulk==d.seam,"Partial pixel opened");
    return out;
}
std::vector<int> pattern(const Result& r) {
    std::vector<int> p;for(const auto& d:r.shell)p.push_back(d.bulk==1.);return p;
}
void probability_law() {
    require(pixel_failure_rate(99,100,20,2)==0 && pixel_failure_rate(100,100,20,2)==0,"Subthreshold hazard is not zero");
    require(pixel_failure_rate(300,100,20,2)>pixel_failure_rate(200,100,20,2),"Pressure does not accelerate failure");
    require(binary_shell_strength(100,.8,0,1)<binary_shell_strength(100,.2,0,1),"Damage does not lower threshold");
    require(binary_shell_strength(100,1,0,1)>0,"Fully damaged barrier has zero residual threshold");
    const auto key=pixel_channel_key("test");int failed=0;const int n=100000;
    for(int i=0;i<n;++i) {
        const double h=pixel_failure_hazard(17,key,i);require(std::isfinite(h) && h>0,"Invalid random threshold");
        if(h<=.4)++failed;
    }
    require(std::abs(double(failed)/n-(-std::expm1(-.4)))<.005,"Exponential survival distribution failed");
}
void sequential_and_retry() {
    const auto s=fixture();const auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh);RuntimeParameters r;
    double first=1e300;for(int i=0;i<144;++i)first=std::min(first,pixel_failure_hazard(17,pixel_channel_key("rupture"),i)/20.);
    const double dt=solver.step(0,.02,.4,r);const auto d=solver.process_diagnostics();
    require(std::abs(dt-first)<1e-14,"Did not land on first random event");
    require(std::abs(d.channels[0].opening_fraction-1./144)<1e-14,"First event opened an entire uniform panel");
    solver.reset();std::vector<ShellDamageState> reset;solver.download_shell_states(reset);
    for(const auto& state:reset)require(state.failure_hazard==0. && state.bulk==0.,"Reset retained a random clock");
    require(solver.step(0,.02,.4,r)==dt,"Reset redrew random pixel thresholds");
    const auto a=run(s,.02,.002,64),b=run(s,.02,.002,256,true),fine=run(s,.02,.001);
    require(pattern(a)==pattern(b) && pattern(a)==pattern(fine),"Seeded pattern depends on timestep or CUDA block size");
    for(std::size_t i=0;i<a.shell.size();++i)require(a.shell[i].failure_hazard==b.shell[i].failure_hazard,"CUDA tile/retry changes stochastic clock");
    require(pattern(a)!=pattern(run(fixture(12,18),.02,.002)),"Changing seed did not change failure pattern");
    const auto& ch=a.diagnostics.channels[0];require(ch.opening_fraction>0 && ch.opening_fraction<1,"No partial panel failure");
    std::cout<<"fixed pressure: first pixel at "<<dt<<" s; open fraction="<<ch.opening_fraction<<"; fixed-seed dt/tile/retry agree\n";
}
void pressure_feedback() {
    auto s=fixture(24,17,.7);s.gas_channels[0].opening.fracture_energy_j_per_m2=100.;
    const auto a=run(s,.03,.001),b=run(s,.06,.001),fine=run(s,.03,.0005);
    require(a.diagnostics.channels[0].opening_fraction>0 && a.diagnostics.channels[0].opening_fraction<1,"Pressure unload did not arrest failure");
    require(a.diagnostics.gas[0].pressure_pa<150000.,"Pressure remained supercritical");
    require(pattern(a)==pattern(b),"New pixels failed below pressure threshold");
    require(pattern(a)==pattern(fine),"Unloading pattern does not converge in timestep fixture");
    const double work=a.diagnostics.channels[0].fracture_work_j;
    require(std::abs(work-100.*gas_channel_area(s,s.gas_channels[0])*a.diagnostics.channels[0].opening_fraction)<1e-12,"Fracture work is not full-pixel area");
    require(work==b.diagnostics.channels[0].fracture_work_j,"Already opened pixels charged again");
    std::cout<<"pressure feedback: opened "<<576*a.diagnostics.channels[0].opening_fraction<<" / 576 pixels; final pressure="<<a.diagnostics.gas[0].pressure_pa<<" Pa; work="<<work<<" J\n";
}
void closed_and_energy_limited() {
    auto cold=fixture(12,17,.7,140000.);for(std::size_t k=1;k<cold.cells.size();++k)cold.cells[k].initial_damage=0.;
    require(!run(cold,.01,.002).diagnostics.channels[0].open,"Subthreshold pixels opened");
    auto damaged=cold;for(std::size_t k=1;k<damaged.cells.size();++k)damaged.cells[k].initial_damage=.8;
    require(run(damaged,.01,.002).diagnostics.channels[0].open,"Thermal damage did not promote fracture");
    auto costly=fixture();costly.gas_channels[0].opening.fracture_energy_j_per_m2=1e12;
    const auto held=run(costly,.01,.002);require(!held.diagnostics.channels[0].open && held.diagnostics.fracture_work_j==0,"Unaffordable pixel opened fractionally or created energy");
}
void grid_geometry() {
    const auto a=fixture(12),b=fixture(24);const auto ma=build_thermal_mesh(a),mb=build_thermal_mesh(b);
    double ca=0.,cb=0.;
    for(std::size_t i=1;i<a.cells.size();++i)ca+=a.cells[i].heat_capacity_j_per_k;
    for(std::size_t i=1;i<b.cells.size();++i)cb+=b.cells[i].heat_capacity_j_per_k;
    require(ca==cb,"Shell refinement changed material inventory");
    require(mb.volumes.size()-27==4*(ma.volumes.size()-27),"Refinement changed normal layers or core grid");
    for(std::size_t j=0;j<b.interfaces.size();++j) {
        double area=0.;for(int k=mb.interface_offsets[j];k<mb.interface_offsets[j+1];++k)area+=mb.links[mb.interface_links[k]].weight;
        require(std::abs(area-1)<1e-12,"Nonmatching shell/core grid loses contact area");
    }
}
int main() {try {probability_law();grid_geometry();sequential_and_retry();pressure_feedback();closed_and_energy_limited();
    std::cout<<"stochastic shell: probability, binary state, independent mesh, retries, pressure feedback and ledgers pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "battery_native/simulation.hpp"
#include "battery_native/shell_opening.hpp"
#include "battery_native/constitutive.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Scenario fixture(int divisions=3,double pressure=180000.) {
    Scenario s;s.gas_model.reference_temperature_k=300.;
    CellConfig c;c.id="B";c.dimensions_m={.02,.03,.04};c.thermal_divisions={divisions,divisions,divisions};
    c.heat_capacity_j_per_k=100;c.initial_temperature_k=300;s.cells={c};
    ShellConfig sh;sh.contents_id="B";sh.material_id="test-case";sh.thickness_m=.001;sh.density_kg_m3=2000;
    sh.specific_heat_j_kg_k=1000;sh.contents_contact_h=1;sh.seam_contact_h=1;add_physical_shell(s,sh);
    GasCompartment g;g.id="inside";g.owner_cell=0;g.volume_m3=1e-6;g.initial_temperature_k=300;
    g.initial_mass_kg=pressure*g.volume_m3/(287*300);s.gas_compartments={g};
    for(int k=1;k<=6;++k) {
        GasChannel ch;ch.id="rupture-"+std::to_string(k);ch.port_id=s.cells[k].id+":outer";
        ch.shell_object=k;ch.from=0;ch.to=-1;ch.rupture=true;ch.outside_pressure_pa=100000;
        ch.opening_pressure_pa=100000;ch.opening.enabled=true;ch.opening.seam_width_m=.001;
        ch.opening.seam_strength_factor=.5;ch.opening.damage_onset=.01;
        // Isolate the constitutive law from unloading in these older unit
        // fixtures. The feedback test below uses the physical Cd=.7 instead.
        ch.discharge_coefficient=.7e-12/gas_channel_area(s,ch);s.gas_channels.push_back(ch);
    }
    return s;
}
ProcessDiagnostics evolve(const Scenario& s,double end,double cap=.001) {
    const auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh);RuntimeParameters r;double time=0;
    while(time<end-1e-14)time+=solver.step(time,std::min(cap,end-time),.4,r);
    const auto d=solver.process_diagnostics();
    require(std::abs(d.mass_residual_kg)<1e-15 && std::abs(d.thermal_residual_j)<1e-7,"Shell opening breaks mass/energy balance");
    return d;
}
void laws() {
    ShellOpeningLaw l;l.enabled=true;
    require(shell_open_fraction(l,0)==0 && shell_open_fraction(l,l.damage_onset)==0 && shell_open_fraction(l,1)==1,"Aperture endpoints");
    double previous=0;for(int i=0;i<=100;++i){double a=shell_open_fraction(l,i/100.);require(a>=previous,"Nonmonotone aperture");previous=a;}
    require(shell_strength(100,400,.5,300,100)<shell_strength(100,300,.5,300,100),"Heating did not weaken shell");
    require(shell_pressure_damage_rate(l,90,100,.2)==0,"Subcritical pressure damages shell");
    require(advance_shell_damage(.4,0,10)==.4,"Damage heals without load");
    require(std::abs(advance_shell_damage(0,2,shell_damage_dt(l,2))-l.max_damage_step)<1e-14,"Wrong damage time-step bound");
    require(std::abs(advance_shell_damage(.6,2,shell_damage_dt(l,2,.6))-.6-l.max_damage_step)<1e-14,"Absolute damage budget is wrong");
    require(shell_damage_dt(l,1e9,1.-1e-14)==1e300,"Saturated seam throttles the solver");
    require(advance_shell_damage(1.-2e-12,0,1)==1.,"Roundoff residue retains intact shell connectivity");
    require(advance_shell_damage(.999,0,1)==.999,"Terminal closure destroys resolved remaining integrity");
    const double feedback=shell_feedback_dt(.01,.2,3.,.004,0.);
    require(std::abs(3.*(.01+.2*feedback)*feedback-.004)<1e-15,"Prospective flow budget is wrong");
    require(shell_feedback_dt(0,.2,3.,.004,1.)==1.,"Closed barrier throttled before opening onset");
    require(shell_feedback_dt(.01,0,3.,.004,0.)==1e300,"Fixed aperture throttled by opening feedback");
    const double slow=shell_slow_aperture_dt(.01,.2,.1,.05);
    require(std::abs(.2*slow/.05-.025)<1e-15,"Slow aperture exceeds whole-element geometric budget");
    require(shell_slow_aperture_dt(.01,.2,.05,.05)==slow/2,"Aperture budget does not refine with mass step");
    require(shell_slow_aperture_dt(0,.2,.1,.05)==0,"Slow-aperture shortcut skips first opening");
    require(shell_slow_aperture_dt(.01,0,.1,.05)==1e300,"Fixed aperture incurs geometric restriction");
    // Pressure-arrested UPV pinhole: do not bind the whole thermal graph to
    // a microscopic relative budget. Area/rate below use a unit whole element.
    const double tiny_area=2.568800803896272e-5,rate=.0023070122552869333*(1-tiny_area);
    const double tiny_dt=shell_slow_aperture_dt(tiny_area,rate,.02,1.);
    require(tiny_dt>2. && tiny_dt<2.3,"Pinhole geometry regressed to relative microstep throttling");
    require(rate*tiny_dt<=.005*(1+1e-14),"Pinhole shortcut exceeds whole-element area budget");
    require(shell_slow_aperture_dt(tiny_area*1e-9,rate,.02,1.)==tiny_dt,"Whole-element budget depends on old hole size");
    require(shell_slow_aperture_dt(tiny_area,rate,.02,2.)==2*tiny_dt,"Whole-element area scaling is wrong");
    for(int n:{1,3,4,8}) {
        double f=0;for(int x=0;x<n;++x)for(int y=0;y<n;++y)
            f+=shell_seam_fraction(-.01+(x+.5)*.02/n,-.015+(y+.5)*.03/n,.02/n,.03/n,.02,.03,.001)/(n*n);
        require(std::abs(f-(1-.018*.028/(.02*.03)))<1e-13,"Seam area depends on grid resolution");
    }
}
void damage_and_geometry() {
    const auto s=fixture();auto d=evolve(s,.1);
    for(const auto& c:d.channels)require(c.open && c.max_bulk_damage==0 && c.max_seam_damage>.01 && c.max_seam_damage<1 && c.opening_fraction>0,"Whole shell weak seams do not open progressively");
    auto finer=evolve(fixture(4),.1);auto finest=evolve(fixture(8),.1);
    for(int k=0;k<6;++k)require(std::abs(d.channels[k].opening_fraction-finer.channels[k].opening_fraction)<1e-12 && std::abs(d.channels[k].opening_fraction-finest.channels[k].opening_fraction)<1e-12,"Uniform-pressure aperture is mesh dependent");
    auto high=evolve(fixture(3,240000),.1);
    for(const auto& c:high.channels)require(c.max_bulk_damage>0 && c.max_seam_damage>c.max_bulk_damage,"Bulk must also tear under sufficient pressure");
    const auto coarse=evolve(s,.1,.002),fine=evolve(s,.1,.0005);
    require(std::abs(coarse.channels[0].max_seam_damage-fine.channels[0].max_seam_damage)<.0002,"Pressure damage failed time refinement");
    std::cout<<"seam D="<<d.channels[0].max_seam_damage<<", bulk D="<<d.channels[0].max_bulk_damage<<", aperture="<<d.channels[0].opening_fraction<<'\n';
}
void flow_and_quench() {
    auto s=fixture();for(auto& ch:s.gas_channels){ch.opening.pressure_damage_rate_per_s=0;ch.discharge_coefficient=.7e-9/gas_channel_area(s,ch);}
    for(auto& ch:s.gas_channels)ch.opening.initial_mechanical_damage=.3;
    const auto lower=evolve(s,.000001);for(auto& ch:s.gas_channels)ch.opening.initial_mechanical_damage=.6;
    const auto higher=evolve(s,.000001);
    require(higher.channels[0].mass_flow_kg_s>lower.channels[0].mass_flow_kg_s,"Larger aperture does not increase flow at fixed pressure");
    auto d=evolve(s,.01);require(d.channels[0].max_bulk_damage==.6 && d.channels[0].max_seam_damage==.6,"Disabled pressure damage healed/changed initial damage");
    // The same already-open channel permits reverse transfer from a receiving cavity.
    auto outside=s.gas_compartments[0];outside.owner_cell=-1;outside.id="receiver";outside.volume_m3=.001;
    outside.initial_mass_kg=300000*outside.volume_m3/(287*300);s.gas_compartments.push_back(outside);
    for(auto& ch:s.gas_channels)ch.to=1;
    d=evolve(s,.000001);require(d.channels[0].mass_flow_kg_s<0,"Opened shell does not pass reverse gas flow");
}
void thermal_damage() {
    auto s=fixture(3,100000);
    for(std::size_t k=1;k<s.cells.size();++k){s.cells[k].damage_onset_k=299;s.cells[k].damage_rate_at_onset_per_s=.5;}
    auto d=evolve(s,.1);for(const auto& c:d.channels)
        require(!c.open && c.max_bulk_damage==0 && c.max_thermal_damage>.01 && c.transferred_kg==0,
            "Thermal weakening opened a barrier without pressure");
    for(std::size_t k=1;k<s.cells.size();++k)s.cells[k].initial_damage=1.;
    // Fully degraded residual barrier, still below its nonzero pressure threshold.
    s.gas_compartments[0].initial_mass_kg*=1.001;
    d=evolve(s,.01);for(const auto& c:d.channels)
        require(!c.open && c.max_thermal_damage==1 && c.fracture_work_j==0,"Dthermal=1 bypassed residual strength");
    s.gas_compartments[0].initial_mass_kg*=1.1;
    d=evolve(s,.01);for(const auto& c:d.channels)
        require(c.open && c.max_bulk_damage>0,"Weakened barrier failed to open above pressure threshold");
}
void unload_and_capacity() {
    auto s=fixture(3,120000);for(auto& ch:s.gas_channels)ch.opening.initial_mechanical_damage=.3;
    auto d=evolve(s,.05);
    for(const auto& c:d.channels)require(std::abs(c.max_bulk_damage-.3)<1e-15 && std::abs(c.max_seam_damage-.3)<1e-15,"Unloaded shell keeps growing damage or heals");
    s=fixture(3,240000);
    for(auto& c:s.cells)c.initial_temperature_k=400;
    auto& law=s.cells[1].heat_capacity_law;law.reference_temperature_k=300;law.factor.count=2;
    law.factor.temperature_k[0]=200;law.factor.temperature_k[1]=1000;law.factor.value[0]=law.factor.value[1]=1;law.damage_factor_at_one=.5;
    const auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh);RuntimeParameters runtime;
    std::vector<ThermalState> before,after;solver.download_field(before);solver.step(0,1e-5,.4,runtime);solver.download_field(after);
    for(int i=mesh.object_offsets[1];i<mesh.object_offsets[2];++i) {
        require(after[i].damage>0 && after[i].stored_energy_j==before[i].stored_energy_j,"Pressure damage created thermal energy");
        require(std::abs(after[i].temperature_k-thermal_temperature(law,mesh.volumes[i].capacity_j_per_k,after[i].stored_energy_j,after[i].damage))<1e-10,"Progressive damage left stale T(U,D)");
    }
}
void validation() {
    for(int k=0;k<9;++k) {
        auto s=fixture();auto& c=s.gas_channels[0];
        if(k==0)c.opening.seam_width_m=.1;
        if(k==1)c.opening.seam_strength_factor=0;
        if(k==2)c.opening.damage_onset=1;
        if(k==3)c.rupture=false;
        if(k==4){auto duplicate=c;duplicate.id="duplicate";s.gas_channels.push_back(duplicate);}
        if(k==5)c.port_id=s.cells[c.shell_object].id+":inner";
        if(k==6)s.gas_compartments[0].owner_cell=-1;
        if(k==7)c.area_m2=1e-12;
        if(k==8)c.opening.initial_mechanical_damage=1.01;
        bool rejected=false;try{build_thermal_mesh(s);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"Invalid progressive shell configuration accepted");
    }
}
void pressure_feedback() {
    auto s=fixture(3,300000.);
    for(auto& c:s.gas_channels)c.discharge_coefficient=.7;
    const auto early=evolve(s,.05,.00005),late=evolve(s,.1,.00005);
    double area=0.,later_area=0.;
    for(std::size_t k=0;k<s.gas_channels.size();++k) {
        const auto& a=early.channels[k];const auto& b=late.channels[k];
        area+=a.effective_area_m2;later_area+=b.effective_area_m2;
        require(a.opening_fraction>0 && a.opening_fraction<1,"Feedback did not arrest partial opening");
        require(std::abs(a.effective_area_m2-gas_channel_area(s,s.gas_channels[k])*a.opening_fraction)<1e-15,"Rupture area is not geometric");
        require(b.max_bulk_damage==a.max_bulk_damage && b.max_seam_damage==a.max_seam_damage,"Damage grows or heals after unloading");
    }
    require(area>1e-8 && area==later_area,"Artificial aperture ceiling or healing");
    require(early.gas[0].pressure_pa<150000.,"Opening failed to relieve pressure");
    const auto fine=evolve(s,.1,.000025);double finer_area=0.;
    for(const auto& c:fine.channels)finer_area+=c.effective_area_m2;
    require(std::abs(finer_area-area)/area<.02,"Pressure-arrested area fails refinement");
    // D=1 exposes the entire panel, regardless of the former tiny orifice.
    for(auto& ch:s.gas_channels)ch.opening.initial_mechanical_damage=1.;
    const auto full=evolve(s,1e-9,1e-9);
    for(std::size_t k=0;k<s.gas_channels.size();++k)
        require(std::abs(full.channels[k].effective_area_m2-gas_channel_area(s,s.gas_channels[k]))<1e-15,"Full damage retains an orifice cap");
    std::cout<<"pressure feedback: P="<<early.gas[0].pressure_pa<<" Pa, arrested area="<<area<<" m2\n";
}
void terminal_connectivity() {
    auto s=fixture(3,100000);for(auto& ch:s.gas_channels)ch.opening.initial_mechanical_damage=1.-2e-12;
    auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh);RuntimeParameters runtime;double time=0;
    for(int i=0;i<2;++i)time+=solver.step(time,1e-5,.4,runtime);
    std::vector<ThermalState> state;std::vector<InterfaceState> edges;solver.download_field(state);solver.download_link_states(edges);
    for(std::size_t i=0;i<state.size();++i)if(mesh.volumes[i].object>0)
        require(state[i].damage==1 && state[i].mode==CellMode::Spent,"Near-total shell damage failed terminal closure on GPU");
    int checked=0;for(std::size_t i=0;i<mesh.links.size();++i) {
        const auto& e=mesh.links[i];if(e.b<0 || mesh.volumes[e.a].object==0 || mesh.volumes[e.a].object!=mesh.volumes[e.b].object)continue;
        require(edges[i].phase==EdgePhase::Destroyed,"Terminal shell retained intact thermal links");++checked;
    }
    require(checked>0,"No shell bulk links tested");
}
void stiff_feedback() {
    auto s=fixture(3,5e6);
    // Relative accuracy of a MICROSCOPIC hole requires a finer absolute budget
    // than the user-selected .5% of the WHOLE element. Keep both existing
    // relative convergence/work assertions, requesting that refinement here.
    for(auto& c:s.gas_channels) {
        c.discharge_coefficient=.7;c.opening.damage_onset=.05;c.opening.exponent=1.;
        c.opening.pressure_damage_rate_per_s=.01;c.opening.fracture_energy_j_per_m2=100.;
    }
    const auto default_budget=evolve(s,.5,.05);
    double whole_area=0.;for(const auto& c:s.gas_channels)whole_area+=gas_channel_area(s,c);
    s.gas_model.max_fraction_per_step=2e-4;
    const auto coarse=evolve(s,.5,.05);
    s.gas_model.max_fraction_per_step*=.5;
    for(auto& c:s.gas_channels)c.opening.max_damage_step*=.5;
    const auto fine=evolve(s,.5,.025);
    double a=0.,b=0.;for(const auto& c:coarse.channels)a+=c.effective_area_m2;
    for(const auto& c:fine.channels)b+=c.effective_area_m2;
    require(a>0 && std::abs(a-b)/b<.05,"Stiff aperture feedback does not converge");
    require(std::abs(coarse.fracture_work_j-fine.fracture_work_j)/fine.fracture_work_j<.05,"Stiff opening work does not converge");
    double default_area=0.;for(const auto& c:default_budget.channels)default_area+=c.effective_area_m2;
    require(std::abs(default_area-b)<=.005*whole_area,"Default opening error exceeds whole-element budget");
    std::cout<<"default whole-element area error="<<std::abs(default_area-b)/whole_area<<'\n';
    std::cout<<"stiff feedback: coarse/fine area="<<a<<'/'<<b<<" m2\n";
}
int main(){try{laws();validation();damage_and_geometry();flow_and_quench();thermal_damage();unload_and_capacity();terminal_connectivity();pressure_feedback();stiff_feedback();
    std::cout<<"progressive whole shell: geometry, pressure/thermal damage, weak seams, refinement, gas flux and conservation pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

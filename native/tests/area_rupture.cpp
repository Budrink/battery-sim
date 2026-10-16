#include "battery_native/simulation.hpp"
#include "battery_native/shell_opening.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool ok,const char* msg) {if(!ok)throw std::runtime_error(msg);}
Scenario fixture(double cd=1e-20,double pressure=200000.) {
    Scenario s;s.gas_model.split_transport=true;s.gas_model.reference_temperature_k=300.;
    CellConfig c;c.id="B";c.dimensions_m={.02,.02,.02};c.thermal_divisions={3,3,3};
    c.heat_capacity_j_per_k=100.;c.initial_temperature_k=300.;s.cells={c};
    ShellConfig sh;sh.contents_id="B";sh.material_id="test";sh.thickness_m=.00015;
    sh.density_kg_m3=1600.;sh.specific_heat_j_kg_k=1000.;sh.contents_contact_h=1.;sh.seam_contact_h=1.;
    sh.surface_divisions={3,3,3};add_physical_shell(s,sh);
    GasCompartment g;g.id="inside";g.owner_cell=0;g.volume_m3=1e-6;g.initial_temperature_k=300.;
    g.initial_mass_kg=pressure*g.volume_m3/(287.*300.);s.gas_compartments={g};
    GasChannel ch;ch.id="rupture";ch.from=0;ch.to=-1;ch.shell_object=6;ch.port_id="B:case:+z:outer";
    ch.rupture=true;ch.outside_pressure_pa=100000.;ch.opening_pressure_pa=50000.;ch.discharge_coefficient=cd;
    ch.opening.enabled=ch.opening.stochastic_area=true;ch.opening.rupture_seed=17;
    ch.opening.pixel_rate_per_s=20.;ch.opening.fragment_mean_area_m2=1e-6;
    s.gas_channels={ch};return s;
}
struct Result {ProcessDiagnostics diagnostics;std::vector<ShellDamageState> shell;};
Result run(const Scenario& s,double end,double cap,int tile=128,bool retry=false) {
    auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh,tile);RuntimeParameters r;double time=0.;
    while(time<end-1e-14) {
        const double dt=solver.prepare_step(time,std::min(cap,end-time),.4,r);
        if(retry) {
            solver.discard_step();solver.prepare_step(time,std::min(cap/3,end-time),.4,r);solver.discard_step();
            require(dt==solver.prepare_step(time,std::min(cap,end-time),.4,r),"Retry changed event time");
        }
        solver.commit_step();time+=dt;
    }
    Result out;out.diagnostics=solver.process_diagnostics();solver.download_shell_states(out.shell);
    require(std::abs(out.diagnostics.thermal_residual_j)<1e-7 && std::abs(out.diagnostics.mass_residual_kg)<1e-13,"Area fracture breaks conservation");
    for(auto d:out.shell)require(d.bulk>=0 && d.bulk<=1 && d.bulk==d.seam,"Invalid area fraction");
    return out;
}
void same(const Result& a,const Result& b) {
    for(std::size_t k=0;k<a.shell.size();++k) {
        require(std::abs(a.shell[k].bulk-b.shell[k].bulk)<1e-12,"Area depends on CUDA tiling/retry/dt at fixed pressure");
        require(a.shell[k].rupture_events==b.shell[k].rupture_events,"Event count depends on CUDA tiling/retry/dt");
    }
}
void distributions() {
    double sum=0,area=0,product=0;const int n=100000;
    for(int k=0;k<n;++k) {
        const double a=area_event_exponential(123,k,0),b=area_event_exponential(123,k,1);
        require(a>0 && b>0 && std::isfinite(a+b),"Invalid event marks");sum+=a;area+=b;product+=a*b;
    }
    require(std::abs(sum/n-1)<.02 && std::abs(area/n-1)<.02 && std::abs(product/n-sum/n*area/n)<.02,"Waiting and size streams are not independent exponential marks");
    require(area_event_rate(0,1,.1,0)==0,"Subthreshold rate nonzero");
    require(area_event_rate(2,1,.1,.8)<area_event_rate(2,1,.1,.2),"Opening does not reduce event probability");
    require(area_event_rate(2,1,.1,1)==0,"Fully open surface still fails");
    for(int cells:{6,18,72}) {
        double mean=0.;
        for(int seed=0;seed<64;++seed)for(int k=0;k<cells;++k) {
            const double patch=.01/cells,fragment=1e-5;double t=0.,a=0.;std::uint64_t event=0,key=pixel_mix(seed)^pixel_mix(k+1000);
            while(a<1.) {
                t+=area_event_exponential(key,event,0)/area_event_rate(1.,patch,fragment,a);
                if(t>.3)break;
                a+=area_event_increment(a,patch,fragment,key,event++);
            }
            mean+=a/cells/64;
        }
        require(std::abs(mean-(-std::expm1(-.3)))<.015,"Refining surface changes expected opened physical area");
        std::cout<<"ensemble cells="<<cells<<" mean opening="<<mean<<'\n';
    }
}
void cuda_checks() {
    auto s=fixture();const auto a=run(s,.01,.002,64),b=run(s,.01,.001,256,true);same(a,b);
    const auto fraction=a.diagnostics.channels[0].opening_fraction;
    require(fraction>0 && fraction<1 && std::abs(9*fraction-std::round(9*fraction))>1e-5,"Still opens whole pixels");
    auto other=s;other.gas_channels[0].opening.rupture_seed=18;
    require(run(other,.01,.002).diagnostics.channels[0].opening_fraction!=fraction,"Seed has no effect");
    auto cold=fixture(.7,140000.);require(!run(cold,.01,.002).diagnostics.channels[0].open,"Subthreshold opening");
    auto costly=s;costly.gas_channels[0].opening.fracture_energy_j_per_m2=1e15;
    require(!run(costly,.005,.001).diagnostics.channels[0].open,"Unaffordable fragment opened");
    auto vent=fixture(.7);vent.gas_channels[0].opening.fracture_energy_j_per_m2=100.;
    const auto x=run(vent,.03,.001),later=run(vent,.06,.001),fine=run(vent,.03,.0005);
    same(x,later);same(x,fine);
    require(x.diagnostics.gas[0].pressure_pa<150000.,"Pressure did not unload");
    require(std::abs(x.diagnostics.channels[0].fracture_work_j-100*gas_channel_area(vent,vent.gas_channels[0])*x.diagnostics.channels[0].opening_fraction)<1e-12,"Wrong incremental fracture work");
    auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh);RuntimeParameters r;
    const double first=solver.step(0,.01,.4,r);std::vector<ShellDamageState> before,after;solver.download_shell_states(before);
    solver.reset();require(solver.step(0,.01,.4,r)==first,"Reset changed first event time");solver.download_shell_states(after);
    for(std::size_t k=0;k<before.size();++k)require(before[k].bulk==after[k].bulk && before[k].rupture_events==after[k].rupture_events,"Reset changed event marks");
    std::cout<<"CUDA: partial area="<<fraction<<"; vent area="<<x.diagnostics.channels[0].opening_fraction<<"; ledgers/dt/retries/reset pass\n";
}
void local_channels() {
    auto s=fixture();
    auto a=s.gas_compartments[0];a.thermal_cell=0;a.thermal_volume=18;a.volume_m3=1e-8;
    a.initial_mass_kg=200000.*a.volume_m3/(287.*300.);
    auto b=a;b.id="low-pressure-core";b.owner_cell=-1;b.thermal_volume=26;b.initial_mass_kg*=.7;
    auto out=a;out.id="nearby";out.owner_cell=out.thermal_cell=-1;out.volume_m3=.001;
    out.initial_mass_kg=100000.*out.volume_m3/(287.*300.);
    s.gas_compartments={a,b,out};
    auto first=s.gas_channels[0];first.to=2;first.shell_thermal_volume=0;
    auto last=first;last.id="second-patch";last.from=1;last.shell_thermal_volume=8;
    s.gas_channels={first,last};
    const auto x=run(s,.01,.001,64),y=run(s,.01,.0005,256,true);same(x,y);
    require(x.diagnostics.channels[0].opening_fraction>0 && x.diagnostics.channels[1].opening_fraction==0,"Local pressures were averaged over the shell face");
    auto mesh=build_thermal_mesh(s);const int begin=mesh.object_offsets[first.shell_object];
    for(int i=1;i<9;++i)require(x.shell[begin+i].bulk==0,"Rupture modified another patch");
    const double full=s.cells[first.shell_object].dimensions_m[0]*s.cells[first.shell_object].dimensions_m[1];
    require(std::abs(gas_channel_area(s,first)*9-full)<1e-15,"Local aperture counted the entire panel");
    s.gas_channels[0].opening.initial_mechanical_damage=.2;
    Simulation initial(s);const auto d=initial.process_diagnostics();
    require(std::abs(d.channels[0].opening_fraction-.2)<1e-12 && d.channels[1].opening_fraction==0,"Initial damage leaked across local channels");
    bool rejected=false;s.gas_channels[1].shell_thermal_volume=0;
    try{build_thermal_mesh(s);}catch(const std::exception&){rejected=true;}
    require(rejected,"Overlapping shell channels accepted");
    s.gas_channels={first,last};s.gas_channels[1].shell_thermal_volume=8;
    GasContact contact;contact.port_id=first.port_id;contact.compartment=2;contact.h_w_per_m2_k=100.;
    s.gas_contacts={contact};auto whole=build_thermal_mesh(s);double whole_g=0;
    for(const auto& p:whole.gas_patches)whole_g+=p.conductance_w_per_k;
    s.gas_contacts.clear();
    for(int k=0;k<9;++k){contact.thermal_volume=k;s.gas_contacts.push_back(contact);}
    auto split=build_thermal_mesh(s);double split_g=0;
    for(const auto& p:split.gas_patches)split_g+=p.conductance_w_per_k;
    require(std::abs(whole_g-split_g)<1e-15,"Local gas contacts duplicated or lost thermal area");
}
void free_passages() {
    auto s=fixture();s.gas_channels.clear();s.gas_compartments.clear();
    for(int i=0;i<4;++i) {
        GasCompartment g;g.id="cavity-"+std::to_string(i);g.volume_m3=1e-5;g.initial_temperature_k=300.;
        g.initial_mass_kg=(i?1e5:2e5)*g.volume_m3/(287.*300.);s.gas_compartments.push_back(g);
    }
    for(int i=0;i<2;++i) {
        GasChannel c;c.id="passage-"+std::to_string(i);c.from=i;c.to=i+1;c.passage=true;c.area_m2=1e-8;
        s.gas_channels.push_back(c);
    }
    Simulation sim(s);sim.step(.0001);auto d=sim.process_diagnostics();
    require(d.channels[0].transferred_kg>0 && d.channels[1].transferred_kg==0,"Gas skipped its adjacent cavity in one transport transaction");
    sim.step(.0001);d=sim.process_diagnostics();
    require(d.channels[1].transferred_kg>0,"Adjacent cavity did not feed its next neighbour");
    require(d.gas[3].mass_kg==s.gas_compartments[3].initial_mass_kg,"Isolated cavity filled through an obstacle");
    require(std::abs(d.mass_residual_kg)<1e-13 && std::abs(d.thermal_residual_j)<1e-7,"Passage breaks conservation");
    s.gas_compartments[0].owner_cell=0;bool rejected=false;
    try{build_thermal_mesh(s);}catch(const std::exception&){rejected=true;}
    require(rejected,"Free-space passage bypassed the battery casing");
}
void fixed_fractions() {
    for(double fraction:{.02,.05}) {
        auto s=fixture();s.gas_channels[0].opening.fixed_area_fraction=fraction;
        s.gas_channels[0].opening.fracture_energy_j_per_m2=100.;
        const auto a=run(s,.01,.002,64),b=run(s,.01,.001,256,true);same(a,b);
        bool opened=false;
        for(auto d:a.shell) {
            require(std::abs(d.bulk/fraction-std::round(d.bulk/fraction))<1e-10,"Opening is not a multiple of the fixed fraction");
            require(std::abs(d.bulk-fmin(1.,d.rupture_events*fraction))<1e-12,"Fixed increment depends on remaining area");
            opened|=d.bulk>0;
        }
        require(opened,"No fixed-fraction event occurred");
        require(std::abs(a.diagnostics.channels[0].fracture_work_j-100.*gas_channel_area(s,s.gas_channels[0])*a.diagnostics.channels[0].opening_fraction)<1e-10,"Wrong fixed-fragment work");
        require(fixed_area_event_rate(2.,fraction,.8)<fixed_area_event_rate(2.,fraction,.2),"Opening did not reduce the hazard");
        require(std::abs(fixed_area_event_rate(2.,fraction,.3)*fraction-1.4)<1e-12,"Changing fixed fraction changed mean aperture drift");
        require(std::abs(fixed_area_event_increment(.99,fraction)-.01)<1e-12,"Last fragment overshoots available area");
    }
}
int main() {try {distributions();cuda_checks();local_channels();free_passages();fixed_fractions();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

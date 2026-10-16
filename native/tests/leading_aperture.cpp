#include "battery_native/simulation.hpp"
#include "battery_native/deterministic_aperture.hpp"
#include <iostream>
#include <stdexcept>
using namespace battery;
bool cpu_backend=false;
bool implicit_backend=false;
bool mc_backend=false;
bool batched_backend=false;
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
Scenario fixture(double cd=1e-20,double pressure=200000.) {
    Scenario s;s.gas_model.split_transport=true;s.gas_model.reference_temperature_k=300.;
    s.gas_model.cpu_local_transport=cpu_backend;
    s.gas_model.cuda_kawasaki_transport=mc_backend;
    s.gas_model.cuda_batched_transport=batched_backend;
    s.gas_model.kawasaki_packet_fraction=.0078125;
    s.gas_model.implicit_local_transport=implicit_backend;
    s.gas_model.local_exchange_interval_s=.001;
    CellConfig c;c.id="B";c.dimensions_m={.02,.02,.02};c.thermal_divisions={3,3,3};
    c.heat_capacity_j_per_k=100.;c.initial_temperature_k=300.;s.cells={c};
    ShellConfig sh;sh.contents_id="B";sh.material_id="test";sh.thickness_m=.00015;
    sh.density_kg_m3=1600.;sh.specific_heat_j_kg_k=1000.;sh.contents_contact_h=1.;sh.seam_contact_h=1.;
    sh.surface_divisions={3,3,3};add_physical_shell(s,sh);
    GasCompartment g;g.id="inside";g.owner_cell=0;g.volume_m3=1e-6;g.initial_temperature_k=300.;
    g.initial_mass_kg=pressure*g.volume_m3/(287.*300.);s.gas_compartments={g};
    GasChannel ch;ch.id="rupture";ch.from=0;ch.to=-1;ch.shell_object=6;ch.port_id="B:case:+z:outer";
    ch.shell_thermal_volume=4;ch.rupture=true;ch.outside_pressure_pa=100000.;ch.opening_pressure_pa=50000.;
    ch.discharge_coefficient=cd;ch.opening.enabled=ch.opening.deterministic_aperture=true;ch.opening.area_rate_per_s=2.;
    if(mc_backend) {
        GasCompartment receiver;receiver.id="chamber";receiver.volume_m3=.001;
        receiver.initial_temperature_k=300;receiver.initial_mass_kg=1e5*.001/(287*300);
        s.gas_compartments.push_back(receiver);ch.to=1;
    }
    s.gas_channels={ch};return s;
}
ProcessDiagnostics run(const Scenario& s,double end,double cap,int tile=128,bool retries=false) {
    const auto mesh=build_thermal_mesh(s);CudaSolver solver(s,mesh,tile);RuntimeParameters r;double t=0.;
    while(t<end-1e-14) {
        const double dt=solver.prepare_step(t,std::min(cap,end-t),.4,r);
        if(retries) {
            const auto untouched=solver.process_diagnostics();
            solver.discard_step();solver.prepare_step(t,std::min(cap/3,end-t),.4,r);solver.discard_step();
            const auto again=solver.process_diagnostics();
            require(untouched.channels[0].opening_fraction==again.channels[0].opening_fraction && untouched.gas[0].energy_j==again.gas[0].energy_j,"prepare leaked prefix state");
            require(dt==solver.prepare_step(t,std::min(cap,end-t),.4,r),"retry changed deterministic timestep");
        }
        solver.commit_step();t+=dt;
    }
    const auto d=solver.process_diagnostics();std::vector<ShellDamageState> states;solver.download_shell_states(states);
    const int changed=mesh.object_offsets[s.gas_channels[0].shell_object]+s.gas_channels[0].shell_thermal_volume;
    for(std::size_t i=0;i<states.size();++i)if(static_cast<int>(i)!=changed)require(states[i].bulk==0,"aperture escaped its patch");
    require(std::abs(d.thermal_residual_j)<1e-7 && std::abs(d.mass_residual_kg)<1e-13,"leading fracture violates balance");
    return d;
}
void geometry() {
    require(std::abs(clipped_disk_area(2,2,0,0,1)-std::acos(-1.))<1e-12,"disk area");
    for(double a:{0.,.01,.5,.9,1.})for(double u:{0.,.4}) {
        const double r=clipped_disk_radius(2,1,u,.1,a);
        require(std::abs(clipped_disk_area(2,1,u,.1,r)/2-a)<1e-12,"clipped radius/area inverse");
    }
    for(auto p:{opening_rectangle(-.5,0,1,2,2,2,.1),opening_rectangle(-.5,-.5,1,1,2,2,.1)}) {
        for(double a:{0.,.02,.2,.7,1.}) {
            double b[4];opened_strips(p,a,b);
            const double area=p.width*p.height-std::max(0.,p.width-b[0]-b[1])*std::max(0.,p.height-b[2]-b[3]);
            require(std::abs(area/(p.width*p.height)-a)<1e-12,"seam rectangle area");
            for(int k=0;k<4;++k)if(p.bands[k]==0)require(b[k]==0,"rectangle spawned on a non-seam edge");
        }
    }
    ShellOpeningLaw law;law.area_rate_per_s=2.;
    const double a=deterministic_fraction(law,0,100000,50000,.2,.3,.5);
    double b=0;for(int i=0;i<100;++i)b=deterministic_fraction(law,b,100000,50000,.2,.3,.005);
    require(std::abs(a-b)<1e-12,"weak stage crossing depends on dt");
    require(deterministic_fraction(law,.2,30000,50000,.2,.3,1.)==.2,"bulk opens below its own threshold");
}
int main(int argc,char** argv){try {
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        cpu_backend|=arg=="--cpu-local";implicit_backend|=arg=="--implicit";mc_backend|=arg=="--cuda-kawasaki";
        batched_backend|=arg=="--cuda-batched";mc_backend|=batched_backend;
    }
    geometry();auto s=fixture();const auto a=run(s,.05,.01,64),b=run(s,.05,.003,256,true);
    require(std::abs(a.channels[0].opening_fraction-(-std::expm1(-.1)))<1e-10,"frozen pressure analytic area");
    require(std::abs(a.channels[0].opening_fraction-b.channels[0].opening_fraction)<1e-12,"dt/tile/retry mismatch");
    auto shut=fixture(.7,140000.);require(!run(shut,.01,.001).channels[0].open,"subthreshold opening");
    auto seam=fixture(1e-20,130000.);seam.gas_channels[0].shell_thermal_volume=0;
    seam.gas_channels[0].opening.seam_width_m=.0005;seam.gas_channels[0].opening.seam_strength_factor=.3;
    const auto sd=run(seam,2.,.02);const auto& panel=seam.cells[6];
    const double w=panel.dimensions_m[0]/3,h=panel.dimensions_m[1]/3;
    const double fraction=1-(w-.0005)*(h-.0005)/(w*h);
    require(std::abs(sd.channels[0].opening_fraction-fraction)<1e-10,"seam must fill before stronger plate opens");
    auto valve=fixture(1e-20,130000.);valve.gas_channels[0].opening.valve_radius_m=.001;
    valve.gas_channels[0].opening.valve_pressure_factor=.3;
    const auto vd=run(valve,2.,.02);
    require(std::abs(vd.channels[0].opening_fraction-std::acos(-1.)*1e-6/(w*h))<1e-10,"valve stage");
    auto vent=fixture(.7);vent.gas_channels[0].opening.fracture_energy_j_per_m2=100.;
    auto mesh=build_thermal_mesh(vent);CudaSolver solver(vent,mesh);RuntimeParameters r;
    solver.step(0,1e-5,.4,r);auto first=solver.process_diagnostics();
    require(first.channels[0].fracture_work_j>0 && first.channels[0].opening_fraction>0,"opening/work did not precede mass permission");
    if(!mc_backend)require(first.channels[0].transferred_kg>0,"opening did not precede first mass transfer");
    else {
        // An open path permits an event but cannot guarantee one in a tiny dt.
        double t=1e-5;
        for(int i=0;i<100 && solver.process_diagnostics().channels[0].transferred_kg==0;++i)
            t+=solver.step(t,1e-5,.4,r);
        require(solver.process_diagnostics().channels[0].transferred_kg>0,"Opened MC path never transported mass");
    }
    const auto x=run(vent,.005,1e-5),y=run(vent,.005,5e-6);
    require(std::abs(x.channels[0].opening_fraction-y.channels[0].opening_fraction)<2e-4,"blowdown area refinement");
    // Tiny pressure imbalance at high absolute pressure: capped mass is below
    // the ordinary inventory budget, but pressure feedback still matters.
    auto near=fixture(.7,3e7);GasCompartment receiver=near.gas_compartments[0];
    if(mc_backend)near.gas_compartments.resize(1);
    receiver.id="receiver";receiver.owner_cell=-1;receiver.initial_mass_kg*=29900000./3e7;
    near.gas_compartments.push_back(receiver);near.gas_channels[0].to=1;
    near.gas_channels[0].opening.area_rate_per_s=20000.;
    near.gas_model.max_fraction_per_step=.001; // pressure-feedback accuracy
    const auto na=run(near,1e-4,1e-4),nb=run(near,1e-4,1e-7);
    const double relative=std::abs(na.channels[0].opening_fraction-nb.channels[0].opening_fraction)/nb.channels[0].opening_fraction;
    std::cout<<"near-equilibrium aperture coarse="<<na.channels[0].opening_fraction<<" fine="<<nb.channels[0].opening_fraction<<" relative="<<relative<<'\n';
    require(relative<.01,"mass cap hid pressure/aperture time coupling");
    auto costly=fixture();costly.gas_channels[0].opening.fracture_energy_j_per_m2=1e12;
    const auto limited=run(costly,1e-4,1e-4);
    require(limited.channels[0].opening_fraction<1e-6 && limited.gas[0].temperature_k>=1.-1e-9,"fracture energy affordability");
    std::cout<<"leading aperture: geometry, seam/valve priority, analytic time law, first-step transfer, balances, tiling, rollback and refinement pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

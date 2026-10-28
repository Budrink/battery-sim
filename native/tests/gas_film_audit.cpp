#include "battery_native/simulation.hpp"
#include "battery_native/gas_thermal_contact.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
using J=nlohmann::json;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
J two_capacity(const CellConfig& material,double conductance,double gas_capacity,double duration,double cap) {
    Scenario s;s.settings={duration,cap,duration,.4};s.settings.minimum_dt_s=.005;
    s.gas_model.split_transport=s.gas_model.cuda_kawasaki_transport=s.gas_model.cuda_batched_transport=true;
    s.gas_model.cuda_network_transport=s.gas_model.coupled_gas_films=true;
    s.gas_model.coupled_transport_interval_s=.01;s.gas_model.reference_temperature_k=300.;
    CellConfig c;c.id=material.id;c.is_battery=false;c.dimensions_m=material.dimensions_m;
    c.heat_capacity_j_per_k=material.heat_capacity_j_per_k;c.conductivity_w_per_m_k=material.conductivity_w_per_m_k;
    c.thermal_divisions=material.thermal_divisions;c.initial_temperature_k=300.;s.cells.push_back(c);
    const double area=c.dimensions_m[1]*c.dimensions_m[2];
    s.ports.push_back({"surface",c.id,0,"contact","+x",area,{c.dimensions_m[0]/2,0,0},{1,0,0},
        {c.dimensions_m[1],c.dimensions_m[2]}});
    GasCompartment g;g.id="gas";g.initial_mass_kg=gas_capacity/s.gas_model.cv_j_per_kg_k;
    g.initial_temperature_k=700.;g.volume_m3=g.initial_mass_kg*287.*700./1e5;s.gas_compartments.push_back(g);
    s.gas_contacts.push_back({"surface",0,conductance/area});
    Simulation sim(s);double heat=0.,residual=0.;
    const double energy0=sim.cells()[0].stored_energy_j;
    while(!sim.finished()) {
        sim.step();for(double q:sim.last_gas_film_heat_j())heat+=q;
        const auto d=sim.process_diagnostics();residual=std::max(residual,std::abs(d.thermal_residual_j));
        require(d.gas[0].temperature_k>=300.-1e-10 && d.gas[0].temperature_k<=700.+1e-10,"Gas violated thermal maximum principle");
        require(sim.cells()[0].temperature_k>=300.-1e-10 && sim.cells()[0].temperature_k<=700.+1e-10,"Solid violated thermal maximum principle");
    }
    const auto d=sim.process_diagnostics();const double cs=c.heat_capacity_j_per_k;
    const double expected=300.+400.*gas_capacity/(gas_capacity+cs)*(-std::expm1(-conductance*(1./gas_capacity+1./cs)*duration));
    const double error=std::abs(sim.cells()[0].temperature_k-expected);
    require(error<.02,"Film disagrees with finite two-capacity analytical solution");
    require(residual<1e-7 && std::abs(heat-(sim.cells()[0].stored_energy_j-energy0))<1e-7,"Per-patch heat lost in scatter/gather");
    return {{"material",c.id},{"solid_capacity_j_k",cs},{"gas_capacity_j_k",gas_capacity},{"conductance_w_k",conductance},
        {"duration_s",duration},{"max_dt_s",cap},{"temperature_k",sim.cells()[0].temperature_k},{"analytic_temperature_k",expected},
        {"error_k",error},{"heat_to_solid_j",heat},{"thermal_residual_j",residual}};
}
int main(int argc,char** argv){try {
    if(argc!=2)throw std::runtime_error("Provide original full-pad scenario for material/contact inspection");
    const auto s=load_scenario(argv[1]);const auto mesh=build_thermal_mesh(s);
    // Explicit gas-facing reverse response is independent of bulk material S.
    // Deliberately unequal source laws ensure no forward/reverse averaging.
    auto reverse=s;auto& contact=reverse.gas_contacts.at(0);
    contact.has_solid_surface_connectivity=true;
    contact.solid_surface_connectivity={};contact.solid_surface_connectivity.linear={1.,1.,1.};
    contact.gas_surface_connectivity={};
    const auto changed=build_thermal_mesh(reverse);
    const auto p=std::find_if(changed.gas_patches.begin(),changed.gas_patches.end(),[&](const auto& q){
        return q.compartment==contact.compartment;});
    require(p!=changed.gas_patches.end(),"Reverse boundary fixture has no patch");
    require(p->law.endpoint_b.linear.x==1.,"Explicit reverse boundary lost in mesher");
    const double forward=gas_contact_conductance(*p,398.15,298.15,0.);
    const double backward=gas_contact_conductance(*p,298.15,398.15,0.);
    require(backward>forward,"Reverse boundary ignored or averaged with gas source");
    for(const auto& e:mesh.links)if(e.law.kind!=TransitionKind::DirectedGeometry)
        throw std::runtime_error("UPV heat bypasses directed connectivity: "+s.cells[mesh.volumes[e.a].object].id);
    for(const auto& p:mesh.gas_patches)require(p.law.kind==TransitionKind::DirectedGeometry,"UPV gas heat bypasses directed connectivity");
    J report={{"contacts",J::object()},{"tests",J::array()}};
    for(const auto& p:mesh.gas_patches) {
        const auto& id=s.gas_compartments[p.compartment].id;if(id.rfind("gap-ISS1-",0)!=0 && id.rfind("gap-TIM1-",0)!=0)continue;
        const auto& object=s.cells[mesh.volumes[p.volume].object];auto& r=report["contacts"][object.id];
        if(r.is_null())r=J::object();
        r["conductance_w_k"]=r.value("conductance_w_k",0.)+p.conductance_w_per_k;
        r["solid_capacity_j_k"]=object.heat_capacity_j_per_k;r["material"]=object.material_id;
    }
    const double conductance=report["contacts"].at("C3:case:-x").at("conductance_w_k");
    for(const auto* name:{"C3:case:-x","ISS1","TIM1"}) {
        const auto it=std::find_if(s.cells.begin(),s.cells.end(),[&](const auto& c){return c.id==name;});
        require(it!=s.cells.end(),"Missing real material");
        const auto g=std::find_if(s.gas_compartments.begin(),s.gas_compartments.end(),[](const auto& c){return c.id=="gap-ISS1-0";});
        require(g!=s.gas_compartments.end(),"Missing gap gas");
        const double cg=8.*g->initial_mass_kg*s.gas_model.cv_j_per_kg_k;
        report["tests"].push_back(two_capacity(*it,conductance,cg,1.,.05));
        report["tests"].push_back(two_capacity(*it,conductance,10.,20.,.05));
        report["tests"].push_back(two_capacity(*it,conductance,10.,20.,.005));
    }
    std::cout<<report.dump(2)<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

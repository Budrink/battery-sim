#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool b,const char* message) {if(!b)throw std::runtime_error(message);}
HeatCapacityLaw law() {
    HeatCapacityLaw l;l.factor.count=3;
    l.factor.temperature_k[0]=298.15;l.factor.temperature_k[1]=318.15;l.factor.temperature_k[2]=343.15;
    l.factor.value[0]=1.;l.factor.value[1]=1.2;l.factor.value[2]=.8;
    return l;
}
Scenario fixture() {
    Scenario s;s.name="Independent energy and transport";s.settings={.1,.001,.01,.2};
    CellConfig c;c.id="cell";c.dimensions_m={.011,.087,.187};c.thermal_divisions={3,3,3};
    c.conductivity_w_per_m_k={.9,28.3,28.3};c.heat_capacity_j_per_k=412.87;
    c.initial_temperature_field_k.assign(27,300);c.initial_temperature_field_k[13]=350;
    c.heat_capacity_law=law();s.cells.push_back(c);return s;
}
int main() {try {
    auto l=law();l.damage_factor_at_one=1.4;
    for(double d:{0.,.4,1.})for(double t:{250.,298.15,300.,318.15,330.,343.15,500.}) {
        const double u=thermal_energy(l,7.,t,d);
        require(std::abs(thermal_temperature(l,7.,u,d)-t)<1e-10,"U(T,D) inversion failed");
        const double derivative=(thermal_energy(l,7.,t+1e-4,d)-thermal_energy(l,7.,t-1e-4,d))/2e-4;
        require(std::abs(derivative-7.*capacity_factor(l,t,d))<1e-5,"dU/dT does not equal node capacity");
    }
    auto s=fixture();Simulation sim(s);auto before=sim.thermal_states();
    const auto& mesh=sim.thermal_mesh();const auto edges=sim.thermal_link_states();
    auto expected=before;sim.step();
    for(std::size_t k=0;k<mesh.links.size();++k) {
        const auto& e=mesh.links[k];const double q=sim.last_dt_s()*edges[k].conductance_w_per_k*(before[e.b].temperature_k-before[e.a].temperature_k);
        expected[e.a].stored_energy_j+=q;expected[e.b].stored_energy_j-=q;
    }
    for(std::size_t i=0;i<expected.size();++i) {
        const double t=thermal_temperature(s.cells[0].heat_capacity_law,mesh.volumes[i].capacity_j_per_k,expected[i].stored_energy_j,0.);
        require(std::abs(t-sim.thermal_states()[i].temperature_k)<1e-10,"CPU/CUDA variable-capacity one-step mismatch");
    }
    sim.run_to_end();require(std::abs(sim.energy_balance_residual_j())<1e-8,"Variable capacity creates energy");
    for(const auto& v:sim.thermal_states())require(v.temperature_k>=300. && v.temperature_k<=350.,"Variable-C maximum principle failed");
    auto fixed=s;fixed.cells[0].heat_capacity_law={};Simulation constant(fixed);
    auto neutral=s;for(int i=0;i<3;++i)neutral.cells[0].heat_capacity_law.factor.value[i]=1.;Simulation unit(neutral);
    constant.run_to_end();unit.run_to_end();
    for(std::size_t i=0;i<before.size();++i)
        require(std::abs(constant.thermal_states()[i].temperature_k-unit.thermal_states()[i].temperature_k)<1e-9,"Unit capacity law changes legacy physics");
    for(std::size_t i=0;i<edges.size();++i)require(edges[i].conductance_w_per_k==constant.thermal_link_states()[i].conductance_w_per_k,"Capacity changed edge law");
    auto isolated=s;auto& c=isolated.cells[0];c.is_battery=false;c.thermal_divisions={1,1,1};c.initial_temperature_field_k={330};
    c.heat_capacity_law=l;c.damage_onset_k=200.;c.damage_rate_at_onset_per_s=.01;c.damage_temperature_scale_k=100.;
    Simulation aging(isolated);aging.run_to_end();require(aging.cells()[0].damage>0,"Defect test did not evolve");
    require(std::abs(aging.energy_balance_residual_j())<1e-9,"Defect constitutive change creates energy");
    c.is_battery=true;c.runaway_energy_j=10.;isolated.primary_failures.push_back({0,"cell",0.,-1});
    Simulation ignition(isolated);require(std::abs(ignition.energy_balance_residual_j())<1e-9,"Ignition silently creates constitutive energy");
    ignition.run_to_end();require(std::abs(ignition.energy_balance_residual_j())<1e-8,"Source and node energy double counted");
    // Tabulated response uses the donor; damage/contrast remain separate.
    SurfaceConnectivity response;response.temperature_log_response=l.factor;
    require(std::abs(directed_log_response(response,0,330,300,0,0)-table_value(l.factor,330))<1e-14,"Wrong tabulated donor response");
    auto transport=fixed;auto& b=transport.cells[0];b.has_bulk_connectivity=b.has_bulk_response=true;
    b.bulk_connectivity.kind=TransitionKind::DirectedGeometry;b.bulk_connectivity.background_conductance_w_per_k=.2;
    b.bulk_connectivity.active_conductance_w_per_k=2.8;b.bulk_connectivity.geometry_reference_logit=std::log(.8/2.);
    b.bulk_response.temperature_log_response=l.factor;
    Simulation table(transport);auto initial=table.thermal_states();auto predicted=initial;const auto& tm=table.thermal_mesh();
    table.step();
    for(std::size_t k=0;k<tm.links.size();++k) {
        const auto& e=tm.links[k];const auto state=evaluate_directed_geometry(e.law,e.initial,e.geometry_axis,
            initial[e.a].temperature_k,initial[e.b].temperature_k,0,0,0,e.weight,e.series_resistance_k_per_w);
        const double q=state.conductance_w_per_k*(initial[e.b].temperature_k-initial[e.a].temperature_k)*table.last_dt_s();
        predicted[e.a].temperature_k+=q/tm.volumes[e.a].capacity_j_per_k;
        predicted[e.b].temperature_k-=q/tm.volumes[e.b].capacity_j_per_k;
    }
    for(std::size_t i=0;i<predicted.size();++i)require(std::abs(predicted[i].temperature_k-table.thermal_states()[i].temperature_k)<1e-10,"CPU/CUDA table flux mismatch");
    table.run_to_end();require(std::abs(table.energy_balance_residual_j())<1e-8,"Tabulated transport creates energy");
    // Different tables must not collapse to the same deduplicated GPU law.
    auto paired=transport;auto other=paired.cells[0];other.id="other";other.position_m.x=1.;
    other.bulk_response.temperature_log_response.value[1]=-.5;
    other.bulk_response.temperature_log_response.value[2]=-.5;paired.cells.push_back(other);
    Simulation distinct(paired);distinct.step();
    require(std::abs(distinct.thermal_states()[13].temperature_k-distinct.thermal_states()[40].temperature_k)>1e-6,"GPU law cache ignored table values");
    // Cached stability uses the minimum allowed capacity, not its reference.
    auto stiff=fixture();stiff.settings={10.,10.,10.,.2};stiff.cells[0].heat_capacity_law.factor.value[2]=.05;
    Simulation bounded(stiff);const auto& bm=bounded.thermal_mesh();auto be=bounded.thermal_link_states();double bound=1e300;
    for(std::size_t i=0;i<bm.volumes.size();++i) {
        double sum=0.;for(int j=bm.adjacency_offsets[i];j<bm.adjacency_offsets[i+1];++j)sum+=be[bm.adjacency_links[j]].conductance_w_per_k;
        if(sum>0)bound=std::min(bound,.2*.05*bm.volumes[i].capacity_j_per_k/sum);
    }
    bounded.step();require(bounded.last_dt_s()<=bound*(1.+1e-12),"CFL ignored the lower capacity bound");
    std::cout<<"Constitutive separation, inversion, CPU/CUDA flux, fixed-C limit, defects and sources PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

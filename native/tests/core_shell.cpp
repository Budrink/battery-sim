#include "battery_native/simulation.hpp"
#include "battery_native/state_geometry.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
Scenario fixture(int n) {
    Scenario s;s.name="Core/shell partition";s.settings={.1,.001,.1,.2};
    CellConfig c;c.id="battery";c.dimensions_m={.011,.087,.187};c.thermal_divisions={n,n,n};
    c.heat_capacity_j_per_k=412.87;c.conductivity_w_per_m_k={.9,28.3,28.3};
    for(int z=0;z<n;++z)for(int y=0;y<n;++y)for(int x=0;x<n;++x)
        c.initial_temperature_field_k.push_back(volume_region(x,y,z,c.thermal_divisions)==ThermalRegion::Core?400:300);
    s.cells.push_back(c);return s;
}
int main() {try {
    for(int n:{3,4}) {
        Simulation sim(fixture(n));const auto& mesh=sim.thermal_mesh();
        int core=0,shell=0,cc=0,cs=0,ss=0;double capacity=0,core_capacity=0;
        for(const auto& v:mesh.volumes) {
            const bool inner=v.region==ThermalRegion::Core;core+=inner;shell+=!inner;
            capacity+=v.capacity_j_per_k;if(inner)core_capacity+=v.capacity_j_per_k;
            if(inner)require(std::abs(v.local_center_m.x)<.011/2-v.size_m.x/2 &&
                             std::abs(v.local_center_m.y)<.087/2-v.size_m.y/2 &&
                             std::abs(v.local_center_m.z)<.187/2-v.size_m.z/2,"Core touches exterior face");
        }
        require(core==(n-2)*(n-2)*(n-2) && shell==n*n*n-core,"Wrong core/shell counts");
        require(std::abs(capacity-412.87)<1e-9,"Region tags changed total capacity");
        require(std::abs(core_capacity/capacity-double(core)/(n*n*n))<1e-12,"Wrong region capacity fraction");
        for(const auto& e:mesh.links) {
            cc+=e.region==ThermalLinkRegion::CoreCore;cs+=e.region==ThermalLinkRegion::CoreShell;ss+=e.region==ThermalLinkRegion::ShellShell;
            if(e.region==ThermalLinkRegion::CoreShell)require(mesh.volumes[e.a].region!=mesh.volumes[e.b].region,"Core-shell link endpoints mismatch");
        }
        require(cc==(n==3?0:12) && cs==(n==3?6:24) && ss==(n==3?48:108),"Wrong region edge counts");
        sim.run_to_end();require(std::abs(sim.energy_balance_residual_j())<1e-8,"Core/shell exchange violates energy balance");
        double core_t=0,shell_t=0;
        for(std::size_t i=0;i<mesh.volumes.size();++i) {
            const double t=sim.thermal_states()[i].temperature_k;
            require(t>=300 && t<=400,"Maximum principle failed");
            if(mesh.volumes[i].region==ThermalRegion::Core)core_t+=t;else shell_t+=t;
        }
        require(core_t/core<400 && shell_t/shell>300,"Core and shell do not exchange heat");
    }
    auto s=fixture(3);s.cells[0].position_m.x=-.0055;
    auto other=s.cells[0];other.id="pad";other.is_battery=false;other.position_m.x=.0055;
    other.thermal_divisions={1,2,4};other.initial_temperature_field_k.clear();s.cells.push_back(other);
    const double area=.087*.187;
    s.ports={{"a","battery",0,"contact","+x",area,{0,0,0},{1,0,0}},
             {"b","pad",1,"contact","-x",area,{0,0,0},{-1,0,0}}};
    InterfaceConfig e;e.id="contact";e.cell_b=1;e.port_a="a";e.port_b="b";e.law.background_conductance_w_per_k=1.;
    s.interfaces.push_back(e);Simulation contact(s);double sum=0;
    for(int i:contact.thermal_mesh().interface_links) {
        const auto& l=contact.thermal_mesh().links[i];sum+=l.weight;
        require(l.region==ThermalLinkRegion::MaterialContact,"External contact misclassified");
        require(contact.thermal_mesh().volumes[l.a].region==ThermalRegion::Shell && contact.thermal_mesh().volumes[l.b].region==ThermalRegion::Shell,"External contact attached to core");
    }
    require(std::abs(sum-1)<1e-12,"Nonconforming 3x3 to 2x4 face lost area");
    contact.run_to_end();require(std::abs(contact.energy_balance_residual_j())<1e-8,"Nonconforming core-shell energy balance");
    // An independent internal law must not inherit the fitted surface response.
    auto internal=fixture(3);auto& c=internal.cells[0];
    c.has_bulk_connectivity=c.has_bulk_response=true;
    c.bulk_connectivity.kind=TransitionKind::DirectedGeometry;
    c.bulk_connectivity.l0_m=1e-4;c.bulk_connectivity.geometry_reference_logit=std::log(.8/2.);
    c.bulk_connectivity.background_conductance_w_per_k=.2;c.bulk_connectivity.active_conductance_w_per_k=2.8;
    c.bulk_response.linear={-2,-2,-2};c.bulk_response.quadratic={1,1,1};
    c.surface_connectivity.linear={12,12,12};Simulation dynamic(internal);
    const auto start=dynamic.thermal_states();auto expected=start;const auto& dm=dynamic.thermal_mesh();
    const auto edges=dynamic.thermal_link_states();dynamic.step();const double dt=dynamic.last_dt_s();
    for(std::size_t index=0;index<dm.links.size();++index) {
        const auto& e=dm.links[index];const double ta=start[e.a].temperature_k,tb=start[e.b].temperature_k;
        const auto v=dm.volumes[e.a].size_m;const double h[]={v.x,v.y,v.z};const int k=e.geometry_axis;
        const double base=c.conductivity_w_per_m_k[k]*h[(k+1)%3]*h[(k+2)%3]/h[k];
        const double theta=(std::max(ta,tb)-298.15)/100.;
        const double factor=.2+2.8/(1.+std::exp(-(std::log(.8/2.)-2.*theta+theta*theta)));
        require(std::abs(edges[index].conductance_w_per_k-base*factor)<1e-12,"Bulk response not independent or donor temperature wrong");
        const double q=dt*base*factor*(ta-tb);
        expected[e.a].temperature_k-=q/dm.volumes[e.a].capacity_j_per_k;
        expected[e.b].temperature_k+=q/dm.volumes[e.b].capacity_j_per_k;
    }
    for(std::size_t i=0;i<start.size();++i)
        require(std::abs(dynamic.thermal_states()[i].temperature_k-expected[i].temperature_k)<1e-10,"Internal CUDA/CPU one-step mismatch");
    dynamic.run_to_end();require(std::abs(dynamic.energy_balance_residual_j())<1e-8,"Dynamic internal law creates heat");
    c.surface_connectivity.linear={-12,-12,-12};Simulation same(internal);same.run_to_end();
    for(std::size_t i=0;i<start.size();++i)
        require(std::abs(dynamic.thermal_states()[i].temperature_k-same.thermal_states()[i].temperature_k)<1e-12,"Surface parameters leaked to internal law");
    c.bulk_response={};Simulation neutral(internal);Simulation constant(fixture(3));neutral.run_to_end();constant.run_to_end();
    for(std::size_t i=0;i<start.size();++i)
        require(std::abs(neutral.thermal_states()[i].temperature_k-constant.thermal_states()[i].temperature_k)<1e-10,"Zero response does not reproduce constant internal k");
    std::cout<<"Core/shell: 3^3 and 4^3 counts, edge classes, capacities, contact area, conservative CUDA exchange PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

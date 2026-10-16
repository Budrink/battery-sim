#include "battery_native/simulation.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
int main(){try {
    Scenario s;s.settings={.1,.001,.1,.4};CellConfig c;c.id="B";c.dimensions_m={.01,.03,.05};
    c.heat_capacity_j_per_k=100;c.thermal_divisions={3,3,3};c.initial_temperature_k=400;c.conductivity_w_per_m_k={1,1,1};s.cells={c};
    ShellConfig sh;sh.contents_id="B";sh.material_id="case-material";sh.thickness_m=.001;
    sh.density_kg_m3=2000;sh.specific_heat_j_kg_k=800;sh.conductivity_w_per_m_k={2,3,4};
    sh.contents_contact_h=100;sh.seam_contact_h=500;add_physical_shell(s,sh);
    require(s.cells.size()==7 && s.interfaces.size()==18,"Shell does not have six panels and twelve seams");
    double capacity=0;for(std::size_t i=1;i<s.cells.size();++i) {
        auto& p=s.cells[i];capacity+=p.heat_capacity_j_per_k;p.initial_temperature_k=300;
        require(!p.is_battery && p.material_id=="case-material" && p.shell_contents_id=="B","Shell is not a distinct material");
    }
    double expected=(.012*.032*.052-.01*.03*.05)*2000*800;
    require(std::abs(capacity-expected)<1e-12,"Shell mass overlaps or leaves gaps");
    auto mesh=build_thermal_mesh(s);
    for(const auto& e:mesh.links)if(e.interface_index>=0) {
        int a=mesh.volumes[e.a].object,b=mesh.volumes[e.b].object;
        require(a!=b,"Physical shell contact became an internal core edge");
    }
    for(std::size_t j=0;j<s.interfaces.size();++j) {
        double w=0;for(int k=mesh.interface_offsets[j];k<mesh.interface_offsets[j+1];++k)w+=mesh.links[mesh.interface_links[k]].weight;
        require(std::abs(w-1)<1e-12,"Nonmatching shell contact loses area");
    }
    Simulation sim(s);sim.run_to_end();require(std::abs(sim.energy_balance_residual_j())<1e-8,"Shell transfer is not conservative");
    require(sim.cells()[0].temperature_k<400 && sim.cells()[1].temperature_k>300,"Shell does not heat independently");
    require(sim.cells()[1].temperature_k<sim.cells()[0].temperature_k,"Shell instantaneously merged with contents");
    auto duplicate=s;bool rejected=false;try{add_physical_shell(duplicate,sh);}catch(...){rejected=true;}
    require(rejected,"Duplicate shell accepted");
    std::cout<<"physical shell: volume, material, all face/seam contacts and CUDA energy balance pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

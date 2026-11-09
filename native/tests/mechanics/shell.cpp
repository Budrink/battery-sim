#include "battery_native/simulation/simulation.hpp"
#include "battery_native/thermal/state_geometry.hpp"
#include "battery_native/thermal/object_grid.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace battery;
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
void plate_checks() {
    Scenario s; s.settings={.1,.001,.1,.4};
    CellConfig core; core.id="B"; core.dimensions_m={.01,.03,.05};
    core.thermal_divisions={3,3,3}; core.heat_capacity_j_per_k=100;
    core.initial_temperature_k=400; s.cells={core};
    ShellConfig sh; sh.contents_id="B"; sh.material_id="plate";
    sh.thickness_m=.001; sh.density_kg_m3=2000; sh.specific_heat_j_kg_k=800;
    sh.contents_contact_h=100; sh.seam_contact_h=500; sh.surface_divisions={6,9,3};
    for(int i=0;i<6;++i) sh.plates[i]={.0001*(i+1),1000.+100*i,500.+50*i};
    add_physical_shell(s,sh);
    double sum_volume=0.;
    for(int i=0;i<6;++i) {
        auto& p=s.cells[i+1]; const int axis=i/2;
        require(p.thermal_divisions[axis]==1,"Plate has multiple thickness layers");
        require(p.dimensions_m[axis]==sh.plates[i].thickness_m,"Wrong per-face thickness");
        double v=1;for(auto d:p.dimensions_m)v*=d;sum_volume+=v;
        require(std::abs(p.heat_capacity_j_per_k-v*sh.plates[i].density_kg_m3*sh.plates[i].specific_heat_j_kg_k)<1e-12,
                "Plate capacity ignores its own material/volume");
        p.damage_onset_k=i==1?350.:600.;p.damage_rate_at_onset_per_s=.05;
        p.damage_temperature_scale_k=50.;
    }
    double outer=1,inner=1;for(int a=0;a<3;++a){inner*=core.dimensions_m[a];outer*=core.dimensions_m[a]+sh.plates[2*a].thickness_m+sh.plates[2*a+1].thickness_m;}
    require(std::abs(sum_volume-(outer-inner))<1e-15,"Unequal plate corners lose or duplicate material");
    for(size_t i=0;i<s.cells.size();++i)for(size_t j=0;j<i;++j){
        const auto&a=s.cells[i];const auto&b=s.cells[j];
        double pa[]{a.position_m.x,a.position_m.y,a.position_m.z},pb[]{b.position_m.x,b.position_m.y,b.position_m.z},v=1;
        for(int d=0;d<3;++d)v*=std::max(0.,std::min(pa[d]+a.dimensions_m[d]/2,pb[d]+b.dimensions_m[d]/2)-std::max(pa[d]-a.dimensions_m[d]/2,pb[d]-b.dimensions_m[d]/2));
        require(v<1e-16,"Physical plates overlap contents or one another");
    }
    const auto mesh=build_thermal_mesh(s);
    for(int i=1;i<7;++i) {
        const auto& panel=s.cells[i];const auto edges=object_grid(s,panel);
        const double origin[]{panel.position_m.x,panel.position_m.y,panel.position_m.z};
        for(int d=0;d<3;++d) if(d!=(i-1)/2) {
            const int factor=panel.thermal_divisions[d]/core.thermal_divisions[d];
            for(int k=1;k<core.thermal_divisions[d];++k)
                require(std::abs(origin[d]+edges[d][k*factor]-core.dimensions_m[d]*(double(k)/core.thermal_divisions[d]-.5))<1e-14,
                        "Shell cuts do not coincide with contents cuts");
        }
        double capacity=0.;
        for(int j=mesh.object_offsets[i];j<mesh.object_offsets[i+1];++j) {
            const auto& v=mesh.volumes[j];capacity+=v.capacity_j_per_k;
            require(std::abs(v.capacity_j_per_k-v.size_m.x*v.size_m.y*v.size_m.z*sh.plates[i-1].density_kg_m3*sh.plates[i-1].specific_heat_j_kg_k)<1e-12,
                    "Edge tile has wrong material capacity");
            int core_links=0;
            for(const auto& e:mesh.links)if(e.interface_index>=0 && ((e.a==j && mesh.volumes[e.b].object==0)||(e.b==j && mesh.volumes[e.a].object==0)))++core_links;
            require(core_links==1,"A casing tile bridges multiple contents volumes");
        }
        require(std::abs(capacity-panel.heat_capacity_j_per_k)<1e-12,"Plate subdivision changes total capacity");
    }
    for(size_t i=0;i<s.interfaces.size();++i){double w=0;for(int k=mesh.interface_offsets[i];k<mesh.interface_offsets[i+1];++k)w+=mesh.links[mesh.interface_links[k]].weight;
        require(std::abs(w-1)<1e-12,"Independent plate grid loses a contents/seam contact");}
    Simulation sim(s);sim.run_to_end();
    require(sim.cells()[2].damage>0,"Front plate did not accumulate its own thermal damage");
    for(int i=1;i<7;++i)if(i!=2)require(sim.cells()[i].damage==0,"Plate damage leaked into a different face");
    require(std::abs(sim.energy_balance_residual_j())<1e-8,"Unequal plates violate energy balance");
}
int main(){try {
    plate_checks();
    Scenario s;s.settings={.1,.001,.1,.4};CellConfig c;c.id="B";c.dimensions_m={.01,.03,.05};
    c.heat_capacity_j_per_k=100;c.thermal_divisions={3,3,3};c.initial_temperature_k=400;c.conductivity_w_per_m_k={1,1,1};s.cells={c};
    ShellConfig sh;sh.contents_id="B";sh.material_id="case-material";sh.thickness_m=.001;
    sh.density_kg_m3=2000;sh.specific_heat_j_kg_k=800;sh.conductivity_w_per_m_k={2,3,4};
    sh.contents_contact_h=100;sh.seam_contact_h=500;add_physical_shell(s,sh);
    require(s.cells.size()==7 && s.interfaces.size()==18,"Shell does not have six panels and twelve seams");
    for(const auto& e:s.interfaces) {
        require(e.law.kind==TransitionKind::DirectedGeometry,"Casing contact bypasses tensor geometry");
        require(e.initial_l_m==0. && e.law.l0_m==0.,"Casing contact retains an obsolete transport length");
        const auto state=evaluate_directed_geometry(e.law,{},0,400.,300.,0.,0.,0.,1.,0.);
        const double h=e.cell_a==0?sh.contents_contact_h:sh.seam_contact_h;
        const auto pa=std::find_if(s.ports.begin(),s.ports.end(),[&](const auto& p){return p.id==e.port_a;});
        require(std::abs(state.conductance_w_per_k-h*pa->area_m2)<1e-12,"Zero casing response changes reference h*A");
    }
    double capacity=0;for(std::size_t i=1;i<s.cells.size();++i) {
        auto& p=s.cells[i];capacity+=p.heat_capacity_j_per_k;p.initial_temperature_k=300;
        require(!p.is_battery && p.material_id=="case-material" && p.shell_contents_id=="B","Shell is not a distinct material");
    }
    double expected=(.012*.032*.052-.01*.03*.05)*2000*800;
    require(std::abs(capacity-expected)<1e-12,"Shell mass overlaps or leaves gaps");
    auto mesh=build_thermal_mesh(s);
    require(mesh.volumes.size()==27+54,"3 cubed contents must have 54 planar casing tiles");
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
    auto responsive=s;
    for(auto& cell:responsive.cells)cell.surface_connectivity.linear={3.,3.,3.};
    Simulation changed(responsive);changed.run_to_end();
    require(changed.cells()[0].temperature_k<sim.cells()[0].temperature_k-1e-5,
        "CUDA ignores state geometry on core/casing contacts");
    require(std::abs(changed.energy_balance_residual_j())<1e-8,"State-dependent casing contact creates heat");
    auto duplicate=s;bool rejected=false;try{add_physical_shell(duplicate,sh);}catch(...){rejected=true;}
    require(rejected,"Duplicate shell accepted");
    std::cout<<"physical shell: volume, material, all face/seam contacts and CUDA energy balance pass\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

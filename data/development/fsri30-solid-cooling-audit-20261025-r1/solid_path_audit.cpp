#include "battery_native/thermal_mesh.hpp"
#include "battery_native/gas_thermal_contact.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
using J=nlohmann::json;
int main(int argc,char**argv) {
    if(argc!=2 && argc!=3)throw std::runtime_error("scene path [report path]; complete observation JSON on stdin");
    const auto s=battery::load_scenario(argv[1]);const auto m=battery::build_thermal_mesh(s);
    J run;std::cin>>run;
    if(run.value("partial",true))throw std::runtime_error("Final fields require completed replay");
    const auto& fields=run.at("final_fields_k");
    std::vector<int> parent(s.cells.size());std::iota(parent.begin(),parent.end(),0);
    auto root=[&](int x){while(parent[x]!=x)x=parent[x];return x;};
    std::set<std::pair<int,int>> object_edges;
    for(const auto&e:m.links)if(e.b>=0) {
        int a=m.volumes[e.a].object,b=m.volumes[e.b].object;
        if(a!=b){parent[root(a)]=root(b);object_edges.emplace(std::min(a,b),std::max(a,b));}
    }
    std::map<int,std::vector<std::string>> groups;
    for(int i=0;i<(int)s.cells.size();++i)groups[root(i)].push_back(s.cells[i].id);
    J out;out["time_s"]=run.at("time_s").back();
    out["solid_components"]=J::array();for(const auto&[r,names]:groups)out["solid_components"].push_back(names);
    out["solid_component_count"]=groups.size();out["object_count"]=s.cells.size();
    out["cross_object_solid_edges"]=J::array();
    for(const auto&[a,b]:object_edges)out["cross_object_solid_edges"].push_back({s.cells[a].id,s.cells[b].id});
    std::map<std::string,double> cap;
    for(const auto&v:m.volumes)cap[s.cells[v.object].id]+=v.capacity_j_per_k;
    for(const auto&[id,c]:cap) {
        const auto&f=fields.at(id);double sum=0.,lo=1e300,hi=-1e300;
        for(const auto&q:f){double t=q;sum+=t;lo=std::min(lo,t);hi=std::max(hi,t);}
        out["objects"][id]={{"reference_capacity_j_k",c},{"mean_temperature_k",sum/f.size()},
            {"minimum_temperature_k",lo},{"maximum_temperature_k",hi}};
    }
    double inside_power=0.,inside_g=0.,outside_cold_power=0.,damage_difference=0.;int ordinary=0,unknown=0,dynamic=0;
    for(const auto&p:m.gas_patches) {
        if(p.channel>=0){++dynamic;continue;}++ordinary;
        const auto&v=m.volumes[p.volume];const auto&id=s.cells[v.object].id;
        const double ts=fields.at(id).at(p.volume-m.object_offsets[v.object]);
        const auto&gasid=s.gas_compartments[p.compartment].id;
        const bool known=run.at("gas").contains(gasid);
        const double tg=known?run.at("gas").at(gasid).at("temperature_k").back().get<double>():298.15;
        const double g=battery::gas_contact_conductance(p,tg,ts,0.);
        const double damaged_g=battery::gas_contact_conductance(p,tg,ts,1.);
        damage_difference=std::max(damage_difference,std::abs(g-damaged_g));
        auto&row=out[known?"observed_gas_contacts":"cold_ambient_upper_bound_NOT_actual_flux"][id];
        if(row.is_null())row={{"conductance_w_per_k",0.},{"gas_to_solid_power_w",0.},{"patch_count",0}};
        row["conductance_w_per_k"]=row["conductance_w_per_k"].get<double>()+g;
        row["gas_to_solid_power_w"]=row["gas_to_solid_power_w"].get<double>()+g*(tg-ts);
        row["patch_count"]=row["patch_count"].get<int>()+1;
        if(id.rfind("ABS:",0)==0) {
            if(known){inside_power+=g*(tg-ts);inside_g+=g;}
            else outside_cold_power+=g*(ts-tg);
        }
        if(!known)++unknown;
    }
    out["abs_interior_gas_to_wall_w"]=inside_power;out["abs_interior_conductance_w_per_k"]=inside_g;
    out["abs_exterior_cold_air_limit_w_NOT_actual"]=outside_cold_power;
    out["ordinary_gas_patches"]=ordinary;out["dynamic_aperture_patches_excluded"]=dynamic;
    out["patches_without_observed_gas_temperature"]=unknown;
    out["max_film_conductance_difference_D0_D1_w_k"]=damage_difference;
    out["limitations"]={"Static final observation audit, NOT restart or trajectory",
        "Solid damage not recorded; D=0 and D=1 film checks reported",
        "Outside local gas temperatures absent: cold-air flux is only an upper-bound counterfactual",
        "Dynamic aperture heat excluded; no synthetic final pressure/state reconstruction"};
    if(argc==3) {
        std::ofstream report(argv[2]);if(!report)throw std::runtime_error("Cannot create report");
        report<<out.dump(2)<<'\n';
        std::cout<<"Solid components="<<groups.size()<<", ABS internal film W="<<inside_power
            <<", ABS exterior cold-air limit (NOT actual) W="<<outside_cold_power<<'\n';
    } else std::cout<<out.dump(2)<<'\n';
}

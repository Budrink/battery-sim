#include "kawasaki_audit_support.hpp"
#include "battery_native/gas_experiment.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
namespace kawasaki_audit {
namespace gx=battery::gas_experiment;
double seconds(Timer::time_point start){return std::chrono::duration<double>(Timer::now()-start).count();}
kg::Node node(double p,double t,double v) {const double m=p*v/(287*t);return {m,1000*m*t,v,0,0,m*.2};}
kg::Network fixture(const std::string& kind) {
    kg::Network n;n.settings.tail_atol_pa=0;const int count=kind=="chain"?6:2;
    for(int i=0;i<count;++i)n.initial.push_back(node(i?1e5:(kind=="reservoir"?8e5:3e5),i?300:900,
        kind=="reservoir" && i==1?1e-3:1e-6));
    n.initial[0].chemical=1.;
    for(int i=1;i<count;++i)n.edges.push_back({i-1,i,2e-8,.7});kg::schedule(n);return n;
}
gx::Network reference_network(const kg::Network& n) {
    battery::Scenario s;s.gas_model.split_transport=true;s.gas_model.reference_temperature_k=300;
    s.gas_model.max_fraction_per_step=.001;s.gas_model.max_temperature_step_k=.01;
    s.gas_model.pressure_absolute_tolerance_pa=.001;s.gas_model.pressure_relative_tolerance=1e-8;
    battery::CellConfig cell;cell.id="inert";cell.is_battery=false;cell.runaway_energy_j=0;
    cell.dimensions_m={.1,.1,.1};cell.heat_capacity_j_per_k=1;cell.thermal_divisions={1,1,1};s.cells={cell};
    for(std::size_t i=0;i<n.initial.size();++i) {
        const auto& v=n.initial[i];battery::GasCompartment g;g.id="v"+std::to_string(i);
        g.initial_mass_kg=v.mass;g.volume_m3=v.volume;g.initial_temperature_k=kg::temperature(v,n.settings);
        g.initial_chemical_j=v.chemical;g.initial_oxidizer_mass_fraction=v.oxidizer/v.mass;s.gas_compartments.push_back(g);
    }
    for(std::size_t i=0;i<n.edges.size();++i) {
        const auto& e=n.edges[i];battery::GasChannel c;c.id="e"+std::to_string(i);c.passage=true;
        c.from=e.a;c.to=e.b;c.area_m2=e.area;c.discharge_coefficient=e.cd;s.gas_channels.push_back(c);
    }
    return gx::make_network(s);
}
Reference reference(const kg::Network& n,double duration,double step_cap) {
    // Independent deterministic fine-step oracle; no CPU Monte Carlo backend.
    const auto network=reference_network(n);gx::Cpu solver(network);Reference result;double t=0;
    const auto start=Timer::now();
    for(int sample=0;sample<=samples;++sample) {
        const double end=duration*sample/samples;
        while(t<end) {
            t+=solver.global_step(std::min(step_cap,end-t));
            if(++result.steps>5000000 || (result.steps%10000==0 && seconds(start)>45))throw std::runtime_error("Reference budget exhausted");
        }
        std::vector<kg::Node> state;
        for(const auto& g:solver.states())state.push_back({g.mass_kg,g.energy_j+g.mass_kg*1000*300,g.free_volume_m3,0,g.chemical_j,g.oxidizer_kg});
        result.samples.push_back(state);
    }
    result.wall=seconds(start);return result;
}
double crossing(const std::vector<double>& values,double threshold,double duration) {
    for(std::size_t i=1;i<values.size();++i)if(values[i]>=threshold && values[i-1]<threshold)
        return duration/samples*(i-1+(threshold-values[i-1])/(values[i]-values[i-1]));
    return -1;
}
J ensemble(kg::Network n,const Reference& ref,double duration,double fraction,int substeps,int replicas) {
    n.settings.packet_fraction=fraction;kg::Cuda solver(n,replicas,20261015);
    std::vector<kg::Node> state;std::vector<kg::Clock> clocks;
    J output=J::array();double max_t=0,max_p=0,max_m=0,max_se=0,wall=0;
    std::vector<double> front,ref_front;
    for(int sample=0;sample<=samples;++sample) {
        const auto start=Timer::now();
        if(sample)for(int k=0;k<substeps;++k)solver.advance(duration/(samples*substeps));
        solver.synchronize();wall+=seconds(start);solver.download(state,clocks);
        J row=J::array();
        for(std::size_t i=0;i<n.initial.size();++i) {
            double t=0,p=0,m=0,chemical=0,t2=0;
            for(int r=0;r<replicas;++r) {
                const auto& v=state[r*n.initial.size()+i];const double temp=kg::temperature(v,n.settings);
                t+=temp;t2+=temp*temp;p+=kg::pressure(v,n.settings);m+=v.mass;chemical+=v.chemical;
            }
            t/=replicas;p/=replicas;m/=replicas;chemical/=replicas;
            const double se=std::sqrt(std::max(0.,t2/replicas-t*t)/(replicas-1));
            const auto& v=ref.samples[sample][i];const double tref=kg::temperature(v,n.settings),pref=kg::pressure(v,n.settings);
            max_t=std::max(max_t,std::abs(t-tref));max_p=std::max(max_p,std::abs(p-pref)/pref);
            max_m=std::max(max_m,std::abs(m-v.mass)/v.mass);max_se=std::max(max_se,se);
            row.push_back(J{{"temperature_k",t},{"temperature_se_k",se},{"pressure_pa",p},{"mass_kg",m},{"chemical_j",chemical},
                {"reference_temperature_k",tref},{"reference_pressure_pa",pref},{"reference_mass_kg",v.mass},{"reference_chemical_j",v.chemical}});
            if(i==1){front.push_back(t);ref_front.push_back(tref);}
        }
        output.push_back(J{{"time_s",duration*sample/samples},{"nodes",row}});
    }
    double m0=0,u0=0,c0=0,o0=0;
    for(const auto& v:n.initial){m0+=v.mass;u0+=v.energy;c0+=v.chemical;o0+=v.oxidizer;}
    double mass_residual=0,energy_residual=0,chemical_residual=0,oxidizer_residual=0;
    for(int r=0;r<replicas;++r) {
        double m=0,u=0,c=0,o=0;for(std::size_t i=0;i<n.initial.size();++i){const auto& v=state[r*n.initial.size()+i];m+=v.mass;u+=v.energy;c+=v.chemical;o+=v.oxidizer;}
        mass_residual=std::max(mass_residual,std::abs(m-m0));energy_residual=std::max(energy_residual,std::abs(u-u0));
        chemical_residual=std::max(chemical_residual,std::abs(c-c0));oxidizer_residual=std::max(oxidizer_residual,std::abs(o-o0));
    }
    if(mass_residual>1e-11*m0 || energy_residual>1e-11*u0 || chemical_residual>1e-11 || oxidizer_residual>1e-11*o0)
        throw std::runtime_error("Ensemble conservation failure");
    std::uint64_t events=0,snaps=0,tails=0;double snapped_mass=0,tail_mass=0;
    for(const auto& c:clocks){events+=c.events;snaps+=c.snaps;snapped_mass+=c.snapped_mass;tails+=c.tails;tail_mass+=c.tail_mass;}
    return J{{"fraction",fraction},{"window_s",duration/(samples*substeps)},{"replicas",replicas},{"gpu_wall_s",wall},
        {"max_mean_temperature_error_k",max_t},{"max_temperature_standard_error_k",max_se},{"max_mean_pressure_relative_error",max_p},
        {"max_mean_mass_relative_error",max_m},{"node1_mean_305k_crossing_s",crossing(front,305,duration)},
        {"node1_reference_305k_crossing_s",crossing(ref_front,305,duration)},
        {"events",events},{"equilibrium_completions",snaps},{"completion_mass_per_replica_kg",snapped_mass/replicas},
        {"tail_atol_pa",n.settings.tail_atol_pa},{"tail_rtol",n.settings.tail_rtol},
        {"tail_updates",tails},{"tail_mass_per_replica_kg",tail_mass/replicas},
        {"max_mass_residual_kg",mass_residual},{"max_energy_residual_j",energy_residual},
        {"max_chemical_residual_j",chemical_residual},{"max_oxidizer_residual_kg",oxidizer_residual},{"samples",output}};
}
kg::Network grid(int side) {
    kg::Network n;n.settings.tail_atol_pa=0;auto id=[&](int x,int y,int z){return (x*side+y)*side+z;};
    for(int x=0;x<side;++x)for(int y=0;y<side;++y)for(int z=0;z<side;++z)n.initial.push_back(node(x<side/2?3e5:1e5,x<side/2?900:300));
    // Axis ordering gives six matchings on this nonperiodic Cartesian grid.
    for(int axis=0;axis<3;++axis)for(int x=0;x<side;++x)for(int y=0;y<side;++y)for(int z=0;z<side;++z) {
        int b[3]={x,y,z};if(++b[axis]<side)n.edges.push_back({id(x,y,z),id(b[0],b[1],b[2]),2e-8,.7});
    }
    kg::schedule(n);return n;
}
J benchmark() {
    auto n=grid(16);kg::Cuda gpu(n);constexpr int windows=16;constexpr double dt=.0005;
    gpu.synchronize();auto start=Timer::now();for(int i=0;i<windows;++i)gpu.advance(dt);gpu.synchronize();const double gpu_s=seconds(start);
    std::vector<kg::Node> state;std::vector<kg::Clock> clocks;gpu.download(state,clocks);
    std::uint64_t events=0,snaps=0;for(const auto& c:clocks){events+=c.events;snaps+=c.snaps;}
    return J{{"nodes",n.initial.size()},{"edges",n.edges.size()},{"colors",n.colors.size()},{"physical_duration_s",windows*dt},
        {"events",events},{"equilibrium_completions",snaps},
        {"cuda_wall_s",gpu_s},
        {"caveat","CUDA-only gas prototype, one realization. Not a full battery simulation or accuracy-matched speedup."}};
}
} // namespace kawasaki_audit
int main(int argc,char** argv) {try {
    using namespace kawasaki_audit;
    if(argc==4 && std::string(argv[1])=="--tail-sweep") {
        const int replicas=std::stoi(argv[3]);
        if(replicas<2 || replicas>1000000)throw std::invalid_argument("Invalid ensemble size");
        tail_sweep(argv[2],replicas);return 0;
    }
    if(argc<2 || argc>3)throw std::invalid_argument("Usage: battery_sim_kawasaki_gas_audit OUTPUT.json [REPLICAS], or --tail-sweep OUTPUT.json REPLICAS");
    const int replicas=argc==3?std::stoi(argv[2]):2048;
    if(replicas<2 || replicas>1000000)throw std::invalid_argument("Invalid ensemble size");
    J report={{"experimental",true},{"production_backend_changed",false},{"time_scale_factor",1.},
        {"seed_base",20261015},{"replicas",replicas},{"cv_j_kg_k",1000},{"R_j_kg_k",287},
        {"equilibrium_atol_pa",1e-6},{"equilibrium_rtol",1e-10},{"cases",J::object()}};
    auto save=[&]{std::ofstream out(argv[1]);if(!out)throw std::runtime_error("Cannot write audit");out<<report.dump(2)<<'\n';};
    for(const std::string kind:{"pair","reservoir","chain"}) {
        const auto n=fixture(kind);const double duration=.12*n.initial[0].mass/kg::flow(n.initial[0],n.initial[1],n.edges[0],n.settings);
        std::cout<<kind<<" duration="<<duration<<" reference..."<<std::endl;
        const auto ref=reference(n,duration,duration/20000),fine=reference(n,duration,duration/40000);
        double ref_t=0,ref_p=0;for(int j=0;j<=samples;++j)for(std::size_t i=0;i<n.initial.size();++i) {
            ref_t=std::max(ref_t,std::abs(kg::temperature(ref.samples[j][i],n.settings)-kg::temperature(fine.samples[j][i],n.settings)));
            ref_p=std::max(ref_p,std::abs(kg::pressure(ref.samples[j][i],n.settings)-kg::pressure(fine.samples[j][i],n.settings)));
        }
        auto& c=report["cases"][kind];c={{"duration_s",duration},{"reference_wall_s",fine.wall},{"reference_steps",fine.steps},
            {"reference_refinement_temperature_k",ref_t},{"reference_refinement_pressure_pa",ref_p},{"runs",J::array()}};
        for(double fraction:{.5,.125,.03125,.0078125}) {
            auto result=ensemble(n,fine,duration,fraction,1,replicas);
            std::cout<<kind<<" fraction="<<fraction<<" max T error="<<result["max_mean_temperature_error_k"]
                <<" K, GPU="<<result["gpu_wall_s"]<<" s"<<std::endl;
            c["runs"].push_back(result);save();
        }
        c["runs"].push_back(ensemble(n,fine,duration,.0078125,4,replicas));save();
    }
    report["grid_benchmark"]=benchmark();save();std::cout<<report["grid_benchmark"].dump(2)<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

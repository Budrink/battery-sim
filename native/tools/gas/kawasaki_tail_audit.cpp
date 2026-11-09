#include "kawasaki_audit_support.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Host-side measurement of CUDA runs, not a CPU stochastic implementation.
namespace kawasaki_audit {

J grid_tail_run(double tolerance,int windows,int replicas) {
    auto n=grid(16);n.settings.tail_atol_pa=tolerance;
    constexpr double duration=.008;
    std::vector<double> timing;
    kg::Cuda measured(n);
    // Warm up separately; every timed repetition starts from the same state.
    measured.advance(duration/windows);measured.synchronize();
    for(int repeat=0;repeat<3;++repeat) {
        measured.reset();measured.synchronize();const auto start=Timer::now();
        for(int step=0;step<windows;++step)measured.advance(duration/windows);
        measured.synchronize();timing.push_back(seconds(start));
    }
    std::vector<kg::Node> state;std::vector<kg::Clock> clocks;
    measured.download(state,clocks);
    std::uint64_t events=0,tails=0;for(const auto& c:clocks){events+=c.events;tails+=c.tails;}
    auto ordered=timing;std::sort(ordered.begin(),ordered.end());
    J result={{"tail_atol_pa",tolerance},{"tail_rtol",n.settings.tail_rtol},
        {"windows",windows},{"physical_duration_s",duration},{"nodes",4096},{"edges",11520},{"colors",n.colors.size()},
        {"fraction",n.settings.packet_fraction},{"single_replica_wall_s",timing},{"median_wall_s",ordered[1]},
        {"single_replica_events",events},{"single_replica_tail_updates",tails},{"trace_replicas",replicas},
        {"samples",J::array()}};

    kg::Cuda trace(n,replicas,20261015);
    double m0=0,u0=0,o0=0,s0=0;
    for(const auto& v:n.initial){m0+=v.mass;u0+=v.energy;o0+=v.oxidizer;s0+=kg::entropy(v,n.settings);}
    double mr=0,ur=0,orr=0,min_entropy=0;
    // Always sample at the same physical times, including window refinements.
    for(int sample=0;sample<=16;++sample) {
        if(sample)for(int k=0;k<windows/16;++k)trace.advance(duration/windows);
        trace.download(state,clocks);J planes=J::array();
        for(int x=0;x<16;++x) {
            double t=0,p=0,t2=0,p2=0;
            for(int r=0;r<replicas;++r) {
                double rt=0,rp=0;
                for(int i=0;i<256;++i){const auto& v=state[r*4096+x*256+i];rt+=kg::temperature(v,n.settings);rp+=kg::pressure(v,n.settings);}
                rt/=256;rp/=256;t+=rt;p+=rp;t2+=rt*rt;p2+=rp*rp;
            }
            t/=replicas;p/=replicas;
            planes.push_back(J{{"temperature_k",t},{"pressure_pa",p},
                {"temperature_se_k",std::sqrt(std::max(0.,t2/replicas-t*t)/(replicas-1))},
                {"pressure_se_pa",std::sqrt(std::max(0.,p2/replicas-p*p)/(replicas-1))}});
        }
        result["samples"].push_back(J{{"time_s",sample*duration/16},{"planes",planes}});
        for(int r=0;r<replicas;++r) {
            double m=0,u=0,o=0,s=0;
            for(int i=0;i<4096;++i) {
                const auto& v=state[r*4096+i];
                if(!(v.mass>0 && v.energy>0 && v.oxidizer>=0 && v.oxidizer<=v.mass))throw std::runtime_error("Grid positivity failure");
                m+=v.mass;u+=v.energy;o+=v.oxidizer;s+=kg::entropy(v,n.settings);
            }
            mr=std::max(mr,std::abs(m-m0)/m0);ur=std::max(ur,std::abs(u-u0)/u0);orr=std::max(orr,std::abs(o-o0)/o0);
            min_entropy=std::min(min_entropy,s-s0);
        }
    }
    if(mr>1e-11 || ur>1e-11 || orr>1e-11 || min_entropy< -1e-10*std::abs(s0))throw std::runtime_error("Grid ledger failure");
    result["max_mass_relative_residual"]=mr;result["max_energy_relative_residual"]=ur;
    result["max_oxidizer_relative_residual"]=orr;result["min_entropy_change_j_k"]=min_entropy;
    return result;
}

void compare_grid(J& run,const J& reference) {
    double max_t=0,max_p=0,max_se=0;
    for(std::size_t k=0;k<run["samples"].size();++k)for(int x=0;x<16;++x) {
        const auto& a=run["samples"][k]["planes"][x];const auto& b=reference["samples"][k]["planes"][x];
        max_t=std::max(max_t,std::abs(a["temperature_k"].get<double>()-b["temperature_k"].get<double>()));
        max_p=std::max(max_p,std::abs(a["pressure_pa"].get<double>()-b["pressure_pa"].get<double>()));
        max_se=std::max(max_se,a["temperature_se_k"].get<double>());
    }
    run["max_plane_mean_temperature_difference_k"]=max_t;
    run["max_plane_mean_pressure_difference_pa"]=max_p;
    run["max_plane_temperature_se_k"]=max_se;
    run["speedup_over_event_only"]=reference["median_wall_s"].get<double>()/run["median_wall_s"].get<double>();
}

void tail_sweep(const char* path,int replicas) {
    J report={{"experimental",true},{"production_backend_changed",false},{"time_scale_factor",1.},
        {"seed_base",20261015},{"replicas",replicas},{"tail_rtol",.001},
        {"caveat","Tail is deterministic, time-limited sqrt(dp) approximation, not instantaneous pressure equalization. Grid errors compare plane ensemble means with event-only MC, not deterministic truth."},
        {"cases",J::object()},{"grid",J::array()}};
    auto save=[&]{std::ofstream out(path);if(!out)throw std::runtime_error("Cannot write tail audit");out<<report.dump(2)<<'\n';};
    const std::vector<double> tolerances={0,.01,.1,1,10,100,1000};
    for(const std::string kind:{"pair","reservoir","chain"}) {
        auto n=fixture(kind);const double duration=.12*n.initial[0].mass/kg::flow(n.initial[0],n.initial[1],n.edges[0],n.settings);
        const auto ref=reference(n,duration,duration/40000);
        auto& c=report["cases"][kind];c=J::array();
        for(double tolerance:tolerances) {
            n.settings.tail_atol_pa=tolerance;
            c.push_back(ensemble(n,ref,duration,.0078125,1,replicas));save();
            std::cout<<kind<<" tail="<<tolerance<<" max mean T error="<<c.back()["max_mean_temperature_error_k"]
                <<" events="<<c.back()["events"]<<std::endl;
        }
        for(double tolerance:{1.,10.,100.}) {
            n.settings.tail_atol_pa=tolerance;c.push_back(ensemble(n,ref,duration,.0078125,4,replicas));save();
        }
    }
    for(double tolerance:tolerances) {
        auto run=grid_tail_run(tolerance,16,8);
        compare_grid(run,report["grid"].empty()?run:report["grid"][0]);
        std::cout<<"grid tail="<<tolerance<<" wall="<<run["median_wall_s"]<<" speedup="<<run["speedup_over_event_only"]
            <<" mean T difference="<<run["max_plane_mean_temperature_difference_k"]<<std::endl;
        report["grid"].push_back(run);save();
    }
    for(double tolerance:{0.,1.,10.,100.}) {
        auto run=grid_tail_run(tolerance,64,8);compare_grid(run,report["grid"][0]);
        report["grid"].push_back(run);save();
        std::cout<<"grid refined tail="<<tolerance<<" wall="<<run["median_wall_s"]<<std::endl;
    }
}
} // namespace kawasaki_audit

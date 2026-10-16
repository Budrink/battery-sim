#include "battery_native/local_gas_transport.hpp"
#include "battery_native/mixture_equilibrium.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace battery {
namespace {
constexpr int max_nodes=9,max_edges=64,max_unknowns=max_nodes+max_edges;
using Vector=std::array<double,max_unknowns>;
struct Matrix {
    std::array<double,max_unknowns*max_unknowns> values;
    int size=0;
    void reset(int n) {size=n;std::fill_n(values.data(),n*n,0.);}
    double* operator[](int row) {return values.data()+row*size;}
};
struct Packet {double mass=0,energy=0,chemical=0,oxidizer=0;};
struct State {
    std::array<GasState,max_nodes> gas;
    std::array<ThermalState,max_nodes> solid;
};
struct Result {State state;std::array<Packet,max_edges> packets{};};
struct NodeLaw {const CellConfig* material=nullptr;double capacity=0,fraction=0;};
struct EdgeLaw {int original,a,b;double area_cd,outside_pressure;};

// Small dense solve, partial pivoting. No external library, global Jacobian,
// or matrix containing thermal contacts. A failed solve rejects the time step.
bool linear_solve(Matrix& a,Vector b,int n,Vector& x) {
    for(int j=0;j<n;++j) {
        int pivot=j;for(int i=j+1;i<n;++i)if(std::abs(a[i][j])>std::abs(a[pivot][j]))pivot=i;
        if(!std::isfinite(a[pivot][j]) || std::abs(a[pivot][j])<1e-26)return false;
        if(pivot!=j){for(int k=0;k<n;++k)std::swap(a[pivot][k],a[j][k]);std::swap(b[pivot],b[j]);}
        for(int i=j+1;i<n;++i) {
            const double factor=a[i][j]/a[j][j];
            for(int k=j+1;k<n;++k)a[i][k]-=factor*a[j][k];
            b[i]-=factor*b[j];
        }
    }
    for(int i=n-1;i>=0;--i) {
        double v=b[i];for(int j=i+1;j<n;++j)v-=a[i][j]*x[j];
        x[i]=v/a[i][i];if(!std::isfinite(x[i]))return false;
    }
    return true;
}
double norm(const Vector& a,int n) {double result=0;for(int i=0;i<n;++i)result=std::max(result,std::abs(a[i]));return result;}

struct Problem {
    const GasModel& model;
    int nodes=0,edges=0;
    std::array<NodeLaw,max_nodes> laws;
    std::array<EdgeLaw,max_edges> paths;
    GasTransportStats& stats;

    double solid_energy(int i,const ThermalState& s,double t) const {
        const auto& law=laws[i];if(!law.material)return 0.;
        const auto& m=*law.material;
        if(m.conversion.enabled) {
            const double c=conversion_capacity(m.conversion,s,law.capacity,law.fraction);
            return c*t+(law.capacity-c)*model.reference_temperature_k;
        }
        return thermal_energy(m.heat_capacity_law,law.capacity,t,s.damage);
    }
    double solid_capacity(int i,const ThermalState& s) const {
        const auto& l=laws[i];if(!l.material)return 0.;
        const auto& m=*l.material;
        return m.conversion.enabled?conversion_capacity(m.conversion,s,l.capacity,l.fraction):
            l.capacity*capacity_factor(m.heat_capacity_law,s.temperature_k,s.damage);
    }
    void closure(int i,State& state) const {
        auto& g=state.gas[i];auto& s=state.solid[i];const auto& l=laws[i];
        if(l.material) {
            const double total=s.stored_energy_j+g.energy_j;
            const auto& m=*l.material;
            g.temperature_k=s.temperature_k=mixture_temperature(m.heat_capacity_law,m.conversion,s,g,l.capacity,l.fraction,model);
            g.energy_j=g.mass_kg*model.cv_j_per_kg_k*(g.temperature_k-model.reference_temperature_k);
            s.stored_energy_j=total-g.energy_j;s.mobile_capacity_j_per_k=g.mass_kg*model.cv_j_per_kg_k;
        } else g.temperature_k=model.reference_temperature_k+g.energy_j/(g.mass_kg*model.cv_j_per_kg_k);
        g.pressure_pa=g.mass_kg*model.specific_r_j_per_kg_k*g.temperature_k/g.free_volume_m3;
    }

    // Unknowns are signed transferred masses and final node temperatures.
    // Mass conservation is eliminated analytically, rather than imposed with
    // a Newton tolerance. Squaring the signed orifice law removes its sqrt(dp)
    // singularity at equilibrium without smoothing/changing the physical law.
    bool backward_euler(const State& old,double dt,Result& result) const {
        const int n=edges+nodes;const double cv=model.cv_j_per_kg_k,r=model.specific_r_j_per_kg_k;
        std::array<double,max_nodes> energy_scale{},temp_scale{},pressure{},energy0{};
        std::array<double,max_edges> mass_scale{},flow_scale{};
        Vector x{};
        for(int i=0;i<nodes;++i) {
            const auto& g=old.gas[i];temp_scale[i]=std::max(1.,g.temperature_k);
            energy_scale[i]=(g.mass_kg*cv+solid_capacity(i,old.solid[i]))*temp_scale[i];
            energy0[i]=g.energy_j+(laws[i].material?old.solid[i].stored_energy_j:0.);
            x[edges+i]=1.;pressure[i]=g.pressure_pa;
        }
        for(int j=0;j<edges;++j) {
            const auto& e=paths[j];const int b=e.b;const auto& a=old.gas[e.a];
            const double pscale=std::max({1.,pressure[e.a],b>=0?pressure[b]:e.outside_pressure});
            const double slope=r*a.temperature_k/a.free_volume_m3+
                (b>=0?r*old.gas[b].temperature_k/old.gas[b].free_volume_m3:0.);
            mass_scale[j]=pscale/slope;
            flow_scale[j]=2*pscale*pscale/(r*temp_scale[e.a]);
        }
        // A positivity-limited explicit predictor avoids the zero derivative
        // of q*|q| at q=0 for a very short step. It is only a Newton guess.
        std::array<double,max_nodes> outgoing{};
        for(int j=0;j<edges;++j) {
            const auto& e=paths[j];const double pb=e.b>=0?pressure[e.b]:e.outside_pressure;
            const bool forward=pressure[e.a]>=pb;const int donor=forward?e.a:e.b;
            if(donor<0)continue;
            const double proposed=dt*gas_orifice_flow(std::max(pressure[e.a],pb),std::min(pressure[e.a],pb),
                old.gas[donor].temperature_k,e.area_cd,1.,model);
            const double hottest=std::max(old.gas[e.a].temperature_k,e.b>=0?old.gas[e.b].temperature_k:0.);
            const double slope=(1+r/cv)*r*hottest*(1/old.gas[e.a].free_volume_m3+
                (e.b>=0?1/old.gas[e.b].free_volume_m3:0.));
            const double dm=(forward?1.:-1.)*std::min(proposed,std::abs(pressure[e.a]-pb)/slope);
            x[j]=dm/mass_scale[j];outgoing[donor]+=std::abs(dm);
        }
        for(int j=0;j<edges;++j) {
            const auto& e=paths[j];const int donor=x[j]>=0?e.a:e.b;
            if(donor>=0 && outgoing[donor]>0)x[j]*=std::min(1.,.1*old.gas[donor].mass_kg/outgoing[donor]);
        }
        const double gamma=1+r/cv,critical=std::pow(2/(gamma+1),gamma/(gamma-1));
        const double choked=gamma*std::pow(2/(gamma+1),(gamma+1)/(gamma-1));
        auto evaluate=[&](const Vector& z,Vector& f,Matrix* jac=nullptr)->bool {
            ++stats.residual_evaluations;
            std::array<double,max_nodes> mass{},t{},p{},energy{};
            std::array<Vector,max_nodes> dp;
            if(jac) {
                jac->reset(n);
                for(int i=0;i<nodes;++i)std::fill_n(dp[i].data(),n,0.);
            }
            for(int i=0;i<nodes;++i){mass[i]=old.gas[i].mass_kg;t[i]=z[edges+i]*temp_scale[i];}
            for(int j=0;j<edges;++j) {
                const auto& e=paths[j];const double dm=z[j]*mass_scale[j];
                mass[e.a]-=dm;if(e.b>=0)mass[e.b]+=dm;
            }
            for(int i=0;i<nodes;++i) {
                if(!(mass[i]>1e-30 && t[i]>0 && std::isfinite(mass[i]) && std::isfinite(t[i])))return false;
                p[i]=mass[i]*r*t[i]/old.gas[i].free_volume_m3;
                energy[i]=solid_energy(i,old.solid[i],t[i])+mass[i]*cv*(t[i]-model.reference_temperature_k)-energy0[i];
                if(jac) {
                    dp[i][edges+i]=mass[i]*r*temp_scale[i]/old.gas[i].free_volume_m3;
                    auto at=old.solid[i];at.temperature_k=t[i];
                    (*jac)[edges+i][edges+i]=(solid_capacity(i,at)+mass[i]*cv)*temp_scale[i];
                }
            }
            if(jac)for(int j=0;j<edges;++j) {
                const auto& e=paths[j];
                dp[e.a][j]-=mass_scale[j]*r*t[e.a]/old.gas[e.a].free_volume_m3;
                (*jac)[edges+e.a][j]-=mass_scale[j]*cv*(t[e.a]-model.reference_temperature_k);
                if(e.b>=0) {
                    dp[e.b][j]+=mass_scale[j]*r*t[e.b]/old.gas[e.b].free_volume_m3;
                    (*jac)[edges+e.b][j]+=mass_scale[j]*cv*(t[e.b]-model.reference_temperature_k);
                }
            }
            for(int j=0;j<edges;++j) {
                const auto& e=paths[j];const double pb=e.b>=0?p[e.b]:e.outside_pressure;
                const bool forward=p[e.a]>=pb;const int donor=forward?e.a:e.b;
                double q=0;
                if(donor>=0)q=(forward?1.:-1.)*gas_orifice_flow(std::max(p[e.a],pb),std::min(p[e.a],pb),t[donor],1.,1.,model);
                const double dm=z[j]*mass_scale[j],flux=dm/(dt*e.area_cd);
                f[j]=(flux*std::abs(flux)-q*std::abs(q))/flow_scale[j];
                const int advector=dm>=0?e.a:e.b;
                // Negative external flux is only an intermediate Newton trial;
                // no reservoir supplies heat/species to the accepted solution.
                const double td=t[advector>=0?advector:e.a];
                const double de=dm*((cv+r)*td-cv*model.reference_temperature_k);
                energy[e.a]+=de;if(e.b>=0)energy[e.b]-=de;
                if(jac) {
                    // Analytic derivative of signed squared compressible-orifice
                    // flow; no finite differencing across 2N+E state vectors.
                    if(donor>=0) {
                        const double up=std::max(p[e.a],pb),down=std::min(p[e.a],pb),ratio=std::max(0.,down/up);
                        double f2=choked,derivative=0;
                        if(ratio>critical) {
                            const double a=2/gamma,b=(gamma+1)/gamma,k=2*gamma/(gamma-1);
                            f2=k*(std::pow(ratio,a)-std::pow(ratio,b));
                            derivative=k*(a*std::pow(ratio,a-1)-b*std::pow(ratio,b-1));
                        }
                        const double du=up/(r*t[donor])*(2*f2-ratio*derivative);
                        const double dd=up/(r*t[donor])*derivative;
                        const double da=forward?du:-dd,db=forward?dd:-du;
                        for(int k=0;k<n;++k)(*jac)[j][k]=-(da*dp[e.a][k]+(e.b>=0?db*dp[e.b][k]:0))/flow_scale[j];
                        (*jac)[j][edges+donor]+=q*std::abs(q)/t[donor]*temp_scale[donor]/flow_scale[j];
                    }
                    // Newton-only diagonal at zero flux resolves cycle null
                    // directions. The residual always uses the unmodified law.
                    const double eps=1e-7*(1e-4+std::abs(z[j]));
                    const double coefficient=mass_scale[j]/(dt*e.area_cd);
                    (*jac)[j][j]+=std::max(2*std::abs(z[j]),eps)*coefficient*coefficient/flow_scale[j];
                    const double enthalpy=(cv+r)*td-cv*model.reference_temperature_k;
                    const int adv=advector>=0?advector:e.a;
                    (*jac)[edges+e.a][j]+=mass_scale[j]*enthalpy;
                    (*jac)[edges+e.a][edges+adv]+=dm*(cv+r)*temp_scale[adv];
                    if(e.b>=0) {
                        (*jac)[edges+e.b][j]-=mass_scale[j]*enthalpy;
                        (*jac)[edges+e.b][edges+adv]-=dm*(cv+r)*temp_scale[adv];
                    }
                }
            }
            for(int i=0;i<nodes;++i) {
                f[edges+i]=energy[i]/energy_scale[i];
                if(jac)for(int j=0;j<n;++j)(*jac)[edges+i][j]/=energy_scale[i];
            }
            for(int i=0;i<n;++i)if(!std::isfinite(f[i]))return false;
            return true;
        };
        Vector f{};if(!evaluate(x,f))return false;
        bool converged=false;
        for(int iteration=0;iteration<32;++iteration) {
            if(norm(f,n)<2e-11){converged=true;break;}
            ++stats.newton_iterations;Matrix jac;
            if(!evaluate(x,f,&jac))return false;
            Vector rhs{},delta{};for(int i=0;i<n;++i)rhs[i]=-f[i];
            if(!linear_solve(jac,rhs,n,delta))return false;
            bool accepted=false;
            for(double alpha=1.;alpha>=1./65536;alpha*=.5) {
                Vector trial=x,ft{};for(int i=0;i<n;++i)trial[i]+=alpha*delta[i];
                if(evaluate(trial,ft) && norm(ft,n)<norm(f,n)) {x=trial;f=ft;accepted=true;break;}
            }
            if(!accepted)return false;
        }
        if(!converged && norm(f,n)>=2e-11)return false;
        // Implicit upwind species transport: M-matrix with nonnegative stocks.
        // The same signed mass packets transport fuel and oxidizer. This also
        // handles gas traversing multiple edges inside a group during one step.
        Matrix tracer;tracer.reset(nodes);
        Vector chemical{},oxidizer{},chem_fraction{},ox_fraction{};
        std::array<double,max_nodes> mass{};
        for(int i=0;i<nodes;++i)mass[i]=old.gas[i].mass_kg;
        for(int j=0;j<edges;++j) {
            const auto& e=paths[j];const double dm=x[j]*mass_scale[j];
            if(e.b<0 && dm<0)return false;
            mass[e.a]-=dm;if(e.b>=0)mass[e.b]+=dm;
        }
        for(int i=0;i<nodes;++i){tracer[i][i]=mass[i];chemical[i]=old.gas[i].chemical_j;oxidizer[i]=old.gas[i].oxidizer_kg;}
        for(int j=0;j<edges;++j) {
            const auto& e=paths[j];const double dm=x[j]*mass_scale[j];
            const int donor=dm>=0?e.a:e.b,receiver=dm>=0?e.b:e.a;
            tracer[donor][donor]+=std::abs(dm);if(receiver>=0)tracer[receiver][donor]-=std::abs(dm);
        }
        Matrix tracer_oxidizer;tracer_oxidizer.size=nodes;
        std::copy_n(tracer.values.data(),nodes*nodes,tracer_oxidizer.values.data());
        if(!linear_solve(tracer,chemical,nodes,chem_fraction) || !linear_solve(tracer_oxidizer,oxidizer,nodes,ox_fraction))return false;
        result.state=old;result.packets={};
        for(int j=0;j<edges;++j) {
            const auto& e=paths[j];const double dm=x[j]*mass_scale[j];const int donor=dm>=0?e.a:e.b;
            const double td=x[edges+donor]*temp_scale[donor];
            const Packet q{dm,dm*((cv+r)*td-cv*model.reference_temperature_k),dm*chem_fraction[donor],dm*ox_fraction[donor]};
            result.packets[j]=q;
            const auto apply=[&](int i,double sign) {auto& g=result.state.gas[i];
                g.mass_kg+=sign*q.mass;g.energy_j+=sign*q.energy;g.chemical_j+=sign*q.chemical;g.oxidizer_kg+=sign*q.oxidizer;};
            apply(e.a,-1);if(e.b>=0)apply(e.b,1);
        }
        for(int i=0;i<nodes;++i) {
            closure(i,result.state);const auto& g=result.state.gas[i];
            if(!(g.mass_kg>0 && g.temperature_k>0 && std::isfinite(g.pressure_pa)) || g.chemical_j< -1e-10 ||
                g.oxidizer_kg< -1e-14 || g.oxidizer_kg>g.mass_kg+1e-14)return false;
        }
        return true;
    }
    double error(const State& a,const State& b,const State& old) const {
        double chemical_scale=1e-12,oxidizer_scale=1e-16;
        for(int i=0;i<nodes;++i){chemical_scale+=old.gas[i].chemical_j;oxidizer_scale+=old.gas[i].oxidizer_kg;}
        double value=0;
        for(int i=0;i<nodes;++i) {
            const auto& u=a.gas[i];const auto& v=b.gas[i];
            value=std::max(value,std::abs(u.pressure_pa-v.pressure_pa)/(model.pressure_absolute_tolerance_pa+
                model.pressure_relative_tolerance*std::max(u.pressure_pa,v.pressure_pa)));
            value=std::max(value,std::abs(u.temperature_k-v.temperature_k)/model.implicit_temperature_tolerance_k);
            value=std::max(value,std::abs(u.chemical_j-v.chemical_j)/(model.implicit_tracer_relative_tolerance*chemical_scale+1e-12));
            value=std::max(value,std::abs(u.oxidizer_kg-v.oxidizer_kg)/(model.implicit_tracer_relative_tolerance*oxidizer_scale+1e-16));
        }
        return value;
    }
};
}

void LocalGasTransport::implicit_group(const GasLocalGroup& p,double duration,std::vector<GasState>& gas,
    std::vector<ThermalState>& solid,std::vector<GasChannelState>& channels,GasTransportStats& stats) const {
    ++stats.groups;Problem problem{scenario_.gas_model,static_cast<int>(p.nodes.size()),0,{},{},stats};State current;
    for(int i=0;i<problem.nodes;++i) {
        const int id=p.nodes[i];current.gas[i]=gas[id];const int j=mass_.bindings[id].thermal_volume;
        if(j>=0) {
            current.solid[i]=solid[j];const auto& v=thermal_.volumes[j];const auto& m=scenario_.cells[v.object];
            problem.laws[i]={&m,v.capacity_j_per_k,v.capacity_j_per_k/m.heat_capacity_j_per_k};
        }
    }
    for(int j=0;j<int(p.edges.size());++j) {
        const int id=p.edges[j];const auto& c=scenario_.gas_channels[id];const auto& state=channels[id];
        if(state.open && state.effective_area_m2>0 && c.discharge_coefficient>0)
            problem.paths[problem.edges++]={id,p.ends[j].first,p.ends[j].second,state.effective_area_m2*c.discharge_coefficient,c.outside_pressure_pa};
    }
    if(problem.edges==0)return;
    std::array<Packet,max_edges> total{};double elapsed=0,h=duration;int attempts=0;
    while(elapsed<duration) {
        h=std::min(h,duration-elapsed);
        if(!(h>0) || elapsed+h==elapsed || ++attempts>2000)throw std::runtime_error("Implicit group time-step convergence failed: first node="+
            std::to_string(p.nodes[0])+", elapsed="+std::to_string(elapsed)+", interval="+std::to_string(duration)+
            ", h="+std::to_string(h)+", rejected="+std::to_string(stats.rejections));
        Result full,half,fine;
        const bool solved=problem.backward_euler(current,h,full) && problem.backward_euler(current,h/2,half) &&
            problem.backward_euler(half.state,h/2,fine);
        const double error=solved?problem.error(full.state,fine.state,current):1e300;
        if(!solved || error>1) {++stats.rejections;h*=solved?std::max(.1,.8/std::sqrt(error)):.5;continue;}
        current=fine.state;elapsed+=h;stats.implicit_steps+=2;
        for(int j=0;j<problem.edges;++j) {
            auto& t=total[j];const auto& a=half.packets[j];const auto& b=fine.packets[j];
            t.mass+=a.mass+b.mass;t.energy+=a.energy+b.energy;t.chemical+=a.chemical+b.chemical;t.oxidizer+=a.oxidizer+b.oxidizer;
        }
        h*=error>0?std::min(2.,.9/std::sqrt(error)):2.;
    }
    for(int i=0;i<problem.nodes;++i) {
        const int id=p.nodes[i];gas[id]=current.gas[i];const int j=mass_.bindings[id].thermal_volume;
        if(j>=0)solid[j]=current.solid[i];
    }
    for(int j=0;j<problem.edges;++j) {
        auto& c=channels[problem.paths[j].original];const auto& q=total[j];
        c.transferred_kg+=q.mass;c.transferred_energy_j+=q.energy;c.transferred_chemical_j+=q.chemical;
        c.transferred_oxidizer_kg+=q.oxidizer;c.mass_flow_kg_s+=q.mass;
    }
}
}

#pragma once
// Whole-graph backward Euler: counterexchange and optional finite surface heat.
// Edge weights remain evaluations of the existing directed S laws.
struct MixSolveState {
    double x[5]{},origin[5]{},r[5]{},d[5]{},ad[5]{};
    double mobile=0.,thermal=0.,sum=0.,film_sum=0.;
    struct Enthalpy {
        double rhs=0.,diagonal=0.,r=0.,shadow=0.,p=0.,v=0.,s=0.,t=0.,z=0.;
    } h;
};
__device__ inline double mix_capacity(const MixSolveState& v,int f){return f==0?v.thermal:v.mobile;}
__device__ inline double mix_diagonal(const MixSolveState& v,int f){return mix_capacity(v,f)+v.sum+(f==0?v.film_sum:0.);}
__device__ inline double mix_scale(int f){return f==0?1000.:((f==1 || f==4)?1e6:1.);}
__device__ inline double mix_tolerance(const GasModel& m,int f){
    return f==0?m.implicit_temperature_tolerance_k/1000.:m.implicit_tracer_relative_tolerance;
}
__device__ inline double mix_payload(const kg::Node& n,const kg::Settings& s,int f){
    if(f==0)return kg::temperature(n,s)/mix_scale(f);
    return (f==1?n.chemical:f==2?n.oxidizer:f==3?n.vapor:n.latent)/(n.mass*mix_scale(f));
}
// The same SPD operator evaluates residuals and search directions.
template<bool direction>
__device__ inline void mix_products(int i,int gas_count,const MixSolveState* work,
    const int* offsets,const int* incidence,const kg::Edge* edges,const Proposal* flux,
    const GasFilmView& films,double film_weight,double result[5]) {
    const auto& own=direction?work[i].d:work[i].x;
    for(int f=0;f<5;++f)result[f]=mix_capacity(work[i],f)*own[f];
    if(i<gas_count){
        for(int j=offsets[i];j<offsets[i+1];++j){const int e=incidence[j];const auto& edge=edges[e];
            const int other=edge.a==i?edge.b:edge.a;const auto& neighbour=direction?work[other].d:work[other].x;
            const double m=flux[e].mixing.mass_each_way;
            for(int f=0;f<5;++f)result[f]+=m*(own[f]-neighbour[f]);}
        if(film_weight>0.)for(int j=films.offsets[i];j<films.offsets[i+1];++j){
            const int p=films.ids[j],other=gas_count+films.patches[p].volume;
            result[0]+=film_weight*films.conductance[p]*(own[0]-(direction?work[other].d[0]:work[other].x[0]));}
    }else{
        for(int j=films.solid_offsets[i-gas_count];j<films.solid_offsets[i-gas_count+1];++j){
            const int p=films.solid_ids[j],other=films.patches[p].compartment;
            result[0]+=film_weight*films.conductance[p]*(own[0]-(direction?work[other].d[0]:work[other].x[0]));}
    }
}
#include "implicit_mixing_bounds.cuh"
#include "advective_surface_energy.cuh"
__device__ inline bool implicit_mix_snapshot(cooperative_groups::grid_group grid,int tid,int stride,
    int gas_count,int edge_count,const int* offsets,const int* incidence,const kg::Edge* edges,
    kg::Node* nodes,Proposal* flux,GasChannelState* channels,const kg::Settings& settings,
    const GasModel& model,double dt,MixSolveState* work,Control* control,double* shared,int* error,
    GasFilmView films={}) {
    const bool joint=model.joint_gas_surface_mixing && films.patches;
    const int count=gas_count+(joint?films.solid_count:0);
    const double film_weight=joint?dt/settings.cv:0.;
    if(tid==0)control->mixing_solid_entropy=0.;
    for(int e=tid;e<edge_count;e+=stride){auto edge=edges[e];
        edge.area=channels[e].open?channels[e].effective_area_m2:0.;
        flux[e].mixing={};flux[e].mixing.mass_each_way=dt*mixing_conductance(nodes[edge.a],nodes[edge.b],edge,settings,
            model.mixing_diffusivity_m2_s,model.mixing_flow_fraction,flux[e].mass/dt);}
    for(int i=tid;i<count;i+=stride){auto& v=work[i];v={};
        if(i<gas_count){v.mobile=nodes[i].mass;v.thermal=v.mobile+nodes[i].solid_capacity/settings.cv;
            for(int f=0;f<5;++f)v.x[f]=v.origin[f]=mix_payload(nodes[i],settings,f);
        }else{const int volume=i-gas_count;double q=0.;
            for(int j=films.solid_offsets[volume];j<films.solid_offsets[volume+1];++j)q+=films.heat[films.solid_ids[j]];
            v.mobile=1.;v.thermal=films.capacity[volume]/settings.cv;
            v.x[0]=v.origin[0]=(films.solid[volume].temperature_k+q/films.capacity[volume])/1000.;}
    }
    grid.sync();
    if(joint)for(int i=tid;i<gas_count;i+=stride)for(int j=films.offsets[i];j<films.offsets[i+1];++j){
        const int p=films.ids[j];const auto& patch=films.patches[p];
        films.conductance[p]=gas_contact_conductance(patch,1000.*work[i].origin[0],
            1000.*work[gas_count+patch.volume].origin[0],films.solid[patch.volume].damage);}
    grid.sync();double dot=0.;
    for(int i=tid;i<count;i+=stride){auto& v=work[i];
        if(i<gas_count){for(int j=offsets[i];j<offsets[i+1];++j)v.sum+=flux[incidence[j]].mixing.mass_each_way;
            if(joint)for(int j=films.offsets[i];j<films.offsets[i+1];++j)v.film_sum+=film_weight*films.conductance[films.ids[j]];
        }else for(int j=films.solid_offsets[i-gas_count];j<films.solid_offsets[i-gas_count+1];++j)
            v.film_sum+=film_weight*films.conductance[films.solid_ids[j]];
        double product[5];mix_products<false>(i,gas_count,work,offsets,incidence,edges,flux,films,film_weight,product);
        for(int f=0;f<5;++f){v.r[f]=mix_capacity(v,f)*v.origin[f]-product[f];
            v.d[f]=v.r[f]/mix_diagonal(v,f);dot+=v.r[f]*v.d[f];}}
    double rho=average_reduce_sum(grid,dot,shared,control),residual=1e300;const double initial_rho=rho;
    unsigned used=0;double accuracy=1.;bool bounded=false;
    for(int iteration=0;iteration<1024 && rho>1e-290;++iteration){
        dot=0.;
        for(int i=tid;i<count;i+=stride){auto& v=work[i];
            mix_products<true>(i,gas_count,work,offsets,incidence,edges,flux,films,film_weight,v.ad);
            for(int f=0;f<5;++f)dot+=v.d[f]*v.ad[f];}
        const double divisor=average_reduce_sum(grid,dot,shared,control);if(!(divisor>0.))break;
        const double alpha=rho/divisor;dot=0.;
        for(int i=tid;i<count;i+=stride){auto& v=work[i];for(int f=0;f<5;++f){
            v.x[f]+=alpha*v.d[f];v.r[f]-=alpha*v.ad[f];dot+=v.r[f]*v.r[f]/mix_diagonal(v,f);}}
        const double next_rho=average_reduce_sum(grid,dot,shared,control),beta=next_rho/rho;
        for(int i=tid;i<count;i+=stride)for(int f=0;f<5;++f)work[i].d[f]=work[i].r[f]/mix_diagonal(work[i],f)+beta*work[i].d[f];
        grid.sync();rho=next_rho;used=iteration+1;
        if(used%8==0 || rho<initial_rho*1e-20 || rho<1e-290){double maximum=0.;
            for(int i=tid;i<count;i+=stride)for(int f=0;f<5;++f){const double r=fabs(work[i].r[f])/(mix_capacity(work[i],f)*mix_tolerance(model,f));
                maximum=isfinite(r)?fmax(maximum,r):INFINITY;}
            residual=-average_reduce_min(grid,-maximum,shared,control);
            if(residual<=accuracy) {
                // Recursive PCG residuals can drift on near-empty gas volumes.
                // Re-evaluate b-Ax before accepting, then inspect the exact
                // conservative gather, including the untracked inert fraction.
                residual=mix_true_residual(grid,tid,stride,count,gas_count,work,offsets,incidence,
                    edges,flux,films,film_weight,model,shared,control);
                if(residual<=1.) {
                    bounded=mix_stage_bounded_flux(grid,tid,stride,count,gas_count,edge_count,work,
                        offsets,incidence,edges,nodes,flux,settings,films,joint,dt,shared,control);
                    if(bounded)break;
                    accuracy*=.1;
                }
                rho=mix_restart_cg(grid,tid,stride,count,work,shared,control);
            }
        }
    }
    if(!bounded) {
        residual=mix_true_residual(grid,tid,stride,count,gas_count,work,offsets,incidence,
            edges,flux,films,film_weight,model,shared,control);
        if(residual<=1.)bounded=mix_stage_bounded_flux(grid,tid,stride,count,gas_count,edge_count,
            work,offsets,incidence,edges,nodes,flux,settings,films,joint,dt,shared,control);
    }
    if(tid==0){control->mixing_iterations=used;control->mixing_residual=residual;}
    if(!(residual<=1.) || !bounded){if(tid==0)*error=residual>1.?25:26;grid.sync();return false;}
    // Counterexchange/species have converged. Reconcile sensible enthalpy of
    // the pressure-driven THROUGH-flow with these same films and mixing links.
    // Otherwise film heating is spuriously capped by end-of-step resident gas.
    if(joint && model.averaged_network_transport){
        if(!advective_surface_energy(grid,tid,stride,count,gas_count,edge_count,
            offsets,incidence,edges,nodes,flux,settings,model,dt,work,films,control,shared,error))return false;
        if(!mix_stage_bounded_flux(grid,tid,stride,count,gas_count,edge_count,work,
            offsets,incidence,edges,nodes,flux,settings,films,joint,dt,shared,control)){
            if(tid==0)*error=35;grid.sync();return false;}
    }
    // Publish only the unmodified conservative proposal checked above.
    // Independent positivity shifts/clips of species potentials can conserve
    // global species masses yet violate their shared local mass budget.
    for(int e=tid;e<edge_count;e+=stride){const auto& p=flux[e].mixing;
        auto& channel=channels[e];channel.mixed_mass_kg+=p.mass_each_way;channel.transferred_energy_j+=p.energy;
        channel.transferred_chemical_j+=p.chemical;channel.transferred_oxidizer_kg+=p.oxidizer;
        channel.transferred_vapor_kg+=p.vapor;channel.transferred_latent_j+=p.latent;}
    for(int e=tid;e<edge_count;e+=stride){auto& p=flux[e];
        channels[e].transferred_energy_j+=(p.donor==edges[e].a?1.:-1.)*p.advective_energy_delta;}
    grid.sync();
    for(int i=tid;i<gas_count;i+=stride){auto n=nodes[i];
        for(int j=offsets[i];j<offsets[i+1];++j){const int e=incidence[j];const auto& p=flux[e].mixing;const double sign=edges[e].a==i?-1.:1.;
            n.energy+=sign*p.energy;n.chemical+=sign*p.chemical;n.oxidizer+=sign*p.oxidizer;n.vapor+=sign*p.vapor;n.latent+=sign*p.latent;}
        for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];
            n.energy+=(p.donor==i?-1.:1.)*p.advective_energy_delta;}
        if(joint)for(int j=films.offsets[i];j<films.offsets[i+1];++j){const int p=films.ids[j];
            const double q=dt*films.conductance[p]*1000.*(work[i].x[0]-work[gas_count+films.patches[p].volume].x[0]);
            n.energy-=q;films.heat[p]+=q;}
        nodes[i]=n;}
    if(joint){double entropy=0.;
        for(int v=tid;v<films.solid_count;v+=stride){double q=0.;
            for(int j=films.solid_offsets[v];j<films.solid_offsets[v+1];++j){const int p=films.solid_ids[j];
                q+=dt*films.conductance[p]*1000.*(work[films.patches[p].compartment].x[0]-work[gas_count+v].x[0]);}
            const double t=work[gas_count+v].origin[0]*1000.;entropy+=films.capacity[v]*log1p(q/(films.capacity[v]*t));}
        const double total=average_reduce_sum(grid,entropy,shared,control);if(tid==0)control->mixing_solid_entropy=total;
    }
    grid.sync();return true;
}

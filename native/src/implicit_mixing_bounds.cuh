#pragma once
#include <cfloat>
// Included after MixSolveState/mix_products. These helpers inspect the actual
// conservative proposal; they never clamp inventories or shorten physical dt.
__device__ inline double mix_true_residual(cooperative_groups::grid_group grid,int tid,int stride,
    int count,int gas_count,MixSolveState* work,const int* offsets,const int* incidence,
    const kg::Edge* edges,const Proposal* flux,const GasFilmView& films,double film_weight,
    const GasModel& model,double* shared,Control* control) {
    double maximum=0.,raw_maximum=0.;
    for(int i=tid;i<count;i+=stride) {
        auto& v=work[i];double residual[5],magnitude[5];int terms=0;
        for(int f=0;f<5;++f) {
            const double capacity=mix_capacity(v,f);
            residual[f]=capacity*(v.origin[f]-v.x[f]);
            magnitude[f]=capacity*(fabs(v.origin[f])+fabs(v.x[f]));
        }
        if(i<gas_count) {
            for(int j=offsets[i];j<offsets[i+1];++j) {
                const int e=incidence[j];const int other=edges[e].a==i?edges[e].b:edges[e].a;
                const double weight=flux[e].mixing.mass_each_way;++terms;
                for(int f=0;f<5;++f) {
                    residual[f]-=weight*(v.x[f]-work[other].x[f]);
                    magnitude[f]+=weight*(fabs(v.x[f])+fabs(work[other].x[f]));
                }
            }
        }
        if(film_weight>0.) {
            const int volume=i-gas_count;
            const int begin=i<gas_count?films.offsets[i]:films.solid_offsets[volume];
            const int end=i<gas_count?films.offsets[i+1]:films.solid_offsets[volume+1];
            for(int j=begin;j<end;++j) {
                const int p=i<gas_count?films.ids[j]:films.solid_ids[j];
                const int other=i<gas_count?gas_count+films.patches[p].volume:films.patches[p].compartment;
                const double weight=film_weight*films.conductance[p];++terms;
                residual[0]-=weight*(v.x[0]-work[other].x[0]);
                magnitude[0]+=weight*(fabs(v.x[0])+fabs(work[other].x[0]));
            }
        }
        for(int f=0;f<5;++f) {
            v.r[f]=residual[f];
            const double requested=mix_capacity(v,f)*mix_tolerance(model,f);
            // A nearly empty control volume can have an unattainable specific
            // residual even when its potential agrees to machine precision.
            // Bound the arithmetic of this row, not a fitted physical scale.
            const double roundoff=(8.+8.*terms)*2.2204460492503131e-16*magnitude[f];
            const double r=fabs(v.r[f])/(requested+roundoff);
            maximum=isfinite(r)?fmax(maximum,r):INFINITY;
            const double raw=fabs(v.r[f])/requested;
            raw_maximum=isfinite(raw)?fmax(raw_maximum,raw):INFINITY;
        }
    }
    const double result=-average_reduce_min(grid,-maximum,shared,control);
    const double raw=-average_reduce_min(grid,-raw_maximum,shared,control);
    if(tid==0)control->mixing_raw_residual=raw;
    return result;
}
__device__ inline bool mix_flux_admissible(cooperative_groups::grid_group grid,int tid,int stride,
    int count,int gas_count,int edge_count,MixSolveState* work,const int* offsets,const int* incidence,
    const kg::Edge* edges,const kg::Node* nodes,Proposal* flux,const kg::Settings& settings,
    const GasFilmView& films,bool joint,double dt,double* shared,Control* control) {
    grid.sync();double admissible=1.;
    for(int i=tid;i<gas_count;i+=stride) {
        auto n=nodes[i];
        for(int j=offsets[i];j<offsets[i+1];++j) {
            const int e=incidence[j];const auto& p=flux[e].mixing;const double sign=edges[e].a==i?-1.:1.;
            n.energy+=sign*p.energy;n.chemical+=sign*p.chemical;n.oxidizer+=sign*p.oxidizer;
            n.vapor+=sign*p.vapor;n.latent+=sign*p.latent;
            const auto& adv=flux[e];n.energy+=(adv.donor==i?-1.:1.)*adv.advective_energy_delta;
        }
        if(joint)for(int j=films.offsets[i];j<films.offsets[i+1];++j) {
            const int p=films.ids[j];n.energy-=dt*films.conductance[p]*1000.*
                (work[i].x[0]-work[gas_count+films.patches[p].volume].x[0]);
        }
        if(!(isfinite(n.energy)&&n.energy>0&&isfinite(n.chemical)&&n.chemical>=0&&
            isfinite(n.oxidizer)&&n.oxidizer>=0&&isfinite(n.vapor)&&n.vapor>=0&&
            n.oxidizer+n.vapor<=n.mass+1e-14&&isfinite(n.latent)&&n.latent>=0))admissible=0.;
    }
    return average_reduce_min(grid,admissible,shared,control)>0.;
}
#include "mixing_underflow.cuh"
__device__ inline bool mix_stage_bounded_flux(cooperative_groups::grid_group grid,int tid,int stride,
    int count,int gas_count,int edge_count,MixSolveState* work,const int* offsets,const int* incidence,
    const kg::Edge* edges,const kg::Node* nodes,Proposal* flux,const kg::Settings& settings,
    const GasFilmView& films,bool joint,double dt,double* shared,Control* control) {
    // Keep all representable incoming fluxes, including subnormal ones: a
    // small incoming flux may fund a larger normal outgoing flux in a chain.
    for(int e=tid;e<edge_count;e+=stride) {
        const auto& edge=edges[e];auto& p=flux[e].mixing;
        double values[5];for(int f=0;f<5;++f)values[f]=p.mass_each_way*mix_scale(f)*
            (work[edge.a].x[f]-work[edge.b].x[f]);
        p.energy=settings.cv*values[0];p.chemical=values[1];p.oxidizer=values[2];p.vapor=values[3];p.latent=values[4];
    }
    for(int pass=0;pass<=32;++pass) {
        if(mix_flux_admissible(grid,tid,stride,count,gas_count,edge_count,work,offsets,incidence,
            edges,nodes,flux,settings,films,joint,dt,shared,control))return true;
        // Rare underflow-only repair of paired transfers, never inventories.
        // A genuine deficit remains an error; normal steps take the fast exit.
        if(pass==32 || !mix_repair_underflow(grid,tid,stride,gas_count,edge_count,work,
            offsets,incidence,edges,nodes,flux,shared,control))return false;
    }
    return false;
}
__device__ inline double mix_restart_cg(cooperative_groups::grid_group grid,int tid,int stride,
    int count,MixSolveState* work,double* shared,Control* control) {
    double dot=0.;
    for(int i=tid;i<count;i+=stride)for(int f=0;f<5;++f) {
        auto& v=work[i];v.d[f]=v.r[f]/mix_diagonal(v,f);dot+=v.r[f]*v.d[f];
    }
    return average_reduce_sum(grid,dot,shared,control);
}

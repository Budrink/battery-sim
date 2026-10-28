#pragma once
// One thermal M-matrix: accepted pressure-carrier enthalpy + counterexchange
// + finite-capacity surfaces. Species/reaction inventories stay untouched.
// Reconciles a split-stage proposal BEFORE publication, not an added heater.
// Algebraic BiCGStab handles directed advection; there are no smaller dt steps.
template<bool search>
__device__ inline double enthalpy_product(int i,int gas_count,const MixSolveState* w,
    const int* offsets,const int* ids,const kg::Edge* edges,const Proposal* flux,
    const GasFilmView& films,double film_weight,double gamma) {
    const double own=search?w[i].h.z:w[i].x[0];double result=w[i].thermal*own;
    if(i<gas_count){
        for(int j=offsets[i];j<offsets[i+1];++j){const int e=ids[j];const auto& p=flux[e];
            const int other=edges[e].a==i?edges[e].b:edges[e].a;
            const double neighbor=search?w[other].h.z:w[other].x[0];
            result+=p.mixing.mass_each_way*(own-neighbor);
            result+=gamma*p.mass*(p.donor==i?own:-neighbor);
        }
        for(int j=films.offsets[i];j<films.offsets[i+1];++j){const int p=films.ids[j],other=gas_count+films.patches[p].volume;
            result+=film_weight*films.conductance[p]*(own-(search?w[other].h.z:w[other].x[0]));}
    }else{
        for(int j=films.solid_offsets[i-gas_count];j<films.solid_offsets[i-gas_count+1];++j){
            const int p=films.solid_ids[j],other=films.patches[p].compartment;
            result+=film_weight*films.conductance[p]*(own-(search?w[other].h.z:w[other].x[0]));}
    }
    return result;
}

__device__ inline double enthalpy_refresh(cooperative_groups::grid_group grid,int tid,int stride,
    int count,int gas_count,MixSolveState* w,const int* offsets,const int* ids,const kg::Edge* edges,
    const Proposal* flux,const GasFilmView& films,double weight,double gamma,const GasModel& model,
    double* shared,Control* control,bool replace=true) {
    double maximum=0.;
    for(int i=tid;i<count;i+=stride){auto& v=w[i];
        const double ax=enthalpy_product<false>(i,gas_count,w,offsets,ids,edges,flux,films,weight,gamma);
        const double actual=v.h.rhs-ax;if(replace)v.h.r=actual;
        // Bound cancellation by this row's coefficients, not a fit parameter.
        const int terms=i<gas_count?offsets[i+1]-offsets[i]+films.offsets[i+1]-films.offsets[i]:
            films.solid_offsets[i-gas_count+1]-films.solid_offsets[i-gas_count];
        double magnitude=fabs(v.h.rhs)+fabs(v.thermal*v.x[0]);
        if(i<gas_count)for(int j=offsets[i];j<offsets[i+1];++j){const int e=ids[j];
            const int other=edges[e].a==i?edges[e].b:edges[e].a;const auto& p=flux[e];
            magnitude+=p.mixing.mass_each_way*(fabs(v.x[0])+fabs(w[other].x[0]));
            magnitude+=gamma*p.mass*fabs(w[p.donor].x[0]);}
        const int begin=i<gas_count?films.offsets[i]:films.solid_offsets[i-gas_count];
        const int end=i<gas_count?films.offsets[i+1]:films.solid_offsets[i-gas_count+1];
        for(int j=begin;j<end;++j){const int p=i<gas_count?films.ids[j]:films.solid_ids[j];
            const int other=i<gas_count?gas_count+films.patches[p].volume:films.patches[p].compartment;
            magnitude+=weight*films.conductance[p]*(fabs(v.x[0])+fabs(w[other].x[0]));}
        const double tolerance=v.thermal*model.implicit_temperature_tolerance_k/1000.+
            (8.+8.*terms)*2.2204460492503131e-16*magnitude;
        const double value=fabs(actual)/tolerance;
        maximum=isfinite(value)?fmax(maximum,value):INFINITY;
    }
    return -average_reduce_min(grid,-maximum,shared,control);
}

__device__ inline bool advective_surface_energy(cooperative_groups::grid_group grid,int tid,int stride,
    int count,int gas_count,int edge_count,const int* offsets,const int* ids,const kg::Edge* edges,
    const kg::Node* nodes,Proposal* flux,const kg::Settings& settings,const GasModel& model,double dt,
    MixSolveState* w,const GasFilmView& films,Control* control,double* shared,int* error) {
    const double gamma=1.+settings.R/settings.cv,weight=dt/settings.cv;
    double transport=0.;for(int e=tid;e<edge_count;e+=stride)transport+=flux[e].mass;
    if(average_reduce_sum(grid,transport,shared,control)==0.)return true;
    for(int i=tid;i<count;i+=stride){auto& v=w[i];v.h={};
        v.h.rhs=v.thermal*v.origin[0];v.h.diagonal=v.thermal+v.sum+v.film_sum;
        if(i<gas_count)for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[ids[j]];
            // Undo only the provisional pressure-enthalpy gather. Reaction
            // heat stays in this RHS. No carrier/species transfer is repeated.
            v.h.rhs+=(p.donor==i?1.:-1.)*p.energy/(settings.cv*1000.);
            if(p.donor==i)v.h.diagonal+=gamma*p.mass;}
    }
    grid.sync();double residual=enthalpy_refresh(grid,tid,stride,count,gas_count,w,offsets,ids,
        edges,flux,films,weight,gamma,model,shared,control);
    double rho_old=1.,alpha=1.,omega=1.;bool restart=true;unsigned used=0;
    for(int iteration=0;iteration<1024 && residual>1.;++iteration){
        if(restart){for(int i=tid;i<count;i+=stride){auto& h=w[i].h;h.shadow=h.r;h.p=h.v=0.;}
            rho_old=alpha=omega=1.;grid.sync();restart=false;}
        double dot=0.;for(int i=tid;i<count;i+=stride)dot+=w[i].h.shadow*w[i].h.r;
        const double rho=average_reduce_sum(grid,dot,shared,control);
        if(!isfinite(rho) || fabs(rho)<1e-290)break;
        const double beta=(rho/rho_old)*(alpha/omega);
        for(int i=tid;i<count;i+=stride){auto& h=w[i].h;h.p=h.r+beta*(h.p-omega*h.v);h.z=h.p/h.diagonal;}
        grid.sync();dot=0.;
        for(int i=tid;i<count;i+=stride){auto& h=w[i].h;
            h.v=enthalpy_product<true>(i,gas_count,w,offsets,ids,edges,flux,films,weight,gamma);dot+=h.shadow*h.v;}
        const double divisor=average_reduce_sum(grid,dot,shared,control);
        if(!isfinite(divisor) || fabs(divisor)<1e-290)break;
        alpha=rho/divisor;
        for(int i=tid;i<count;i+=stride){auto& h=w[i].h;h.s=h.r-alpha*h.v;h.z=h.s/h.diagonal;}
        grid.sync();dot=0.;double denominator=0.;
        for(int i=tid;i<count;i+=stride){auto& h=w[i].h;
            h.t=enthalpy_product<true>(i,gas_count,w,offsets,ids,edges,flux,films,weight,gamma);
            dot+=h.t*h.s;denominator+=h.t*h.t;}
        const double numerator=average_reduce_sum(grid,dot,shared,control);
        const double norm=average_reduce_sum(grid,denominator,shared,control);
        omega=norm>1e-290?numerator/norm:0.;
        for(int i=tid;i<count;i+=stride){auto& v=w[i];auto& h=v.h;
            v.x[0]+=(alpha*h.p+omega*h.s)/h.diagonal;h.r=h.s-omega*h.t;}
        grid.sync();rho_old=rho;++used;
        if(used%4==0 || fabs(omega)<1e-30){
            residual=enthalpy_refresh(grid,tid,stride,count,gas_count,w,offsets,ids,edges,flux,films,
                weight,gamma,model,shared,control,false);
            // Inspect the TRUE residual without destroying Krylov history.
            // Restart only a degenerate recurrence, not at fixed intervals.
            if(fabs(omega)<1e-30 && residual>1.){
                enthalpy_refresh(grid,tid,stride,count,gas_count,w,offsets,ids,edges,flux,films,
                    weight,gamma,model,shared,control);restart=true;}
        }
    }
    residual=enthalpy_refresh(grid,tid,stride,count,gas_count,w,offsets,ids,edges,flux,films,
        weight,gamma,model,shared,control);
    if(tid==0){control->enthalpy_iterations=used;control->enthalpy_residual=residual;}
    if(!(residual<=1.)){if(tid==0)*error=34;grid.sync();return false;}
    for(int e=tid;e<edge_count;e+=stride){auto& p=flux[e];
        p.advective_energy_delta=p.mass*(settings.cv+settings.R)*1000.*w[p.donor].x[0]-p.energy;}
    grid.sync();return true;
}

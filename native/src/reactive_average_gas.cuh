#pragma once
// Included inside batched_gas AFTER the shared reductions. Pressure/carrier
// fluxes are already accepted. Analytically integrate reaction DURING that
// transit; downstream enthalpy sees its heat in the SAME physical window.
__device__ inline bool reacting_payload(cooperative_groups::grid_group grid,int tid,int stride,
    int count,int edge_count,const int* offsets,const int* incidence,const kg::Edge* edges,
    const kg::Node* old,kg::Node* next,Proposal* flux,const kg::Settings& settings,
    const GasModel& model,double dt,AverageState* a,AverageState* b,
    double* shared,Control* control,int* error,GasCombustionView chemistry) {
    namespace rt=battery::reactive_transit;
    bool valid=true;
    for(int i=tid;i<count;i+=stride){
        auto& r=a[i].reaction;r={};
        for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];
            (p.donor==i?r.outgoing:r.incoming)+=p.mass;}
        valid=rt::exposure(old[i].mass,r.incoming,r.outgoing,r.exposure) && valid;
        const auto law=chemistry.laws?chemistry.laws[i]:GasCombustionLaw{};
        // Arriving enthalpy is included: use the already computed transport
        // temperature, not the cold pre-arrival resident temperature.
        r.rate_dt=dt*combustion_rate(law,kg::temperature(next[i],settings));
    }
    if(average_reduce_min(grid,valid?1.:0.,shared,control)==0.){if(tid==0)*error=29;grid.sync();return false;}
    double residual=1.;
    // Pressure-directed advection is a DAG. Snapshot iterations propagate
    // sources along that DAG; convergence is algebraic, NEVER smaller dt.
    const int budget=model.averaged_payload_iterations>128?model.averaged_payload_iterations:128;
    for(int iteration=0;iteration<budget;++iteration){
        double change=0.;
        for(int i=tid;i<count;i+=stride){
            const auto& r=a[i].reaction;auto v=a[i];auto& result=v.reaction;
            double fuel=0.,oxygen=0.,vapor=0.,latent=0.;
            for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];
                if(p.donor==i)continue;
                const auto& donor=a[p.donor].reaction;
                const double w=donor.outgoing>0.?p.mass/donor.outgoing:0.;
                fuel+=w*donor.fuel;oxygen+=w*donor.oxygen;vapor+=w*donor.vapor;latent+=w*donor.latent;
            }
            const double heat_per_oxygen=chemistry.laws?chemistry.laws[i].oxidizer_heat_j_per_kg:13.1e6;
            const auto reaction=rt::integrate(old[i].chemical,old[i].oxidizer*heat_per_oxygen,
                fuel,oxygen*heat_per_oxygen,r.exposure,r.rate_dt);
            result.heat=reaction.heat;result.fuel=reaction.fuel_out;
            result.oxygen=reaction.oxygen_out/heat_per_oxygen;
            const double from_resident=-expm1(-r.exposure),from_arrivals=1.-rt::phi(r.exposure);
            const double burned=settings.vapor_fuel_heat_j_kg>0. && old[i].chemical+fuel>0.?
                fmin(1.,reaction.heat/(old[i].chemical+fuel)):0.;
            result.vapor=(1.-burned)*(old[i].vapor*from_resident+vapor*from_arrivals);
            result.vapor_burned=burned*(old[i].vapor+vapor);
            // Phase-reference energy remains with products (existing R3 rule).
            result.latent=old[i].latent*from_resident+latent*from_arrivals;
            change=fmax(change,fabs(result.fuel-r.fuel)/(1e-10+old[i].chemical+fuel));
            change=fmax(change,fabs(result.heat-r.heat)/(1e-10+old[i].chemical+fuel));
            change=fmax(change,fabs(result.oxygen-r.oxygen)/(1e-16+old[i].mass+r.incoming));
            change=fmax(change,fabs(result.vapor-r.vapor)/(1e-16+old[i].mass+r.incoming));
            change=fmax(change,fabs(result.latent-r.latent)/(1e-10+old[i].latent+latent));
            b[i]=v;
        }
        grid.sync();auto* swap=a;a=b;b=swap;
        if((iteration+1)%4==0){residual=-average_reduce_min(grid,-change,shared,control);
            if(tid==0){control->reactive_iterations=iteration+1;control->reactive_residual=residual;}
            if(residual<1e-9)break;}
    }
    if(!(residual<1e-9)){if(tid==0)*error=30;grid.sync();return false;}
    for(int e=tid;e<edge_count;e+=stride){auto& p=flux[e];const auto& r=a[p.donor].reaction;
        const double w=r.outgoing>0.?p.mass/r.outgoing:0.;
        p.chemical=w*r.fuel;p.oxidizer=w*r.oxygen;p.vapor=w*r.vapor;p.latent=w*r.latent;
    }
    grid.sync();valid=true;
    for(int i=tid;i<count;i+=stride){auto n=old[i];const auto& r=a[i].reaction;
        n.chemical-=r.heat;n.oxidizer-=r.heat/(chemistry.laws?chemistry.laws[i].oxidizer_heat_j_per_kg:13.1e6);
        n.vapor-=r.vapor_burned;n.energy+=r.heat;
        n.reaction_released_j+=r.heat;
        n.reaction_oxidizer_kg+=r.heat/(chemistry.laws?chemistry.laws[i].oxidizer_heat_j_per_kg:13.1e6);
        for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];const double sign=p.donor==i?-1.:1.;
            n.mass+=sign*p.mass;n.chemical+=sign*p.chemical;n.oxidizer+=sign*p.oxidizer;
            n.vapor+=sign*p.vapor;n.latent+=sign*p.latent;}
        // Reject genuine negative inventory. Only erase harmless summation
        // roundoff; no reaction or missing heat is repaired by a clamp.
        valid=valid && n.mass>0. && n.chemical>=-1e-10 && n.oxidizer>=-1e-15 &&
            n.vapor>=-1e-15 && n.oxidizer+n.vapor<=n.mass+1e-14 && n.latent>=-1e-10;
        n.chemical=fmax(0.,n.chemical);n.oxidizer=fmax(0.,n.oxidizer);n.vapor=fmax(0.,n.vapor);n.latent=fmax(0.,n.latent);
        next[i]=n;a[i].temperature=0.;
    }
    if(average_reduce_min(grid,valid?1.:0.,shared,control)==0.){if(tid==0)*error=31;grid.sync();return false;}
    const double cp=settings.cv+settings.R;
    for(int iteration=0;iteration<model.averaged_payload_iterations;++iteration){
        for(int i=tid;i<count;i+=stride){double diagonal=next[i].solid_capacity+settings.cv*next[i].mass;
            double rhs=old[i].energy+a[i].reaction.heat;
            for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];
                if(p.donor==i)diagonal+=cp*p.mass;else rhs+=cp*p.mass*a[p.donor].temperature;}
            b[i]=a[i];b[i].temperature=rhs/diagonal;
        }
        grid.sync();auto* swap=a;a=b;b=swap;
    }
    for(int e=tid;e<edge_count;e+=stride)flux[e].energy=flux[e].mass*cp*a[flux[e].donor].temperature;
    grid.sync();valid=true;
    for(int i=tid;i<count;i+=stride){auto& n=next[i];
        for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];n.energy+=(p.donor==i?-1.:1.)*p.energy;}
        valid=valid && isfinite(n.energy) && n.energy>0.;
    }
    if(average_reduce_min(grid,valid?1.:0.,shared,control)==0.){if(tid==0)*error=31;grid.sync();return false;}
    return true;
}

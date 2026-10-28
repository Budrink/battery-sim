#pragma once
#include "pressure_preconditioner.cuh"
// Included inside batched_gas. Fixed-work, coarse implicit NETWORK integration.
// Iterations are algebraic, not hidden physical time substeps. No edge-local
// equilibrium cap and no gross-outflow/inventory cap: transit is permitted.
// Every accepted packet is applied identically at its two endpoints.
__device__ inline double average_reduce_min(cooperative_groups::grid_group grid,
    double value,double* shared,Control* control) {
    shared[threadIdx.x]=value;__syncthreads();
    for(int w=blockDim.x/2;w;w/=2){if(threadIdx.x<w)shared[threadIdx.x]=fmin(shared[threadIdx.x],shared[threadIdx.x+w]);__syncthreads();}
    if(threadIdx.x==0)control->pressure_gap_by_block[blockIdx.x]=shared[0];
    grid.sync();
    double result=1.;for(int b=0;b<int(gridDim.x);++b)result=fmin(result,control->pressure_gap_by_block[b]);
    grid.sync();return result;
}
__device__ inline double average_reduce_sum(cooperative_groups::grid_group grid,
    double value,double* shared,Control* control) {
    shared[threadIdx.x]=value;__syncthreads();
    for(int w=blockDim.x/2;w;w/=2){if(threadIdx.x<w)shared[threadIdx.x]+=shared[threadIdx.x+w];__syncthreads();}
    if(threadIdx.x==0)control->pressure_gap_by_block[blockIdx.x]=shared[0];
    grid.sync();double sum=0.;for(int block=0;block<int(gridDim.x);++block)sum+=control->pressure_gap_by_block[block];
    grid.sync();return sum;
}
// Pressure-equation residual in units of the larger of the old and predicted
// local inventories. A near-vacuum cavity can receive substantial gas in this
// very solve; normalizing solely by its almost-zero OLD pressure demands an
// impossible residual smaller than floating-point cancellation. Retain the old
// scale for draining cavities and a local norm for tiny cavities beside large
// reservoirs. The true residual is still recomputed before acceptance.
__device__ inline double average_pressure_residual(cooperative_groups::grid_group grid,
    int tid,int stride,int count,const AverageState* state,double* shared,Control* control) {
    double value=0.;
    for(int i=tid;i<count;i+=stride){const auto& a=state[i];
        const double scale=fmax(1.,fmax(a.pressure,a.pressure+a.correction));
        value=fmax(value,fabs(a.residual)/(a.compliance*scale));}
    return -average_reduce_min(grid,-value,shared,control);
}
#include "reactive_average_gas.cuh"
#include "pressure_mobility.cuh"
// Recompute b-A*x in correction coordinates. Recursive CG residuals can drift
// beside the many-decade pore/far-field compliance contrast. This is algebraic
// work only: no relaxed tolerance, new physical substep or discarded flux.
__device__ inline void refresh_average_residual(cooperative_groups::grid_group grid,int tid,int stride,
    int count,AverageState* a,const Proposal* flux,const int* offsets,const int* incidence,
    const kg::Edge* edges,double dt) {
    for(int i=tid;i<count;i+=stride) {
        auto& v=a[i];double residual=-v.compliance*v.correction;
        for(int j=offsets[i];j<offsets[i+1];++j) {
            const int e=incidence[j];const auto& edge=edges[e];const int other=edge.a==i?edge.b:edge.a;
            residual-=dt*flux[e].cap*((v.pressure-a[other].pressure)+(v.correction-a[other].correction));
        }
        v.residual=residual;
    }
    grid.sync();
}
__device__ inline bool average_transport(cooperative_groups::grid_group grid,int tid,int stride,
    int count,int edge_count,const int* offsets,const int* incidence,const kg::Edge* edges,
    const kg::Node* old,kg::Node* next,Proposal* flux,GasChannelState* channels,
    const kg::Settings& settings,const GasModel& model,double dt,double duration,Control* control,
    AverageState* a,AverageState* b,double* shared,int* error,GasCombustionView chemistry) {
    const double cp=settings.cv+settings.R;
    for(int i=tid;i<count;i+=stride){const double pattern=control->pressure_pattern_valid?a[i].pressure_pattern:0.;
        a[i]={};a[i].pressure_pattern=pattern;a[i].pressure=kg::pressure(old[i],settings);}
    for(int e=tid;e<edge_count;e+=stride){
        Proposal p;const auto& edge=edges[e];auto open=edge;
        open.area=channels[e].open?channels[e].effective_area_m2:0.;
        const bool forward=kg::pressure(old[edge.a],settings)>=kg::pressure(old[edge.b],settings);
        p.donor=forward?edge.a:edge.b;const int receiver=forward?edge.b:edge.a;
        const auto& d=old[p.donor];const auto& r=old[receiver];
        const double td=kg::temperature(d,settings),tr=kg::temperature(r,settings);
        const double pd=kg::pressure(d,settings),pr=kg::pressure(r,settings),gap=pd-pr;
        // Frozen secant mobility. A finite 1e-6 relative pressure scale only
        // regularizes the sqrt(dp) tangent at equilibrium; no artificial flow.
        const double epsilon=fmax(1e-6,1e-6*fmax(pd,pr));
        p.rate=kg::flow(d,r,open,settings);
        if(gap>=epsilon)p.cap=p.rate/gap;
        else if(open.area>0. && open.cd>0.) {
            auto test=d;test.energy*=1+epsilon/fmax(1.,pd);
            p.cap=kg::flow(test,r,open,settings)/epsilon;
        }
        const double cd=d.solid_capacity+settings.cv*d.mass,cr=r.solid_capacity+settings.cv*r.mass;
        const double wd=settings.R*td/d.volume*(1+d.mass*settings.R/cd);
        // Frozen local polytropic compliance, shared by every edge of a node.
        // Incoming enthalpy is handled by the conservative energy solve below,
        // NOT double-counted as an unlimited pressure-supported mass source.
        const double wr=settings.R*tr/r.volume*(1+r.mass*settings.R/cr);
        // These two weights are a/b ordered in this module, not donor/receiver.
        p.donor_pressure_weight=forward?wd:wr;p.receiver_pressure_weight=forward?wr:wd;
        flux[e]=p;
    }
    grid.sync();
    // Nonlinear channel consistency at the SAME physical dt. A frozen
    // equilibrium tangent alone can flush an arriving pulse through a quiet
    // downstream restriction hundreds of times faster than its orifice law.
    // Each Picard iterate is an SPD network solve, never a physical substep.
    unsigned total_pressure_used=0;
    bool flow_converged=false;
    bool strict_linear=false;
    for(int nonlinear=0;nonlinear<32;++nonlinear) {
    // Inexact early Picard solves are only predictors. Before accepting any
    // flux, solve the same last operator to the original strict tolerance and
    // recheck the nonlinear law; no physical substeps or relaxed final gate.
    const double pressure_tolerance=strict_linear?1e-6:1e-4;
    // Diagonally preconditioned CG on the symmetric compliance matrix.
    // Solve for pressure CORRECTIONS to avoid cancellation of large absolute
    // pressures in tiny cavities. Bounded algebraic work, no time refinement.
    double dot=0.;
    for(int i=tid;i<count;i+=stride){auto& v=a[i];const auto& n=old[i];
        const double capacity=n.solid_capacity+settings.cv*n.mass;
        v.compliance=n.volume/(settings.R*kg::temperature(n,settings)*(1+n.mass*settings.R/capacity));
        v.diagonal=v.compliance;v.residual=0.;
        if(nonlinear==0)v.correction=0.;
        for(int j=offsets[i];j<offsets[i+1];++j){const int e=incidence[j];const auto& edge=edges[e];
            const int other=edge.a==i?edge.b:edge.a;const double k=dt*flux[e].cap;
            v.diagonal+=k;v.residual-=k*(v.pressure-a[other].pressure);}
        v.direction=v.residual/v.diagonal;dot+=v.residual*v.direction;}
    grid.sync();
    // One-dimensional online reduced basis: fit the previous accepted spatial
    // correction to THIS matrix/rhs by an energy projection. It is only a CG
    // initial guess; all fluxes still require the unchanged true local residual.
    // Skip cheap solves, and discard a pattern that worsens the local norm.
    refresh_average_residual(grid,tid,stride,count,a,flux,offsets,incidence,edges,dt);
    if(nonlinear==0 && model.pressure_pattern_predictor && control->pressure_pattern_valid && control->averaged_pressure_used>=32) {
        const double zero_residual=average_pressure_residual(grid,tid,stride,count,a,shared,control);
        double numerator=0.,denominator=0.;
        for(int i=tid;i<count;i+=stride){const double v=a[i].pressure_pattern;
            numerator+=v*a[i].residual;denominator+=a[i].compliance*v*v;}
        for(int e=tid;e<edge_count;e+=stride){const auto& edge=edges[e];
            const double dv=a[edge.a].pressure_pattern-a[edge.b].pressure_pattern;
            denominator+=dt*flux[e].cap*dv*dv;}
        const double num=average_reduce_sum(grid,numerator,shared,control);
        const double den=average_reduce_sum(grid,denominator,shared,control);
        const double alpha=den>1e-290?num/den:0.;
        for(int i=tid;i<count;i+=stride)a[i].correction=isfinite(alpha)?alpha*a[i].pressure_pattern:0.;
        grid.sync();refresh_average_residual(grid,tid,stride,count,a,flux,offsets,incidence,edges,dt);
        const double predicted=average_pressure_residual(grid,tid,stride,count,a,shared,control);
        const bool accept=isfinite(predicted) && predicted<zero_residual;
        if(tid==0){++control->pressure_pattern_attempts;control->pressure_pattern_accepted+=accept;}
        if(!accept){for(int i=tid;i<count;i+=stride)a[i].correction=0.;
            grid.sync();refresh_average_residual(grid,tid,stride,count,a,flux,offsets,incidence,edges,dt);}
    }
    for(int i=tid;i<count;i+=stride){a[i].pressure_peer=-1;a[i].pressure_coupling=0.;
        pressure_match(i,a,flux,offsets,incidence,edges,dt);}
    grid.sync();dot=0.;
    for(int i=tid;i<count;i+=stride){a[i].direction=pressure_precondition(i,a);dot+=a[i].residual*a[i].direction;}
    double rho=average_reduce_sum(grid,dot,shared,control);const double initial_rho=rho;
    unsigned pressure_used=0;
    const bool predicted_converged=average_pressure_residual(grid,tid,stride,count,a,shared,control)<=pressure_tolerance;
    for(int iteration=0;iteration<model.averaged_pressure_iterations && rho>1e-290 && !predicted_converged;++iteration){
        dot=0.;
        for(int i=tid;i<count;i+=stride){auto& v=a[i];v.matvec=v.compliance*v.direction;
            for(int j=offsets[i];j<offsets[i+1];++j){const int e=incidence[j];const auto& edge=edges[e];const int other=edge.a==i?edge.b:edge.a;
                v.matvec+=dt*flux[e].cap*(v.direction-a[other].direction);}
            dot+=v.direction*v.matvec;}
        const double divisor=average_reduce_sum(grid,dot,shared,control);
        if(!(divisor>0.))break;
        const double alpha=rho/divisor;dot=0.;
        for(int i=tid;i<count;i+=stride){auto& v=a[i];v.correction+=alpha*v.direction;v.residual-=alpha*v.matvec;}
        grid.sync();
        for(int i=tid;i<count;i+=stride)dot+=a[i].residual*pressure_precondition(i,a);
        const double next_rho=average_reduce_sum(grid,dot,shared,control);const double beta=next_rho/rho;
        for(int i=tid;i<count;i+=stride){auto& v=a[i];v.direction=pressure_precondition(i,a)+beta*v.direction;}
        grid.sync();rho=next_rho;
        ++pressure_used;
        // Algebraic convergence, NOT physical subcycling. Do not silently use
        // the last iterate after exhausting a small fixed work budget.
        if(pressure_used%8==0 || rho<=initial_rho*1e-20){
            const double local=average_pressure_residual(grid,tid,stride,count,a,shared,control);
            // A fixed restart destroys accumulated conjugate directions on
            // a graded mesh. Replace only an apparently converged recursive
            // residual, then restart if the actual equation still disagrees.
            if(local<=pressure_tolerance) {
                refresh_average_residual(grid,tid,stride,count,a,flux,offsets,incidence,edges,dt);
                if(average_pressure_residual(grid,tid,stride,count,a,shared,control)<=pressure_tolerance)break;
                dot=0.;
                for(int i=tid;i<count;i+=stride){a[i].direction=pressure_precondition(i,a);dot+=a[i].residual*a[i].direction;}
                grid.sync();rho=average_reduce_sum(grid,dot,shared,control);
            }
        }
    }
    refresh_average_residual(grid,tid,stride,count,a,flux,offsets,incidence,edges,dt);
    const double residual=average_pressure_residual(grid,tid,stride,count,a,shared,control);
    total_pressure_used+=pressure_used;
    if(tid==0){control->averaged_pressure_used=total_pressure_used;control->averaged_pressure_residual=residual;}
    if(!(isfinite(residual) && residual<=pressure_tolerance)){
        if(tid==0)*error=24;grid.sync();return false;
    }
    // The predictor belongs to the first frozen operator, not the final
    // nonlinear one. Repeating that operator must still skip its CG solve.
    if(nonlinear==0){
        if(tid==0)control->averaged_first_linear_used=pressure_used;
        for(int i=tid;i<count;i+=stride)a[i].pressure_pattern=a[i].correction;
        grid.sync();
    }
    double defect=0.;
    for(int e=tid;e<edge_count;e+=stride){const auto& edge=edges[e];auto& p=flux[e];auto open=edge;
        open.area=channels[e].open?channels[e].effective_area_m2:0.;
        const double pa=a[edge.a].pressure+a[edge.a].correction;
        const double pb=a[edge.b].pressure+a[edge.b].correction;
        if(!(pa>0. && pb>0.)){defect=1e300;continue;}
        p.next_pressure_cap=pressure_mobility(old[edge.a],old[edge.b],pa,pb,open,settings);
        defect=fmax(defect,pressure_flow_defect(p.cap,p.next_pressure_cap,pa-pb,dt,old[edge.a].mass,old[edge.b].mass));
    }
    const double worst=-average_reduce_min(grid,-defect,shared,control);
    if(tid==0){control->averaged_nonlinear_used=nonlinear+1;control->averaged_flow_defect=worst;}
    if(isfinite(worst) && worst<=1.){
        if(strict_linear){flow_converged=true;break;}
        strict_linear=true;continue;
    }
    strict_linear=false;
    for(int e=tid;e<edge_count;e+=stride)flux[e].cap=flux[e].next_pressure_cap;
    grid.sync();
    }
    if(!flow_converged){if(tid==0)*error=33;grid.sync();return false;}
    for(int i=tid;i<count;i+=stride)a[i].pressure+=a[i].correction;
    grid.sync();
    if(tid==0){control->averaged_scale=1.;control->averaged_backtracks=0;}
    grid.sync();
    for(int attempt=0;attempt<8;++attempt){
        if(tid==0){control->averaged_invalid_mask[attempt]=0;
            control->averaged_first_invalid[attempt]=count;
            control->averaged_attempt_scale[attempt]=control->averaged_scale;}
        grid.sync();
        for(int e=tid;e<edge_count;e+=stride){const auto& edge=edges[e];auto& p=flux[e];
            const double mass=control->averaged_scale*dt*p.cap*(a[edge.a].pressure-a[edge.b].pressure);
            p.donor=mass>=0?edge.a:edge.b;p.mass=fabs(mass);p.energy=p.chemical=p.oxidizer=p.vapor=p.latent=0.;}
        for(int i=tid;i<count;i+=stride){a[i].mass=a[i].chemical=a[i].oxygen=a[i].vapor=a[i].latent=0.;}
        grid.sync();
        // Local conservative flow reconciliation. Start from ZERO availability
        // and include incoming gas, not just the resident inventory. Monotone
        // iterates guarantee out <= .95*old + incoming, even when stopped early.
        // A deficient node limits only its own outflow, never the whole graph.
        for(int iteration=0;iteration<model.averaged_payload_iterations;++iteration){
            for(int i=tid;i<count;i+=stride){auto v=a[i];double incoming=.95*old[i].mass,outgoing=0.;
                for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];
                    if(p.donor==i)outgoing+=p.mass;else incoming+=p.mass*a[p.donor].mass;}
                v.mass=outgoing>0.?fmin(1.,incoming/outgoing):1.;b[i]=v;}
            grid.sync();auto* swap=a;a=b;b=swap;
        }
        for(int e=tid;e<edge_count;e+=stride)flux[e].mass*=a[flux[e].donor].mass;
        grid.sync();
        for(int i=tid;i<count;i+=stride)a[i].mass=0.;
        grid.sync();
        // Implicit well-mixed transit. Iteration from zero is monotone, so
        // even an unfinished solve has nonnegative conservative remainders.
        // Transport the carrier too: payloads and the untracked inert remainder
        // then use the SAME positive operator and preserve constituent bounds.
        for(int iteration=0;iteration<model.averaged_payload_iterations;++iteration){
            for(int i=tid;i<count;i+=stride){
                auto v=a[i];double diagonal=old[i].mass;
                v.mass=old[i].mass;v.chemical=old[i].chemical;v.oxygen=old[i].oxidizer;v.vapor=old[i].vapor;v.latent=old[i].latent;
                for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];if(p.donor==i)continue;
                    const auto& d=a[p.donor];diagonal+=p.mass;
                    v.mass+=p.mass*d.mass;v.chemical+=p.mass*d.chemical;v.oxygen+=p.mass*d.oxygen;
                    v.vapor+=p.mass*d.vapor;v.latent+=p.mass*d.latent;}
                v.mass/=diagonal;v.chemical/=diagonal;v.oxygen/=diagonal;v.vapor/=diagonal;v.latent/=diagonal;b[i]=v;
            }
            grid.sync();auto* swap=a;a=b;b=swap;
        }
        for(int e=tid;e<edge_count;e+=stride){auto& p=flux[e];const auto& d=a[p.donor];const double mass=p.mass;
            p.mass=mass*d.mass;p.chemical=mass*d.chemical;p.oxidizer=mass*d.oxygen;p.vapor=mass*d.vapor;p.latent=mass*d.latent;}
        grid.sync();
        for(int i=tid;i<count;i+=stride){auto n=old[i];
            for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];const double sign=p.donor==i?-1.:1.;
                n.mass+=sign*p.mass;n.chemical+=sign*p.chemical;n.oxidizer+=sign*p.oxidizer;n.vapor+=sign*p.vapor;n.latent+=sign*p.latent;}
            next[i]=n;a[i].temperature=0.;}
        grid.sync();
        // Implicit upwind enthalpy, not cv*T packets. This retains pressure
        // work, positive sensible energy and heat exchange with an LTE matrix.
        for(int iteration=0;iteration<model.averaged_payload_iterations;++iteration){
            for(int i=tid;i<count;i+=stride){double diagonal=next[i].solid_capacity+settings.cv*next[i].mass,rhs=old[i].energy;
                for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];
                    if(p.donor==i)diagonal+=cp*p.mass;else rhs+=cp*p.mass*a[p.donor].temperature;}
                b[i]=a[i];b[i].temperature=rhs/diagonal;}
            grid.sync();auto* swap=a;a=b;b=swap;
        }
        for(int e=tid;e<edge_count;e+=stride)flux[e].energy=flux[e].mass*cp*a[flux[e].donor].temperature;
        grid.sync();
        double entropy=0.;bool valid=true;
        for(int i=tid;i<count;i+=stride){auto n=next[i];
            for(int j=offsets[i];j<offsets[i+1];++j){const auto& p=flux[incidence[j]];n.energy+=(p.donor==i?-1.:1.)*p.energy;}
            const unsigned mask=(!(n.mass>0.)?1u:0u) |
                (!(isfinite(n.energy) && n.energy>0.)?2u:0u) |
                (!(n.chemical>=0.)?4u:0u) | (!(n.oxidizer>=0.)?8u:0u) |
                (!(n.vapor>=0.)?16u:0u) | (!(n.oxidizer+n.vapor<=n.mass+1e-14)?32u:0u) |
                (!(n.latent>=0.)?64u:0u);
            valid=valid && mask==0;
            if(mask){atomicOr(&control->averaged_invalid_mask[attempt],mask);
                atomicMin(&control->averaged_first_invalid[attempt],i);}
            else entropy+=kg::entropy(n,settings)-kg::entropy(old[i],settings);
            next[i]=n;}
        shared[threadIdx.x]=entropy;__syncthreads();
        for(int w=blockDim.x/2;w;w/=2){if(threadIdx.x<w)shared[threadIdx.x]+=shared[threadIdx.x+w];__syncthreads();}
        if(threadIdx.x==0)control->entropy_by_block[blockIdx.x]=shared[0];
        const double all_valid=average_reduce_min(grid,valid?1.:0.,shared,control);
        if(tid==0){double sum=0.;for(int block=0;block<int(gridDim.x);++block)sum+=control->entropy_by_block[block];
            control->averaged_entropy[attempt]=sum;
            if(all_valid>0. && sum>=-1e-9)control->averaged_backtracks=attempt;
            else {control->averaged_scale*=.5;control->averaged_backtracks=8;}}
        grid.sync();
        if(control->averaged_backtracks!=8){
            // The unreactive transport entropy test above remains separate:
            // reaction heat must not conceal a bad conservative transfer.
            if(model.reactive_transit && !reacting_payload(grid,tid,stride,count,edge_count,
                offsets,incidence,edges,old,next,flux,settings,model,dt,a,b,shared,control,error,chemistry))return false;
            // Both ping-pong buffers carry the same accepted pressure pattern,
            // irrespective of odd payload iteration counts or reactive swaps.
            if(model.pressure_pattern_predictor){
                for(int i=tid;i<count;i+=stride)b[i].pressure_pattern=a[i].pressure_pattern;
                grid.sync();if(tid==0)control->pressure_pattern_valid=true;grid.sync();}
            if(tid==0){++control->averaged_windows;control->averaged_limited_windows+=control->averaged_scale<1.-1e-12;
                control->averaged_entropy_backtracks+=control->averaged_backtracks;
                control->averaged_min_scale=fmin(control->averaged_min_scale,control->averaged_scale);}
            for(int e=tid;e<edge_count;e+=stride){const auto& p=flux[e];auto& c=channels[e];const double sign=p.donor==edges[e].a?1.:-1.;
                c.transferred_kg+=sign*p.mass;c.transferred_energy_j+=sign*(p.energy-p.mass*settings.cv*model.reference_temperature_k);
                c.transferred_chemical_j+=sign*p.chemical;c.transferred_oxidizer_kg+=sign*p.oxidizer;
                c.transferred_vapor_kg+=sign*p.vapor;c.transferred_latent_j+=sign*p.latent;c.mass_flow_kg_s+=sign*p.mass/duration;}
            grid.sync();return true;
        }
    }
    if(tid==0)*error=23;grid.sync();return false;
}

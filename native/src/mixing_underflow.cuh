#pragma once
// Rare, conservative resolution of subnormal tracer cancellation. No physical
// tolerance, chemistry change, inventory clipping, or new time substep.
__host__ __device__ inline double mixing_underflow_factor(double result,double available,double outgoing) {
    if(!(result<0. && result>=-DBL_MIN && available>=0. && outgoing>0. &&
        isfinite(available) && isfinite(outgoing)))return 1.;
    return fmin(1.,available/outgoing)*(1.-32.*DBL_EPSILON);
}
__device__ inline bool mix_repair_underflow(cooperative_groups::grid_group grid,int tid,int stride,
    int gas_count,int edge_count,MixSolveState* work,const int* offsets,const int* incidence,
    const kg::Edge* edges,const kg::Node* nodes,Proposal* flux,double* shared,Control* control) {
    double changed=0.;
    for(int i=tid;i<gas_count;i+=stride) {
        const auto& n=nodes[i];const double initial[4]={n.chemical,n.oxidizer,n.vapor,n.latent};
        double result[4],incoming[4]{},outgoing[4]{};
        for(int f=0;f<4;++f)result[f]=initial[f];
        for(int j=offsets[i];j<offsets[i+1];++j) {
            const int e=incidence[j];const auto& p=flux[e].mixing;
            const double sign=edges[e].a==i?-1.:1.;
            const double q[4]={p.chemical,p.oxidizer,p.vapor,p.latent};
            for(int f=0;f<4;++f) {
                const double dq=sign*q[f];result[f]+=dq;
                if(dq>0.)incoming[f]+=dq;else outgoing[f]-=dq;
            }
        }
        for(int f=0;f<4;++f) {
            const double factor=mixing_underflow_factor(result[f],initial[f]+incoming[f],outgoing[f]);
            work[i].ad[f+1]=factor;if(factor<1.)changed=1.;
        }
    }
    if(average_reduce_sum(grid,changed,shared,control)==0.)return false;
    for(int e=tid;e<edge_count;e+=stride) {
        const auto& edge=edges[e];auto& p=flux[e].mixing;
        double* q[4]={&p.chemical,&p.oxidizer,&p.vapor,&p.latent};
        for(int f=0;f<4;++f) {
            const int donor=*q[f]>=0.?edge.a:edge.b;const double factor=work[donor].ad[f+1];
            // nextafter is essential: multiplication alone cannot reduce a
            // subnormal by less than one representable unit.
            if(factor<1.)*q[f]=nextafter(*q[f]*factor,0.);
        }
    }
    grid.sync();return true;
}

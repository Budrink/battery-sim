#pragma once
// Disjoint 2x2 principal blocks of the frozen SPD pressure matrix. Matching is
// numerical only: no physical edges, flows, tensor averages or time steps change.
__device__ inline double pressure_precondition(int i,const AverageState* a) {
    const auto& v=a[i];const int p=v.pressure_peer;
    if(p<0 || a[p].pressure_peer!=i)return v.residual/v.diagonal;
    const double k=v.pressure_coupling;
    const double det=(v.diagonal-k)*a[p].diagonal+k*(a[p].diagonal-k);
    return (a[p].diagonal*v.residual+k*a[p].residual)/det;
}
__device__ inline void pressure_match(int i,AverageState* a,const Proposal* flux,
    const int* offsets,const int* incidence,const kg::Edge* edges,double dt) {
    double best=0.;int chosen=-1;
    for(int j=offsets[i];j<offsets[i+1];++j){const int e=incidence[j];const auto& edge=edges[e];
        const int p=edge.a==i?edge.b:edge.a;const double k=dt*flux[e].cap;
        const double strength=k*k/(a[i].diagonal*a[p].diagonal);
        if(strength>best || (strength==best && strength>0. && e<chosen)){
            best=strength;chosen=e;a[i].pressure_peer=p;a[i].pressure_coupling=k;}}
}

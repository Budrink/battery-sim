// Included inside cuda_solver.cu's device namespace. One fixed 128-thread block
// owns each independent electrical object: deterministic reductions, no atomics,
// no host iterations, no reads of another block's partial state.
template<bool Maximum=false> __device__ double electrical_reduce(double value) {
    __shared__ double partial[128];partial[threadIdx.x]=value;__syncthreads();
    for(int stride=64;stride;stride/=2) {
        if(threadIdx.x<stride) {
            if constexpr(Maximum)partial[threadIdx.x]=fmax(partial[threadIdx.x],partial[threadIdx.x+stride]);
            else partial[threadIdx.x]+=partial[threadIdx.x+stride];
        }
        __syncthreads();
    }
    const double result=partial[0];__syncthreads();return result;
}
__device__ double electrical_apply(int i,const double* x,DeviceRows rows,const ElectricalEdge* edges,const double* g) {
    double value=0.,diagonal=0.;
    for(int k=rows.offsets[i];k<rows.offsets[i+1];++k) {
        const int j=rows.ids[k];const auto& e=edges[j];const int other=e.a==i?e.b:e.a;
        value+=g[j]*(x[i]-(other>=0?x[other]:0.));diagonal+=g[j];
    }
    return diagonal>0.?value:x[i]; // pin isolated zero-conductivity nodes, not a hidden conducting floor
}
__global__ void electrical_kernel(int count,const int* objects,const int* object_offsets,const int* edge_offsets,
    const DeviceVolume* volumes,const Material* materials,const ThermalState* solid,
    DeviceRows rows,const ElectricalEdge* edges,double* conductance,double* sigma,double* diagonal,
    double* residual,double* z,double* direction,double* applied,const double* committed_potential,double* potential,
    double* heat_per_i2,const ElectricalState* committed,ElectricalState* diagnostics,double time,double horizon) {
    const int group=blockIdx.x;if(group>=count)return;
    const int object=objects[group],begin=object_offsets[object],end=object_offsets[object+1],tid=threadIdx.x;
    const auto& m=materials[object];const auto& law=m.electrical;
    const double maximum=maximum_current(law,m.current,time,time+horizon);
    const double current=current_at(law,m.current,time);
    ElectricalState report;report.time_s=time;report.current_a=current;
    if(law.voltage_limited_current) {
        report.positive_charge_c=committed[object].positive_charge_c;
        report.terminal_work_j=committed[object].terminal_work_j;
        report.voltage_limited_time_s=committed[object].voltage_limited_time_s;
    }
    if(maximum==0.) {
        for(int i=begin+tid;i<end;i+=128) {potential[i]=committed_potential[i];heat_per_i2[i]=0.;}
        if(tid==0)diagnostics[object]=report;return;
    }
    double bad=0.;
    for(int i=begin+tid;i<end;i+=128) {
        const auto& s=solid[i];const double fraction=volumes[i].capacity_j_per_k/m.capacity;
        sigma[i]=electrical_mixture_conductivity(law,m.conversion,s.temperature_k,s.reactant_kg,
            s.activated_kg,s.residue_kg,m.conversion.mass_kg*fraction,s.healthy_charged_kg);
        if(!isfinite(sigma[i]) || sigma[i]<0.) {bad=1.;sigma[i]=0.;}
        potential[i]=committed_potential[i];heat_per_i2[i]=0.;
    }
    bad=electrical_reduce<true>(bad);
    for(int j=edge_offsets[group]+tid;j<edge_offsets[group+1];j+=128) {
        const auto& e=edges[j];
        conductance[j]=sigma[e.a]>0. && (e.b<0 || sigma[e.b]>0.)?
            1./(e.half_a/sigma[e.a]+(e.b>=0?e.half_b/sigma[e.b]:0.)):0.;
    }
    __syncthreads();
    double b2=0.,r2=0.;
    for(int i=begin+tid;i<end;i+=128) {
        double d=0.,rhs=0.;
        for(int k=rows.offsets[i];k<rows.offsets[i+1];++k) {
            const int j=rows.ids[k];const auto& e=edges[j];d+=conductance[j];
            if(e.b<0)rhs+=conductance[j]*e.terminal_potential;
        }
        diagonal[i]=d>0.?d:1.;
        residual[i]=rhs-electrical_apply(i,potential,rows,edges,conductance);
        z[i]=residual[i]/diagonal[i];direction[i]=z[i];
        b2+=rhs*rhs;r2+=residual[i]*residual[i];
    }
    b2=electrical_reduce(b2);r2=electrical_reduce(r2);
    const double target=fmax(1e-300,b2*1e-24); // relative linear residual 1e-12
    double rz=0.;for(int i=begin+tid;i<end;i+=128)rz+=residual[i]*z[i];
    rz=electrical_reduce(rz);
    int iterations=0;
    for(;r2>target && iterations<4*(end-begin)+32;++iterations) {
        double denominator=0.;
        for(int i=begin+tid;i<end;i+=128) {
            applied[i]=electrical_apply(i,direction,rows,edges,conductance);denominator+=direction[i]*applied[i];
        }
        denominator=electrical_reduce(denominator);
        if(!(denominator>0.) || !isfinite(denominator)) {bad=1.;break;}
        const double alpha=rz/denominator;r2=0.;double next_rz=0.;
        for(int i=begin+tid;i<end;i+=128) {
            potential[i]+=alpha*direction[i];residual[i]-=alpha*applied[i];
            z[i]=residual[i]/diagonal[i];r2+=residual[i]*residual[i];next_rz+=residual[i]*z[i];
        }
        r2=electrical_reduce(r2);next_rz=electrical_reduce(next_rz);
        const double beta=next_rz/rz;
        for(int i=begin+tid;i<end;i+=128)direction[i]=z[i]+beta*direction[i];
        __syncthreads();rz=next_rz;
    }
    report.iterations=iterations;
    double inlet=0.,outlet=0.,power=0.,max_error=0.;
    for(int i=begin+tid;i<end;i+=128) {
        double q=0.,rhs=0.;
        for(int k=rows.offsets[i];k<rows.offsets[i+1];++k) {
            const int j=rows.ids[k];const auto& e=edges[j];const int other=e.a==i?e.b:e.a;
            const double delta=potential[i]-(other>=0?potential[other]:e.terminal_potential);
            const double flow=conductance[j]*delta;
            if(conductance[j]>0.)q+=flow*flow*(e.a==i?e.half_a:e.half_b)/sigma[i];
            if(e.b<0) {
                rhs+=conductance[j]*e.terminal_potential;
                if(e.terminal_potential==1.)inlet-=flow;else outlet+=flow;
            }
        }
        heat_per_i2[i]=q;power+=q;
        max_error=fmax(max_error,fabs(electrical_apply(i,potential,rows,edges,conductance)-rhs));
    }
    inlet=electrical_reduce(inlet);outlet=electrical_reduce(outlet);power=electrical_reduce(power);
    max_error=electrical_reduce<true>(max_error);
    report.kcl_relative_residual=max_error/fmax(inlet,1e-300);
    report.power_relative_residual=fabs(power-inlet)/fmax(inlet,1e-300);
    if(bad || r2>target || !(inlet>0.) || !isfinite(inlet))report.error=1; // open circuit or unconverged solve
    else {
        report.resistance_ohm=1./inlet;
        const double actual=law.voltage_limited_current?
            compliant_current(law,current,sample_drive(m.voltage,time),report.resistance_ohm):current;
        report.current_a=actual;report.ohmic_drop_v=actual/inlet;report.power_w=actual*actual/inlet;
        report.outlet_current_a=actual*outlet/inlet;
        if(report.kcl_relative_residual>1e-9 || report.power_relative_residual>1e-9 ||
            fabs(outlet/inlet-1.)>1e-9)report.error=2;
        if(report.resistance_ohm<law.min_resistance_ohm*(1.-1e-9) ||
            report.resistance_ohm>law.max_resistance_ohm*(1.+1e-9))report.error=3;
    }
    for(int i=begin+tid;i<end;i+=128)heat_per_i2[i]=report.error?0.:heat_per_i2[i]/(inlet*inlet);
    if(tid==0)diagnostics[object]=report;
}
// One thread per macro-object, once per accepted candidate interval, NOT once
// per voxel. Ordered before all chemistry; a retry reads the committed ledger.
__global__ void electrical_interval_kernel(int count,const int* objects,const ElectricalState* committed,
    ElectricalState* candidate,const Material* materials,const double* step,double time) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const int object=objects[i];const auto& m=materials[object];
    if(!m.electrical.voltage_limited_current)return;
    auto& e=candidate[object];const auto& old=committed[object];
    const auto interval=compliant_interval(m.electrical,m.current,m.voltage,e.resistance_ohm,time,*step);
    e.step_charge_c=interval.charge_c;e.step_i2_a2_s=interval.i2_a2_s;e.step_work_j=interval.work_j;
    e.positive_charge_c=old.positive_charge_c+interval.charge_c;
    e.terminal_work_j=old.terminal_work_j+interval.work_j;
    e.voltage_limited_time_s=old.voltage_limited_time_s+interval.limited_time_s;
}
__global__ void electrical_validate_kernel(int count,const int* objects,const ElectricalState* states,const Material* materials,
    const double* step,double time,StepResult* result) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const int object=objects[i];const auto& e=states[object];const auto& m=materials[object];
    if(e.error)atomicOr(&result->invalid,128u);
    if(m.electrical.voltage_limited_current &&
        (!isfinite(e.positive_charge_c) || e.positive_charge_c<0. || !isfinite(e.terminal_work_j) || e.terminal_work_j<0. ||
         !isfinite(e.step_charge_c) || e.step_charge_c<0. || !isfinite(e.step_i2_a2_s) || e.step_i2_a2_s<0. ||
         !isfinite(e.step_work_j) || e.step_work_j<0.))atomicOr(&result->invalid,128u);
    if(m.voltage.count && minimum_current(m.electrical,m.current,time,time+*step)>=0.) {
        const bool limited=m.electrical.voltage_limited_current;
        const double work=limited?e.step_work_j:electrical_work_integral(m.electrical,m.current,m.voltage,time,*step);
        const double budget=limited?work-m.electrical.reversible_voltage_floor_v*e.step_charge_c:
            electrical_dissipative_budget(m.electrical,m.current,m.voltage,time,*step);
        const double heat=e.resistance_ohm*(limited?e.step_i2_a2_s:squared_current_integral(m.electrical,m.current,time,*step));
        if(heat>budget+1e-9+1e-9*fabs(work))atomicOr(&result->invalid,256u);
    }
}

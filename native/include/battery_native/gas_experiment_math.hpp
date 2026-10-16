#pragma once
#include "battery_native/gas_experiment.hpp"
#include "battery_native/conversion.hpp"
#include "battery_native/gas_timestep.hpp"
#ifdef __CUDACC__
#define GAS_HD __host__ __device__
#else
#define GAS_HD
#endif
namespace battery::gas_experiment {
struct Proposal {double rate=0,cap=0,pressure_dt=1e300;};
struct Budget {double outgoing=0,power=0,pressure_dt=1e300;};
GAS_HD inline Proposal propose(const GasState& a,const GasState& b,int da,int db,
    const Edge& e,const GasModel& model) {
    Proposal q;
    const bool forward=a.pressure_pa>=b.pressure_pa;
    const auto& donor=forward?a:b;
    q.rate=(forward?1.:-1.)*gas_orifice_flow(fmax(a.pressure_pa,b.pressure_pa),fmin(a.pressure_pa,b.pressure_pa),
        donor.temperature_k,e.area,e.cd,model);
    if(q.rate==0)return q;
    const double slope=(1+model.specific_r_j_per_kg_k/model.cv_j_per_kg_k)*model.specific_r_j_per_kg_k*donor.temperature_k*
        (da/a.free_volume_m3+db/b.free_volume_m3);
    q.cap=fabs(a.pressure_pa-b.pressure_pa)/slope;
    q.pressure_dt=gas_pressure_dt(a.pressure_pa,b.pressure_pa,q.rate,q.cap,model.pressure_relaxation_fraction,
        model.pressure_absolute_tolerance_pa,model.pressure_relative_tolerance);
    return q;
}
GAS_HD inline void add_budget(Budget& budget,const GasState& node,const GasState& donor,bool outgoing,
    const Proposal& q,const GasModel& model) {
    if(q.rate==0)return;
    if(outgoing)budget.outgoing+=fabs(q.rate);
    const double coefficient=outgoing?model.specific_r_j_per_kg_k*node.temperature_k:
        fabs(gas_enthalpy(donor,model)-model.cv_j_per_kg_k*(node.temperature_k-model.reference_temperature_k));
    budget.power+=fabs(q.rate)*coefficient;
    budget.pressure_dt=fmin(budget.pressure_dt,q.pressure_dt);
}
GAS_HD inline double limit(const Budget& b,const GasState& s,const GasModel& model) {
    double dt=b.pressure_dt;
    if(b.outgoing>0)dt=fmin(dt,model.max_fraction_per_step*s.mass_kg/b.outgoing);
    if(b.power>0)dt=fmin(dt,fmin(model.max_temperature_step_k,.05*s.temperature_k)*s.mass_kg*model.cv_j_per_kg_k/b.power);
    return dt;
}
GAS_HD inline Packet transfer(const Proposal& q,const GasState& a,const GasState& b,double dt,const GasModel& model) {
    const double dm=copysign(fmin(fabs(q.rate)*dt,q.cap),q.rate);
    const auto& donor=dm>=0?a:b;
    return {dm,dm*gas_enthalpy(donor,model),dm*donor.chemical_j/donor.mass_kg,dm*donor.oxidizer_kg/donor.mass_kg};
}
GAS_HD inline void accumulate(Packet& total,const Packet& q) {
    total.mass+=q.mass;total.energy+=q.energy;total.chemical+=q.chemical;total.oxidizer+=q.oxidizer;
}
GAS_HD inline void apply(GasState& state,const Packet& q,double sign) {
    state.mass_kg+=sign*q.mass;state.energy_j+=sign*q.energy;
    state.chemical_j+=sign*q.chemical;state.oxidizer_kg+=sign*q.oxidizer;
}
GAS_HD inline void refresh(GasState& state,const GasModel& model) {
    state.temperature_k=model.reference_temperature_k+state.energy_j/(state.mass_kg*model.cv_j_per_kg_k);
    state.pressure_pa=state.mass_kg*model.specific_r_j_per_kg_k*state.temperature_k/state.free_volume_m3;
}
GAS_HD inline bool valid(const GasState& s) {
    return s.mass_kg>0 && s.temperature_k>0 && s.pressure_pa>0 &&
        s.mass_kg<1e290 && s.temperature_k<1e290 && s.chemical_j>=-1e-12 && s.oxidizer_kg>=-1e-15;
}
} // namespace battery::gas_experiment
#undef GAS_HD

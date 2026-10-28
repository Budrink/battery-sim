#include "battery_native/simulation.hpp"
#include "battery_native/checkpoint_io.hpp"
#include <cmath>
#include <utility>

namespace battery {
void Simulation::rebase_unvented_from(const Simulation& source,const std::vector<GasRemap>& map) {
    if(source.time_s_>scenario_.settings.duration_s || cells_.size()!=source.cells_.size() ||
       interfaces_.size()!=source.interfaces_.size())throw std::runtime_error("Incompatible unvented rebase");
    if(scenario_.heating_protocols.size()!=source.scenario_.heating_protocols.size())
        throw std::runtime_error("Incompatible heating protocol rebase");
    gpu_->rebase_unvented_from(*source.gpu_,map);
    runtime_=source.runtime_;time_s_=source.time_s_;next_history_s_=source.next_history_s_;next_failure_=source.next_failure_;
    step_count_=source.step_count_;outer_step_count_=source.outer_step_count_;last_substep_count_=source.last_substep_count_;
    last_dt_s_=source.last_dt_s_;last_interval_s_=source.last_interval_s_;
    initial_energy_j_=source.initial_energy_j_;boundary_energy_out_j_=source.boundary_energy_out_j_;
    events_=source.events_;runaway_order_=source.runaway_order_;history_=source.history_;
    heating_states_=source.heating_states_;
    gpu_->download_macros(cells_,interfaces_);std::vector<GasState> gas;gpu_->download_gas_states(gas,gas_channels_);
    field_cache_.clear();field_dirty_=true;
}
std::string Simulation::save_checkpoint(const std::string& contract) const {
    if(contract.empty())throw std::invalid_argument("Restart physics contract is required");
    checkpoint::Writer w;w.text("battery-simulation-v1");w.text(contract);
    w.pod(runtime_);w.pod(time_s_);w.pod(next_history_s_);w.pod(next_failure_);
    w.pod(step_count_);w.pod(outer_step_count_);w.pod(last_substep_count_);
    w.pod(last_dt_s_);w.pod(last_interval_s_);w.pod(initial_energy_j_);w.pod(boundary_energy_out_j_);
    w.pod(std::uint64_t(events_.size()));
    for(const auto& e:events_){w.pod(e.time_s);w.text(e.kind);w.text(e.entity_id);}
    w.vector(runaway_order_);w.vector(history_.time_s);
    w.rows(history_.temperature_k);w.rows(history_.cell_damage);
    w.rows(history_.conductance_w_per_k);w.rows(history_.edge_l_ratio);
    if(!scenario_.heating_protocols.empty())w.vector(heating_states_);
    w.text(gpu_->save_state());return w.finish();
}
void Simulation::restore_checkpoint(const std::string& bytes,const std::string& contract) {
    checkpoint::Reader r(bytes);
    if(r.text()!="battery-simulation-v1" || r.text()!=contract)
        throw std::runtime_error("Restart physics/topology contract mismatch");
    const auto runtime=r.pod<RuntimeParameters>();const double time=r.pod<double>(),next_history=r.pod<double>();
    const auto next_failure=r.pod<std::size_t>(),steps=r.pod<std::size_t>();
    const auto outer=r.pod<std::size_t>(),substeps=r.pod<std::size_t>();
    const double last_dt=r.pod<double>(),interval=r.pod<double>(),initial=r.pod<double>(),boundary=r.pod<double>();
    const auto event_count=r.pod<std::uint64_t>();std::vector<EventRecord> events;
    for(std::uint64_t i=0;i<event_count;++i){const double at=r.pod<double>();auto kind=r.text(),id=r.text();events.push_back({at,kind,id});}
    auto order=r.vector<std::size_t>();History history;history.time_s=r.vector<float>();
    history.temperature_k=r.rows<float>(cells_.size());history.cell_damage=r.rows<float>(cells_.size());
    history.conductance_w_per_k=r.rows<float>(interfaces_.size());history.edge_l_ratio=r.rows<float>(interfaces_.size());
    std::vector<HeatingProtocolState> heating;
    if(!scenario_.heating_protocols.empty()) {
        heating=r.vector<HeatingProtocolState>(scenario_.heating_protocols.size());
        if(heating.size()!=scenario_.heating_protocols.size())throw std::runtime_error("Heating restart count");
        for(const auto& h:heating)if(!std::isfinite(h.previous_time_s)||h.previous_time_s>time ||
            !std::isfinite(h.previous_temperature_k)||h.previous_temperature_k<=0.||
            !std::isfinite(h.commanded_power_w)||h.commanded_power_w<0.||
            !std::isfinite(h.cutoff_time_s)||h.cutoff_time_s>time)
            throw std::runtime_error("Invalid heating restart state");
    }
    const auto device=r.text();r.finish();
    if(!std::isfinite(time) || time<0. || time>scenario_.settings.duration_s ||
       !std::isfinite(next_history) || next_history<=time || next_failure>scenario_.primary_failures.size() ||
       !std::isfinite(initial) || !std::isfinite(boundary) || !std::isfinite(runtime.max_dt_s) || runtime.max_dt_s<=0.)
        throw std::runtime_error("Invalid restart time/ledger");
    gpu_->restore_state(device);
    runtime_=runtime;time_s_=time;next_history_s_=next_history;next_failure_=next_failure;
    step_count_=steps;outer_step_count_=outer;last_substep_count_=substeps;
    last_dt_s_=last_dt;last_interval_s_=interval;initial_energy_j_=initial;boundary_energy_out_j_=boundary;
    events_=std::move(events);runaway_order_=std::move(order);history_=std::move(history);
    heating_states_=std::move(heating);
    gpu_->download_macros(cells_,interfaces_);std::vector<GasState> gas;gpu_->download_gas_states(gas,gas_channels_);
    field_cache_.clear();field_dirty_=true;
}
}

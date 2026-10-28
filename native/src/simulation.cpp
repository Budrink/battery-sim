#include "battery_native/simulation.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace battery {
// Host orchestration only. Thermal evolution and macro reductions are CUDA kernels.
Simulation::Simulation(Scenario scenario):scenario_(std::move(scenario)),
    mesh_(build_thermal_mesh(scenario_)),gpu_(std::make_unique<CudaSolver>(scenario_,mesh_)) {
    runtime_.max_dt_s=scenario_.settings.max_dt_s;reset();
}
Simulation::~Simulation()=default;
void Simulation::reset() {
    gpu_->reset();time_s_=0;step_count_=0;last_dt_s_=0;next_failure_=0;
    outer_step_count_=0;last_substep_count_=0;last_interval_s_=0;
    next_history_s_=scenario_.settings.output_interval_s;
    events_.clear();runaway_order_.clear();history_={};
    gas_channels_.assign(scenario_.gas_channels.size(),{});
    history_.temperature_k.resize(scenario_.cells.size());history_.cell_damage.resize(scenario_.cells.size());
    history_.conductance_w_per_k.resize(scenario_.interfaces.size());history_.edge_l_ratio.resize(scenario_.interfaces.size());
    primary_cells_.assign(scenario_.cells.size(),false);
    for(const auto& event:scenario_.primary_failures)primary_cells_[event.cell_index]=true;
    gpu_->download_macros(cells_,interfaces_);
    heating_states_.clear();
    for(const auto& protocol:scenario_.heating_protocols) {
        validate_heating_protocol(protocol);
        if(protocol.heater_object<0 || protocol.sensor_object<0 ||
           protocol.heater_object>=int(cells_.size()) || protocol.sensor_object>=int(cells_.size()))
            throw std::invalid_argument("Unknown heating protocol object");
        HeatingProtocolState state;state.previous_temperature_k=cells_[protocol.sensor_object].temperature_k;
        heating_states_.push_back(state);
        gpu_->set_resistive_heater_power(protocol.heater_object,0.);
    }
    initial_energy_j_=stored_energy_j();boundary_energy_out_j_=0;
    auto before=cells_;
    for(auto& c:before)c.runaway_started_s=-1;
    apply_scheduled();gpu_->download_macros(cells_,interfaces_);
    collect_events(before,interfaces_);field_dirty_=true;record_history();
}
void Simulation::apply_scheduled() {
    while(next_failure_<scenario_.primary_failures.size() && scenario_.primary_failures[next_failure_].time_s<=time_s_+1e-10) {
        const auto& e=scenario_.primary_failures[next_failure_++];
        gpu_->ignite(e.cell_index,e.thermal_volume,e.time_s);
    }
}
void Simulation::collect_events(const std::vector<CellState>& old,const std::vector<InterfaceState>& old_edges) {
    for(std::size_t i=0;i<cells_.size();++i) {
        const auto& s=cells_[i];
        if(scenario_.cells[i].is_battery && s.runaway_started_s>=0 && old[i].runaway_started_s<0) {
            runaway_order_.push_back(i);
            bool prescribed=false;
            for(const auto& e:scenario_.primary_failures)
                prescribed=prescribed||(e.cell_index==i && std::abs(e.time_s-s.runaway_started_s)<1e-9);
            events_.push_back({s.runaway_started_s,prescribed?"runaway_primary":"runaway_secondary",scenario_.cells[i].id});
        }
        if(s.mode==CellMode::Spent && old[i].mode!=CellMode::Spent)
            events_.push_back({time_s_,scenario_.cells[i].is_battery?"runaway_complete":"material_destroyed",scenario_.cells[i].id});
        else if(s.spent_fraction>old[i].spent_fraction+1e-12)
            events_.push_back({time_s_,"thermal_volume_destroyed",scenario_.cells[i].id});
        if(s.mode==CellMode::Quenched && old[i].mode!=CellMode::Quenched)
            events_.push_back({time_s_,"reaction_quenched",scenario_.cells[i].id});
        if(s.mode==CellMode::Runaway && old[i].mode==CellMode::Quenched)
            events_.push_back({time_s_,"reaction_resumed",scenario_.cells[i].id});
    }
    for(std::size_t i=0;i<interfaces_.size();++i) {
        const auto& s=interfaces_[i];
        if(s.destroyed_fraction>old_edges[i].destroyed_fraction+1e-12)
            events_.push_back({time_s_,"interface_destroyed",scenario_.interfaces[i].id});
        else if(s.phase!=old_edges[i].phase)
            events_.push_back({time_s_,std::string("phase_")+name(old_edges[i].phase)+"_to_"+name(s.phase),scenario_.interfaces[i].id});
    }
}
bool Simulation::finished() const {return time_s_>=scenario_.settings.duration_s-1e-10;}
bool Simulation::step(double dt_cap_s) {
    if(finished())return false;
    if(std::isnan(dt_cap_s)||dt_cap_s<=0)throw std::runtime_error("Invalid substep cap");
    for(double x:{runtime_.max_dt_s,runtime_.interface_conductance_scale,runtime_.ambient_conductance_scale,
                  runtime_.damage_rate_scale,runtime_.runaway_power_scale})
        if(!std::isfinite(x)||x<0)throw std::runtime_error("Invalid CUDA runtime parameter");
    if(runtime_.max_dt_s==0)throw std::runtime_error("max_dt_s must be positive");
    if(scenario_.settings.minimum_dt_s>0. && runtime_.max_dt_s<scenario_.settings.minimum_dt_s)
        throw std::runtime_error("Runtime max_dt_s cannot be below minimum_dt_s");
    double cap=std::min({dt_cap_s,runtime_.max_dt_s,scenario_.settings.duration_s-time_s_,next_history_s_-time_s_});
    if(next_failure_<scenario_.primary_failures.size())cap=std::min(cap,scenario_.primary_failures[next_failure_].time_s-time_s_);
    if(cap<=0)throw std::runtime_error("CUDA host clock collapsed");
    auto old=cells_;auto old_edges=interfaces_;
    auto proposed_heating=heating_states_;
    for(std::size_t i=0;i<proposed_heating.size();++i) {
        const auto& protocol=scenario_.heating_protocols[i];
        proposed_heating[i]=heating_command(protocol,heating_states_[i],time_s_,cells_[protocol.sensor_object].temperature_k);
        gpu_->set_resistive_heater_power(protocol.heater_object,proposed_heating[i].commanded_power_w);
    }
    last_dt_s_=gpu_->step(time_s_,cap,scenario_.settings.stability_factor,runtime_);
    if(!std::isfinite(last_dt_s_)||last_dt_s_<=0 || time_s_+last_dt_s_==time_s_)
        throw std::runtime_error("CUDA timestep collapsed");
    time_s_+=last_dt_s_;++step_count_;
    for(std::size_t i=0;i<proposed_heating.size();++i)
        if(proposed_heating[i].stopped && !heating_states_[i].stopped)
            events_.push_back({proposed_heating[i].cutoff_time_s,"heater_cutoff",scenario_.heating_protocols[i].id});
    heating_states_=std::move(proposed_heating);
    apply_scheduled();gpu_->download_macros(cells_,interfaces_);field_dirty_=true;
    if(!scenario_.gas_compartments.empty()) {
        std::vector<GasState> gas;std::vector<GasChannelState> channels;gpu_->download_gas_states(gas,channels);
        for(const auto& g:gas)if(!std::isfinite(g.temperature_k)||g.temperature_k<=0 || !std::isfinite(g.pressure_pa)||g.mass_kg<=0 || g.chemical_j< -1e-10)
            throw std::runtime_error("Nonphysical gas state");
        for(std::size_t i=0;i<channels.size();++i)if(channels[i].open && !gas_channels_[i].open)
            events_.push_back({channels[i].opened_s,scenario_.gas_channels[i].rupture?"shell_rupture":"vent_open",scenario_.gas_channels[i].id});
        gas_channels_=channels;
    }
    boundary_energy_out_j_=0;
    for(const auto& c:cells_) {
        if(!std::isfinite(c.temperature_k)||c.min_temperature_k<=0)throw std::runtime_error("Nonphysical CUDA temperature");
        boundary_energy_out_j_+=c.boundary_energy_out_j;
    }
    collect_events(old,old_edges);
    if(next_history_s_<=time_s_+1e-10) {record_history();next_history_s_+=scenario_.settings.output_interval_s;}
    return true;
}
bool Simulation::advance(const std::function<bool()>& stop_after_substep) {
    last_substep_count_=0;last_interval_s_=0;
    if(finished())return false;
    const double interval=scenario_.settings.step_interval_s>0?scenario_.settings.step_interval_s:runtime_.max_dt_s;
    if(!std::isfinite(interval)||interval<=0)throw std::runtime_error("Invalid outer step interval");
    const double begin=time_s_,end=std::min(begin+interval,scenario_.settings.duration_s);
    if(end<=begin)throw std::runtime_error("Outer step clock collapsed");
    const auto before=step_count_;
    while(!finished() && time_s_<end) {
        if(!step(end-time_s_))break;
        if(stop_after_substep && stop_after_substep())break;
    }
    last_substep_count_=step_count_-before;last_interval_s_=time_s_-begin;
    if(last_substep_count_==0)throw std::runtime_error("Outer step made no progress");
    ++outer_step_count_;return true;
}
void Simulation::run_to_end() {while(advance()) {}}
std::size_t Simulation::battery_count() const {
    return std::count_if(scenario_.cells.begin(),scenario_.cells.end(),[](const auto& c){return c.is_battery;});
}
bool Simulation::all_batteries_destroyed() const {
    bool found=false;
    for(std::size_t i=0;i<cells_.size();++i)if(scenario_.cells[i].is_battery) {
        found=true;if(cells_[i].mode!=CellMode::Spent)return false;
    }
    return found;
}
const std::vector<ThermalState>& Simulation::thermal_states() const {
    if(field_dirty_) {gpu_->download_field(field_cache_);field_dirty_=false;}
    return field_cache_;
}
std::vector<InterfaceState> Simulation::thermal_link_states() const {
    std::vector<InterfaceState> result;gpu_->download_link_states(result);return result;
}
void Simulation::record_history() {
    history_.time_s.push_back(static_cast<float>(time_s_));
    for(std::size_t i=0;i<cells_.size();++i) {
        history_.temperature_k[i].push_back(static_cast<float>(cells_[i].temperature_k));
        history_.cell_damage[i].push_back(static_cast<float>(cells_[i].damage));
    }
    for(std::size_t i=0;i<interfaces_.size();++i) {
        history_.conductance_w_per_k[i].push_back(static_cast<float>(interfaces_[i].conductance_w_per_k));
        history_.edge_l_ratio[i].push_back(static_cast<float>(interfaces_[i].l_m/std::max(interfaces_[i].l0_m,1e-12)));
    }
}
bool Simulation::propagated() const {
    return std::any_of(runaway_order_.begin(),runaway_order_.end(),[&](std::size_t i){return !primary_cells_[i];});
}
double Simulation::max_temperature_k() const {
    double value=0;for(const auto& c:cells_)value=std::max(value,c.max_temperature_k);return value;
}
double Simulation::stored_energy_j() const {
    double value=0;for(const auto& c:cells_)value+=c.stored_energy_j;return value;
}
double Simulation::released_energy_j() const {
    double value=0;for(const auto& c:cells_)value+=c.released_energy_j;return value;
}
double Simulation::joule_energy_j() const {
    double value=0;for(const auto& c:cells_)value+=c.joule_energy_j;return value;
}
double Simulation::energy_balance_residual_j() const {
    if(!scenario_.gas_compartments.empty())return gpu_->process_diagnostics().thermal_residual_j;
    return stored_energy_j()-initial_energy_j_-(released_energy_j()+joule_energy_j()-boundary_energy_out_j_-vent_heat_out_j());
}
double Simulation::vent_heat_out_j() const {
    double value=0;for(const auto& c:cells_)value+=c.vent_heat_out_j;return value;
}
double Simulation::chemical_energy_remaining_j() const {
    double value=0;for(const auto& c:cells_)value+=c.chemical_energy_remaining_j;return value;
}
double Simulation::ejected_chemical_energy_j() const {
    double value=0;for(const auto& c:cells_)value+=c.ejected_chemical_energy_j;return value;
}
double Simulation::chemical_balance_residual_j() const {
    if(!scenario_.gas_compartments.empty())return gpu_->process_diagnostics().chemical_residual_j;
    double value=0;for(std::size_t i=0;i<cells_.size();++i)
        value+=cells_[i].chemical_energy_remaining_j+cells_[i].released_energy_j+cells_[i].ejected_chemical_energy_j-scenario_.cells[i].runaway_energy_j;
    return value;
}
} // namespace battery

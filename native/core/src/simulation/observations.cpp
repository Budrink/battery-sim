#include "battery_native/simulation/simulation.hpp"
#include <algorithm>

namespace battery {
std::size_t Simulation::battery_count() const {
    return std::count_if(scenario_.cells.begin(), scenario_.cells.end(),
                         [](const auto& c) { return c.is_battery; });
}
bool Simulation::all_batteries_destroyed() const {
    bool found = false;
    for (std::size_t i = 0; i < cells_.size(); ++i)
        if (scenario_.cells[i].is_battery) {
            found = true;
            if (cells_[i].mode != CellMode::Spent)
                return false;
        }
    return found;
}
const std::vector<ThermalState>& Simulation::thermal_states() const {
    if (field_dirty_) {
        gpu_->download_field(field_cache_);
        field_dirty_ = false;
    }
    return field_cache_;
}
std::vector<InterfaceState> Simulation::thermal_link_states() const {
    std::vector<InterfaceState> result;
    gpu_->download_link_states(result);
    return result;
}
void Simulation::record_history() {
    history_.time_s.push_back(static_cast<float>(time_s_));
    for (std::size_t i = 0; i < cells_.size(); ++i) {
        history_.temperature_k[i].push_back(static_cast<float>(cells_[i].temperature_k));
        history_.cell_damage[i].push_back(static_cast<float>(cells_[i].damage));
    }
    for (std::size_t i = 0; i < interfaces_.size(); ++i) {
        history_.conductance_w_per_k[i].push_back(static_cast<float>(interfaces_[i].conductance_w_per_k));
        history_.edge_l_ratio[i].push_back(
            static_cast<float>(interfaces_[i].l_m / std::max(interfaces_[i].l0_m, 1e-12)));
    }
}
bool Simulation::propagated() const {
    return std::any_of(runaway_order_.begin(), runaway_order_.end(),
                       [&](std::size_t i) { return !primary_cells_[i]; });
}
double Simulation::max_temperature_k() const {
    double value = 0;
    for (const auto& c : cells_)
        value = std::max(value, c.max_temperature_k);
    return value;
}
double Simulation::stored_energy_j() const {
    double value = 0;
    for (const auto& c : cells_)
        value += c.stored_energy_j;
    return value;
}
double Simulation::released_energy_j() const {
    double value = 0;
    for (const auto& c : cells_)
        value += c.released_energy_j;
    return value;
}
double Simulation::joule_energy_j() const {
    double value = 0;
    for (const auto& c : cells_)
        value += c.joule_energy_j;
    return value;
}
double Simulation::energy_balance_residual_j() const {
    if (!scenario_.gas_compartments.empty())
        return gpu_->process_diagnostics().thermal_residual_j;
    return stored_energy_j() - initial_energy_j_ -
           (released_energy_j() + joule_energy_j() - boundary_energy_out_j_ - vent_heat_out_j());
}
double Simulation::vent_heat_out_j() const {
    double value = 0;
    for (const auto& c : cells_)
        value += c.vent_heat_out_j;
    return value;
}
double Simulation::chemical_energy_remaining_j() const {
    double value = 0;
    for (const auto& c : cells_)
        value += c.chemical_energy_remaining_j;
    return value;
}
double Simulation::ejected_chemical_energy_j() const {
    double value = 0;
    for (const auto& c : cells_)
        value += c.ejected_chemical_energy_j;
    return value;
}
double Simulation::chemical_balance_residual_j() const {
    if (!scenario_.gas_compartments.empty())
        return gpu_->process_diagnostics().chemical_residual_j;
    double value = 0;
    for (std::size_t i = 0; i < cells_.size(); ++i)
        value += cells_[i].chemical_energy_remaining_j + cells_[i].released_energy_j +
                 cells_[i].ejected_chemical_energy_j - scenario_.cells[i].runaway_energy_j;
    return value;
}
} // namespace battery

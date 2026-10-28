#pragma once

#include "battery_native/model.hpp"
#include "battery_native/cuda_solver.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace battery {

struct RuntimeParameters {
    double max_dt_s = 0.05;
    double interface_conductance_scale = 1.0;
    double ambient_conductance_scale = 1.0;
    double damage_rate_scale = 1.0;
    double runaway_power_scale = 1.0;
};

struct History {
    std::vector<float> time_s;
    std::vector<std::vector<float>> temperature_k;
    std::vector<std::vector<float>> cell_damage;
    std::vector<std::vector<float>> conductance_w_per_k;
    std::vector<std::vector<float>> edge_l_ratio;
};

class Simulation {
public:
    explicit Simulation(Scenario scenario);
    ~Simulation();
    Simulation(const Simulation&)=delete;
    Simulation& operator=(const Simulation&)=delete;

    void reset();
    // Complete committed state; explicit caller contract binds physics/topology.
    std::string save_checkpoint(const std::string& contract) const;
    void restore_checkpoint(const std::string& bytes,const std::string& contract);
    // Caller must validate unchanged solid physics. Gas inventory is remapped,
    // never reconstructed from temperatures or initial composition.
    void rebase_unvented_from(const Simulation&,const std::vector<GasRemap>&);
    // One adaptive physical substep (also retained for numerical tests).
    bool step(double dt_cap_s = std::numeric_limits<double>::infinity());
    // One outer interval; observer runs after EVERY completed substep and may stop early.
    bool advance(const std::function<bool()>& stop_after_substep = {});
    void run_to_end();

    const Scenario& scenario() const { return scenario_; }
    const std::vector<CellState>& cells() const { return cells_; }
    const std::vector<InterfaceState>& interfaces() const { return interfaces_; }
    const std::vector<EventRecord>& events() const { return events_; }
    const std::vector<std::size_t>& runaway_order() const { return runaway_order_; }
    const History& history() const { return history_; }
    const std::vector<HeatingProtocolState>& heating_protocol_states() const { return heating_states_; }
    const ThermalMesh& thermal_mesh() const { return mesh_; }
    const std::vector<ThermalState>& thermal_states() const;
    // Local history for diagnostics; G was evaluated on the last substep's input.
    std::vector<InterfaceState> thermal_link_states() const;
    std::vector<ElectricalState> electrical_states() const {
        std::vector<ElectricalState> states;gpu_->download_electrical_states(states);return states;
    }
    const std::string& device_name() const { return gpu_->device_name(); }
    double last_dt_s() const { return last_dt_s_; }
    const ThermalState* cuda_field() const { return gpu_->device_field(); }

    RuntimeParameters& runtime() { return runtime_; }
    const RuntimeParameters& runtime() const { return runtime_; }

    double time_s() const { return time_s_; }
    std::size_t step_count() const { return step_count_; }
    std::size_t outer_step_count() const { return outer_step_count_; }
    std::size_t last_substep_count() const { return last_substep_count_; }
    double last_interval_s() const { return last_interval_s_; }
    std::size_t battery_count() const;
    bool all_batteries_destroyed() const;
    bool finished() const;
    bool propagated() const;
    double max_temperature_k() const;
    double initial_stored_energy_j() const { return initial_energy_j_; }
    double stored_energy_j() const;
    double released_energy_j() const;
    double joule_energy_j() const;
    double vent_heat_out_j() const;
    double chemical_energy_remaining_j() const;
    double ejected_chemical_energy_j() const;
    double chemical_balance_residual_j() const;
    double boundary_energy_out_j() const { return boundary_energy_out_j_; }
    double energy_balance_residual_j() const;
    ProcessDiagnostics process_diagnostics() const { return gpu_->process_diagnostics(); }
    SolverProfile profile() const { return gpu_->profile(); }
    std::vector<double> last_gas_film_heat_j() const { return gpu_->last_gas_film_heat_j(); }

private:
    void apply_scheduled();
    void collect_events(const std::vector<CellState>&, const std::vector<InterfaceState>&);
    void record_history();

    Scenario scenario_;
    ThermalMesh mesh_;
    std::unique_ptr<CudaSolver> gpu_;
    mutable std::vector<ThermalState> field_cache_;
    mutable bool field_dirty_ = true;
    double last_dt_s_ = 0;
    RuntimeParameters runtime_;
    std::vector<CellState> cells_;
    std::vector<InterfaceState> interfaces_;
    std::vector<GasChannelState> gas_channels_;
    std::vector<HeatingProtocolState> heating_states_;
    std::vector<EventRecord> events_;
    std::vector<std::size_t> runaway_order_;
    std::vector<bool> primary_cells_;
    History history_;
    double time_s_ = 0.0;
    double next_history_s_ = 0.0;
    std::size_t next_failure_ = 0;
    std::size_t step_count_ = 0;
    std::size_t outer_step_count_ = 0, last_substep_count_ = 0;
    double last_interval_s_ = 0;
    double initial_energy_j_ = 0.0;
    double boundary_energy_out_j_ = 0.0;
};

}  // namespace battery

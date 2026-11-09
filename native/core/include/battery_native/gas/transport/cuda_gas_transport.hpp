#pragma once
#include "battery_native/simulation/cuda_solver.hpp"
#include "battery_native/gas/transport/mass_mesh.hpp"
#include "battery_native/gas/transport/gas_direction_memory.hpp"

namespace battery {
// CUDA-only transactional adapter for the closed, constant-local-capacity model.
// Conversion can change capacity between stages, not during pressure transport.
// This is the shared adapter, not the choice of numerical method: GasModel
// selects pairwise Kawasaki, batched sweeps or deterministic network transport.
class CudaGasTransport {
  public:
    CudaGasTransport(const Scenario&, const ThermalMesh&, const MassMesh&);
    ~CudaGasTransport();
    void reset();
    void prepare(const GasState* input, GasState* output, ThermalState* solid, GasChannelState* channels,
                 double duration);
    void commit();
    // Changes only the numerical transfer operator; committed RNG, mixture and
    // physical momentum history remain in the shared adapter.
    void set_colored_kawasaki(bool enabled);
    bool colored_kawasaki() const;
    bool can_switch_transport() const;
    void statistics(SolverProfile&) const;
    std::string save_state() const;
    void restore_state(const std::string&);
    // Read-only heat records from the last completed prepare (accepted stage).
    std::vector<double> last_film_heat_j() const;
    std::vector<gas_direction::Memory> directional_memory() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace battery

#pragma once
#include "battery_native/cuda_solver.hpp"
#include "battery_native/mass_mesh.hpp"

namespace battery {
// CUDA-only transactional adapter for the closed, constant-local-capacity model.
// Conversion can change capacity between stages, not during pressure transport.
class CudaKawasakiTransport {
public:
    CudaKawasakiTransport(const Scenario&,const ThermalMesh&,const MassMesh&);
    ~CudaKawasakiTransport();
    void reset();
    void prepare(const GasState* input,GasState* output,ThermalState* solid,
                 GasChannelState* channels,double duration);
    void commit();
    void statistics(SolverProfile&) const;
    std::string save_state() const;
    void restore_state(const std::string&);
    // Read-only heat records from the last completed prepare (accepted stage).
    std::vector<double> last_film_heat_j() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

#pragma once
#include "battery_native/cuda_solver.hpp"
#include "battery_native/mass_mesh.hpp"
#include "battery_native/gas_partition.hpp"

namespace battery {
// Stateless with respect to accepted simulation time: all writes are to the
// caller's candidate. A discarded CUDA transaction cannot advance CPU history.
class LocalGasTransport {
public:
    LocalGasTransport(const Scenario&,const ThermalMesh&,const MassMesh&);
    void advance(std::vector<GasState>&,std::vector<ThermalState>&,
                 std::vector<GasChannelState>&,double duration,GasTransportStats* stats=nullptr) const;
private:
    const Scenario& scenario_;
    const ThermalMesh& thermal_;
    const MassMesh& mass_;
    GasPartition partition_;
    double capacity(int,const GasState&,const std::vector<ThermalState>&) const;
    void equilibrate(int,GasState&,std::vector<ThermalState>&) const;
    void group(const GasLocalGroup&,double,std::vector<GasState>&,
               std::vector<ThermalState>&,std::vector<GasChannelState>&,GasTransportStats&) const;
    void implicit_group(const GasLocalGroup&,double,std::vector<GasState>&,
               std::vector<ThermalState>&,std::vector<GasChannelState>&,GasTransportStats&) const;
};
}

#pragma once
#include "battery_native/thermal_mesh.hpp"
#include "battery_native/gas_transport_stats.hpp"
#include <memory>
#include <string>

namespace battery {
struct RuntimeParameters;
struct ShellDamageState;
struct ThermalState {
    double temperature_k = 298.15;
    double stored_energy_j = 0; // authoritative when a node capacity law is enabled
    double damage = 0;
    double released_energy_j = 0;
    double joule_energy_j = 0; // supplied electrically, never charged to chemical reserve
    double boundary_energy_out_j = 0;
    double runaway_started_s = -1;
    double reaction_progress_s = 0; // reaction clock; runtime power scale pauses/speeds it
    double chemical_energy_remaining_j = 0;
    double ejected_chemical_energy_j = 0;
    double vent_heat_out_j = 0;
    double chemical_power_w = 0; // current gross chemical rate, before discharge
    double conductivity_factor[3]{1.,1.,1.}; // cached local constitutive state
    CellMode mode = CellMode::Healthy;
    double reactant_kg = 0., activated_kg = 0., residue_kg = 0.;
    double gross_reaction_energy_j = 0.;
    // Per-substep source to the owner's gas compartment.
    double generated_gas_kg = 0., generated_gas_energy_j = 0., generated_gas_chemical_j = 0.;
    double mobile_capacity_j_per_k = 0.; // derived LTE cache; gas inventory owns its energy
};
struct GasState {
    double mass_kg = 0., energy_j = 0., chemical_j = 0.;
    double temperature_k = 298.15, pressure_pa = 0.;
    double oxidizer_kg = 0., consumed_oxidizer_kg = 0., released_energy_j = 0.;
    double free_volume_m3 = 0.;
};
struct GasChannelState {
    bool open = false;
    // Signed in configured from -> to orientation; internal channels can reverse.
    double opened_s = -1., mass_flow_kg_s = 0.;
    double proposed_mass_flow_kg_s = 0.; // same substep, before equalization cap
    double transferred_kg = 0., transferred_energy_j = 0., transferred_chemical_j = 0.;
    double transferred_oxidizer_kg = 0.;
    double surface_energy_j = 0.; // signed heat diverted from arriving enthalpy
    double fracture_work_j = 0.; // irreversible work, not a second heat source
    // End-state aperture; mass_flow is the accepted previous substep's rate.
    double opening_fraction = 0., effective_area_m2 = 0.;
    double max_bulk_damage = 0., max_seam_damage = 0.; // progressive law only
    double max_thermal_damage = 0.; // weakening, distinct from mechanical opening
    double aperture_radius_m = 0.; // circle/valve; derived on diagnostic download
    std::array<double,4> aperture_strips_m{}; // seam fronts: -u,+u,-v,+v
};
struct ProcessDiagnostics {
    std::vector<GasState> gas;
    std::vector<GasChannelState> channels;
    double mass_residual_kg = 0., thermal_residual_j = 0., chemical_residual_j = 0.;
    double escaped_mass_kg = 0., escaped_energy_j = 0., escaped_chemical_j = 0.;
    double oxidizer_residual_kg = 0., escaped_oxidizer_kg = 0.;
    double fracture_work_j = 0.;
    double joule_energy_j = 0.;
};
class CudaSolver {
public:
    // Pointwise work may be tiled differently without changing summation order.
    // Reductions retain a fixed tree. Supported work tile sizes: 64,128,256.
    CudaSolver(const Scenario&, const ThermalMesh&, int pointwise_block_size=128);
    ~CudaSolver();
    CudaSolver(const CudaSolver&)=delete;
    CudaSolver& operator=(const CudaSolver&)=delete;
    void reset();
    void ignite(std::size_t object, int local_volume, double time_s);
    double step(double time_s, double dt_cap, double stability, const RuntimeParameters&);
    // Transactional substep: prepare may be repeated/replaced without changing
    // the committed state. Only commit advances inventories/history. A second
    // commit is an error. reset/ignite discard any pending proposal.
    double prepare_step(double time_s, double dt_cap, double stability, const RuntimeParameters&);
    void commit_step();
    void discard_step();
    void download_macros(std::vector<CellState>&, std::vector<InterfaceState>&);
    void download_field(std::vector<ThermalState>&) const;
    void download_shell_states(std::vector<ShellDamageState>&) const;
    void download_link_states(std::vector<InterfaceState>&) const;
    const std::string& device_name() const;
    const ThermalState* device_field() const; // read-only, valid until the next step/reset
    ProcessDiagnostics process_diagnostics() const;
    SolverProfile profile() const;
    void download_gas_states(std::vector<GasState>&, std::vector<GasChannelState>&) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace battery

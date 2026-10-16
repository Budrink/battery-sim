#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace battery {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

enum class ShapeKind { Cylinder, Prismatic, Pouch };
enum class TransitionKind { Constant, Smooth, FirstOrder, SecondOrder, StateGeometry, DirectedGeometry };
enum class EdgePhase { Low, High, Destroyed };
enum class CellMode { Healthy, Runaway, Spent, PartiallySpent, Quenched };
enum class RunawayProfile { ConstantPower, BurstTail, Kinetic };
// Effective one-reaction closure, NOT a chemistry-specific reaction mechanism.
struct ReactionKinetics {
    double reference_temperature_k = 500.;
    double rate_at_reference_per_s = .1; // at activation_damage
    double activation_energy_j_per_mol = 60000.;
    double activation_damage = .2;
    double damage_sensitivity = 2.;
    double damage_per_conversion = 1.;
    double quench_rate_per_s = 1e-5; // reporting threshold only; does not stop chemistry
    double max_fraction_per_step = .02;
    double max_temperature_rise_per_step_k = 2.;
};
struct ReactionStructure {
    bool enabled = false;
    double active_factor[3]{1.,1.,1.};
    double residue_factor[3]{1.,1.,1.};
    double activity_rate_per_s = .1;
};
// Energy-only, unresolved discharge to OUTSIDE the modeled thermal domain.
// A named, declared vent port is mandatory. Not a pressure/jet/mass solver.
struct EnergyOutletLaw {
    bool enabled = false;
    double opening_start_conversion = 0.;
    double opening_end_conversion = .1;
    double closure_start_conversion = 1.; // 1/1 disables closure
    double closure_end_conversion = 1.;
    double heat_fraction = 0.;
    double ejected_energy_per_released_energy = 0.;
};
struct RunawayPulse {
    RunawayProfile profile = RunawayProfile::ConstantPower;
    double fast_fraction = 0.8;
    double tail_time_s = 30.0;
};
struct BulkFailureTransient {
    double peak_factor = 1.0; // relative to reference bulk k, not to the residual
    double relaxation_time_s = 0.0; // zero disables the transient (legacy switch)
};

// R1: heat-neutral aging; R2: exothermic decomposition; R3: gas combustion.
// Chemical and sensible inventories share the gas reference temperature.
struct ConversionLaw {
    bool enabled = false;
    double mass_kg = 0., reactant_cp = 1000., activated_cp = 1000., residue_cp = 1000.;
    double net_heat_j_per_kg = 0.;
    double degradation_rate_per_s = .1, reaction_rate_per_s = .1;
    double reaction_reference_k = 500., reaction_activation_j_per_mol = 0.;
    double degradation_reference_k = 298.15, degradation_activation_j_per_mol = 40000.;
    double initial_precursor_fraction = 0.;
    double overcharge_conversion_per_charge = 0.; // fraction transformed per nominal charge, at electrical T_ref
    // Reporting only; never a cutoff in the R2 kinetic law (fraction of m0/s).
    double runaway_reporting_rate_per_s = 1e-5;
    double gas_yield = 0., gas_chemical_fraction = 0.;
    // Effective gas/evaporation yield: y0 exp(bT*(T-T0)) (rho/rho0)^brho.
    // rho is condensed inventory per ORIGINAL local volume, not mesh density.
    double gas_yield_reference_k = 500., gas_yield_temperature_per_k = 0.;
    double gas_yield_density_exponent = 0., gas_yield_max = .95;
    double max_fraction_per_step = .01, max_temperature_step_k = .5;
    double residue_conductivity[3]{1.,1.,1.};
    int gas_compartment = -1;
};
// Whole-cell resistance, not one resistor per thermal voxel. Positive Joule
// power is partitioned by reference thermal volume; current is externally driven.
struct ElectricalLaw {
    bool enabled = false;
    double current_a = 0., resistance_ohm = .01, reference_temperature_k = 298.15;
    double temperature_log_slope_per_k = 0., damage_log_slope = 0.;
    double min_resistance_ohm = 1e-6, max_resistance_ohm = 1.;
    double max_temperature_step_k = .5;
    double overcharge_fraction = 0., nominal_capacity_coulomb = 0.;
};
struct GasModel {
    bool split_transport = false; // thermal/chemistry stage, then pressure-driven mass stage
    bool cpu_local_transport = false;
    bool cuda_kawasaki_transport = false;
    bool cuda_batched_transport = false; // shares the CUDA mixture adapter, not edge coloring
    bool cuda_deterministic_transport = false; // bounded, fixed-count conservative graph sweeps
    bool cuda_network_transport = false; // simultaneous whole-graph deterministic analytic edge flux
    int deterministic_sweeps_per_step = 4;
    double deterministic_pressure_fraction = .75;
    double network_pressure_fraction = 1.; // limiter on simultaneous network pressure change
    double batch_pressure_fraction = .2, batch_mean_threshold = 32.;
    double kawasaki_packet_fraction = .03125, kawasaki_tail_atol_pa = 100.;
    double kawasaki_tail_rtol = .001;
    std::uint64_t kawasaki_seed = 20261015;
    bool implicit_local_transport = false; // opt-in conservative backward Euler
    double implicit_temperature_tolerance_k = .02;
    double implicit_tracer_relative_tolerance = .001;
    double local_exchange_interval_s = 1e-5; // numerical splitting interval, NOT a fit parameter
    double reference_temperature_k = 298.15;
    double cv_j_per_kg_k = 1000., specific_r_j_per_kg_k = 287.;
    double max_fraction_per_step = .01, max_temperature_step_k = .5;
    // Numerical pressure-relaxation control, not material/calibration parameters.
    double pressure_relaxation_fraction = .2;
    double pressure_absolute_tolerance_pa = 1., pressure_relative_tolerance = 1e-5;
};
struct GasCompartment {
    std::string id;
    int owner_cell = -1; // battery contents, or -1 for a receiving cavity
    double volume_m3 = 1e-4, initial_mass_kg = 0., initial_temperature_k = 298.15;
    // Opt-in, already-ignited effective chemistry (not a flame/ignition model).
    // Oxidizer is a constituent of total gas mass; reaction changes its label,
    // not total mass. A common pseudo-species R/cv is retained after reaction.
    double initial_chemical_j = 0., initial_oxidizer_mass_fraction = 0.;
    double reaction_rate_per_s = 0., oxidizer_heat_j_per_kg = 13.1e6;
    double ignition_temperature_k = 0.;
    // Optional mobile phase sharing ONE numerical solid volume (local LTE).
    // Independent of owner_cell, which is the legacy lumped source association.
    int thermal_cell = -1, thermal_volume = 0;
    // Optional occupied-volume closure for the reacting inventory. Zero keeps
    // an explicitly fixed pore volume; densities are never inferred from cp.
    double reactant_density_kg_m3 = 0., residue_density_kg_m3 = 0.;
};
struct ShellOpeningLaw {
    bool enabled = false;
    bool deterministic_aperture = false; // leading, continuous area operator; no random events
    double area_rate_per_s = 1., pressure_exponent = 1.;
    // Optional circular weak valve, coordinates relative to its local shell patch.
    double valve_radius_m = 0., valve_u_m = 0., valve_v_m = 0., valve_pressure_factor = 1.;
    bool binary_pixels = false; // whole local panel elements, no fractional opening kinetics
    bool stochastic_pixels = false; // independent cumulative-hazard clocks, binary aperture
    bool stochastic_area = false; // repeated events removing a random physical area
    double fragment_mean_area_m2 = 1e-5;
    double fixed_area_fraction = 0.; // >0: fixed original-patch fraction per stochastic event
    double pixel_rate_per_s = 1., pixel_pressure_exponent = 1.;
    std::uint64_t rupture_seed = 1;
    // Pre-existing mechanical opening history, NOT thermal degradation.
    double initial_mechanical_damage = 0.;
    double damage_onset = .2, exponent = 2.;
    double pressure_damage_rate_per_s = 1., max_damage_step = .02; // maximum absolute pressure-driven delta D
    // Optional weak perimeter band, resolved by physical area, not voxel count.
    double seam_width_m = 0., seam_strength_factor = 1.;
    // Effective work per newly opened area (fracture/plastic deformation).
    // Zero is legacy compatibility, not a claim that real rupture is free.
    double fracture_energy_j_per_m2 = 0.;
};
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool discrete_shell_events(const ShellOpeningLaw& law) {
    return law.binary_pixels || law.stochastic_area;
}
struct GasChannel {
    std::string id, port_id;
    int from = -1, to = -1; // -1 destination is an explicit outside sink
    int shell_object = -1;
    bool internal = false; // adjacent local pore volumes; no exterior face port
    // Fixed orifices only. Progressive rupture derives area from its panel;
    // zero means unspecified, a supplied value must equal that physical face.
    double area_m2 = 0., discharge_coefficient = .7;
    double opening_pressure_pa = 0., outside_pressure_pa = 101325.;
    double strength_reference_k = 298.15, strength_temperature_scale_k = 1e6;
    bool rupture = false; // latched opening, separate from chemical D
    ShellOpeningLaw opening;
    // Opt-in directional thermal connection through this aperture. Geometry
    // chooses the first solid surface; zero preserves historical scenarios.
    double surface_link_h_w_per_m2_k = 0.;
    // Local shell voxel, x-fastest; -1 retains a whole-panel channel.
    int shell_thermal_volume = -1;
    bool passage = false; // free-space connection, never bypasses a casing
};
struct GasContact {
    std::string port_id;
    int compartment = -1;
    double h_w_per_m2_k = 0.;
    int thermal_volume = -1; // optional local face patch, same h, no area duplication
};

const char* name(ShapeKind value);
const char* name(TransitionKind value);
const char* name(EdgePhase value);
const char* name(CellMode value);

struct PortConfig {
    std::string id;
    std::string cell_id;
    std::size_t cell_index = 0;
    std::string kind;
    std::string surface;
    double area_m2 = 0.0;
    Vec3 centroid_m;
    Vec3 normal;
    std::array<double,2> rectangle_m{0.,0.}; // ordered axes (normal+1)%3, (normal+2)%3
};

// Small, device-copyable constitutive table. Linear interpolation, constant
// endpoint extension. An empty table preserves the legacy constitutive law.
struct ScalarTable {
    int count = 0;
    double temperature_k[8]{};
    double value[8]{};
};

struct HeatCapacityLaw {
    ScalarTable factor; // C(T,D=0)/C_ref, strictly positive
    double reference_temperature_k = 298.15;
    double damage_factor_at_one = 1.0; // disabled until independently identified
};

// Log-eigenvalues of a local, diagonal structural metric in the object's frame.
// Used by contact geometry only; does not modify measured bulk k or heat capacity.
struct SurfaceConnectivity {
    double reference_temperature_k = 298.15;
    double temperature_scale_k = 100.0;
    Vec3 linear, quadratic, damage;
    // Source-owned sensitivities to signed sender-minus-receiver state contrasts.
    // Unlike the legacy diagonal metric, these define an ordered transport law.
    Vec3 temperature_contrast, damage_contrast;
    ScalarTable temperature_log_response; // replaces the polynomial, isotropic
};

struct ConnectivityLaw {
    TransitionKind kind = TransitionKind::Constant;
    // Dimensionless directed tensor response; no independent length coordinate.
    double destroyed_geometry_reference_logit = 0.;
    double l0_m = 0.0;
    double l_low_m = 0.0;
    double l_high_m = 0.0;
    double transition_temperature_k = 400.0;
    double transition_width_k = 10.0;
    double hysteresis_k = 0.0;
    double relaxation_time_s = 0.0;
    double background_conductance_w_per_k = 0.0;
    double active_conductance_w_per_k = 0.0;
    double activation_width_m = 1.0e-5;
    double damage_onset_k = 450.0;
    double damage_rate_at_onset_per_s = 0.0;
    double damage_temperature_scale_k = 25.0;
    double damage_l_shift_m = 0.0;
    double damage_conductance_factor_at_one = 1.0;
    double endpoint_damage_weight = 0.0; // coupling to mean adjacent-volume damage
    double destruction_damage = 1.0;
    double destroyed_l0_m = 0.0;
    double destroyed_l_m = 0.0;
    double destroyed_background_conductance_w_per_k = 0.0;
    double destroyed_active_conductance_w_per_k = 0.0;
    double geometry_reference_logit = 0.0;
    SurfaceConnectivity endpoint_a, endpoint_b; // populated from the node material profiles by the mesher
};

struct CellConfig {
    std::string id;
    bool is_battery = true; // passive solids carry heat/damage but never initiate runaway
    std::string material_id = "unspecified";
    ShapeKind shape = ShapeKind::Prismatic;
    std::vector<double> dimensions_m;
    Vec3 position_m;
    std::array<double, 4> orientation_xyzw{0.0, 0.0, 0.0, 1.0};
    double heat_capacity_j_per_k = 1.0;
    HeatCapacityLaw heat_capacity_law;
    double initial_temperature_k = 298.15;
    // Independent numerical resolution, NOT the connectivity threshold l0.
    std::array<int, 3> thermal_divisions{4, 4, 4};
    std::array<double, 3> conductivity_w_per_m_k{1.0, 1.0, 1.0};
    // Exact infinite INTERNAL conductivity limit; external contacts stay finite.
    bool isothermal = false;
    std::vector<double> initial_temperature_field_k;
    bool has_bulk_connectivity = false;
    ConnectivityLaw bulk_connectivity; // dimensionless factors are converted to face G by the mesher
    SurfaceConnectivity surface_connectivity;
    // Optional independent internal-edge response. Legacy bulk laws keep the
    // surface response unless this field is explicitly supplied.
    bool has_bulk_response = false;
    SurfaceConnectivity bulk_response;
    // Opt-in: use the same destroyed bulk factor in boundary half-volumes.
    bool bulk_destruction_on_boundary = false;
    BulkFailureTransient bulk_failure_transient;
    double aging_fraction = 0.0;
    double initial_damage = 0.0;
    double damage_onset_k = 420.0;
    double damage_rate_at_onset_per_s = 0.0;
    double damage_temperature_scale_k = 25.0;
    double aging_damage_multiplier = 0.0;
    double runaway_energy_j = 0.0;
    double runaway_duration_s = 1.0;
    RunawayPulse runaway_pulse;
    ReactionKinetics reaction_kinetics;
    ReactionStructure reaction_structure;
    EnergyOutletLaw energy_outlet;
    std::string energy_outlet_port;
    ConversionLaw conversion;
    ElectricalLaw electrical;
    std::vector<std::array<double,2>> current_profile_a; // [physical seconds, amperes]
    std::string shell_contents_id; // physical casing panel, NOT a numerical region
};

struct InterfaceConfig {
    std::string id;
    std::string port_a;
    std::string port_b;
    std::size_t cell_a = 0;
    std::size_t cell_b = 0;
    Vec3 point_a_m;
    Vec3 point_b_m;
    std::string material_a;
    std::string material_b;
    std::string interface_material;
    ConnectivityLaw law;
    double initial_l_m = 0.0;
    double initial_damage = 0.0;
    EdgePhase initial_phase = EdgePhase::Low;
};

struct AmbientBoundary {
    std::string id;
    std::string port_id;
    std::size_t cell_index = 0;
    double temperature_k = 298.15;
    double conductance_w_per_k = 0.0;
    // Optional measured forcing: piecewise-linear [time_s, temperature_k].
    // Must cover the complete scenario interval; no silent extrapolation.
    std::vector<std::array<double, 2>> temperature_profile;
    bool has_connectivity = false;
    ConnectivityLaw connectivity;
    SurfaceConnectivity remote_surface_connectivity;
};

struct FailureEvent {
    std::size_t cell_index = 0;
    std::string cell_id;
    double time_s = 0.0;
    int thermal_volume = -1; // local x-fast index; -1 ignites the whole object
};

struct SimulationSettings {
    double duration_s = 1.0;
    double max_dt_s = 0.01;
    double output_interval_s = 0.1;
    double stability_factor = 0.2;
    double step_interval_s = 0.0; // outer step; 0 preserves legacy max_dt_s cadence
};

struct Scenario {
    std::string name;
    std::string source;
    std::string calibration_status;
    std::string geometry_status;
    SimulationSettings settings;
    std::vector<CellConfig> cells;
    std::vector<PortConfig> ports;
    std::vector<InterfaceConfig> interfaces;
    std::vector<AmbientBoundary> boundaries;
    std::vector<FailureEvent> primary_failures;
    GasModel gas_model;
    std::vector<GasCompartment> gas_compartments;
    std::vector<GasChannel> gas_channels;
    std::vector<GasContact> gas_contacts;
    std::vector<std::string> expected_runaway_order;
    bool has_expected_propagation = false;
    bool expected_propagation = false;
};

Scenario load_scenario(const std::string& path);
struct ShellConfig {
    std::string contents_id, material_id;
    double thickness_m = 0., density_kg_m3 = 0., specific_heat_j_kg_k = 0.;
    std::array<double,3> conductivity_w_per_m_k{1.,1.,1.};
    double contents_contact_h = 0., seam_contact_h = 0.;
    double damage_onset_k = 500., damage_rate_per_s = 0., damage_scale_k = 25.;
    double destroyed_conductance_factor = .01;
    double initial_temperature_k = -1.; // inherit contents only if unspecified
    std::array<int,3> surface_divisions{0,0,0}; // zero triple inherits contents; normal axis stays one
};
// Adds six non-overlapping physical panels, twelve seam contacts, and six
// contents contacts. Contents dimensions/capacity are EXCLUSIVE of the casing.
void add_physical_shell(Scenario&, const ShellConfig&);
// Refine passive surroundings only when present; battery zone counts are a model choice.
void refine_environment_mesh(Scenario& scenario, int scale);

struct CellState {
    double temperature_k = 298.15;
    double damage = 0.0;
    CellMode mode = CellMode::Healthy;
    double runaway_started_s = -1.0;
    double released_energy_j = 0.0;
    double joule_energy_j = 0.0;
    double min_temperature_k = 298.15;
    double max_temperature_k = 298.15;
    double max_damage = 0.0;
    double spent_fraction = 0.0;
    double runaway_fraction = 0.0;
    double stored_energy_j = 0.0;
    double boundary_energy_out_j = 0.0;
    double heat_capacity_j_per_k = 0.0;
    double chemical_energy_remaining_j = 0.0;
    double ejected_chemical_energy_j = 0.0;
    double vent_heat_out_j = 0.0;
    double chemical_power_w = 0.0;
    double quenched_fraction = 0.0;
};

struct InterfaceState {
    double l0_m = 0.0;
    double l_m = 0.0;
    double damage = 0.0;
    EdgePhase phase = EdgePhase::Low;
    double conductance_w_per_k = 0.0;
    double destroyed_fraction = 0.0;
};

struct EventRecord {
    double time_s = 0.0;
    std::string kind;
    std::string entity_id;
};

}  // namespace battery

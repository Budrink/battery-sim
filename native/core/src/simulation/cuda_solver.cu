#include "battery_native/simulation/cuda_solver.hpp"
#include "battery_native/chemistry/reaction_step_bounds.hpp"
#include "battery_native/numerics/time_step_policy.hpp"
#include "battery_native/io/checkpoint_io.hpp"
#include "battery_native/simulation/simulation.hpp"
#include "battery_native/thermal/state_geometry.hpp"
#include "battery_native/gas/mixing/gas_thermal_contact.hpp"
#include "battery_native/mechanics/failure.hpp"
#include "battery_native/chemistry/reaction.hpp"
#include "battery_native/chemistry/conversion.hpp"
#include "battery_native/chemistry/volatile_phase.hpp"
#include "battery_native/electrical/joule.hpp"
#include "battery_native/electrical/circuit_solve.hpp"
#include "battery_native/electrical/circuit_validation.hpp"
#include "battery_native/gas/transport/gas_timestep.hpp"
#include "battery_native/mechanics/shell_opening.hpp"
#include "battery_native/mechanics/deterministic_aperture.hpp"
#include "battery_native/mechanics/plate_fracture.hpp"
#include "battery_native/mechanics/aperture_transfer.hpp"
#include "battery_native/gas/transport/mass_mesh.hpp"
#include "battery_native/gas/transport/local_gas_transport.hpp"
#include "battery_native/gas/transport/cuda_gas_transport.hpp"
#include "battery_native/chemistry/gas_combustion.hpp"
#include "battery_native/chemistry/mixture_equilibrium.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace battery {
namespace {
void check(cudaError_t result, const char* context) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(result));
}
template <class T> struct Buffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Buffer(std::size_t count) : n(count) {
        if (n)
            check(cudaMalloc(&p, n * sizeof(T)), "CUDA allocation");
    }
    ~Buffer() {
        if (p)
            cudaFree(p);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& v) {
        if (n)
            check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice), "CUDA upload");
    }
    void download(std::vector<T>& v) const {
        v.resize(n);
        if (n)
            check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "CUDA download");
    }
};
struct DeviceRows {
    const int* offsets;
    const int* ids;
};
struct StepResult {
    double dt;
    unsigned invalid;
};
// Stable CSR incidence lists: gather touches neighbours only, in input order.
// No device-side searching, appending, sorting or floating-point atomics.
struct Incidence {
    Buffer<int> offsets;
    std::unique_ptr<Buffer<int>> ids;
    explicit Incidence(const std::vector<std::vector<int>>& rows) : offsets(rows.size() + 1) {
        std::vector<int> starts{0}, flat;
        for (const auto& row : rows) {
            flat.insert(flat.end(), row.begin(), row.end());
            starts.push_back(static_cast<int>(flat.size()));
        }
        offsets.upload(starts);
        ids = std::make_unique<Buffer<int>>(flat.size());
        ids->upload(flat);
    }
    DeviceRows view() const {
        return {offsets.p, ids->p};
    }
};
struct Material {
    double capacity, energy, duration, onset, rate, temperature_scale, aging_multiplier;
    bool isothermal;
    bool is_battery;
    HeatCapacityLaw capacity_law;
    RunawayPulse pulse;
    BulkBoundaryPhase boundary_phase;
    ReactionKinetics kinetics;
    ReactionStructure structure;
    EnergyOutletLaw outlet;
    ConversionLaw conversion;
    GasModel gas_model;
    ElectricalLaw electrical;
    CurrentView current;
    CurrentView voltage;
};
__device__ double maximum_structure(const Material& m, int axis) {
    if (m.conversion.enabled)
        return fmax(1., m.conversion.residue_conductivity[axis]);
    return m.structure.enabled
               ? fmax(1., fmax(m.structure.active_factor[axis], m.structure.residue_factor[axis]))
               : 1.;
}
// Geometry and verbose law records are cold data; kernels stream only this
// compact connectivity record. Dynamic material laws are shared by index.
struct DeviceVolume {
    int object;
    double capacity_j_per_k;
};
#include "../electrical/electrical_kernels.cuh"
#include "../electrical/circuit_kernels.cuh"
// Close derived constitutive fields from authoritative energy/composition.
// Used after thermal evolution AND after discrete structural events.
__device__ void close_thermal_state(const Material& m, const DeviceVolume& v, ThermalState& s, double speed) {
    if (m.conversion.enabled) {
        const double tref = m.gas_model.reference_temperature_k;
        s.temperature_k = tref + (s.stored_energy_j - v.capacity_j_per_k * tref) /
                                     conversion_capacity(m.conversion, s, v.capacity_j_per_k,
                                                         v.capacity_j_per_k / m.capacity);
        refresh_conversion(m.conversion, s, v.capacity_j_per_k / m.capacity, speed);
    } else {
        if (m.capacity_law.factor.count)
            s.temperature_k =
                thermal_temperature(m.capacity_law, v.capacity_j_per_k, s.stored_energy_j, s.damage);
        if (m.pulse.profile == RunawayProfile::Kinetic)
            refresh_reaction(m.kinetics, m.structure, s, m.energy * v.capacity_j_per_k / m.capacity, speed);
    }
}
#include "../thermal/solid_transport_kernels.cuh"
#include "../gas/transport/gas_kernels.cuh"
#include "../chemistry/mixture_kernels.cuh"
#include "../mechanics/shell_kernels.cuh"
#include "../mechanics/shell_feedback.cuh"
#include "../mechanics/aperture_kernels.cuh"
#include "../mechanics/shell_work_kernels.cuh"
#include "../mechanics/shell_event_kernels.cuh"
#include "../mechanics/leading_shell_kernels.cuh"
#include "../mechanics/plate_front_kernels.cuh"
#include "../chemistry/solid_reaction_kernels.cuh"
#include "macro_reduction_kernels.cuh"
int blocks(std::size_t count) {
    return static_cast<int>((count + 127) / 128);
}
// A rejected candidate is never published. Only integer error flags are atomic;
// no floating-point inventory/energy reduction depends on scheduling.
__global__ void validate_candidate_kernel(int nn, const ThermalState* nodes, int ng, const GasState* gas,
                                          int ne, const InterfaceState* edges, const double* dt,
                                          StepResult* result) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned error = 0;
    if (i == 0)
        result->dt = *dt;
    if (i < nn) {
        const auto& s = nodes[i];
        if (!isfinite(s.liquid_kg) || s.liquid_kg < 0)
            error |= 1;
        if (!isfinite(s.temperature_k) || s.temperature_k <= 0 || !isfinite(s.stored_energy_j) ||
            !isfinite(s.damage) || s.damage < 0 || s.damage > 1 || !isfinite(s.reactant_kg) ||
            s.reactant_kg < 0 || !isfinite(s.activated_kg) || s.activated_kg < 0 || !isfinite(s.residue_kg) ||
            s.residue_kg < 0 || !isfinite(s.healthy_charged_kg) || s.healthy_charged_kg < 0 ||
            s.healthy_charged_kg > s.activated_kg + 1e-14 || !isfinite(s.degraded_mass_kg) ||
            s.degraded_mass_kg < 0 || !isfinite(s.joule_energy_j) || s.joule_energy_j < 0 ||
            !isfinite(s.precursor_charge_energy_j) || s.precursor_charge_energy_j < 0 ||
            !isfinite(s.electrical_chemical_input_j) || s.electrical_chemical_input_j < 0)
            error |= 1;
    }
    if (i < ng) {
        const auto& g = gas[i];
        if (!isfinite(g.vapor_kg) || g.vapor_kg < 0 || g.vapor_kg > g.mass_kg + 1e-14 ||
            !isfinite(g.latent_j) || g.latent_j < 0)
            error |= 2;
        if (!isfinite(g.temperature_k) || g.temperature_k <= 0 || !isfinite(g.pressure_pa) ||
            g.pressure_pa < 0 || !isfinite(g.mass_kg) || g.mass_kg <= 0 || !isfinite(g.energy_j) ||
            !isfinite(g.chemical_j) || g.chemical_j < -1e-10 || !isfinite(g.oxidizer_kg) ||
            g.oxidizer_kg < -1e-14 || g.oxidizer_kg > g.mass_kg + 1e-14)
            error |= 2;
    }
    if (i < ne) {
        const auto& e = edges[i];
        if (!isfinite(e.conductance_w_per_k) || e.conductance_w_per_k < 0 || !isfinite(e.damage) ||
            !isfinite(e.l_m))
            error |= 4;
    }
    if (error)
        atomicOr(&result->invalid, error);
}
} // namespace

#include "solver_storage.cuh"
#include "cuda_step.cuh"
#include "../mechanics/leading_shell_step.cuh"
#include "mass_transfer_stages.cuh"
#include "mass_step.cuh"
CudaSolver::CudaSolver(const Scenario& s, const ThermalMesh& m, int tile)
    : impl_(std::make_unique<Impl>(s, m, tile)) {
    reset();
}
CudaSolver::~CudaSolver() = default;
const std::string& CudaSolver::device_name() const {
    return impl_->device;
}
const ThermalState* CudaSolver::device_field() const {
    return impl_->state_a.p;
}
void CudaSolver::reset() {
    auto& p = *impl_;
    p.prepared = false;
    p.profiling = {};
    std::vector<ThermalState> state;
    p.materials.upload(p.reference_materials);
    if (p.cuda_gas)
        p.cuda_gas->reset();
    for (const auto& v : p.mesh.volumes) {
        ThermalState s;
        s.temperature_k = v.initial_temperature_k;
        s.damage = shell_material_damage(
            initial_shell_state(p.scenario, v.object,
                                static_cast<int>(state.size()) - p.mesh.object_offsets[v.object]),
            0.);
        s.stored_energy_j = thermal_energy(p.scenario.cells[v.object].heat_capacity_law, v.capacity_j_per_k,
                                           s.temperature_k, s.damage);
        const auto& c = p.scenario.cells[v.object];
        s.chemical_energy_remaining_j = c.runaway_energy_j * v.capacity_j_per_k / c.heat_capacity_j_per_k;
        if (s.damage >= (c.runaway_pulse.profile == RunawayProfile::Kinetic
                             ? c.reaction_kinetics.activation_damage
                             : 1.)) {
            if (p.scenario.cells[v.object].is_battery) {
                s.mode = CellMode::Runaway;
                s.runaway_started_s = 0;
            } else
                s.mode = CellMode::Spent;
        }
        if (c.runaway_pulse.profile == RunawayProfile::Kinetic)
            refresh_reaction(c.reaction_kinetics, c.reaction_structure, s, s.chemical_energy_remaining_j, 1.);
        if (c.conversion.enabled) {
            const double fraction = v.capacity_j_per_k / c.heat_capacity_j_per_k;
            const double mass = c.conversion.mass_kg * fraction;
            s.healthy_charged_kg = mass * c.conversion.initial_charged_fraction;
            s.degraded_mass_kg = mass * c.conversion.initial_precursor_fraction;
            s.activated_kg = s.degraded_mass_kg + s.healthy_charged_kg;
            // Energy stored BEFORE the modeled experiment is initial reserve,
            // not electrical work received during this trajectory.
            s.precursor_charge_energy_j = s.healthy_charged_kg * c.conversion.net_heat_j_per_kg;
            s.reactant_kg = mass - s.activated_kg;
            s.liquid_kg = c.conversion.volatile_phase.initial_mass_kg * fraction;
            const double cap = conversion_capacity(c.conversion, s, v.capacity_j_per_k, fraction);
            s.stored_energy_j = v.capacity_j_per_k * p.scenario.gas_model.reference_temperature_k +
                                cap * (s.temperature_k - p.scenario.gas_model.reference_temperature_k);
            refresh_conversion(c.conversion, s, v.capacity_j_per_k / c.heat_capacity_j_per_k, 1.);
        }
        state.push_back(s);
    }
    p.state_a.upload(state);
    if (p.electrical_objects.n) {
        p.electrical_potential_a.upload(p.electrical_mesh.initial_potential);
        p.electrical_power.upload(std::vector<double>(p.electrical_power.n, 0.));
    }
    p.electrical_a.upload(std::vector<ElectricalState>(p.scenario.cells.size()));
    p.electrical_b.upload(std::vector<ElectricalState>(p.scenario.cells.size()));
    std::vector<GasState> gas;
    for (const auto& c : p.scenario.gas_compartments) {
        GasState g;
        g.mass_kg = c.initial_mass_kg;
        g.temperature_k = c.initial_temperature_k;
        g.chemical_j = c.initial_chemical_j;
        g.oxidizer_kg = c.initial_mass_kg * c.initial_oxidizer_mass_fraction;
        g.energy_j = g.mass_kg * p.scenario.gas_model.cv_j_per_kg_k *
                     (g.temperature_k - p.scenario.gas_model.reference_temperature_k);
        g.free_volume_m3 = c.volume_m3;
        g.pressure_pa =
            g.mass_kg * p.scenario.gas_model.specific_r_j_per_kg_k * g.temperature_k / c.volume_m3;
        gas.push_back(g);
    }
    p.gas_a.upload(gas);
    p.channel_state.upload(std::vector<GasChannelState>(p.channels.n));
    if (p.scenario.gas_model.split_transport)
        p.equilibrate(p.state_a.p, p.gas_a.p, 1.);
    std::vector<InterfaceState> edges;
    for (const auto& e : p.mesh.links)
        edges.push_back(e.initial);
    p.edge_state.upload(edges);
    p.dt.upload({0});
    if (p.shell_patches) {
        std::vector<ShellDamageState> damage;
        for (const auto& v : p.mesh.volumes)
            damage.push_back(initial_shell_state(
                p.scenario, v.object, static_cast<int>(damage.size()) - p.mesh.object_offsets[v.object]));
        p.shell_damage->upload(damage);
        shell_aperture_kernel<<<p.work_blocks(p.channels.n), p.work_threads>>>(
            p.channels.n, p.channels.p, p.object_offsets.p, p.shell_patches->p, p.shell_damage->p,
            p.channel_state.p, 0., p.dt.p);
    }
    p.propose_edges(RuntimeParameters{}, 0);
    std::swap(p.edge_state.p, p.edge_next.p);
    check(cudaGetLastError(), "CUDA reset");
    p.aggregate();
}
void CudaSolver::ignite(std::size_t object, int local_volume, double time) {
    discard_step();
    if (!impl_->scenario.cells.at(object).is_battery)
        throw std::runtime_error("Cannot ignite a passive material");
    auto& p = *impl_;
    int begin = p.mesh.object_offsets.at(object), end = p.mesh.object_offsets.at(object + 1);
    if (local_volume >= 0) {
        begin += local_volume;
        end = begin + 1;
    }
    ignition_kernel<<<p.work_blocks(end - begin), p.work_threads>>>(p.state_a.p, p.volumes.p, p.materials.p,
                                                                    begin, end, time);
    check(cudaGetLastError(), "CUDA ignition");
    p.aggregate();
}
bool CudaSolver::can_switch_gas_transport() const {
    return impl_->cuda_gas && impl_->cuda_gas->can_switch_transport();
}
bool CudaSolver::colored_kawasaki() const {
    return impl_->cuda_gas && impl_->cuda_gas->colored_kawasaki();
}
void CudaSolver::set_colored_kawasaki(bool enabled) {
    if (impl_->prepared)
        throw std::logic_error("Switch gas transport only between committed steps");
    if (!can_switch_gas_transport())
        throw std::logic_error("This scenario does not support live gas transport switching");
    impl_->cuda_gas->set_colored_kawasaki(enabled);
}
double CudaSolver::prepare_step(double time, double cap, double safety, const RuntimeParameters& runtime) {
    auto& p = *impl_;
    p.prepared = false;
    struct Timer {
        double& total;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        ~Timer() {
            total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    } timer{p.profiling.prepare_s};
    if (!std::isfinite(time) || !std::isfinite(cap) || cap <= 0 || !std::isfinite(safety) || safety <= 0 ||
        safety > 1)
        throw std::invalid_argument("Invalid CUDA substep controls");
    for (double x : {runtime.interface_conductance_scale, runtime.ambient_conductance_scale,
                     runtime.damage_rate_scale, runtime.runaway_power_scale})
        if (!std::isfinite(x) || x < 0)
            throw std::invalid_argument("Invalid CUDA runtime multiplier");
    const auto accept = [&](double accepted) {
        const double floor = p.scenario.settings.minimum_dt_s;
        if (floor > 0.) {
            if (cap >= floor * (1. - 1e-12) && accepted < floor * (1. - 1e-12))
                throw std::runtime_error("Adaptive step violated minimum_dt_s");
            if (accepted < floor * (1. - 1e-12))
                ++p.profiling.aligned_short_steps;
            else if (p.profiling.minimum_regular_step_s == 0. ||
                     accepted < p.profiling.minimum_regular_step_s)
                p.profiling.minimum_regular_step_s = accepted;
        }
        p.prepared = true;
        return accepted;
    };
    if (p.scenario.gas_model.split_transport)
        return accept(p.prepare_split_step(time, cap, safety, runtime));
    p.propose_gas_transport(time);
    p.select_solid_step(time, cap, safety, runtime);
    p.select_gas_step(time, safety, runtime);
    p.apply_step_floor(cap);
    p.propose_processes(time, runtime);
    p.gather_candidate(time, runtime);
    const double dt = p.close_and_validate_candidate(time, runtime);
    return accept(dt);
}
void CudaSolver::discard_step() {
    impl_->prepared = false;
}
void CudaSolver::set_resistive_heater_power(std::size_t object, double power) {
    auto& p = *impl_;
    if (p.prepared || object >= p.scenario.cells.size() || !std::isfinite(power) || power < 0.)
        throw std::invalid_argument("Invalid heater drive update");
    const auto& c = p.scenario.cells[object];
    const auto& e = c.electrical;
    if (c.is_battery || c.conversion.enabled || !e.enabled || e.voltage_limited_current ||
        e.charge_acceptance_fraction != 0. || e.temperature_log_slope_per_k != 0. ||
        e.temperature_log_response.count || e.charge_log_slope != 0. || e.resistance_ohm <= 0. ||
        e.min_resistance_ohm != e.resistance_ohm || e.max_resistance_ohm != e.resistance_ohm)
        throw std::invalid_argument("Controlled heater must be a fixed-R passive resistor");
    auto m = p.reference_materials[object];
    const double current = std::sqrt(power / e.resistance_ohm);
    const CurrentPoint voltage{0., current * e.resistance_ohm};
    check(cudaMemcpy(p.heater_voltage_points->p + object, &voltage, sizeof(voltage), cudaMemcpyHostToDevice),
          "heater voltage");
    m.electrical.current_a = current;
    m.current = {nullptr, 0};
    m.voltage = {p.heater_voltage_points->p + object, 1};
    check(cudaMemcpy(p.materials.p + object, &m, sizeof(m), cudaMemcpyHostToDevice), "heater material drive");
}
std::string CudaSolver::save_state() const {
    const auto& p = *impl_;
    if (p.prepared || p.prefix_inputs_active || p.cpu_gas)
        throw std::runtime_error("Restart requires committed CUDA state");
    checkpoint::Writer w;
    w.text("cuda-state-v2");
    const auto save = [&](const auto& buffer) {
        using T = std::remove_pointer_t<decltype(buffer.p)>;
        std::vector<T> values;
        buffer.download(values);
        w.vector(values);
    };
    save(p.state_a);
    save(p.edge_state);
    save(p.gas_a);
    save(p.channel_state);
    if (p.shell_damage)
        save(*p.shell_damage);
    else
        w.vector(std::vector<ShellDamageState>{});
    save(p.electrical_a);
    save(p.electrical_potential_a);
    w.text(p.cuda_gas ? p.cuda_gas->save_state() : std::string{});
    return w.finish();
}
void CudaSolver::restore_state(const std::string& bytes) {
    auto& p = *impl_;
    if (p.cpu_gas)
        throw std::runtime_error("CPU transport restart is unsupported");
    checkpoint::Reader r(bytes);
    if (r.text() != "cuda-state-v2")
        throw std::runtime_error("Unsupported CUDA restart version");
    auto field = r.vector<ThermalState>(p.state_a.n);
    auto edges = r.vector<InterfaceState>(p.edge_state.n);
    auto gas = r.vector<GasState>(p.gas_a.n);
    auto channels = r.vector<GasChannelState>(p.channel_state.n);
    auto shell = r.vector<ShellDamageState>(p.shell_damage ? p.shell_damage->n : 0);
    auto electrical = r.vector<ElectricalState>(p.electrical_a.n);
    auto potential = r.vector<double>(p.electrical_potential_a.n);
    const auto transport = r.text();
    r.finish();
    for (const auto& v : field)
        if (!std::isfinite(v.temperature_k) || v.temperature_k <= 0. || !std::isfinite(v.stored_energy_j))
            throw std::runtime_error("Invalid thermal restart state");
    for (const auto& v : gas)
        if (!std::isfinite(v.temperature_k) || v.temperature_k <= 0. || !std::isfinite(v.mass_kg) ||
            v.mass_kg <= 0.)
            throw std::runtime_error("Invalid gas restart state");
    if (bool(p.cuda_gas) != !transport.empty())
        throw std::runtime_error("Restart gas backend mismatch");
    if (p.cuda_gas)
        p.cuda_gas->restore_state(transport);
    p.prepared = false;
    p.state_a.upload(field);
    p.state_b.upload(field);
    p.edge_state.upload(edges);
    p.edge_next.upload(edges);
    p.gas_a.upload(gas);
    p.gas_b.upload(gas);
    p.channel_state.upload(channels);
    p.channel_next.upload(channels);
    if (p.shell_damage) {
        p.shell_damage->upload(shell);
        p.shell_next->upload(shell);
    }
    p.electrical_a.upload(electrical);
    p.electrical_b.upload(electrical);
    p.electrical_potential_a.upload(potential);
    p.electrical_potential_b.upload(potential);
    p.profiling = {};
    p.cached_interface_scale = p.cached_ambient_scale = p.cached_safety = -1.;
    p.aggregate();
}
void CudaSolver::rebase_unvented_from(const CudaSolver& source, const std::vector<GasRemap>& map) {
    const auto& old = *source.impl_;
    auto& p = *impl_;
    if (old.scenario.gas_model.directional_inertia || p.scenario.gas_model.directional_inertia)
        throw std::runtime_error("Gas graph rebase requires an explicit directional-memory remap");
    if (old.prepared || p.prepared || old.prefix_inputs_active || old.cpu_gas || p.cpu_gas ||
        !old.scenario.gas_model.cuda_network_transport || !p.scenario.gas_model.cuda_network_transport)
        throw std::runtime_error("Unvented rebase requires committed deterministic CUDA networks");
    std::vector<GasState> original;
    old.gas_a.download(original);
    std::vector<GasChannelState> previous;
    old.channel_state.download(previous);
    for (std::size_t i = 0; i < previous.size(); ++i)
        if (old.scenario.gas_channels[i].rupture &&
            (previous[i].open || previous[i].opening_fraction != 0. || previous[i].transferred_kg != 0.))
            throw std::runtime_error("Cannot rebase gas geometry after a rupture");
    std::vector<GasState> gas(p.gas_a.n);
    std::vector<double> fractions(original.size()), volumes(gas.size());
    for (const auto& e : map) {
        if (e.from >= original.size() || e.to >= gas.size() || !std::isfinite(e.fraction) ||
            e.fraction <= 0. || e.fraction > 1. + 1e-10)
            throw std::runtime_error("Invalid gas remap fraction");
        const auto& a = old.scenario.gas_compartments[e.from];
        const auto& b = p.scenario.gas_compartments[e.to];
        if (a.owner_cell != b.owner_cell || a.thermal_cell != b.thermal_cell ||
            a.thermal_volume != b.thermal_volume ||
            (a.thermal_cell >= 0 && (a.id != b.id || e.fraction != 1.)))
            throw std::runtime_error("Gas remap changes local solid binding");
        fractions[e.from] += e.fraction;
        volumes[e.to] += e.fraction * a.volume_m3;
        auto& g = gas[e.to];
        const auto& v = original[e.from];
        const double f = e.fraction;
        g.mass_kg += f * v.mass_kg;
        g.energy_j += f * v.energy_j;
        g.chemical_j += f * v.chemical_j;
        g.oxidizer_kg += f * v.oxidizer_kg;
        g.consumed_oxidizer_kg += f * v.consumed_oxidizer_kg;
        g.released_energy_j += f * v.released_energy_j;
        g.vapor_kg += f * v.vapor_kg;
        g.latent_j += f * v.latent_j;
        g.free_volume_m3 += f * v.free_volume_m3;
        g.ambient_mass_out_kg += f * v.ambient_mass_out_kg;
        g.ambient_energy_out_j += f * v.ambient_energy_out_j;
        g.ambient_chemical_out_j += f * v.ambient_chemical_out_j;
        g.ambient_oxidizer_out_kg += f * v.ambient_oxidizer_out_kg;
        g.ambient_gross_in_kg += f * v.ambient_gross_in_kg;
        g.ambient_gross_out_kg += f * v.ambient_gross_out_kg;
        g.ambient_entropy_j_k += f * v.ambient_entropy_j_k;
    }
    for (double f : fractions)
        if (std::abs(f - 1.) > 1e-10)
            throw std::runtime_error("Gas remap loses source inventory");
    for (std::size_t i = 0; i < gas.size(); ++i) {
        const auto& c = p.scenario.gas_compartments[i];
        auto& g = gas[i];
        if (std::abs(volumes[i] - c.volume_m3) > 1e-10 * c.volume_m3 + 1e-16 || g.mass_kg <= 0.)
            throw std::runtime_error("Gas remap does not cover receiver volume");
        g.temperature_k = p.scenario.gas_model.reference_temperature_k +
                          g.energy_j / (g.mass_kg * p.scenario.gas_model.cv_j_per_kg_k);
        g.pressure_pa =
            g.mass_kg * p.scenario.gas_model.specific_r_j_per_kg_k * g.temperature_k / g.free_volume_m3;
    }
    // Direct one-to-one nodes (especially LTE pores) retain every derived bit.
    for (const auto& e : map)
        if (e.fraction == 1. &&
            old.scenario.gas_compartments[e.from].id == p.scenario.gas_compartments[e.to].id &&
            old.scenario.gas_compartments[e.from].volume_m3 == p.scenario.gas_compartments[e.to].volume_m3)
            gas[e.to] = original[e.from];
    std::vector<GasChannelState> channels;
    p.channel_state.download(channels);
    for (std::size_t j = 0; j < p.scenario.gas_channels.size(); ++j) {
        const auto& c = p.scenario.gas_channels[j];
        if (c.id.rfind("void-face-", 0) == 0)
            continue;
        for (std::size_t i = 0; i < old.scenario.gas_channels.size(); ++i)
            if (old.scenario.gas_channels[i].id == c.id) {
                channels[j] = previous[i];
                break;
            }
    }
    checkpoint::Writer w;
    w.text("cuda-state-v2");
    const auto copy = [&](const auto& buffer, std::size_t expected) {
        if (buffer.n != expected)
            throw std::runtime_error("Gas rebase changed solid mesh");
        using T = std::remove_pointer_t<decltype(buffer.p)>;
        std::vector<T> v;
        buffer.download(v);
        w.vector(v);
    };
    copy(old.state_a, p.state_a.n);
    copy(old.edge_state, p.edge_state.n);
    w.vector(gas);
    w.vector(channels);
    if (old.shell_damage && p.shell_damage)
        copy(*old.shell_damage, p.shell_damage->n);
    else if (!old.shell_damage && !p.shell_damage)
        w.vector(std::vector<ShellDamageState>{});
    else
        throw std::runtime_error("Gas rebase changed shell mesh");
    copy(old.electrical_a, p.electrical_a.n);
    copy(old.electrical_potential_a, p.electrical_potential_a.n);
    // Deterministic network has no stochastic clocks to retain across new
    // edges. Fresh adapter clocks/caches are explicit; no hidden RNG re-seed.
    w.text(p.cuda_gas ? p.cuda_gas->save_state() : std::string{});
    restore_state(w.finish());
}
SolverProfile CudaSolver::profile() const {
    auto result = impl_->profiling;
    if (impl_->cuda_gas)
        impl_->cuda_gas->statistics(result);
    return result;
}
std::vector<double> CudaSolver::last_gas_film_heat_j() const {
    if (impl_->prepared || !impl_->cuda_gas)
        throw std::runtime_error("Film trace requires committed coupled CUDA state");
    return impl_->cuda_gas->last_film_heat_j();
}
void CudaSolver::commit_step() {
    auto& p = *impl_;
    if (!p.prepared)
        throw std::logic_error("No prepared CUDA substep");
    p.prepared = false;
    std::swap(p.state_a.p, p.state_b.p);
    std::swap(p.gas_a.p, p.gas_b.p);
    std::swap(p.edge_state.p, p.edge_next.p);
    std::swap(p.channel_state.p, p.channel_next.p);
    if (p.shell_patches)
        std::swap(p.shell_damage->p, p.shell_next->p);
    std::swap(p.electrical_potential_a.p, p.electrical_potential_b.p);
    std::swap(p.electrical_a.p, p.electrical_b.p);
    if (p.cuda_gas)
        p.cuda_gas->commit();
    p.aggregate();
}
double CudaSolver::step(double time, double cap, double safety, const RuntimeParameters& runtime) {
    const double dt = prepare_step(time, cap, safety, runtime);
    commit_step();
    return dt;
}
void CudaSolver::download_macros(std::vector<CellState>& cells, std::vector<InterfaceState>& edges) {
    impl_->macro_cells.download(cells);
    impl_->macro_edges.download(edges);
}
void CudaSolver::download_field(std::vector<ThermalState>& state) const {
    impl_->state_a.download(state);
}
void CudaSolver::download_electrical_states(std::vector<ElectricalState>& state) const {
    impl_->electrical_a.download(state);
}
void CudaSolver::download_shell_states(std::vector<ShellDamageState>& state) const {
    if (impl_->shell_damage)
        impl_->shell_damage->download(state);
    else
        state.clear();
}
void CudaSolver::download_link_states(std::vector<InterfaceState>& state) const {
    impl_->edge_state.download(state);
}
ProcessDiagnostics CudaSolver::process_diagnostics() const {
    const auto& p = *impl_;
    ProcessDiagnostics out;
    p.gas_a.download(out.gas);
    p.channel_state.download(out.channels);
    for (std::size_t i = 0; i < out.channels.size(); ++i) {
        const auto& c = p.scenario.gas_channels[i];
        if (!c.opening.deterministic_aperture)
            continue;
        const auto& patch = p.shell_geometry[p.mesh.object_offsets[c.shell_object] + c.shell_thermal_volume];
        auto& d = out.channels[i];
        if (patch.seam_fraction > 0)
            opened_strips(patch.rectangle, d.opening_fraction, d.aperture_strips_m.data());
        else
            d.aperture_radius_m =
                clipped_disk_radius(patch.rectangle.width, patch.rectangle.height, c.opening.valve_u_m,
                                    c.opening.valve_v_m, d.opening_fraction);
    }
    std::vector<ThermalState> field;
    p.state_a.download(field);
    double initial_thermal = 0, current_thermal = 0, released = 0, boundary = 0, vent = 0, chemical = 0,
           initial_chemical = 0, initial_mass = 0, current_mass = 0, legacy_ejected = 0;
    double initial_oxidizer = 0, accounted_oxidizer = 0;
    for (std::size_t i = 0; i < field.size(); ++i) {
        const auto& v = p.mesh.volumes[i];
        const auto& c = p.scenario.cells[v.object];
        const auto& st = field[i];
        const double fraction = v.capacity_j_per_k / c.heat_capacity_j_per_k;
        double initial_node_energy = thermal_energy(
            c.heat_capacity_law, v.capacity_j_per_k, v.initial_temperature_k,
            shell_material_damage(initial_shell_state(p.scenario, v.object,
                                                      static_cast<int>(i) - p.mesh.object_offsets[v.object]),
                                  0.));
        if (c.conversion.enabled)
            initial_node_energy +=
                c.conversion.mass_kg * fraction *
                (c.conversion.initial_precursor_fraction + c.conversion.initial_charged_fraction) *
                (c.conversion.activated_cp - c.conversion.reactant_cp) *
                (v.initial_temperature_k - p.scenario.gas_model.reference_temperature_k);
        initial_thermal += initial_node_energy;
        current_thermal += st.stored_energy_j;
        released += st.released_energy_j;
        boundary += st.boundary_energy_out_j;
        vent += st.vent_heat_out_j;
        out.joule_energy_j += st.joule_energy_j;
        out.electrical_chemical_input_j += st.electrical_chemical_input_j;
        chemical += st.chemical_energy_remaining_j + st.sterilized_chemical_j;
        legacy_ejected += st.ejected_chemical_energy_j;
        initial_chemical +=
            fraction * (c.conversion.enabled ? c.conversion.mass_kg * c.conversion.net_heat_j_per_kg
                                             : c.runaway_energy_j);
        if (c.conversion.enabled)
            initial_chemical += fraction * c.conversion.mass_kg * c.conversion.initial_charged_fraction *
                                c.conversion.net_heat_j_per_kg;
        if (c.conversion.enabled)
            initial_chemical += fraction * c.conversion.volatile_phase.initial_mass_kg *
                                c.conversion.volatile_phase.chemical_j_per_kg;
        if (c.conversion.enabled) {
            initial_mass += (c.conversion.mass_kg + c.conversion.volatile_phase.initial_mass_kg) * fraction;
            current_mass += st.reactant_kg + st.activated_kg + st.residue_kg + st.liquid_kg;
            if (st.reactant_kg < 0 || st.activated_kg < 0 || st.residue_kg < 0)
                throw std::runtime_error("Negative reacting inventory");
        }
    }
    for (std::size_t i = 0; i < out.gas.size(); ++i) {
        const auto& g = out.gas[i];
        const auto& c = p.scenario.gas_compartments[i];
        if (!std::isfinite(g.temperature_k) || g.temperature_k <= 0 || !std::isfinite(g.pressure_pa) ||
            g.mass_kg <= 0 || g.chemical_j < -1e-10 || !std::isfinite(g.chemical_j) ||
            !std::isfinite(g.oxidizer_kg) || g.oxidizer_kg < -1e-14 || g.oxidizer_kg > g.mass_kg + 1e-14)
            throw std::runtime_error("Nonphysical gas state");
        initial_mass += c.initial_mass_kg;
        current_mass += g.mass_kg;
        chemical += g.chemical_j;
        current_thermal += g.energy_j + g.latent_j;
        initial_chemical += c.initial_chemical_j;
        released += g.released_energy_j;
        initial_oxidizer += c.initial_mass_kg * c.initial_oxidizer_mass_fraction;
        accounted_oxidizer += g.oxidizer_kg + g.consumed_oxidizer_kg;
        initial_thermal += c.initial_mass_kg * p.scenario.gas_model.cv_j_per_kg_k *
                           (c.initial_temperature_k - p.scenario.gas_model.reference_temperature_k);
        out.escaped_mass_kg += g.ambient_mass_out_kg;
        out.escaped_energy_j += g.ambient_energy_out_j;
        out.escaped_chemical_j += g.ambient_chemical_out_j;
        out.escaped_oxidizer_kg += g.ambient_oxidizer_out_kg;
    }
    for (std::size_t i = 0; i < out.channels.size(); ++i)
        if (p.scenario.gas_channels[i].to < 0) {
            const auto& ch = out.channels[i];
            out.escaped_mass_kg += ch.transferred_kg;
            out.escaped_energy_j += ch.transferred_energy_j;
            out.escaped_chemical_j += ch.transferred_chemical_j;
            out.escaped_oxidizer_kg += ch.transferred_oxidizer_kg;
        }
    out.mass_residual_kg = current_mass + out.escaped_mass_kg - initial_mass;
    out.oxidizer_residual_kg = accounted_oxidizer + out.escaped_oxidizer_kg - initial_oxidizer;
    for (const auto& channel : out.channels)
        out.fracture_work_j += channel.fracture_work_j;
    if (p.cuda_gas && p.scenario.gas_model.directional_inertia)
        for (const auto& m : p.cuda_gas->directional_memory())
            out.gas_kinetic_j += m.kinetic_j;
    out.thermal_residual_j = current_thermal + out.gas_kinetic_j + out.fracture_work_j - initial_thermal -
                             (released + out.joule_energy_j - boundary - vent - out.escaped_energy_j);
    out.chemical_residual_j = chemical + released + legacy_ejected + out.escaped_chemical_j -
                              initial_chemical - out.electrical_chemical_input_j;
    return out;
}
void CudaSolver::download_gas_states(std::vector<GasState>& gas,
                                     std::vector<GasChannelState>& channels) const {
    impl_->gas_a.download(gas);
    impl_->channel_state.download(channels);
}
} // namespace battery

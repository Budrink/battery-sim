#include "battery_native/model/validation.hpp"
#include "battery_native/electrical/circuit_validation.hpp"
#include "battery_native/chemistry/reaction.hpp"
#include "battery_native/chemistry/conversion.hpp"
#include "battery_native/thermal/constitutive.hpp"
#include <cmath>
#include <stdexcept>

namespace battery {
namespace {
void positive(double value, const char* field) {
    if (!std::isfinite(value) || value <= 0.)
        throw std::runtime_error(std::string("Expected positive ") + field);
}

void validate_conversion(const CellConfig& c, const Scenario& s, std::size_t object) {
    if (c.conversion.enabled) {
        const auto& p = c.conversion;
        if (!c.is_battery || c.isothermal || c.runaway_energy_j != 0 ||
            c.runaway_pulse.profile != RunawayProfile::ConstantPower || c.heat_capacity_law.factor.count ||
            c.energy_outlet.enabled || c.reaction_structure.enabled || c.bulk_destruction_on_boundary ||
            c.initial_damage != 0 || c.damage_rate_at_onset_per_s != 0 ||
            c.bulk_failure_transient.relaxation_time_s != 0)
            throw std::runtime_error("Conversion has its own composition/capacity/source; legacy "
                                     "chemistry, aging and bulk failure cannot overlap");
        for (double x : {p.mass_kg, p.reactant_cp, p.activated_cp, p.residue_cp, p.degradation_rate_per_s,
                         p.reaction_rate_per_s, p.reaction_reference_k, p.max_fraction_per_step,
                         p.max_temperature_step_k})
            positive(x, "conversion parameter");
        for (double x : {p.net_heat_j_per_kg, p.reaction_activation_j_per_mol, p.gas_yield,
                         p.gas_chemical_fraction, p.charge_conversion_per_nominal_charge})
            if (!std::isfinite(x) || x < 0)
                throw std::runtime_error("Negative/nonfinite conversion parameter");
        for (double x : p.residue_conductivity)
            positive(x, "residue conductivity");
        const auto& phase = p.volatile_phase;
        if (!std::isfinite(phase.initial_mass_kg) || phase.initial_mass_kg < 0.)
            throw std::runtime_error("Invalid volatile mass");
        if (phase.initial_mass_kg > 0.) {
            for (double x :
                 {phase.liquid_cp, phase.density_kg_m3, phase.latent_j_per_kg, phase.boiling_temperature_k,
                  phase.saturation_reference_pa, phase.vapor_specific_r, phase.relaxation_per_s})
                positive(x, "volatile phase parameter");
            if (!s.gas_model.cuda_network_transport || !s.gas_model.split_transport)
                throw std::runtime_error("Volatile phase requires CUDA network + split LTE; unsupported "
                                         "backends must not drop its inventories");
            if (c.heat_capacity_j_per_k <=
                p.mass_kg * p.reactant_cp + phase.initial_mass_kg * phase.liquid_cp)
                throw std::runtime_error(
                    "Total contents heat capacity must be greater than reactive mass x specific heat "
                    "plus liquid mass x specific heat; the remaining inert heat capacity must be positive.");
            if (std::abs(phase.vapor_specific_r - s.gas_model.specific_r_j_per_kg_k) > 1e-12)
                throw std::runtime_error("Current pseudo-gas requires a common vapor gas constant");
            if (!std::isfinite(phase.chemical_j_per_kg) || phase.chemical_j_per_kg < 0. ||
                phase.chemical_j_per_kg != s.gas_model.vapor_fuel_heat_j_kg)
                throw std::runtime_error("Volatile fuel heat must equal the common gas-network declaration");
            for (const auto& other : s.cells)
                if (other.conversion.volatile_phase.initial_mass_kg > 0.) {
                    const auto& q = other.conversion.volatile_phase;
                    if (q.liquid_cp != phase.liquid_cp || q.latent_j_per_kg != phase.latent_j_per_kg ||
                        q.boiling_temperature_k != phase.boiling_temperature_k ||
                        q.saturation_reference_pa != phase.saturation_reference_pa ||
                        q.density_kg_m3 != phase.density_kg_m3 ||
                        q.vapor_specific_r != phase.vapor_specific_r ||
                        q.chemical_j_per_kg != phase.chemical_j_per_kg)
                        throw std::runtime_error(
                            "One common volatile pseudo-species per gas network is supported");
                }
        }
        {
            for (double x : {p.degradation_reference_k, p.runaway_reporting_rate_per_s,
                             p.gas_yield_reference_k, p.gas_yield_max})
                positive(x, "R1/R2 parameter");
            for (double x : {p.degradation_activation_j_per_mol, p.initial_precursor_fraction,
                             p.initial_charged_fraction, p.gas_yield_temperature_per_k,
                             p.gas_yield_density_exponent, p.precursor_yield_temperature_per_k})
                if (!std::isfinite(x))
                    throw std::runtime_error("Nonfinite R1/R2 parameter");
            // Independent reactions: either activation energy may be steeper.
            // Their relative ordering is a calibration hypothesis, not an invariant.
            if (p.degradation_activation_j_per_mol < 0 || p.initial_precursor_fraction < 0 ||
                p.initial_precursor_fraction > 1 || p.initial_charged_fraction < 0 ||
                p.initial_charged_fraction > 1 ||
                p.initial_precursor_fraction + p.initial_charged_fraction > 1 || p.gas_yield_max > 1 ||
                p.gas_yield > p.gas_yield_max || !std::isfinite(p.precursor_yield) || p.precursor_yield < 0 ||
                p.precursor_yield > 1 || p.precursor_temperature_ref_k <= 0)
                throw std::runtime_error(
                    "R1/R2 activation energies must be nonnegative; precursor/yield fractions bounded");
        }
        if (p.gas_yield > 1 || p.gas_chemical_fraction > 1 ||
            (p.gas_yield == 0 && p.gas_chemical_fraction > 0) || p.max_fraction_per_step > .1 ||
            c.heat_capacity_j_per_k <= p.mass_kg * p.reactant_cp || p.gas_compartment < 0 ||
            p.gas_compartment >= static_cast<int>(s.gas_compartments.size()) ||
            s.gas_compartments[p.gas_compartment].owner_cell != static_cast<int>(object))
            throw std::runtime_error("Invalid conversion fractions, inert capacity or gas owner");
    }
}

void validate_electrical(const CellConfig& c, bool network_driven) {
    if (c.electrical.enabled) {
        const auto& e = c.electrical;
        bool has_current = e.current_a != 0.;
        for (const auto& point : c.current_profile_a)
            has_current |= point[1] != 0.;
        // A passive resistive heater uses the same conservative I2R
        // source, but cannot own battery charging/polarization chemistry.
        if (!c.is_battery && (e.charge_acceptance_fraction != 0. || e.charge_log_slope != 0.))
            throw std::runtime_error("Passive resistive heaters cannot store battery charge");
        for (double x : {e.resistance_ohm, e.reference_temperature_k, e.min_resistance_ohm,
                         e.max_resistance_ohm, e.max_temperature_step_k})
            positive(x, "electrical parameter");
        for (double x : {e.current_a, e.temperature_log_slope_per_k})
            if (!std::isfinite(x))
                throw std::runtime_error("Nonfinite electrical law");
        const auto& table = e.temperature_log_response;
        if (table.count) {
            if (table.count < 2 || table.count > 8 || e.temperature_log_slope_per_k != 0.)
                throw std::runtime_error(
                    "Electrical temperature table requires 2..8 knots and no scalar slope");
            for (int j = 0; j < table.count; ++j)
                if (!std::isfinite(table.temperature_k[j]) || table.temperature_k[j] <= 0. ||
                    !std::isfinite(table.value[j]) ||
                    (j && table.temperature_k[j] <= table.temperature_k[j - 1]))
                    throw std::runtime_error("Invalid electrical temperature table");
            if (std::abs(table_value(table, e.reference_temperature_k)) > 1e-10)
                throw std::runtime_error(
                    "Electrical temperature response must be zero at reference temperature");
        }
        if (!std::isfinite(e.product_conductivity_ratio) || e.product_conductivity_ratio <= 0. ||
            e.product_conductivity_ratio > 1. || e.terminal_axis < 0 || e.terminal_axis > 2)
            throw std::runtime_error("Invalid electrical product conductivity or terminal axis");
        if (!std::isfinite(e.reversible_voltage_floor_v) || e.reversible_voltage_floor_v < 0. ||
            (e.reversible_voltage_floor_v > 0. && c.voltage_profile_v.empty() && !network_driven &&
             has_current))
            throw std::runtime_error("Electrical reversible-voltage prior requires a voltage trace");
        if (e.min_resistance_ohm > e.resistance_ohm || e.resistance_ohm > e.max_resistance_ohm)
            throw std::runtime_error("Reference resistance outside declared bounds");
        if (!std::isfinite(e.charge_acceptance_fraction) || e.charge_acceptance_fraction < 0 ||
            e.charge_acceptance_fraction > 1 || !std::isfinite(e.nominal_capacity_coulomb) ||
            e.nominal_capacity_coulomb < 0 ||
            (e.charge_acceptance_fraction > 0 && (e.nominal_capacity_coulomb <= 0 || !c.conversion.enabled)))
            throw std::runtime_error("Charging requires a finite nominal capacity and R1/R2 chemistry");
        if (!std::isfinite(e.charge_log_slope) || e.charge_log_slope < 0. ||
            (e.charge_log_slope > 0. && (!c.conversion.enabled || e.nominal_capacity_coulomb <= 0.)))
            throw std::runtime_error("Charge-dependent resistance requires stored-precursor chemistry "
                                     "and a finite nominal capacity");
        if (e.charge_acceptance_fraction > 0. && c.conversion.net_heat_j_per_kg > 0. &&
            c.voltage_profile_v.empty() && !network_driven && has_current)
            throw std::runtime_error(
                "Charge precursor storage requires terminal voltage to account for electrical work");
        double last = -1.;
        for (const auto& q : c.current_profile_a) {
            if (!std::isfinite(q[0]) || q[0] < 0 || q[0] <= last || !std::isfinite(q[1]))
                throw std::runtime_error("Invalid current profile");
            last = q[0];
        }
        if (e.voltage_limited_current) {
            if (c.voltage_profile_v.empty() || e.current_a < 0.)
                throw std::runtime_error(
                    "Voltage-limited setpoint requires voltage and nonnegative charging current");
            for (const auto& q : c.current_profile_a)
                if (q[1] < 0.)
                    throw std::runtime_error("Voltage-limited setpoint does not support reverse current");
        }
        last = -1.;
        for (const auto& q : c.voltage_profile_v) {
            if (!std::isfinite(q[0]) || q[0] < 0 || q[0] <= last || !std::isfinite(q[1]) || q[1] < 0)
                throw std::runtime_error("Invalid voltage profile");
            last = q[0];
        }
    }
}

void validate_legacy_material(const CellConfig& c) {
    positive(c.heat_capacity_j_per_k, "heat capacity");
    positive(c.runaway_duration_s, "runaway duration");
    positive(c.damage_temperature_scale_k, "damage temperature scale");
    if (!std::isfinite(c.runaway_pulse.fast_fraction) || c.runaway_pulse.fast_fraction < 0 ||
        c.runaway_pulse.fast_fraction > 1)
        throw std::runtime_error("Invalid fast energy fraction");
    positive(c.runaway_pulse.tail_time_s, "pulse tail time");
    if (c.runaway_pulse.profile == RunawayProfile::BurstTail &&
        c.runaway_pulse.tail_time_s < c.runaway_duration_s)
        throw std::runtime_error("Pulse tail must not be faster than burst");
    if (c.bulk_destruction_on_boundary) {
        const auto& b = c.bulk_connectivity;
        if (!c.has_bulk_connectivity || b.kind != TransitionKind::DirectedGeometry ||
            b.endpoint_damage_weight != 1 || b.damage_rate_at_onset_per_s != 0 || b.destruction_damage != 1 ||
            b.destroyed_active_conductance_w_per_k != 0 || c.isothermal)
            throw std::runtime_error(
                "Boundary bulk switch requires a D=1 directed constant destroyed bulk branch");
        positive(b.destroyed_background_conductance_w_per_k, "destroyed bulk factor");
    }
    positive(c.bulk_failure_transient.peak_factor, "bulk transient peak factor");
    if (!std::isfinite(c.bulk_failure_transient.relaxation_time_s) ||
        c.bulk_failure_transient.relaxation_time_s < 0 ||
        (c.bulk_failure_transient.relaxation_time_s > 0 &&
         (!c.bulk_destruction_on_boundary || !c.is_battery)))
        throw std::runtime_error("Bulk transient requires battery rupture clock and boundary bulk switch");
    if (!std::isfinite(c.runaway_energy_j) || c.runaway_energy_j < 0 || c.initial_damage < 0 ||
        c.initial_damage > 1 || c.aging_fraction < 0 || c.aging_fraction > 1 ||
        c.damage_rate_at_onset_per_s < 0)
        throw std::runtime_error("Invalid material state: " + c.id);
    if (c.runaway_pulse.profile == RunawayProfile::Kinetic) {
        if (!c.is_battery || c.isothermal || c.bulk_destruction_on_boundary ||
            c.bulk_failure_transient.relaxation_time_s > 0)
            throw std::runtime_error(
                "Kinetic reaction requires a spatial battery, without legacy timed bulk destruction");
        positive(c.runaway_energy_j, "kinetic energy reserve");
        const auto& p = c.reaction_kinetics;
        positive(p.reference_temperature_k, "reaction reference temperature");
        positive(p.rate_at_reference_per_s, "reaction reference rate");
        positive(p.activation_damage, "reaction activation damage");
        positive(p.quench_rate_per_s, "quench reporting rate");
        positive(p.max_fraction_per_step, "reaction fraction step");
        positive(p.max_temperature_rise_per_step_k, "reaction temperature step");
        if (p.activation_damage > 1 || p.max_fraction_per_step > .1 ||
            !std::isfinite(p.activation_energy_j_per_mol) || p.activation_energy_j_per_mol < 0 ||
            !std::isfinite(p.damage_sensitivity) || p.damage_sensitivity < 0 ||
            !std::isfinite(p.damage_per_conversion) || p.damage_per_conversion < 0)
            throw std::runtime_error("Invalid reaction kinetics");
        if (c.reaction_structure.enabled) {
            positive(c.reaction_structure.activity_rate_per_s, "structure activity rate");
            for (int axis = 0; axis < 3; ++axis) {
                positive(c.reaction_structure.active_factor[axis], "active structure factor");
                positive(c.reaction_structure.residue_factor[axis], "residue structure factor");
            }
            if (c.has_bulk_connectivity && (c.bulk_connectivity.endpoint_damage_weight != 0 ||
                                            c.bulk_connectivity.damage_rate_at_onset_per_s != 0))
                throw std::runtime_error(
                    "Reaction structure cannot be combined with a second bulk destruction law");
        }
        if (c.energy_outlet.enabled) {
            const auto& o = c.energy_outlet;
            for (double x : {o.opening_start_conversion, o.opening_end_conversion, o.closure_start_conversion,
                             o.closure_end_conversion, o.heat_fraction})
                if (!std::isfinite(x) || x < 0 || x > 1)
                    throw std::runtime_error("Invalid outlet fraction");
            if (o.opening_start_conversion >= o.opening_end_conversion ||
                o.closure_start_conversion < o.opening_end_conversion ||
                (o.closure_start_conversion < 1 && o.closure_end_conversion <= o.closure_start_conversion) ||
                !std::isfinite(o.ejected_energy_per_released_energy) ||
                o.ejected_energy_per_released_energy < 0)
                throw std::runtime_error("Invalid outlet conversion window or ejected reserve ratio");
        }
    } else if (c.reaction_structure.enabled || c.energy_outlet.enabled)
        throw std::runtime_error("Reaction structure/outlet requires kinetic reaction");
}
} // namespace

void validate_material_models(const Scenario& s) {
    validate_circuit(s);
    for (std::size_t object = 0; object < s.cells.size(); ++object) {
        const auto& c = s.cells[object];
        validate_conversion(c, s, object);
        bool network_driven = false;
        for (const auto& branch : s.electrical_circuit.branches)
            network_driven |= branch.kind == CircuitKind::Battery && branch.object == int(object);
        validate_electrical(c, network_driven);
        validate_legacy_material(c);
    }
}
} // namespace battery

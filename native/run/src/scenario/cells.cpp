#include "scenario_internal.hpp"
#include "battery_native/thermal/constitutive.hpp"
#include "battery_native/thermal/state_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace battery::scenario_detail {
void parse_cells(const json& root, Scenario& scenario, CellIndex& cell_index) {
    for (const auto& value : root.at("cells")) {
        CellConfig cell;
        cell.id = value.at("id").get<std::string>();
        const auto role = value.value("role", std::string("battery"));
        if (role != "battery" && role != "passive")
            throw std::runtime_error("Unknown object role: " + role);
        cell.is_battery = role == "battery";
        cell.material_id = value.value("material_id", std::string("unspecified"));
        if (!cell_index.emplace(cell.id, scenario.cells.size()).second) {
            throw std::runtime_error("duplicate cell id: " + cell.id);
        }
        const auto& geometry = value.at("geometry");
        cell.shape = shape_kind(geometry.at("kind").get<std::string>());
        cell.dimensions_m = geometry.at("dimensions_m").get<std::vector<double>>();
        const std::size_t expected = cell.shape == ShapeKind::Cylinder ? 2 : 3;
        if (cell.dimensions_m.size() != expected) {
            throw std::runtime_error("wrong dimension count for cell " + cell.id);
        }
        const auto pose = value.value("pose", json::object());
        cell.position_m = vec3(pose.value("position_m", json::array({0.0, 0.0, 0.0})), "pose.position_m");
        const auto orientation = pose.value("orientation_xyzw", json::array({0.0, 0.0, 0.0, 1.0}));
        if (!orientation.is_array() || orientation.size() != 4) {
            throw std::runtime_error("orientation_xyzw must contain four numbers");
        }
        for (std::size_t index = 0; index < 4; ++index) {
            cell.orientation_xyzw[index] = orientation.at(index).get<double>();
        }
        const auto& thermal = value.at("thermal");
        cell.heat_capacity_j_per_k = thermal.at("heat_capacity_j_per_k").get<double>();
        cell.initial_temperature_k = thermal.value("initial_temperature_k", 298.15);
        cell.thermal_divisions = thermal.value("divisions", std::array<int, 3>{4, 4, 4});
        cell.conductivity_w_per_m_k = thermal.value("conductivity_w_per_m_k", std::array<double, 3>{1, 1, 1});
        cell.initial_temperature_field_k =
            thermal.value("initial_temperature_field_k", std::vector<double>{});
        cell.surface_connectivity = parse_surface(thermal.value("surface_connectivity", json::object()));
        cell.bulk_destruction_on_boundary = thermal.value("bulk_destruction_on_boundary", false);
        if (thermal.contains("bulk_failure_transient")) {
            const auto& f = thermal.at("bulk_failure_transient");
            cell.bulk_failure_transient = {f.at("peak_factor"), f.at("relaxation_time_s")};
            if (cell.bulk_failure_transient.relaxation_time_s <= 0)
                throw std::runtime_error("Explicit bulk transient requires positive relaxation time");
        }
        if (thermal.contains("reaction_structure")) {
            const auto& r = thermal.at("reaction_structure");
            auto& p = cell.reaction_structure;
            p.enabled = true;
            const auto a = r.at("active_factor").get<std::array<double, 3>>();
            const auto b = r.at("residue_factor").get<std::array<double, 3>>();
            for (int axis = 0; axis < 3; ++axis) {
                p.active_factor[axis] = a[axis];
                p.residue_factor[axis] = b[axis];
            }
            p.activity_rate_per_s = r.at("activity_rate_per_s");
        }
        const auto transfer = thermal.value("internal_heat_transfer", std::string("finite_volume"));
        if (transfer != "finite_volume" && transfer != "isothermal")
            throw std::runtime_error("Unknown internal_heat_transfer for cell " + cell.id);
        cell.isothermal = transfer == "isothermal";
        if (thermal.contains("heat_capacity_law")) {
            const auto& law = thermal.at("heat_capacity_law");
            auto& l = cell.heat_capacity_law;
            l.factor = parse_table(law.at("factor"), true);
            l.reference_temperature_k = law.value("reference_temperature_k", 298.15);
            l.damage_factor_at_one = law.value("damage_factor_at_one", 1.);
            if (!std::isfinite(l.reference_temperature_k) || l.reference_temperature_k <= 0 ||
                !std::isfinite(l.damage_factor_at_one) || l.damage_factor_at_one <= 0 ||
                std::abs(table_value(l.factor, l.reference_temperature_k) - 1.) > 1e-12)
                throw std::runtime_error("Capacity law must be positive and anchored at C_ref");
            if (cell.isothermal)
                throw std::runtime_error("Variable capacity currently requires finite_volume transfer");
        }
        if (thermal.contains("bulk_connectivity")) {
            auto bulk = thermal.at("bulk_connectivity");
            bulk["background_conductance_w_per_k"] = bulk.value("background_factor", 1.0);
            bulk["active_conductance_w_per_k"] = bulk.value("active_factor", 0.0);
            bulk["destroyed_background_conductance_w_per_k"] =
                bulk.value("destroyed_background_factor", bulk.value("background_factor", 1.0));
            bulk["destroyed_active_conductance_w_per_k"] =
                bulk.value("destroyed_active_factor", bulk.value("active_factor", 0.0));
            cell.bulk_connectivity = parse_law(bulk);
            cell.has_bulk_connectivity = true;
        }
        if (thermal.contains("bulk_response")) {
            if (!cell.has_bulk_connectivity)
                throw std::runtime_error("bulk_response requires bulk_connectivity: " + cell.id);
            cell.bulk_response = parse_surface(thermal.at("bulk_response"));
            cell.has_bulk_response = true;
        }
        const auto degradation = value.value("degradation", json::object());
        cell.aging_fraction = degradation.value("aging_fraction", 0.0);
        cell.initial_damage = degradation.value("initial_damage", 0.0);
        cell.damage_onset_k = degradation.value("onset_k", 420.0);
        cell.damage_rate_at_onset_per_s = degradation.value("rate_at_onset_per_s", 0.0);
        cell.damage_temperature_scale_k = degradation.value("temperature_scale_k", 25.0);
        cell.aging_damage_multiplier = degradation.value("aging_damage_multiplier", 0.0);
        const auto& runaway = value.at("runaway");
        cell.runaway_energy_j = runaway.at("energy_j").get<double>();
        const auto profile = runaway.value("profile", std::string("constant_power"));
        if (profile != "constant_power" && profile != "burst_tail" && profile != "kinetic")
            throw std::runtime_error("Unknown runaway profile");
        cell.runaway_pulse.profile = profile == "kinetic"      ? RunawayProfile::Kinetic
                                     : profile == "burst_tail" ? RunawayProfile::BurstTail
                                                               : RunawayProfile::ConstantPower;
        if (profile == "kinetic") {
            if (runaway.contains("duration_s"))
                throw std::runtime_error("Kinetic reaction has no prescribed duration");
            const auto& r = runaway.at("kinetics");
            auto& p = cell.reaction_kinetics;
            p.reference_temperature_k = r.at("reference_temperature_k");
            p.rate_at_reference_per_s = r.at("rate_at_reference_per_s");
            p.activation_energy_j_per_mol = r.at("activation_energy_j_per_mol");
            p.activation_damage = r.at("activation_damage");
            p.damage_sensitivity = r.value("damage_sensitivity", 2.);
            p.damage_per_conversion = r.value("damage_per_conversion", 1.);
            p.quench_rate_per_s = r.value("quench_rate_per_s", 1e-5);
            p.max_fraction_per_step = r.value("max_fraction_per_step", .02);
            p.max_temperature_rise_per_step_k = r.value("max_temperature_rise_per_step_k", 2.);
            if (runaway.contains("energy_outlet")) {
                const auto& v = runaway.at("energy_outlet");
                auto& o = cell.energy_outlet;
                o.enabled = true;
                if (v.at("sink") != "outside_domain")
                    throw std::runtime_error(
                        "Energy outlet currently supports only outside_domain, not a neighboring cell");
                cell.energy_outlet_port = v.at("port_id");
                o.opening_start_conversion = v.at("opening_start_conversion");
                o.opening_end_conversion = v.at("opening_end_conversion");
                o.closure_start_conversion = v.value("closure_start_conversion", 1.);
                o.closure_end_conversion = v.value("closure_end_conversion", 1.);
                o.heat_fraction = v.at("heat_fraction");
                o.ejected_energy_per_released_energy = v.value("ejected_energy_per_released_energy", 0.);
            }
        } else {
            cell.runaway_duration_s = runaway.at("duration_s");
            if (runaway.contains("kinetics") || runaway.contains("energy_outlet") ||
                cell.reaction_structure.enabled)
                throw std::runtime_error(
                    "Reaction kinetics, structure and energy outlet require kinetic profile");
        }
        cell.runaway_pulse.fast_fraction = runaway.value("fast_fraction", .8);
        cell.runaway_pulse.tail_time_s = runaway.value("tail_time_s", 30.);
        if (value.contains("electrical")) {
            const auto& e = value.at("electrical");
            auto& p = cell.electrical;
            p.enabled = true;
            const auto drive = e.value("current_input", std::string("measured"));
            if (drive != "measured" && drive != "voltage_limited_setpoint")
                throw std::runtime_error("Unknown electrical current_input");
            p.voltage_limited_current = drive == "voltage_limited_setpoint";
            if (e.contains("overcharge_fraction"))
                throw std::runtime_error(
                    "Retired overcharge_fraction: use charge_acceptance_fraction for ordinary charging");
            p.current_a = e.value("current_a", 0.);
            p.resistance_ohm = e.at("resistance_ohm");
            p.reference_temperature_k = e.value("reference_temperature_k", 298.15);
            p.temperature_log_slope_per_k = e.value("temperature_log_slope_per_k", 0.);
            if (e.contains("temperature_log_response")) {
                if (e.contains("temperature_log_slope_per_k"))
                    throw std::runtime_error(
                        "Specify one electrical temperature response: table OR scalar slope");
                p.temperature_log_response = parse_table(e.at("temperature_log_response"), false);
            }
            if (e.contains("damage_log_slope"))
                throw std::runtime_error("Electrical damage_log_slope was replaced by "
                                         "product_conductivity_ratio and Kirchhoff conduction");
            p.product_conductivity_ratio = e.value("product_conductivity_ratio", .2);
            const auto axis = e.value("terminal_axis", std::string("z"));
            if (axis != "x" && axis != "y" && axis != "z")
                throw std::runtime_error("Electrical terminal_axis must be x, y or z");
            p.terminal_axis = axis[0] - 'x';
            p.reversible_voltage_floor_v = e.value("reversible_voltage_floor_v", 0.);
            p.min_resistance_ohm = e.value("min_resistance_ohm", 1e-6);
            p.max_resistance_ohm = e.value("max_resistance_ohm", 1.);
            p.max_temperature_step_k = e.value("max_temperature_step_k", .5);
            p.charge_acceptance_fraction = e.value("charge_acceptance_fraction", 0.);
            p.nominal_capacity_coulomb = e.value("nominal_capacity_coulomb", 0.);
            p.charge_log_slope = e.value("charge_log_slope", 0.);
            if (e.contains("charge_state_input") && e.at("charge_state_input") != "stored_precursor")
                throw std::runtime_error("Electrical charge_state_input must be stored_precursor");
            if (p.charge_log_slope > 0. && !e.contains("charge_state_input"))
                throw std::runtime_error(
                    "Charge polarization requires explicit charge_state_input=stored_precursor; "
                    "throughput-based profiles need recalibration");
            if (e.contains("overcharge_formation_fraction"))
                throw std::runtime_error("Retired overcharge_formation_fraction: charging precursor must be "
                                         "fully paid from electrical work");
            cell.current_profile_a = e.value("current_profile_a", std::vector<std::array<double, 2>>{});
            cell.voltage_profile_v = e.value("voltage_profile_v", std::vector<std::array<double, 2>>{});
            if (!e.contains("current_a") && cell.current_profile_a.empty())
                throw std::runtime_error("Electrical heating requires a current drive");
            if (e.contains("current_a") && !cell.current_profile_a.empty())
                throw std::runtime_error("Specify constant current OR its profile, not both");
        }
        if (profile != "burst_tail" && (runaway.contains("fast_fraction") || runaway.contains("tail_time_s")))
            throw std::runtime_error("Pulse shape parameters require burst_tail");
        if (cell.heat_capacity_j_per_k <= 0.0 || cell.runaway_duration_s <= 0.0) {
            throw std::runtime_error("invalid thermal parameters for cell " + cell.id);
        }
        scenario.cells.push_back(std::move(cell));
    }
}
} // namespace battery::scenario_detail

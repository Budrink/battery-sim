#include "scenario_internal.hpp"
#include "battery_native/contracts/gas_combustion_model.hpp"
#include "battery_native/thermal/constitutive.hpp"
#include "battery_native/thermal/state_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace battery::scenario_detail {
void parse_gas_network(const json& root, Scenario& scenario, const CellIndex& cell_index,
                       const CellIndex& port_index) {
    if (root.contains("gas_network")) {
        const auto& net = root.at("gas_network");
        auto& gm = scenario.gas_model;
        const std::string transport = net.value("transport", std::string("legacy_coupled"));
        if (transport != "legacy_coupled" && transport != "split_local_equilibrium")
            throw std::runtime_error("Unknown gas transport scheme");
        gm.split_transport = transport == "split_local_equilibrium";
        const auto backend = net.value("mass_backend", std::string("cuda_global"));
        if (backend != "cuda_global" && backend != "cpu_local" && backend != "cuda_kawasaki" &&
            backend != "cuda_batched" && backend != "cuda_deterministic" && backend != "cuda_network")
            throw std::runtime_error("Unknown mass backend");
        gm.cpu_local_transport = backend == "cpu_local";
        gm.cuda_deterministic_transport = backend == "cuda_deterministic";
        gm.cuda_network_transport = backend == "cuda_network";
        const auto network_integrator = net.value("network_integrator", std::string("pairwise"));
        if (network_integrator != "pairwise" && network_integrator != "averaged")
            throw std::runtime_error("Unknown network gas integrator");
        gm.averaged_network_transport = network_integrator == "averaged";
        gm.reactive_transit = net.value("reactive_transit", false);
        if (gm.reactive_transit && !gm.averaged_network_transport)
            throw std::runtime_error("Reactive transit requires the averaged network integrator");
        const auto mixing_integrator = net.value("mixing_integrator", std::string("pairwise"));
        if (mixing_integrator != "pairwise" && mixing_integrator != "implicit" &&
            mixing_integrator != "implicit_surfaces")
            throw std::runtime_error("Unknown gas mixing integrator");
        gm.implicit_network_mixing = mixing_integrator != "pairwise";
        gm.joint_gas_surface_mixing = mixing_integrator == "implicit_surfaces";
        gm.free_gas_response_only = net.value("free_gas_response_only", false);
        if (gm.implicit_network_mixing && !gm.averaged_network_transport)
            throw std::runtime_error("Implicit mixing requires averaged network transport");
        gm.averaged_pressure_iterations = net.value("averaged_pressure_iterations", 512);
        gm.averaged_payload_iterations = net.value("averaged_payload_iterations", 16);
        // Algebraic work budget only. Large volume/area contrasts can need
        // more CG iterations; retain the kernel's unchanged 1e-6 residual
        // acceptance and the old default rather than taking smaller dt.
        if ((gm.averaged_network_transport && !gm.cuda_network_transport) ||
            gm.averaged_pressure_iterations < 4 ||
            gm.averaged_pressure_iterations > GasModel::maximum_pressure_iterations ||
            gm.averaged_payload_iterations < 4 || gm.averaged_payload_iterations > 128)
            throw std::runtime_error("Invalid averaged CUDA network controls");
        gm.coupled_gas_films = net.value("coupled_gas_films", false);
        if (gm.reactive_transit && gm.coupled_gas_films && !gm.joint_gas_surface_mixing)
            throw std::runtime_error(
                "Reactive transit needs joint surface mixing when gas films are enabled");
        if (gm.joint_gas_surface_mixing && !gm.coupled_gas_films)
            throw std::runtime_error("Joint surface mixing needs coupled gas films");
        gm.coupled_transport_interval_s = net.value("coupled_transport_interval_s", .01);
        if (!std::isfinite(gm.coupled_transport_interval_s) || gm.coupled_transport_interval_s <= 0.)
            throw std::runtime_error("Invalid coupled gas transport interval");
        if (gm.coupled_gas_films && !(gm.cuda_network_transport && gm.split_transport))
            throw std::runtime_error("Coupled gas films require CUDA network split transport");
        gm.cuda_batched_transport =
            backend == "cuda_batched" || gm.cuda_deterministic_transport || gm.cuda_network_transport;
        gm.cuda_kawasaki_transport = backend == "cuda_kawasaki" || gm.cuda_batched_transport;
        gm.deterministic_sweeps_per_step =
            net.value("deterministic_sweeps_per_step", gm.deterministic_sweeps_per_step);
        gm.deterministic_pressure_fraction =
            net.value("deterministic_pressure_fraction", gm.deterministic_pressure_fraction);
        gm.network_pressure_fraction = net.value("network_pressure_fraction", gm.network_pressure_fraction);
        gm.mixing_diffusivity_m2_s = net.value("mixing_diffusivity_m2_s", 0.);
        gm.mixing_flow_fraction = net.value("mixing_flow_fraction", 0.);
        gm.thermal_stirring_coefficient = net.value("thermal_stirring_coefficient", 0.);
        gm.mixing_length_cap_m = net.value("mixing_length_cap_m", .05);
        gm.thermal_stirring_length_m = net.value("thermal_stirring_length_m", .02);
        if (gm.thermal_stirring_coefficient > 0. && !net.contains("thermal_stirring_length_m"))
            throw std::runtime_error("Thermal stirring requires an explicit physical "
                                     "thermal_stirring_length_m, independent of mesh spacing");
        gm.pressure_pattern_predictor = net.value("pressure_pattern_predictor", false);
        gm.dynamic_viscosity_pa_s = net.value("dynamic_viscosity_pa_s", 0.);
        if (net.contains("directional_memory")) {
            const auto& memory = net.at("directional_memory");
            if (!memory.is_object())
                throw std::runtime_error("directional_memory must be an object");
            for (const auto& [key, value] : memory.items())
                if (key != "enabled")
                    throw std::runtime_error(
                        "Directional memory now uses momentum, not strength/tau; unknown field: " + key);
            gm.directional_inertia = memory.value("enabled", false);
        }
        if (gm.directional_inertia &&
            (!(std::isfinite(gm.dynamic_viscosity_pa_s) && gm.dynamic_viscosity_pa_s > 0.) ||
             !(gm.split_transport && gm.cuda_network_transport && gm.averaged_network_transport)))
            throw std::runtime_error(
                "Momentum needs positive dynamic_viscosity_pa_s and split averaged CUDA network");
        if (gm.pressure_pattern_predictor && !gm.averaged_network_transport)
            throw std::runtime_error("Pressure pattern predictor requires averaged network transport");
        if (net.contains("mixing_connectivity")) {
            scenario.has_gas_mixing_connectivity = true;
            scenario.gas_mixing_connectivity = parse_law(net.at("mixing_connectivity"));
        }
        gm.vapor_fuel_heat_j_kg = net.value("vapor_fuel_heat_j_kg", 0.);
        gm.batch_pressure_fraction = net.value("batch_pressure_fraction", gm.batch_pressure_fraction);
        gm.batch_mean_threshold = net.value("batch_mean_threshold", gm.batch_mean_threshold);
        gm.kawasaki_packet_fraction = net.value("kawasaki_packet_fraction", gm.kawasaki_packet_fraction);
        gm.kawasaki_tail_atol_pa = net.value("kawasaki_tail_atol_pa", gm.kawasaki_tail_atol_pa);
        gm.kawasaki_tail_rtol = net.value("kawasaki_tail_rtol", gm.kawasaki_tail_rtol);
        gm.kawasaki_seed = net.value("kawasaki_seed", gm.kawasaki_seed);
        const auto integrator = net.value("local_integrator", std::string("explicit"));
        if (integrator != "explicit" && integrator != "implicit")
            throw std::runtime_error("Unknown local gas integrator");
        gm.implicit_local_transport = integrator == "implicit";
        gm.implicit_temperature_tolerance_k =
            net.value("implicit_temperature_tolerance_k", gm.implicit_temperature_tolerance_k);
        gm.implicit_tracer_relative_tolerance =
            net.value("implicit_tracer_relative_tolerance", gm.implicit_tracer_relative_tolerance);
        gm.local_exchange_interval_s = net.value("local_exchange_interval_s", gm.local_exchange_interval_s);
        gm.reference_temperature_k = net.at("reference_temperature_k");
        gm.cv_j_per_kg_k = net.at("cv_j_per_kg_k");
        gm.specific_r_j_per_kg_k = net.at("specific_r_j_per_kg_k");
        gm.max_fraction_per_step = net.value("max_fraction_per_step", .01);
        gm.max_temperature_step_k = net.value("max_temperature_step_k", .5);
        gm.pressure_relaxation_fraction =
            net.value("pressure_relaxation_fraction", gm.pressure_relaxation_fraction);
        gm.pressure_absolute_tolerance_pa =
            net.value("pressure_absolute_tolerance_pa", gm.pressure_absolute_tolerance_pa);
        gm.pressure_relative_tolerance =
            net.value("pressure_relative_tolerance", gm.pressure_relative_tolerance);
        std::unordered_map<std::string, int> gas_ids;
        const auto common_r3 =
            net.contains("combustion") ? normalized_gas_combustion(net.at("combustion"), true) : json();
        for (const auto& v : net.at("compartments")) {
            GasCompartment g;
            g.id = v.at("id");
            g.volume_m3 = v.at("volume_m3");
            g.initial_mass_kg = v.at("initial_mass_kg");
            g.initial_temperature_k = v.at("initial_temperature_k");
            g.initial_chemical_j = v.value("initial_chemical_j", 0.);
            g.initial_oxidizer_mass_fraction = v.value("initial_oxidizer_mass_fraction", 0.);
            const auto& r3 = common_r3.is_null() ? v : common_r3;
            g.reaction_rate_per_s = r3.value("reaction_rate_per_s", 0.);
            g.oxidizer_heat_j_per_kg = r3.value("oxidizer_heat_j_per_kg", 13.1e6);
            g.ignition_temperature_k = r3.value("ignition_temperature_k", 0.);
            g.reaction_reference_k = r3.value("reaction_reference_k", 600.);
            g.reaction_activation_j_per_mol = r3.value("reaction_activation_j_per_mol", 0.);
            if (v.contains("ambient_boundary")) {
                const auto& a = v.at("ambient_boundary");
                auto& b = g.ambient;
                b.enabled = true;
                for (const auto& item : a.items())
                    if (item.key() != "pressure_pa" && item.key() != "temperature_k" &&
                        item.key() != "oxidizer_mass_fraction" && item.key() != "area_m2" &&
                        item.key() != "discharge_coefficient" && item.key() != "renewal_flow_m3_s")
                        throw std::runtime_error("Unknown ambient boundary property: " + item.key());
                b.pressure_pa = a.at("pressure_pa");
                b.temperature_k = a.at("temperature_k");
                b.oxidizer_mass_fraction = a.at("oxidizer_mass_fraction");
                b.area_m2 = a.at("area_m2");
                b.discharge_coefficient = a.value("discharge_coefficient", .7);
                b.renewal_flow_m3_s = a.value("renewal_flow_m3_s", 0.);
            }
            if (g.reaction_rate_per_s > 0. &&
                (!r3.contains("reaction_reference_k") || !r3.contains("reaction_activation_j_per_mol")))
                throw std::runtime_error(
                    "R3 requires its own reaction_reference_k and reaction_activation_j_per_mol");
            if (v.contains("local_equilibrium")) {
                const auto& binding = v.at("local_equilibrium");
                for (const auto& item : binding.items())
                    if (item.key() != "cell_id" && item.key() != "thermal_volume" &&
                        item.key() != "reactant_density_kg_m3" && item.key() != "residue_density_kg_m3")
                        throw std::runtime_error("Unknown local_equilibrium property: " + item.key());
                g.thermal_cell = static_cast<int>(cell_index.at(binding.at("cell_id").get<std::string>()));
                g.thermal_volume = binding.at("thermal_volume");
                g.reactant_density_kg_m3 = binding.value("reactant_density_kg_m3", 0.);
                g.residue_density_kg_m3 = binding.value("residue_density_kg_m3", 0.);
            }
            if (v.contains("contents_id"))
                g.owner_cell = static_cast<int>(cell_index.at(v.at("contents_id").get<std::string>()));
            if (!gas_ids.emplace(g.id, static_cast<int>(scenario.gas_compartments.size())).second)
                throw std::runtime_error("Duplicate gas compartment");
            scenario.gas_compartments.push_back(g);
        }
        for (const auto& v : net.value("channels", json::array())) {
            GasChannel ch;
            ch.id = v.at("id");
            ch.from = gas_ids.at(v.at("from").get<std::string>());
            const std::string to = v.at("to");
            ch.to = to == "outside_domain" ? -1 : gas_ids.at(to);
            const std::string kind = v.at("kind");
            if (kind != "vent" && kind != "rupture" && kind != "internal" && kind != "passage")
                throw std::runtime_error("Unknown gas channel kind");
            ch.internal = kind == "internal";
            ch.passage = kind == "passage";
            ch.shell_thermal_volume = v.value("shell_thermal_volume", -1);
            ch.mixing_axis = v.value("mixing_axis", 0);
            if (v.contains("flow_direction")) {
                const auto direction = v.at("flow_direction").get<std::vector<double>>();
                if (direction.size() != 3)
                    throw std::runtime_error("flow_direction requires three components");
                const double length = std::hypot(direction[0], direction[1], direction[2]);
                if (!std::isfinite(length) || length <= 0.)
                    throw std::runtime_error("Invalid flow_direction");
                ch.flow_direction = {direction[0] / length, direction[1] / length, direction[2] / length};
            }
            if (gm.directional_inertia && ch.flow_direction.x == 0. && ch.flow_direction.y == 0. &&
                ch.flow_direction.z == 0.)
                throw std::runtime_error("Directional memory requires explicit flow_direction for channel " +
                                         ch.id);
            if (ch.mixing_axis < 0 || ch.mixing_axis > 2)
                throw std::runtime_error("Invalid gas mixing axis");
            if (ch.internal || ch.passage) {
                if (v.contains("port_id"))
                    throw std::runtime_error("Internal pore path must not masquerade as an exterior port");
                ch.shell_object = scenario.gas_compartments[ch.from].thermal_cell;
            } else {
                ch.port_id = v.at("port_id");
                const auto& p = scenario.ports.at(port_index.at(ch.port_id));
                ch.shell_object = static_cast<int>(p.cell_index);
            }
            ch.area_m2 = v.value("area_m2", 0.);
            ch.discharge_coefficient = v.value("discharge_coefficient", .7);
            ch.mixing_length_m = v.value("mixing_length_m", .01);
            ch.inertial_length_m = v.value("inertial_length_m", 0.);
            if (gm.directional_inertia && !(std::isfinite(ch.inertial_length_m) && ch.inertial_length_m > 0.))
                throw std::runtime_error("Momentum requires explicit positive inertial_length_m for " +
                                         ch.id);
            ch.hydraulic_resistance_factor = v.value("hydraulic_resistance_factor", 1.);
            ch.opening_pressure_pa = v.at("opening_pressure_pa");
            ch.outside_pressure_pa = v.value("outside_pressure_pa", 101325.);
            ch.rupture = kind == "rupture";
            ch.strength_reference_k = v.value("strength_reference_k", 298.15);
            ch.strength_temperature_scale_k = v.value("strength_temperature_scale_k", 1e6);
            if (v.contains("surface_link")) {
                const auto& link = v.at("surface_link");
                if (!link.is_object() || link.size() != 1 || !link.contains("h_w_per_m2_k"))
                    throw std::runtime_error("surface_link requires only h_w_per_m2_k");
                ch.surface_link_h_w_per_m2_k = link.at("h_w_per_m2_k");
                if (!std::isfinite(ch.surface_link_h_w_per_m2_k) || ch.surface_link_h_w_per_m2_k <= 0)
                    throw std::runtime_error("surface_link h must be positive");
            }
            if (v.contains("damage_opening")) {
                const auto& o = v.at("damage_opening");
                if (!o.is_object())
                    throw std::runtime_error("damage_opening must be an object");
                for (const auto& item : o.items())
                    if (item.key() != "damage_onset" && item.key() != "exponent" &&
                        item.key() != "pressure_damage_rate_per_s" && item.key() != "max_damage_step" &&
                        item.key() != "seam_width_m" && item.key() != "seam_strength_factor" &&
                        item.key() != "fracture_energy_j_per_m2" &&
                        item.key() != "initial_mechanical_damage" && item.key() != "mode" &&
                        item.key() != "pixel_rate_per_s" && item.key() != "pixel_pressure_exponent" &&
                        item.key() != "seed" && item.key() != "fragment_mean_area_m2" &&
                        item.key() != "fixed_area_fraction" && item.key() != "area_rate_per_s" &&
                        item.key() != "pressure_exponent" && item.key() != "valve" &&
                        item.key() != "whole_plate")
                        throw std::runtime_error("Unknown damage_opening property: " + item.key());
                ch.opening.enabled = true;
                const auto mode = o.value("mode", std::string("progressive"));
                if (mode != "progressive" && mode != "binary_pixels" && mode != "stochastic_pixels" &&
                    mode != "stochastic_area" && mode != "stochastic_fraction" &&
                    mode != "deterministic_aperture")
                    throw std::runtime_error("Unknown shell opening mode");
                ch.opening.deterministic_aperture = mode == "deterministic_aperture";
                ch.opening.whole_plate = o.value("whole_plate", false);
                if (ch.opening.whole_plate && !ch.opening.deterministic_aperture)
                    throw std::runtime_error("Whole-plate mechanics requires deterministic_aperture");
                if (!ch.opening.deterministic_aperture &&
                    (o.contains("area_rate_per_s") || o.contains("pressure_exponent") || o.contains("valve")))
                    throw std::runtime_error("Aperture parameters require deterministic_aperture");
                if (ch.opening.deterministic_aperture) {
                    for (const auto* key :
                         {"seed", "pixel_rate_per_s", "pixel_pressure_exponent", "fragment_mean_area_m2",
                          "fixed_area_fraction", "damage_onset", "exponent", "pressure_damage_rate_per_s"})
                        if (o.contains(key))
                            throw std::runtime_error("Inapplicable deterministic aperture property: " +
                                                     std::string(key));
                    ch.opening.area_rate_per_s = o.value("area_rate_per_s", 1.);
                    ch.opening.pressure_exponent = o.value("pressure_exponent", 1.);
                    if (o.contains("valve")) {
                        const auto& valve = o.at("valve");
                        if (!valve.is_object())
                            throw std::runtime_error("Valve must be an object");
                        for (const auto& item : valve.items())
                            if (item.key() != "radius_m" && item.key() != "center_m" &&
                                item.key() != "pressure_factor")
                                throw std::runtime_error("Unknown valve property");
                        const auto xy = valve.at("center_m").get<std::array<double, 2>>();
                        ch.opening.valve_u_m = xy[0];
                        ch.opening.valve_v_m = xy[1];
                        ch.opening.valve_radius_m = valve.at("radius_m");
                        ch.opening.valve_pressure_factor = valve.at("pressure_factor");
                        if (ch.opening.valve_radius_m <= 0)
                            throw std::runtime_error("Valve radius must be positive");
                    }
                }
                ch.opening.binary_pixels = mode == "binary_pixels" || mode == "stochastic_pixels";
                ch.opening.stochastic_pixels = mode == "stochastic_pixels";
                ch.opening.stochastic_area = mode == "stochastic_area" || mode == "stochastic_fraction";
                if (o.contains("fixed_area_fraction") && mode != "stochastic_fraction")
                    throw std::runtime_error("fixed_area_fraction requires stochastic_fraction mode");
                if (mode == "stochastic_fraction" && o.contains("fragment_mean_area_m2"))
                    throw std::runtime_error("Fixed-fraction mode has no random fragment area");
                ch.opening.fixed_area_fraction =
                    mode == "stochastic_fraction" ? o.value("fixed_area_fraction", .02) : 0.;
                if (mode == "stochastic_fraction" &&
                    (!std::isfinite(ch.opening.fixed_area_fraction) || ch.opening.fixed_area_fraction <= 0 ||
                     ch.opening.fixed_area_fraction > 1))
                    throw std::runtime_error("Invalid fixed area fraction");
                ch.opening.fragment_mean_area_m2 = o.value("fragment_mean_area_m2", 1e-5);
                ch.opening.pixel_rate_per_s = o.value("pixel_rate_per_s", 1.);
                ch.opening.pixel_pressure_exponent = o.value("pixel_pressure_exponent", 1.);
                if (o.contains("seed") && !o.at("seed").is_number_unsigned())
                    throw std::runtime_error("Shell seed must be an unsigned integer");
                ch.opening.rupture_seed = o.value("seed", std::uint64_t{1});
                ch.opening.initial_mechanical_damage = o.value("initial_mechanical_damage", 0.);
                ch.opening.damage_onset = o.value("damage_onset", .2);
                ch.opening.exponent = o.value("exponent", 2.);
                ch.opening.pressure_damage_rate_per_s = o.value("pressure_damage_rate_per_s", 1.);
                ch.opening.max_damage_step = o.value("max_damage_step", .02);
                ch.opening.seam_width_m = o.value("seam_width_m", 0.);
                ch.opening.seam_strength_factor = o.value("seam_strength_factor", 1.);
                ch.opening.fracture_energy_j_per_m2 = o.value("fracture_energy_j_per_m2", 0.);
            }
            scenario.gas_channels.push_back(ch);
        }
        for (const auto& v : net.value("thermal_contacts", json::array())) {
            GasContact c;
            c.compartment = gas_ids.at(v.at("compartment").get<std::string>());
            c.port_id = v.at("port_id");
            if (v.contains("connectivity") && v.contains("h_w_per_m2_k"))
                throw std::runtime_error("Supply gas connectivity or reference h, not both");
            c.has_connectivity = v.contains("connectivity");
            if (c.has_connectivity)
                c.connectivity = parse_law(v.at("connectivity"));
            else
                c.h_w_per_m2_k = v.at("h_w_per_m2_k");
            c.gas_surface_connectivity = parse_surface(
                v.value("gas_surface_connectivity", net.value("surface_connectivity", json::object())));
            c.has_solid_surface_connectivity = v.contains("solid_surface_connectivity");
            if (c.has_solid_surface_connectivity)
                c.solid_surface_connectivity = parse_surface(v.at("solid_surface_connectivity"));
            c.thermal_volume = v.value("thermal_volume", -1);
            scenario.gas_contacts.push_back(c);
        }
        scenario.gas_surface_connectivity = parse_surface(net.value("surface_connectivity", json::object()));
        for (const auto& v : root.at("cells"))
            if (v.contains("conversion")) {
                auto& c = scenario.cells[cell_index.at(v.at("id").get<std::string>())];
                auto& p = c.conversion;
                const auto& x = v.at("conversion");
                p.enabled = true;
                if (x.at("chemistry") != "r1-r2-r3")
                    throw std::runtime_error("Only R1/R2/R3 chemistry is supported");
                for (const char* old : {"l0_m", "lcrit_m", "energy_scale_j_per_kg", "activation_rate_per_s",
                                        "absorption_j_per_kg", "overcharge_conversion_per_charge",
                                        "precursor_yield_voltage_per_v", "precursor_voltage_ref_v"})
                    if (x.contains(old))
                        throw std::runtime_error(std::string("Obsolete chemistry field: ") + old);
                p.mass_kg = x.at("mass_kg");
                p.reactant_cp = x.at("reactant_cp");
                p.activated_cp = x.at("activated_cp");
                p.residue_cp = x.at("residue_cp");
                p.net_heat_j_per_kg = x.at("net_heat_j_per_kg");
                p.degradation_rate_per_s = x.at("degradation_rate_per_s");
                p.reaction_rate_per_s = x.at("reaction_rate_per_s");
                p.reaction_reference_k = x.at("reaction_reference_k");
                p.reaction_activation_j_per_mol = x.at("reaction_activation_j_per_mol");
                p.degradation_reference_k = x.at("degradation_reference_k");
                p.degradation_activation_j_per_mol = x.at("degradation_activation_j_per_mol");
                p.initial_precursor_fraction = x.value("initial_precursor_fraction", 0.);
                p.initial_charged_fraction = x.value("initial_charged_fraction", 0.);
                p.charge_conversion_per_nominal_charge = x.value("charge_conversion_per_nominal_charge", 0.);
                p.precursor_yield = x.value("precursor_yield", 1.);
                p.precursor_yield_temperature_per_k = x.value("precursor_yield_temperature_per_k", 0.);
                p.precursor_temperature_ref_k = x.value("precursor_temperature_ref_k", 298.15);
                p.runaway_reporting_rate_per_s = x.value("runaway_reporting_rate_per_s", 1e-5);
                p.gas_yield_reference_k = x.value("gas_yield_reference_k", p.reaction_reference_k);
                p.gas_yield_temperature_per_k = x.value("gas_yield_temperature_per_k", 0.);
                p.gas_yield_density_exponent = x.value("gas_yield_density_exponent", 0.);
                p.gas_yield_max = x.value("gas_yield_max", .95);
                p.gas_yield = x.at("gas_yield");
                p.gas_chemical_fraction = x.at("gas_chemical_fraction");
                if (x.contains("volatile_phase")) {
                    const auto& q = x.at("volatile_phase");
                    auto& v = p.volatile_phase;
                    v.initial_mass_kg = q.at("initial_mass_kg");
                    v.liquid_cp = q.at("liquid_cp");
                    v.density_kg_m3 = q.at("density_kg_m3");
                    v.latent_j_per_kg = q.at("latent_j_per_kg");
                    v.boiling_temperature_k = q.at("boiling_temperature_k");
                    v.saturation_reference_pa = q.at("saturation_reference_pa");
                    v.vapor_specific_r = q.at("vapor_specific_r");
                    v.relaxation_per_s = q.at("relaxation_per_s");
                    v.chemical_j_per_kg = q.value("chemical_j_per_kg", 0.);
                    // The shared volatile pseudo-species has the same chemical
                    // reserve before and after evaporation. This overrides the
                    // legacy duplicate, not heat capacity or latent heat.
                    if (!common_r3.is_null() && net.contains("vapor_fuel_heat_j_kg"))
                        v.chemical_j_per_kg = gm.vapor_fuel_heat_j_kg;
                }
                p.max_fraction_per_step = x.value("max_fraction_per_step", .01);
                p.max_temperature_step_k = x.value("max_temperature_step_k", .5);
                const auto f = x.at("residue_conductivity").get<std::array<double, 3>>();
                for (int k = 0; k < 3; ++k)
                    p.residue_conductivity[k] = f[k];
                p.gas_compartment = gas_ids.at(x.at("gas_compartment").get<std::string>());
            }
    } else
        for (const auto& v : root.at("cells"))
            if (v.contains("conversion"))
                throw std::runtime_error("Conversion requires an explicit gas network");
}
} // namespace battery::scenario_detail

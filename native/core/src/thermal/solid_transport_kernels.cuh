// Solid/boundary heat exchange, directed connectivity and explicit stability bounds.
// Private implementation fragment, included only by cuda_solver.cu.
struct DeviceLink {
    int a, b, interface_index, law_index;
    double weight, series_resistance_k_per_w, boundary_temperature_k, maximum_g;
    int boundary_index;
    int geometry_axis;
    double series_resistance_a_k_per_w;
};
struct ProfilePoint {
    double time, temperature;
};
__device__ double boundary_temperature(const DeviceLink& e, double time, const int* offsets,
                                       const ProfilePoint* points) {
    if (e.boundary_index < 0)
        return e.boundary_temperature_k;
    int lo = offsets[e.boundary_index], hi = offsets[e.boundary_index + 1];
    if (hi == lo)
        return e.boundary_temperature_k;
    if (time <= points[lo].time)
        return points[lo].temperature;
    if (time >= points[hi - 1].time)
        return points[hi - 1].temperature;
    --hi;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (points[mid].time <= time)
            lo = mid;
        else
            hi = mid;
    }
    const auto a = points[lo], b = points[hi];
    return a.temperature + (b.temperature - a.temperature) * (time - a.time) / (b.time - a.time);
}
auto law_key(const ConnectivityLaw& l) {
    std::vector<double> key{static_cast<double>(l.kind),
                            l.l0_m,
                            l.l_low_m,
                            l.l_high_m,
                            l.transition_temperature_k,
                            l.transition_width_k,
                            l.hysteresis_k,
                            l.relaxation_time_s,
                            l.background_conductance_w_per_k,
                            l.active_conductance_w_per_k,
                            l.activation_width_m,
                            l.damage_onset_k,
                            l.damage_rate_at_onset_per_s,
                            l.damage_temperature_scale_k,
                            l.damage_l_shift_m,
                            l.damage_conductance_factor_at_one,
                            l.destruction_damage,
                            l.destroyed_l0_m,
                            l.destroyed_l_m,
                            l.destroyed_background_conductance_w_per_k,
                            l.destroyed_active_conductance_w_per_k,
                            l.endpoint_damage_weight,
                            l.geometry_reference_logit,
                            l.destroyed_geometry_reference_logit,
                            l.radiation_coefficient_w_per_k4,
                            l.radiation_temperature_limit_k,
                            l.interface_damage_log_response};
    for (const auto& r : {l.endpoint_a, l.endpoint_b}) {
        key.push_back(r.reference_temperature_k);
        key.push_back(r.temperature_scale_k);
        for (const auto v : {r.linear, r.quadratic, r.damage, r.temperature_contrast, r.damage_contrast})
            for (double x : {v.x, v.y, v.z})
                key.push_back(x);
        key.push_back(r.temperature_log_response.count);
        for (int i = 0; i < r.temperature_log_response.count; ++i) {
            key.push_back(r.temperature_log_response.temperature_k[i]);
            key.push_back(r.temperature_log_response.value[i]);
        }
    }
    return key;
}
__device__ double clamp(double x, double lo, double hi) {
    return fmin(fmax(x, lo), hi);
}
__device__ double logistic(double x) {
    return 1.0 / (1.0 + exp(-clamp(x, -80, 80)));
}
__device__ double damage_rate(double t, double onset, double rate, double scale) {
    return t < onset || rate <= 0 ? 0 : rate * exp(clamp((t - onset) / scale, -80, 80));
}
__device__ void minimum_positive(double* address, double value) {
    auto ptr = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = atomicCAS(ptr, 0ULL, 0ULL), assumed;
    while (value < __longlong_as_double(old)) {
        assumed = old;
        old = atomicCAS(ptr, assumed, __double_as_longlong(value));
        if (old == assumed)
            break;
    }
}
__device__ void block_minimum(double value, double* output) {
    __shared__ double partial[128];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] = fmin(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        minimum_positive(output, partial[0]);
}
__global__ void cfl_kernel(int count, const DeviceVolume* volume, const DeviceLink* links, const int* offsets,
                           const int* adjacency, const Material* materials, RuntimeParameters runtime,
                           double safety, double* cfl, DeviceRows patch_rows,
                           const GasThermalPatch* patches) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool spatial = i < count && !materials[volume[i].object].isothermal;
    double sum = 0;
    for (int j = spatial ? offsets[i] : 0; j < (spatial ? offsets[i + 1] : 0); ++j) {
        const auto& e = links[adjacency[j]];
        double scale = e.b < 0 ? runtime.ambient_conductance_scale
                               : (e.interface_index >= 0 ? runtime.interface_conductance_scale : 1.0);
        double g = e.maximum_g * scale;
        const auto pa = materials[volume[e.a].object].boundary_phase;
        const auto pb = e.b >= 0 ? materials[volume[e.b].object].boundary_phase : BulkBoundaryPhase{};
        if (g > 0)
            g = 1.0 /
                (1.0 / g +
                 minimum_phase_resistance(
                     e.series_resistance_k_per_w, e.series_resistance_a_k_per_w, pa, pb,
                     maximum_structure(materials[volume[e.a].object], e.geometry_axis),
                     e.b >= 0 ? maximum_structure(materials[volume[e.b].object], e.geometry_axis) : 1.));
        sum += g;
    }
    // Ordinary gas contacts are static. Rupture links are bounded separately
    // from their CURRENT opening and flow, never from a fully open panel.
    if (spatial)
        for (int j = patch_rows.offsets[i]; j < patch_rows.offsets[i + 1]; ++j) {
            const auto& patch = patches[patch_rows.ids[j]];
            if (patch.channel < 0)
                sum += patch.conductance_w_per_k;
        }
    const double factor = spatial ? minimum_capacity_factor(materials[volume[i].object].capacity_law) : 1.;
    double capacity = spatial ? volume[i].capacity_j_per_k * factor : 1.;
    if (spatial && materials[volume[i].object].conversion.enabled) {
        const auto& m = materials[volume[i].object];
        capacity = conversion_min_capacity(m.conversion, volume[i].capacity_j_per_k,
                                           volume[i].capacity_j_per_k / m.capacity);
    }
    double limit = sum > 0 ? safety * capacity / sum : 1e300;
    block_minimum(limit, cfl);
}
// Isothermal objects exchange heat through their WHOLE heat capacity, not the
// capacity of a boundary voxel. The bound must not shrink when the grid changes.
__global__ void isothermal_cfl_kernel(const DeviceVolume* volumes, const DeviceLink* links,
                                      const int* object_offsets, const int* offsets, const int* adjacency,
                                      const Material* materials, RuntimeParameters runtime, double safety,
                                      double* cfl) {
    const int object = blockIdx.x, t = threadIdx.x;
    if (!materials[object].isothermal)
        return;
    double sum = 0;
    for (int i = object_offsets[object] + t; i < object_offsets[object + 1]; i += blockDim.x)
        for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
            const auto& e = links[adjacency[j]];
            double scale = e.b < 0 ? runtime.ambient_conductance_scale : runtime.interface_conductance_scale;
            double g = e.maximum_g * scale;
            const auto pa = materials[volumes[e.a].object].boundary_phase;
            const auto pb = e.b >= 0 ? materials[volumes[e.b].object].boundary_phase : BulkBoundaryPhase{};
            sum +=
                g > 0
                    ? 1.0 / (1.0 / g +
                             minimum_phase_resistance(
                                 e.series_resistance_k_per_w, e.series_resistance_a_k_per_w, pa, pb,
                                 maximum_structure(materials[volumes[e.a].object], e.geometry_axis),
                                 e.b >= 0 ? maximum_structure(materials[volumes[e.b].object], e.geometry_axis)
                                          : 1.))
                    : 0;
        }
    __shared__ double partial[128];
    partial[t] = sum;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (t < stride)
            partial[t] += partial[t + stride];
        __syncthreads();
    }
    if (t == 0 && partial[0] > 0)
        minimum_positive(cfl, safety * materials[object].capacity / partial[0]);
}
__global__ void stability_kernel(int count, const ThermalState* state, const DeviceVolume* volume,
                                 const Material* materials, RuntimeParameters runtime, const double* cfl,
                                 double* dt, double time, double safety, DeviceRows patch_rows,
                                 const GasThermalPatch* patches, const double* conductance,
                                 const double* joule_per_i2, double horizon, double* hard_dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double limit = *cfl, hard = *cfl;
    if (i < count) {
        const auto& m = materials[volume[i].object];
        const auto p = m.boundary_phase;
        double joule_power = 0.;
        if (m.electrical.enabled) {
            const double current = maximum_current(m.electrical, m.current, time, time + horizon);
            joule_power = current * current * joule_per_i2[i];
            const double capacity =
                m.conversion.enabled ? conversion_min_capacity(m.conversion, volume[i].capacity_j_per_k,
                                                               volume[i].capacity_j_per_k / m.capacity)
                                     : volume[i].capacity_j_per_k * minimum_capacity_factor(m.capacity_law);
            if (joule_power > 0)
                limit = fmin(limit, m.electrical.max_temperature_step_k * capacity / joule_power);
        }
        double aperture_g = 0.;
        for (int j = patch_rows.offsets[i]; j < patch_rows.offsets[i + 1]; ++j) {
            const int patch = patch_rows.ids[j];
            if (patches[patch].channel >= 0)
                aperture_g += conductance[patch];
        }
        if (aperture_g > 0.) {
            const double capacity =
                m.conversion.enabled ? conversion_min_capacity(m.conversion, volume[i].capacity_j_per_k,
                                                               volume[i].capacity_j_per_k / m.capacity)
                                     : volume[i].capacity_j_per_k * minimum_capacity_factor(m.capacity_law);
            // 1/cfl bounds every static G/(safety*C). ADD the current
            // aperture contribution: min(dt_static,dt_aperture) is not enough
            // when both exchanges consume the same solid heat capacity.
            limit = 1. / (1. / limit + aperture_g / (safety * capacity));
            hard = 1. / (1. / hard + aperture_g / (safety * capacity));
        }
        if (m.conversion.enabled) {
            const auto& c = m.conversion;
            const auto& s = state[i];
            const auto& g = m.gas_model;
            const double fraction = volume[i].capacity_j_per_k / m.capacity;
            const double current =
                m.electrical.enabled ? maximum_charging_current(m.electrical, m.current, time, time + horizon)
                                     : 0.;
            const double charge_rate =
                charging_precursor_rate(c, m.electrical, current, charge_capacity_fraction(c, s, fraction)) *
                runtime.runaway_power_scale;
            const double r1 = degradation_rate(c, s.temperature_k) * runtime.runaway_power_scale;
            const double kb = completion_rate(c, s.temperature_k) * runtime.runaway_power_scale;
            const double phi = precursor_branch(c, s.temperature_k);
            const double capmin = conversion_min_capacity(c, volume[i].capacity_j_per_k, fraction);
            const double ra = phi * r1 * healthy_mass(s) + charge_rate * s.reactant_kg;
            const double fresh = fresh_completion_bound(kb, horizon);
            const double rb = completion_rate_bound(kb, s.activated_kg, ra, horizon);
            const double yield = maximum_gas_yield(c);
            const double capacity_change = fmax(std::abs(c.residue_cp - c.activated_cp),
                                                std::abs((1 - yield) * c.residue_cp - c.activated_cp));
            // Bound total transformed mass, not the fractional turnover of an
            // arbitrarily small quasi-steady intermediate. Include new feed in
            // the heat bound because it can complete within this same substep.
            const double turnover = r1 * healthy_mass(s) + charge_rate * s.reactant_kg + rb;
            if (turnover > 0)
                limit = fmin(limit, c.max_fraction_per_step * c.mass_kg * fraction / turnover);
            const double power =
                joule_power + kb * s.precursor_charge_energy_j +
                charge_rate * s.reactant_kg * c.net_heat_j_per_kg * fresh +
                r1 * healthy_mass(s) *
                    fmax(std::abs(c.activated_cp - c.reactant_cp), std::abs(c.residue_cp - c.reactant_cp)) *
                    std::abs(s.temperature_k - g.reference_temperature_k) +
                charge_rate * s.reactant_kg * std::abs(c.activated_cp - c.reactant_cp) *
                    std::abs(s.temperature_k - g.reference_temperature_k) +
                rb * (c.net_heat_j_per_kg +
                      yield * g.cv_j_per_kg_k * std::abs(s.temperature_k - g.reference_temperature_k) +
                      capacity_change * std::abs(s.temperature_k - g.reference_temperature_k));
            if (power > 0)
                limit = fmin(limit, fmin(c.max_temperature_step_k, .1 * s.temperature_k) * capmin / power);
        }
        if (m.pulse.profile == RunawayProfile::Kinetic && state[i].mode != CellMode::Spent) {
            const auto& s = state[i];
            const auto& k = m.kinetics;
            const double aging = damage_rate(s.temperature_k, m.onset, m.rate, m.temperature_scale) *
                                 m.aging_multiplier * runtime.damage_rate_scale;
            if (aging > 0 && s.damage < 1)
                limit = fmin(limit, k.max_fraction_per_step / aging);
            if (reaction_live(s) && runtime.runaway_power_scale > 0) {
                const double rate = reaction_rate(k, s.temperature_k, s.damage) * runtime.runaway_power_scale;
                const double budget = m.energy * volume[i].capacity_j_per_k / m.capacity;
                const double opened = outlet_opening(m.outlet, reaction_conversion(s, budget));
                const double total_rate = rate * (1 + opened * m.outlet.ejected_energy_per_released_energy);
                limit = fmin(limit, -log1p(-k.max_fraction_per_step) / total_rate);
                const double power = rate * s.chemical_energy_remaining_j;
                if (power > 0)
                    limit = fmin(limit, k.max_temperature_rise_per_step_k * volume[i].capacity_j_per_k *
                                            minimum_capacity_factor(m.capacity_law) / power);
            }
        }
        if (p.relaxation_time_s > 0 && state[i].runaway_started_s >= 0) {
            const double left = 6 * p.relaxation_time_s - (time - state[i].runaway_started_s);
            if (left > 1e-12)
                limit = fmin(limit, fmin(.1 * p.relaxation_time_s, left));
        }
    }
    if (i < count && state[i].mode == CellMode::Runaway && runtime.runaway_power_scale > 0) {
        const auto& m = materials[volume[i].object];
        if (m.pulse.profile == RunawayProfile::Kinetic) { /* limited above, including quenched nodes */
        } else if (m.pulse.profile == RunawayProfile::BurstTail) {
            // Resolve a rapid pulse in the coupled transport, even though its
            // energy integral itself is exact for arbitrarily large dt.
            const double age = state[i].reaction_progress_s;
            const double tau =
                m.pulse.fast_fraction > 0 && age < 6 * m.duration ? m.duration : m.pulse.tail_time_s;
            const double left = pulse_end(m.pulse, m.duration) - age;
            if (left > 1e-12)
                limit = fmin(limit, fmin(.1 * tau, left) / runtime.runaway_power_scale);
        } else {
            double energy = m.energy * volume[i].capacity_j_per_k / m.capacity;
            double remaining = fmax(0.0, energy - state[i].released_energy_j);
            double power = energy / m.duration * runtime.runaway_power_scale;
            if (power > 0 && remaining > energy * 1e-12)
                limit = fmin(limit, remaining / power);
        }
    }
    block_minimum(limit, dt);
    block_minimum(hard, hard_dt);
}
__device__ InterfaceState evolve(const DeviceLink& edge, const ConnectivityLaw& l, InterfaceState s, double t,
                                 double damage, double dt, double scale) {
    s.damage = fmax(s.damage, l.endpoint_damage_weight * damage);
    s.damage = fmin(1.0, s.damage + damage_rate(t, l.damage_onset_k, l.damage_rate_at_onset_per_s,
                                                l.damage_temperature_scale_k) *
                                        dt);
    if (s.damage >= l.destruction_damage) {
        s.phase = EdgePhase::Destroyed;
        s.l0_m = l.destroyed_l0_m;
    }
    double target = l.l_low_m;
    if (s.phase == EdgePhase::Destroyed)
        target = l.destroyed_l_m;
    else {
        if (l.kind == TransitionKind::Smooth)
            target +=
                (l.l_high_m - l.l_low_m) * logistic((t - l.transition_temperature_k) / l.transition_width_k);
        if (l.kind == TransitionKind::FirstOrder) {
            if (s.phase == EdgePhase::Low && t >= l.transition_temperature_k + l.hysteresis_k / 2)
                s.phase = EdgePhase::High;
            else if (s.phase == EdgePhase::High && t <= l.transition_temperature_k - l.hysteresis_k / 2)
                s.phase = EdgePhase::Low;
            target = s.phase == EdgePhase::High ? l.l_high_m : l.l_low_m;
        }
        if (l.kind == TransitionKind::SecondOrder) {
            double reduced = fmax(0.0, (t - l.transition_temperature_k) / l.transition_width_k);
            target += (l.l_high_m - l.l_low_m) * fmin(1.0, sqrt(reduced));
            s.phase = reduced > 0 ? EdgePhase::High : EdgePhase::Low;
        }
        target += l.damage_l_shift_m * s.damage;
    }
    target = fmax(0.0, target);
    s.l_m = l.relaxation_time_s > 0 ? s.l_m + (target - s.l_m) * (-expm1(-dt / l.relaxation_time_s)) : target;
    bool destroyed = s.phase == EdgePhase::Destroyed;
    double bg = destroyed ? l.destroyed_background_conductance_w_per_k : l.background_conductance_w_per_k;
    double active = destroyed ? l.destroyed_active_conductance_w_per_k : l.active_conductance_w_per_k;
    double g = (bg + active * logistic((s.l_m - s.l0_m) / l.activation_width_m)) *
               pow(l.damage_conductance_factor_at_one, s.damage) * edge.weight * scale;
    s.conductance_w_per_k = g > 0 ? 1.0 / (1.0 / g + edge.series_resistance_k_per_w) : 0;
    s.destroyed_fraction = destroyed ? 1.0 : 0.0;
    return s;
}
// Dispatch is done once on the host into fixed/directed/state/transition pools.
template <int Kind>
__global__ void edge_kernel(int count, const int* ids, const DeviceLink* links, const InterfaceState* old,
                            InterfaceState* states, const ThermalState* volumes, RuntimeParameters runtime,
                            const double* dt, double* flux, const double* constant_g,
                            const ConnectivityLaw* laws, double time, const int* profile_offsets,
                            const ProfilePoint* profile_points, const DeviceVolume* mesh_volumes,
                            const Material* materials) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count)
        return;
    const int i = ids[j];
    states[i] = old[i];
    const auto& e = links[i];
    const double series_r = phase_resistance(
        e.series_resistance_k_per_w, e.series_resistance_a_k_per_w, volumes[e.a].damage,
        e.b >= 0 ? volumes[e.b].damage : 0., materials[mesh_volumes[e.a].object].boundary_phase,
        e.b >= 0 ? materials[mesh_volumes[e.b].object].boundary_phase : BulkBoundaryPhase{}, time,
        volumes[e.a].runaway_started_s, e.b >= 0 ? volumes[e.b].runaway_started_s : -1.,
        volumes[e.a].conductivity_factor[e.geometry_axis],
        e.b >= 0 ? volumes[e.b].conductivity_factor[e.geometry_axis] : 1.);
    double ta = volumes[e.a].temperature_k;
    double tb = e.b >= 0 ? volumes[e.b].temperature_k
                         : boundary_temperature(e, time, profile_offsets, profile_points);
    double scale = e.b < 0 ? runtime.ambient_conductance_scale
                           : (e.interface_index >= 0 ? runtime.interface_conductance_scale : 1.0);
    if (e.b >= 0 && mesh_volumes[e.a].object == mesh_volumes[e.b].object)
        scale *= reaction_bulk_factor(volumes[e.a], volumes[e.b], e.geometry_axis);
    double g;
    if constexpr (Kind == 0) {
        g = constant_g[i] * scale;
        g = g > 0 ? 1.0 / (1.0 / g + series_r) : 0;
        states[i].conductance_w_per_k = g;
    } else {
        const double damage =
            e.b >= 0 ? 0.5 * (volumes[e.a].damage + volumes[e.b].damage) : volumes[e.a].damage;
        const auto& law = laws[e.law_index];
        if constexpr (Kind == 1) {
            const double multiplier =
                e.b >= 0 && mesh_volumes[e.a].object == mesh_volumes[e.b].object
                    ? bulk_failure_multiplier(materials[mesh_volumes[e.a].object].boundary_phase, ta, tb,
                                              volumes[e.a].damage, volumes[e.b].damage,
                                              volumes[e.a].runaway_started_s, volumes[e.b].runaway_started_s,
                                              time)
                    : 1.;
            double radiative_ra = -1.;
            if (law.radiation_coefficient_w_per_k4 > 0. && e.b >= 0)
                radiative_ra = phase_resistance(e.series_resistance_a_k_per_w, e.series_resistance_a_k_per_w,
                                                volumes[e.a].damage, 0.,
                                                materials[mesh_volumes[e.a].object].boundary_phase, {}, time,
                                                volumes[e.a].runaway_started_s, -1.,
                                                volumes[e.a].conductivity_factor[e.geometry_axis], 1.);
            states[i] =
                evaluate_directed_geometry(law, states[i], e.geometry_axis, ta, tb, volumes[e.a].damage,
                                           e.b >= 0 ? volumes[e.b].damage : 0., *dt, e.weight, series_r,
                                           scale, 0, multiplier, radiative_ra);
        } else if constexpr (Kind == 2)
            states[i] =
                evaluate_state_geometry(law, states[i], e.geometry_axis, ta, tb, volumes[e.a].damage,
                                        e.b >= 0 ? volumes[e.b].damage : 0., *dt, e.weight, series_r, scale);
        else {
            auto changed = e;
            changed.series_resistance_k_per_w = series_r;
            states[i] = evolve(changed, law, states[i], 0.5 * (ta + tb), damage, *dt, scale);
        }
        g = states[i].conductance_w_per_k;
    }
    flux[i] = g * (tb - ta); // one flux, gathered with opposite signs
}

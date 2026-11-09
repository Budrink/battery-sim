// Local heat/chemistry proposals and conservative gathering into solid candidates.
// Private implementation fragment, included only by cuda_solver.cu.
__device__ double release_source(const Material& m, ThermalState& s, double budget, double dt, double speed) {
    if (m.pulse.profile == RunawayProfile::Kinetic)
        return advance_reaction(m.kinetics, m.outlet, s, budget, dt, speed);
    if (s.mode != CellMode::Runaway)
        return 0.;
    double source;
    if (m.pulse.profile == RunawayProfile::ConstantPower)
        source = fmin(fmax(0., budget - s.released_energy_j), budget / m.duration * speed * dt);
    else {
        s.reaction_progress_s = fmin(pulse_end(m.pulse, m.duration), s.reaction_progress_s + speed * dt);
        const double target = budget * pulse_fraction(m.pulse, m.duration, s.reaction_progress_s);
        source = fmin(fmax(0., budget - s.released_energy_j), fmax(0., target - s.released_energy_j));
    }
    s.released_energy_j += source;
    s.chemical_energy_remaining_j = fmax(0., budget - s.released_energy_j);
    if (s.released_energy_j >= budget * (1 - 1e-12)) {
        s.mode = CellMode::Spent;
        s.damage = 1;
    }
    return source;
}
// Local mechanisms own proposal buffers, never the committed node state.
// Gas production is an explicit transfer packet, not a read of next[] by a
// different subsystem. Exact two-stage chemistry is one coupled mechanism.
template <bool Conversion>
__global__ void source_kernel(int count, const int* ids, const DeviceVolume* volumes,
                              const Material* materials, const ThermalState* old, ThermalState* local,
                              double* heat, GasTransfer* gas, RuntimeParameters runtime, double time,
                              const double* step, const double* joule_per_i2,
                              const ElectricalState* electrical) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count)
        return;
    const int i = ids[j];
    const auto& v = volumes[i];
    const auto& m = materials[v.object];
    auto s = old[i];
    double q;
    if constexpr (Conversion) {
        const double charging =
            m.electrical.enabled ? (m.electrical.voltage_limited_current
                                        ? electrical[v.object].step_charge_c
                                        : positive_current_integral(m.electrical, m.current, time, *step)) /
                                       (*step)
                                 : 0.;
        double charge_rate = charging_precursor_rate(
            m.conversion, m.electrical, charging,
            charge_capacity_fraction(m.conversion, s, v.capacity_j_per_k / m.capacity));
        if (charge_rate > 0.) {
            // Chemical storage draws from terminal work, INCLUDING EMF work;
            // the irreversible-voltage floor is a heat bound, not a second tax
            // on energy explicitly stored by charging. No trace => no payment.
            const auto& e = electrical[v.object];
            const double budget =
                m.electrical.voltage_limited_current
                    ? e.step_work_j
                    : electrical_work_integral(m.electrical, m.current, m.voltage, time, *step);
            const double cell_heat =
                e.resistance_ohm * (m.electrical.voltage_limited_current
                                        ? e.step_i2_a2_s
                                        : squared_current_integral(m.electrical, m.current, time, *step));
            const double room = (v.capacity_j_per_k / m.capacity) * fmax(0., budget - cell_heat);
            charge_rate = charging_rate_within_budget(m.conversion, charge_rate, s.reactant_kg, *step,
                                                      runtime.runaway_power_scale, room);
        }
        q = advance_conversion(m.conversion, m.gas_model, s, v.capacity_j_per_k / m.capacity, *step,
                               runtime.runaway_power_scale, time, charge_rate);
        gas[i] = {s.generated_gas_kg, s.generated_gas_energy_j, s.generated_gas_chemical_j, 0.};
    } else {
        q = release_source(m, s, m.energy * v.capacity_j_per_k / m.capacity, *step,
                           runtime.runaway_power_scale);
        gas[i] = {};
    }
    if (m.electrical.enabled) {
        const double joule =
            joule_per_i2[i] * (m.electrical.voltage_limited_current
                                   ? electrical[v.object].step_i2_a2_s
                                   : squared_current_integral(m.electrical, m.current, time, *step));
        s.joule_energy_j += joule;
        q += joule;
    }
    local[i] = s;
    heat[i] = q;
}
__global__ void volume_kernel(int count, const DeviceVolume* volumes, const Material* materials,
                              const DeviceLink* links, const int* offsets, const int* adjacency,
                              const double* flux, const ThermalState* old, ThermalState* next,
                              RuntimeParameters runtime, double time, const double* step,
                              DeviceRows patch_rows, const double* gas_flux, const ThermalState* local,
                              const double* heat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const double dt = *step;
    const auto& v = volumes[i];
    const auto& m = materials[v.object];
    if (m.isothermal)
        return;
    ThermalState s = local[i];
    double power = 0, boundary_power = 0;
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
        int k = adjacency[j];
        const auto& e = links[k];
        power += (e.a == i ? 1.0 : -1.0) * flux[k];
        if (e.b < 0)
            boundary_power -= flux[k];
    }
    for (int j = patch_rows.offsets[i]; j < patch_rows.offsets[i + 1]; ++j)
        power += gas_flux[patch_rows.ids[j]];
    const double budget = m.energy * v.capacity_j_per_k / m.capacity;
    const double source = heat[i];
    if (m.conversion.enabled) {
        s.stored_energy_j += power * dt + source;
        s.boundary_energy_out_j += boundary_power * dt;
        close_thermal_state(m, v, s, runtime.runaway_power_scale);
        next[i] = s;
        return;
    } else if (m.capacity_law.factor.count) {
        s.stored_energy_j += power * dt + source;
        s.temperature_k =
            thermal_temperature(m.capacity_law, v.capacity_j_per_k, s.stored_energy_j, old[i].damage);
    } else
        s.temperature_k += (power * dt + source) / v.capacity_j_per_k;
    s.boundary_energy_out_j += boundary_power * dt;
    // A bound mixture must age at its shared temperature, not at the
    // intermediate solid-only temperature before local equilibration.
    if (old[i].mobile_capacity_j_per_k == 0 &&
        (s.mode == CellMode::Healthy || (m.pulse.profile == RunawayProfile::Kinetic && reaction_live(s)))) {
        double t = 0.5 * (s.temperature_k + old[i].temperature_k);
        s.damage = fmin(1.0, s.damage + damage_rate(t, m.onset, m.rate, m.temperature_scale) *
                                            m.aging_multiplier * runtime.damage_rate_scale * dt);
        const double threshold =
            m.pulse.profile == RunawayProfile::Kinetic ? m.kinetics.activation_damage : 1.;
        if (s.mode == CellMode::Healthy && s.damage >= threshold) {
            if (m.is_battery)
                s.runaway_started_s = time + dt;
            s.mode = m.is_battery && budget > 0 ? CellMode::Runaway : CellMode::Spent;
        }
    }
    if (!m.capacity_law.factor.count)
        s.stored_energy_j = v.capacity_j_per_k * s.temperature_k;
    close_thermal_state(m, v, s, runtime.runaway_power_scale);
    next[i] = s;
}
// Infinite internal conductivity as an algebraic constraint. Contact fluxes
// still use old-state temperatures; sum all heat before evaluating local damage.
// Each thread stages only its own volumes, and the block synchronizes before
// broadcasting the common temperature. No in-place field reads or atomic sums.
__global__ void isothermal_volume_kernel(const DeviceVolume* volumes, const Material* materials,
                                         const DeviceLink* links, const int* object_offsets,
                                         const int* offsets, const int* adjacency, const double* flux,
                                         const ThermalState* old, ThermalState* next,
                                         RuntimeParameters runtime, double time, const double* step,
                                         const ThermalState* local, const double* heat) {
    const int object = blockIdx.x, t = threadIdx.x;
    const auto& m = materials[object];
    if (!m.isothermal)
        return;
    const double dt = *step;
    double capacity = 0, delta = 0;
    for (int i = object_offsets[object] + t; i < object_offsets[object + 1]; i += blockDim.x) {
        const auto& v = volumes[i];
        ThermalState s = local[i];
        double power = 0, boundary_power = 0;
        for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
            const int k = adjacency[j];
            const auto& e = links[k];
            power += (e.a == i ? 1.0 : -1.0) * flux[k];
            if (e.b < 0)
                boundary_power -= flux[k];
        }
        const double source = heat[i];
        s.boundary_energy_out_j += boundary_power * dt;
        next[i] = s;
        capacity += v.capacity_j_per_k;
        delta += power * dt + source;
    }
    __shared__ double capacities[128], deltas[128];
    capacities[t] = capacity;
    deltas[t] = delta;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (t < stride) {
            capacities[t] += capacities[t + stride];
            deltas[t] += deltas[t + stride];
        }
        __syncthreads();
    }
    // Initial fields are validated uniform, and every update preserves this.
    // Avoid roundoff from repeatedly multiplying/dividing the stored baseline.
    const double temperature = old[object_offsets[object]].temperature_k + deltas[0] / capacities[0];
    for (int i = object_offsets[object] + t; i < object_offsets[object + 1]; i += blockDim.x) {
        ThermalState s = next[i];
        s.temperature_k = temperature;
        s.stored_energy_j = volumes[i].capacity_j_per_k * temperature;
        if (s.mode == CellMode::Healthy) {
            const double average = 0.5 * (old[i].temperature_k + temperature);
            s.damage = fmin(1.0, s.damage + damage_rate(average, m.onset, m.rate, m.temperature_scale) *
                                                m.aging_multiplier * runtime.damage_rate_scale * dt);
            if (s.damage >= 1) {
                if (m.is_battery)
                    s.runaway_started_s = time + dt;
                s.mode = m.is_battery && m.energy > 0 ? CellMode::Runaway : CellMode::Spent;
            }
        }
        next[i] = s;
    }
}
__global__ void ignition_kernel(ThermalState* state, const DeviceVolume* volumes, const Material* materials,
                                int begin, int end, double time) {
    int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end)
        return;
    if (state[i].mode == CellMode::Healthy) {
        const auto& m = materials[volumes[i].object];
        state[i].mode = CellMode::Runaway;
        state[i].runaway_started_s = time;
        state[i].damage = m.pulse.profile == RunawayProfile::Kinetic
                              ? fmax(state[i].damage, m.kinetics.activation_damage)
                              : 1.;
        const auto& l = m.capacity_law;
        if (l.factor.count)
            state[i].temperature_k = thermal_temperature(l, volumes[i].capacity_j_per_k,
                                                         state[i].stored_energy_j, state[i].damage);
        if (m.pulse.profile == RunawayProfile::Kinetic)
            refresh_reaction(m.kinetics, m.structure, state[i],
                             m.energy * volumes[i].capacity_j_per_k / m.capacity, 1.);
    }
}

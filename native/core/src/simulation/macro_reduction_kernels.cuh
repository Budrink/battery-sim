// Read-only reductions from thermal volumes to battery/interface observations.
// Private implementation fragment, included only by cuda_solver.cu.
struct Aggregate {
    double capacity, energy, damage, max_damage, tmin, tmax, spent, active, released, boundary, start,
        weighted_temperature, reference_capacity;
    double chemical, ejected, vent, power, quenched, joule;
};
__global__ void macro_kernel(const DeviceVolume* volumes, const Material* materials,
                             const ThermalState* state, const int* offsets, CellState* macros) {
    const int object = blockIdx.x, t = threadIdx.x;
    Aggregate a{};
    a.tmin = 1e300;
    a.start = 1e300;
    for (int i = offsets[object] + t; i < offsets[object + 1]; i += blockDim.x) {
        const auto& s = state[i];
        double c = volumes[i].capacity_j_per_k;
        const auto& l = materials[object].capacity_law;
        const auto& mat = materials[object];
        const double actual =
            (mat.conversion.enabled ? conversion_capacity(mat.conversion, s, c, c / mat.capacity)
                                    : c * capacity_factor(l, s.temperature_k, s.damage)) +
            s.mobile_capacity_j_per_k;
        a.capacity += actual;
        a.reference_capacity += c;
        a.weighted_temperature += actual * s.temperature_k;
        a.energy += l.factor.count || mat.conversion.enabled ? s.stored_energy_j : c * s.temperature_k;
        a.damage += c * s.damage;
        a.max_damage = fmax(a.max_damage, s.damage);
        a.tmin = fmin(a.tmin, s.temperature_k);
        a.tmax = fmax(a.tmax, s.temperature_k);
        a.spent += s.mode == CellMode::Spent ? c : 0;
        a.active += s.mode == CellMode::Runaway ? c : 0;
        a.released += s.released_energy_j;
        a.boundary += s.boundary_energy_out_j;
        a.joule += s.joule_energy_j;
        a.chemical += s.chemical_energy_remaining_j;
        a.ejected += s.ejected_chemical_energy_j;
        a.vent += s.vent_heat_out_j;
        a.power += s.chemical_power_w;
        a.quenched += s.mode == CellMode::Quenched ? c : 0.;
        if (s.runaway_started_s >= 0)
            a.start = fmin(a.start, s.runaway_started_s);
    }
    __shared__ Aggregate partial[128];
    partial[t] = a;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (t < stride) {
            auto& x = partial[t];
            const auto& y = partial[t + stride];
            x.capacity += y.capacity;
            x.energy += y.energy;
            x.damage += y.damage;
            x.max_damage = fmax(x.max_damage, y.max_damage);
            x.reference_capacity += y.reference_capacity;
            x.weighted_temperature += y.weighted_temperature;
            x.tmin = fmin(x.tmin, y.tmin);
            x.tmax = fmax(x.tmax, y.tmax);
            x.spent += y.spent;
            x.active += y.active;
            x.released += y.released;
            x.boundary += y.boundary;
            x.start = fmin(x.start, y.start);
            x.joule += y.joule;
            x.chemical += y.chemical;
            x.ejected += y.ejected;
            x.vent += y.vent;
            x.power += y.power;
            x.quenched += y.quenched;
        }
        __syncthreads();
    }
    if (t == 0) {
        const auto& x = partial[0];
        CellState out;
        out.temperature_k = x.weighted_temperature / x.capacity;
        out.damage = x.damage / x.reference_capacity;
        out.min_temperature_k = x.tmin;
        out.max_temperature_k = x.tmax;
        out.max_damage = x.max_damage;
        out.spent_fraction = x.spent / x.reference_capacity;
        out.runaway_fraction = x.active / x.reference_capacity;
        out.mode =
            x.active > 0
                ? CellMode::Runaway
                : (x.quenched > 0 ? CellMode::Quenched
                                  : (out.spent_fraction > 1 - 1e-12
                                         ? CellMode::Spent
                                         : (x.spent > 0 ? CellMode::PartiallySpent : CellMode::Healthy)));
        out.runaway_started_s = x.start < 1e299 ? x.start : -1;
        out.released_energy_j = x.released;
        out.stored_energy_j = x.energy;
        out.boundary_energy_out_j = x.boundary;
        out.heat_capacity_j_per_k = x.capacity;
        out.joule_energy_j = x.joule;
        out.chemical_energy_remaining_j = x.chemical;
        out.ejected_chemical_energy_j = x.ejected;
        out.vent_heat_out_j = x.vent;
        out.chemical_power_w = x.power;
        out.quenched_fraction = x.quenched / x.reference_capacity;
        macros[object] = out;
    }
}
__global__ void interface_macro_kernel(int count, const DeviceLink* links, const InterfaceState* states,
                                       const int* offsets, const int* ids, InterfaceState* macros) {
    int i = blockIdx.x, t = threadIdx.x;
    InterfaceState out{};
    double high = 0;
    for (int j = offsets[i] + t; j < offsets[i + 1]; j += blockDim.x) {
        int k = ids[j];
        const auto& s = states[k];
        double w = links[k].weight;
        out.l_m += w * s.l_m;
        out.l0_m += w * s.l0_m;
        out.damage += w * s.damage;
        out.conductance_w_per_k += s.conductance_w_per_k;
        out.destroyed_fraction += w * s.destroyed_fraction;
        if (s.phase == EdgePhase::High)
            high += w;
    }
    __shared__ InterfaceState partial[128];
    __shared__ double highs[128];
    partial[t] = out;
    highs[t] = high;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (t < stride) {
            auto& a = partial[t];
            const auto& b = partial[t + stride];
            a.l_m += b.l_m;
            a.l0_m += b.l0_m;
            a.damage += b.damage;
            a.conductance_w_per_k += b.conductance_w_per_k;
            a.destroyed_fraction += b.destroyed_fraction;
            highs[t] += highs[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        out = partial[0];
        out.phase = out.destroyed_fraction > 1 - 1e-12 ? EdgePhase::Destroyed
                                                       : (highs[0] > 0 ? EdgePhase::High : EdgePhase::Low);
        macros[i] = out;
    }
}

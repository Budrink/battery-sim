// One worker per physical face. Frozen-pressure event integration is sequential
// only within its small planar graph; faces run in parallel on the same stream.
struct PlateFrontAccess {
    int begin, end;
    const DeviceShellPatch* patches;
    const DeviceChannel* channels;
    const ShellDamageState* damage;
    const GasState* gas;
    double* delta;
    int* order;
    double speed;
    __device__ int size() const {
        return end - begin;
    }
    __device__ double area(int i) const {
        return patches[begin + i].area_m2;
    }
    __device__ double open(int i) const {
        return fmin(1., damage[begin + i].bulk + delta[begin + i]);
    }
    __device__ double weak_fraction(int i) const {
        return patches[begin + i].weak_fraction;
    }
    __device__ double load_ratio(int i) const {
        const int k = begin + i;
        const auto& p = patches[k];
        const auto& c = channels[p.channel];
        const double dp = gas[c.from].pressure_pa - (c.to < 0 ? c.outside_pressure : gas[c.to].pressure_pa);
        const double factor = open(i) < p.weak_fraction ? p.weak_factor : 1.;
        return dp / (c.threshold * shell_remaining_barrier(damage[k].thermal) * factor);
    }
    __device__ double rate(int i) const {
        const auto& law = channels[patches[begin + i].channel].opening;
        return speed * law.area_rate_per_s * pow(fmax(0., load_ratio(i) - 1.), law.pressure_exponent);
    }
    __device__ bool touches_open(int i) const {
        for (int n : patches[begin + i].neighbors)
            if (n >= 0 && open(n - begin) >= 1.)
                return true;
        return false;
    }
    __device__ void advance(int i, double da, int rank) {
        const int k = begin + i;
        if (order[k] < 0)
            order[k] = rank;
        delta[k] = fmin(1. - damage[k].bulk, delta[k] + da);
    }
};
__global__ void plate_front_proposal_kernel(int count, const int* offsets, const DeviceShellPatch* patches,
                                            const DeviceChannel* channels, const ShellDamageState* damage,
                                            const GasState* gas, double* delta, int* order,
                                            const double* step, double speed) {
    const int object = blockIdx.x * blockDim.x + threadIdx.x;
    if (object >= count)
        return;
    const int begin = offsets[object], end = offsets[object + 1];
    for (int k = begin; k < end; ++k) {
        delta[k] = 0.;
        order[k] = -1;
    }
    if (begin == end || patches[begin].channel < 0 || !channels[patches[begin].channel].opening.whole_plate)
        return;
    PlateFrontAccess access{begin, end, patches, channels, damage, gas, delta, order, speed};
    advance_plate_front(access, *step);
}
// Every gas donor budgets ALL its outgoing fracture requests once, including
// independent legacy patches. This prevents two plates spending the same gas U.
__global__ void plate_fracture_budget_kernel(int count, DeviceRows rows, const DeviceChannel* channels,
                                             const int* offsets, const DeviceShellPatch* patches,
                                             const ShellDamageState* damage, const GasState* gas,
                                             GasModel model, const double* delta, double* scales,
                                             const double* step, double speed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    double request = 0.;
    for (int j = rows.offsets[i]; j < rows.offsets[i + 1]; ++j) {
        const auto& c = channels[rows.ids[j]];
        if (c.from != i || !c.opening.deterministic_aperture)
            continue;
        const int k = offsets[c.shell] + c.shell_thermal_volume;
        const double da =
            c.opening.whole_plate ? delta[k] : leading_increment(c, patches[k], damage[k], gas, *step, speed);
        request += c.opening.fracture_energy_j_per_m2 * patches[k].area_m2 *
                   shell_remaining_barrier(damage[k].thermal) * da;
    }
    const double available = fmax(0., gas[i].mass_kg * model.cv_j_per_kg_k * (gas[i].temperature_k - 1.));
    scales[i] = request > available ? available / request : 1.;
}
// Keep only an affordable PREFIX of the proposed connected crack. Scaling each
// tile independently could create a hole beyond a predecessor that never broke.
__global__ void plate_front_afford_kernel(int count, const int* offsets, const DeviceShellPatch* patches,
                                          const DeviceChannel* channels, double* delta, const int* order,
                                          const double* scales) {
    const int object = blockIdx.x * blockDim.x + threadIdx.x;
    if (object >= count)
        return;
    const int begin = offsets[object], end = offsets[object + 1];
    if (begin == end || patches[begin].channel < 0 || !channels[patches[begin].channel].opening.whole_plate)
        return;
    int cutoff = 2 * (end - begin) + 1;
    for (int k = begin; k < end; ++k)
        if (delta[k] > 0 && scales[channels[patches[k].channel].from] < 1.)
            cutoff = min(cutoff, order[k]);
    for (int k = begin; k < end; ++k) {
        if (order[k] > cutoff)
            delta[k] = 0.;
        else if (order[k] == cutoff)
            delta[k] *= scales[channels[patches[k].channel].from];
    }
}

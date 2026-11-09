// Optional leading fracture, then two conservative transfer operators in one
// transaction. Thermal kernels/contact laws are unchanged. Pressure transport
// reads the completed thermal/production/LTE snapshot, never partial neighbours.
double CudaSolver::Impl::prepare_split_step(double time, double cap, double safety,
                                            const RuntimeParameters& runtime) {
    const double retry_floor = std::min(cap, scenario.settings.minimum_dt_s);
    for (int retry = 0; retry < 32; ++retry) {
        struct RestoreInputs {
            Impl& owner;
            ~RestoreInputs() {
                owner.restore_prefix_inputs();
            }
        } restore{*this};
        begin_prefix_inputs();
        // 1. Mechanics comes before either transport geometry.
        if (!prepare_fracture_candidate(time, cap, safety, retry_floor, runtime))
            continue;

        // 2. Solid heat exchange, R1/R2, Joule heat and gas production.
        prepare_thermal_candidate(time, runtime);

        // 3. Mass/enthalpy/oxygen transport and gas chemistry. These backends
        // share the transaction, not partially updated neighbour states.
        if (cuda_gas)
            return prepare_cuda_mass_candidate(time, runtime);
        if (cpu_gas)
            return prepare_cpu_mass_candidate(time, runtime);

        // The explicit reference backend may reject its proposed interval.
        double accepted_dt = 0.;
        if (prepare_explicit_mass_candidate(time, cap, safety, retry_floor, runtime, accepted_dt))
            return accepted_dt;
    }
    throw std::runtime_error("Split thermal/mass timestep failed to converge");
}

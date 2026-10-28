#pragma once
#include <cstdint>
namespace battery {
struct GasTransportStats {
    std::uint64_t windows=0,groups=0,explicit_substeps=0;
    std::uint64_t implicit_steps=0,rejections=0,newton_iterations=0,residual_evaluations=0;
};
struct SolverProfile {
    // Wall clock, including attempted/discarded candidates. No physical state.
    double prepare_s=0,cpu_gas_s=0,download_s=0,upload_s=0;
    double cuda_gas_s=0;
    std::uint64_t coarse_floor_activations=0,aligned_short_steps=0;
    std::uint64_t split_step_retries=0,floor_bounded_retries=0;
    double minimum_regular_step_s=0.;
    double deterministic_pressure_gap_pa=0;
    std::uint64_t kawasaki_events=0,kawasaki_tail_updates=0;
    std::uint64_t gas_batches=0,gas_limited_batches=0;
    std::uint64_t gas_last_reaction_intervals=0,gas_max_local_reactor_iterations=0;
    double gas_last_reaction_step_s=0.;
    unsigned gas_last_pressure_iterations=0,gas_last_mixing_iterations=0;
    std::uint64_t gas_pressure_pattern_attempts=0,gas_pressure_pattern_accepted=0;
    double gas_last_mixing_residual=0.;
    double gas_last_mixing_raw_residual=0.;
    double gas_averaged_min_scale=1.;
    std::uint64_t gas_averaged_windows=0,gas_averaged_limited_windows=0,gas_averaged_entropy_backtracks=0;
    GasTransportStats gas;
};
}

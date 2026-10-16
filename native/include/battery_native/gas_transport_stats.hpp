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
    double deterministic_pressure_gap_pa=0;
    std::uint64_t kawasaki_events=0,kawasaki_tail_updates=0;
    std::uint64_t gas_batches=0,gas_limited_batches=0;
    GasTransportStats gas;
};
}

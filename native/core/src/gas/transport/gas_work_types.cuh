#pragma once
// Shared work records for the cooperative gas solver; field layout is unchanged.
#include "battery_native/gas/transport/kawasaki_gas_math.hpp"
#include "battery_native/gas/mixing/gas_mixing.hpp"

namespace battery::batched_gas {
namespace kg = kawasaki;
struct Proposal {
    int donor = 0;
    double cap = 0, rate = 0, mass = 0;
    double next_pressure_cap = 0.;
    double momentum_bias = 0., next_momentum_bias = 0., kinetic_j = 0., kinetic_delta_j = 0.;
    double advective_energy_delta = 0.;
    double energy = 0, chemical = 0, oxidizer = 0;
    double donor_pressure_weight = 0, receiver_pressure_weight = 0;
    double vapor = 0, latent = 0;
    MixingTransfer mixing;
    bool tail = false;
};
struct Budget {
    double scale = 1, outgoing = 0, energy = 0;
};
struct Control {
    double entropy_by_block[32];
    double pressure_gap_by_block[32];
    double reaction_step_by_block[32], reaction_step;
    double accepted_reaction_step = 0.;
    double last_max_pressure_gap_pa;
    unsigned reaction_intervals = 1;
    unsigned max_local_reactor_iterations = 0;
    double averaged_scale = 1.;
    unsigned averaged_backtracks = 0;
    double averaged_min_scale = 1.;
    unsigned averaged_windows = 0, averaged_limited_windows = 0, averaged_entropy_backtracks = 0;
    // Failure-only diagnostics. No host transfer on an accepted window.
    unsigned averaged_invalid_mask[8]{};
    int averaged_first_invalid[8]{};
    double averaged_entropy[8]{}, averaged_attempt_scale[8]{};
    double averaged_window_s = 0., averaged_elapsed_s = 0.;
    bool averaged_old_is_initial = true;
    unsigned averaged_pressure_used = 0;
    double averaged_pressure_residual = 0.;
    unsigned averaged_nonlinear_used = 0;
    unsigned averaged_first_linear_used = 0;
    double averaged_flow_defect = 0.;
    unsigned enthalpy_iterations = 0;
    double enthalpy_residual = 0.;
    bool pressure_pattern_valid = false;
    unsigned pressure_pattern_attempts = 0, pressure_pattern_accepted = 0;
    unsigned mixing_iterations = 0;
    double mixing_residual = 0.;
    double mixing_raw_residual = 0.;
    double mixing_solid_entropy = 0.;
    unsigned reactive_iterations = 0;
    double reactive_residual = 0.;
};
struct AverageState {
    double pressure = 0., mass = 0., chemical = 0., oxygen = 0., vapor = 0., latent = 0., temperature = 0.;
    double compliance = 0., diagonal = 0., residual = 0., direction = 0., matvec = 0., correction = 0.;
    double pressure_pattern = 0.;
    int pressure_peer = -1;
    double pressure_coupling = 0.;
    struct Reactive {
        double incoming = 0., outgoing = 0., exposure = 0., rate_dt = 0., heat = 0., vapor_burned = 0.;
        double fuel = 0., oxygen = 0., vapor = 0., latent = 0.; // integrated outgoing amounts
    } reaction;
};
} // namespace battery::batched_gas

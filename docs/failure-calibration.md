# Post-event calibration: chemical reserve and changing medium

The process-based successor is now available as an opt-in
[`kinetic` profile](reaction-process.md). Its first
[UPV fit and frozen validation](kinetic-calibration.md) give 6.861 / 12.820 K
late-witness RMSE, with important energy/contact and reaction-rate ambiguity.
The v3 results below remain historical pulse-model results, not validation of
the new temperature-dependent kinetics. Neither the profiles nor their recorded
model fingerprints have been rewritten.

## Previous fitted model: transient internal structure (v3)

The CUDA core now separates chemical heat release from structural evolution.
There is no mass/gas-flow solver. This is an effective constitutive closure, not
resolved chemistry, venting or a measured high-temperature material law.

For each failed numerical volume, retain its local rupture time. Before rupture,
the accepted temperature/state-dependent directed internal law is unchanged.
At D=1, switch `l0` irreversibly and permit an initial enhancement of transport,
followed by relaxation to a weak residual structure. For age `a >= 0`:

```text
F(a, tau) = (1 - exp(-a/tau)) / (1 - exp(-6)),  0 <= a < 6*tau
F(a, tau) = 1,                                a >= 6*tau
bulk_factor(a) = residual + (peak - residual) * (1 - F(a, tau_structure))
```

`peak` and `residual` multiply the reference conductivity tensor. A peak of three
means three times reference k, not necessarily three times the intact k(T,D).
This allows a sudden upward jump followed by a nonlinear fall; it does not force
every material to have that trajectory. With the optional transient omitted,
the previous constant destroyed branch is reproduced.

Internal edges select the failed donor's local structural age. If only the
receiver has ruptured, its rupture state controls that edge. Forward/reverse
branches are never averaged. Each edge still transfers one conservative heat
quantity with opposite signs into its endpoints. This minimal ordered closure
does not identify anisotropic high-temperature mixing from the experiments.

The same local factor modifies the battery's half-volume resistance on each
boundary and the surface-temperature reconstruction. Material-pair contact h
remains a distinct resistance, not a replacement for that half-volume. CUDA CFL
bounds include the largest possible transient conductance, and the timestep also
resolves the structural relaxation. Cooling cannot restore the intact branch.
The rupture clock keeps advancing after the chemical reserve is exhausted; it
is independent of the runtime chemical-power multiplier.

Chemical energy uses a finite reserve and a fast pulse plus a slower tail:

```text
released(age) = E * [f * F(reaction_age, tau_fast)
                     + (1-f) * F(reaction_age, tau_tail)]
```

CUDA deposits differences of this cumulative function, capped by the remaining
reserve. No double counting between burst/tail and no energy from changing links.
`runaway.duration_s` denotes the fast time constant only for `profile=burst_tail`;
its historical constant-power meaning is unchanged. All capacities remain fixed.

Scene opt-in for the calibrated C2 source (other required bulk-law fields are
shown in the exported scenarios):

```json
{
  "thermal": {
    "bulk_destruction_on_boundary": true,
    "bulk_failure_transient": {"peak_factor": 3.0, "relaxation_time_s": 5.0}
  },
  "runaway": {
    "profile": "burst_tail", "energy_j": 70000.0,
    "duration_s": 0.5, "fast_fraction": 0.8, "tail_time_s": 30.0
  }
}
```

These illustrative numbers are **not measured constants**. The comparison fixes
the two structural-transient parameters and the three pulse-shape parameters as
priors. Only deposited E, residual internal bulk factor, and two destroyed
C2 contact conductances are fitted. Residual-factor bounds are exploratory.
This separates implemented capability from identified parameters.

### Current trial result

| Case | Previous v1, same late window | Transient v3 |
|---|---:|---:|
| No cooling, fit | 34.226 K | 6.862 K |
| Active cooling, frozen check | 19.065 K | 12.805 K |

This compares the previously fitted v1 profile without refitting it to the new
objective; it is not an equal-objective model-selection study. The intermediate
constant-destroyed-bulk v2, fitted on the same late-only objective, gave
6.863/12.790 K. Adding the transient therefore does **not** establish improved
accuracy from these data. The v3 profile is **not accepted as a complete
destruction calibration**; the active-cooling discrepancy remains mainly C3
underprediction (TIM RMSE 18.60 K, ISS 16.68 K).

With peak=3 and structural tau=5 s held fixed, the fit gives deposited E=134.57 kJ,
residual factor=0.005008, ISS contact h=1.0005 and TIM h=1.4930 W/m²/K.
Residual factor and ISS h approach their lower search bounds. These are not
established material properties. The previous constant-structure trial needed
68.91 kJ while giving almost the same late curves (maximum change across four
witnesses is 0.127 K in the fit case and 0.077 K in validation). This is direct
evidence of dependence on the assumed rapid-stage law, not a precise energy
identification. Bounds have not been expanded to improve the score.

Halving max dt changes any diagnostic temperature by less than 0.015 K; energy
residuals are below 3e-9 J. Both accepted pre-event temperature histories replay
with zero change, retaining RMSE 1.003729/1.272585 K. All 75 Python tests and
14 registered native/CUDA tests pass.

### Observation contract

Simulate 1800 s from the last pre-event scan, preserving original scan times.
Score **only** 101/102/107/108 from 300 s after the first event scan to the end
(about 1799 s after the event). Keep the same window for no cooling and active
cooling. Source C2 readings and the first 300 s are diagnostic, never fitting
targets. Exclusion does not establish that every early reading is wrong: it
avoids treating uncertain gas-exposed surface spikes as bulk temperatures.

Measured ambient/cooling temperatures remain boundary inputs, including during
the event. Thus fitted E is heat deposited in the modeled battery solid; it is
not total chemical heat or a chamber calorimetric energy estimate. Primary time
is prescribed from the pressure event; secondary-ignition kinetics are not fitted.
No new nonlocal gas-heating path is added by this trial.

Results and fixed-versus-fitted parameters are recorded in
[`upv-failure-transient-v3/assessment.json`](../data/calibration/upv-failure-transient-v3/assessment.json).
Compare old/new predictions only on that identical late-witness objective, not
against the historical 109 K peak-window score below. The report replays both
accepted intact experiments and verifies every predicted temperature is unchanged.
Prior model hashes are preserved, not rewritten to claim old profiles were fitted
with the new executable. No default viewer scene is replaced.

```bash
cmake --build /tmp/battery-sim-cuda-v1 -j 4
python3 scripts/calibrate_upv_failure_transition.py fit --probe /tmp/battery-sim-cuda-v1/battery_sim_failure_probe --max-evaluations 65
python3 scripts/calibrate_upv_failure_transition.py validate --probe /tmp/battery-sim-cuda-v1/battery_sim_failure_probe
MPLCONFIGDIR=/tmp/battery-sim-matplotlib python3 scripts/report_upv_failure_transition.py --intact-probe /tmp/battery-sim-cuda-v1/battery_sim_experiment_probe
```

Outputs include [late-witness comparison](../data/calibration/upv-failure-transient-v3/comparison.png),
[the structural transient and finite energy reserve](../data/calibration/upv-failure-transient-v3/rupture-law.png),
full histories, native property audits and directly runnable `*-fitted-scenario.json`.
Native regressions cover energy accounting, reset, pause/resume, secondary thermal
ignition, transient enhancement/decay after energy exhaustion, directional ages,
maximum principle and the high-conductance boundary CFL. Python tests cover
frozen-property contracts, late-only scoring, two consistent surface observers,
parameter guards and identical exported replay.

## Historical constant-source/contact trial (v1)

The accepted pre-event transport calibration remains unchanged. The new trial
uses the same CUDA solver, fixed 3×3×3 batteries, all fixed capacities and intact
directed links. Python prepares observations and selects parameters; it does not
integrate heat flow. No accepted example or default viewer scene is replaced.

## Protocol

The UPV module experiments deliberately overcharge C2. Therefore the primary
event is prescribed, not explained by fitting a thermal-damage threshold to an
electrical-abuse experiment. Detect the first pressure increase of at least
0.5 bar between scans no more than two seconds apart, corroborated by a thermal
jump within the next two seconds (multiplexed channels need not jump in the same
scan). The interval between that pressure scan and the preceding scan is the event bracket,
not sub-scan knowledge of rupture time. C2's last pre-event surface readings must
all be valid. Pressure is channel 114 voltage × 2.5 bar/V per the source.

1. Warm up the existing conditional CUDA scene to the last pre-event sample;
   retain every numerical volume's temperature, not just macro averages.
2. Insert C2 in its actual 11 mm gap, with the same material, reference properties
   and internal connectivity as C1/C3. Initialize its 27 volumes from the four
   measured surface temperatures, interpolated through thickness and vertically.
   Exact sensor positions are unavailable: this interpolation is an assumption.
3. Replace the two measured C2 boundaries by real C2–ISS2 / C2–TIM2 contacts.
   Trigger the uniform, finite-energy CUDA source at the measured event time.
4. Fit only deposited energy, pulse duration, and the two contacts' destroyed
   conductances. Their irreversible switch uses the existing D=1 branch.
   All intact contact laws and all heat capacities stay fixed. No secondary
   ignition or accumulated-defect kinetics is fitted in this trial.
5. Score a fixed 300 s window at the original instrument scan times: four witness
   channels and two C2 face means, in Kelvin, with equal weights. Invalid data remain masked. There are
   no fitted offsets, time shifts, channel exclusions or interpolated gaps.
6. Freeze the profile and run active cooling. Temperature of the environment and
   the cooling contact remain measured inputs in both cases; this is not a closed
   gas/chamber energy model. `fit` opens no active-cooling record or scene.

Only the pulse and new source-contact destruction parameters are fitted. They are exploratory
effective parameters, not measured material constants. The current rectangular
pulse deposits energy into the modeled solid: it is not total chemical heat,
ejected-gas heat or calorimetry. Known Cp/k are held fixed, but their continuation
through destruction is an unvalidated closure, not a measured high-T law.

## First result

The best of two starts, scored at original scan times, gave E ≈ 112.17 kJ,
duration ≈ 1.02 s, destroyed ISS contact h ≈ 21.63 and TIM h ≈ 782.71 W/m²/K.
These numbers are **not accepted as a destruction calibration**:

| Case | All-channel RMSE | Witness-only RMSE | C2-face RMSE |
|---|---:|---:|---:|
| No cooling, fit | 109.10 K | 82.98 K | 148.65 K |
| Active cooling, unchanged parameters | 78.62 K | 21.89 K | 132.60 K |

Halving dt changes predictions by less than 0.057 K. The energy ledger closes
within 1e-5 J and exported scenarios reproduce the parameterized probe exactly.
This points to model/observation inadequacy in this trial, not time-step error.
It does not establish which additional physical mechanism is dominant.

An initial one-second-resampled diagnostic was sensitive to the event's alignment
with the output grid, particularly its fitted pulse duration. The recorded final
protocol therefore preserves original instrument times for both observations
and boundary curves. No sub-scan temperature shape or pulse duration is claimed
as experimentally resolved. The fitted TIM contact also differs between starts
with nearly identical cost: these estimates are not uniquely identified.

The raw records contain simultaneous pressure and ambient-temperature spikes,
strongly different source-face peaks, and a short, very hot C3/ISS signal only
in the uncooled case. A single hot thermocouple is not proof that the witness's
whole battery ran away. Candidate missing blocks are localized vent/gas heat
transfer, ejected mass/radiation, post-event sensor contact, spatial source
distribution and irreversible internal geometry. They cannot be absorbed into
the previously validated intact conductances or heat capacities.

## Reproduction

```bash
cmake --build /tmp/battery-sim-cuda-v1 --target battery_sim_failure_probe -j 4
python3 scripts/calibrate_upv_failure.py fit --probe /tmp/battery-sim-cuda-v1/battery_sim_failure_probe
python3 scripts/calibrate_upv_failure.py validate --probe /tmp/battery-sim-cuda-v1/battery_sim_failure_probe
python3 scripts/report_upv_failure.py
```

Outputs live in `data/calibration/upv-failure-v1/`. Use a new `--output` directory
for subsequent model trials. Profiles retain source/model/probe hashes and both
optimization starts; validation rejects a changed model or binary. The report
retains masked observations, model histories, actual native properties, energy
and timestep checks. `comparison.png` and `event-observations.png` distinguish
the model discrepancy from the raw event evidence.

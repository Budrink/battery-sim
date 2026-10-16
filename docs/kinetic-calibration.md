# UPV kinetic-process calibration (v4)

This is a **conditional post-event fit**, not an identification of full cell
chemistry, venting or secondary-ignition thresholds. Integration is CUDA;
Python performs bounded optimization, provenance checks and reporting.

## Frozen experiment split and objective

- Fit: no-cooling UPV module record only.
- Validation: active-cooling record with the fitted parameter file frozen.
- Original timestamps and missing observations are retained.
- Exactly the previous v3 objective: four witness channels 101, 102, 107, 108,
  from 300 to 1800 s after the measured pressure/thermal event.
- Source-face thermocouples and the first 300 s remain diagnostic, not fit targets.
- No channel offsets, time shifts, sensor relocation or topology changes are fitted.

Both records were seen during previous development; this is held-out parameter
validation, not a claim of a pristine blind experiment. Historical C1/C3 numbering
uncertainty is unchanged. Chamber pressure supplies an event marker, not internal
cell pressure. The event is prescribed, not predicted from overcharge chemistry.

## What can change

Six positive parameters are optimized in log coordinates, from two distinct starts:

| Parameter | Meaning | Search bounds |
| --- | --- | --- |
| `energy_j` | Effective chemical reserve accessible as modeled solid heat | 1–1000 kJ |
| `rate_400_per_s` | Reaction rate at 400 K and activation damage | 0.0001–10 /s |
| `destroyed_iss_h` | Effective C2–ISS contact h after local D reaches 1 | 0.01–10000 W/(m² K) |
| `destroyed_tim_h` | Effective C2–TIM contact h after local D reaches 1 | 0.01–10000 W/(m² K) |
| `residue_through` | Through-plane residual structural multiplier | 0.005–10 |
| `residue_inplane` | Shared in-plane y/z residual structural multiplier | 0.005–10 |

These are exploratory numerical bounds, not confidence intervals or measurements.
Post-rupture h is an effective coefficient over the original contact area. Its
lower bound permits a nearly disconnected contact; it is not the intrinsic
conductivity of an intact ISS/TIM material.

All measured battery Cp and reference k, all interlayer properties, geometry,
3×3×3 battery discretization, intact contact/internal state laws, warmup fields,
surface observers, measured ambient and cooling boundary histories are frozen.
The witness cells carry no chemical source in this experiment fit. Their
secondary-failure behavior has therefore **not** been calibrated.

The source keeps a local finite reserve and actual temperature/damage feedback;
there is no fitted pulse duration. Damage reaches the pre-existing D=1 contact
branch progressively. Structural multipliers apply both internally and to the
correct bulk halves of contact resistances. See [the implemented law](reaction-process.md).

## Explicit unmeasured assumptions

Activation energy 60 kJ/mol, activation damage 0.2, damage sensitivity 2 and
reaction damage gain 1 are fixed illustrative priors, not identified chemical
constants. The active structural factors [3, 8, 8] and activity rate scale 0.1 /s
are also fixed. An independent rate/activation-energy fit would be poorly
constrained by the excluded early interval and uncertain bulk source temperature.

No vent was invented for the pouch and no energy outlet is enabled. Consequently
the fitted reserve is **not total cell chemical heat including ejecta**. It is
the effective energy accessible to this solid-only conditional response model.
Measured chamber ambient temperature is still used as boundary forcing; that
does not establish a spatially uniform gas temperature after rupture.

## Reproduction and audit

```bash
python3 scripts/calibrate_upv_kinetic.py fit \
  --probe /tmp/battery-sim-cuda-v1/battery_sim_failure_probe \
  --starts 2 --max-evaluations 30
python3 scripts/calibrate_upv_kinetic.py validate \
  --probe /tmp/battery-sim-cuda-v1/battery_sim_failure_probe
python3 scripts/report_upv_kinetic.py \
  --intact-probe /tmp/battery-sim-cuda-v1/battery_sim_experiment_probe
```

Outputs live in `data/calibration/upv-kinetic-v4/`: `profile.json`, independent
`fit.json`/`validate.json`, exported native scenarios, `comparison.png`,
`process.png`, and `assessment.json`. The profile records input-data, source-code
and native-binary hashes. Validation refuses a changed model or executable.

Each candidate checks both energy ledgers, allowed material/parameter changes
and exact observation alignment. Final exports must replay exactly. Final runs
are compared with all three numerical controls halved: max_dt, consumed fraction
and source temperature-rise cap. Scored-window and early/source differences are
reported separately, not conflated. Chemical power is recorded at experimental
scan times and its sampled maximum is not a resolved peak-power measurement.

Rate and in-plane-residue perturbations are evaluated on the **training record
only**, without compensating other parameters. These are sensitivity checks,
not confidence intervals or profile-likelihood inference. Multiple solutions
with similar residuals must not be presented as a unique physical identification.

## Result, 2026-10-13

Both optimizer starts terminated successfully (219 evaluated candidates total).
The profile was then frozen before running the active-cooling experiment.

| Same 300–1800 s witness objective | Previous pulse/structure v3 | Kinetic v4 |
| --- | ---: | ---: |
| No-cooling fit RMSE | 6.8624 K | 6.8609 K |
| Active-cooling held-out RMSE | 12.8051 K | 12.8204 K |

This is effectively unchanged predictive error, not an improvement in model
adequacy. The largest held-out errors remain on C3: TIM RMSE 18.625 K and ISS
16.698 K, with mean biases -17.220 K and -16.211 K respectively (underheating).
In the training case C3/ISS is instead overpredicted, bias +11.447 K.

The selected numerical profile is E = 276.999 kJ, rate at 400 K = 4.91434 /s,
destroyed h(ISS) = 0.415861 and h(TIM) = 0.559769 W/(m² K), residual structural
multipliers 0.00586018 through-plane and 0.00520465 in-plane. **These are not
uniquely recovered physical material constants.** In particular:

- A second start gives E = 82.297 kJ and different destroyed contacts, yet RMSE
  is 6.8631 K: just 0.0022 K worse with an energy reserve smaller by a factor 3.37.
- Holding all other selected parameters fixed, rates from 0.491434 to 10 /s
  change training RMSE only from 6.86156 to 6.86089 K (span 0.000676 K).
- Such differences are comparable to or smaller than the numerical refinement
  check, and far smaller than the structured residual to experiment.
- Equal sampled t10/t50/t90 values in these exports mean that the release occurs
  between instrument scans, not that its physical duration was measured as zero
  or as exactly one scan interval. The fast process is not identified here.

Halving all three step controls changes scored witness temperatures by at most
0.001101 K (fit) / 0.002789 K (validation). Across all six channels, including
the unscored early source faces, maxima are 0.024411 / 0.032361 K. Thermal balance
residuals are below 7e-8 J; chemical balance residuals below 1e-9 J. Exported
scenarios replay exactly. The accepted intact curves retain 0 K change.

[Comparison](../data/calibration/upv-kinetic-v4/comparison.png),
[modeled source histories](../data/calibration/upv-kinetic-v4/process.png), and
[machine-readable assessment](../data/calibration/upv-kinetic-v4/assessment.json).

Conclusion: the fitting procedure works and preserves the original material
properties, but these late temperatures do not pin down chemical reserve and
reaction speed separately from post-rupture contacts. The profile is retained
as an exploratory conditional fit, **not promoted as calibrated failure physics**.
The remaining systematic error is not explained by CUDA energy imbalance or
time-step accuracy. This result alone does not establish that a full mass-flow
solver is necessary; unresolved exposure/heat-transfer paths and observation
assumptions still need independent constraints.

The old accepted intact scenes and v3 failure artifacts are retained unchanged.
The default viewer scene is not replaced by this exploratory fit.

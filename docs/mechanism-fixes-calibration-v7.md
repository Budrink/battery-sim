# Mechanism fixes, staged CUDA refactor and UPV v7 recalibration

This revision fixes two demonstrated implementation defects; it does not certify
that all physical closures describe the real battery. Earlier frozen fits are
preserved. The default viewer scene is not replaced.

## Corrections and regression coverage

1. Gas channels used to cap the flow rate against a provisional dt. An unrelated
   process could reduce the accepted dt afterwards, leaving an obsolete rate.
   A controlled fixture accepted the same 5 microsecond step yet transported
   100 times different masses for proposed caps 0.1 and 0.001 s. The channel now
   proposes a raw rate and an extensive equilibrium mass cap independently of dt.
   Shared mass/thermal budgets select the final step; a single accepted packet
   updates both endpoints and the channel ledger. Both controls now give the
   same analytically expected transfer, approximately 4.5122e-11 kg.
2. Rupture changed shell damage to one after temperature closure. For
   damage-dependent capacity, stored energy and reported temperature disagreed.
   The regression fixture stayed at 400 K where its constitutive law required
   500 K. Rupture now calls the same energy/composition closure as thermal
   evolution. Stored energy is unchanged; temperature is refreshed.

Both targeted GPU tests failed on the previous executable and pass after the
fix. The shared capped-budget helper also passes 2,000 deterministic randomized
CPU cases, including unbounded and saturated demands. This tests numerical
accounting, not a statistical physical model.

## Refactor

`cuda_step.cuh` owns six named orchestration stages: solid step bounds, gas
transport proposals, gas step bounds, process proposals, candidate gather and
closure/validation. Physical laws remain in their specialized kernels; the
shared budget algebra lives in `capped_budget.hpp`.

Proposals read committed state and write scratch/candidate buffers. One owner
gathers each output in fixed order. All kernels use the ordered CUDA stream;
there is no host barrier between stages. Prepare/discard/commit remains
transactional. The same-device result is exact for 64 and 256-thread pointwise
tiles and repeated discarded proposals. This is not a promise of bitwise
portability between GPUs or of physical time evolution being idempotent.

Verification after the fixes: 19/19 native tests passed; Python discovery ran
91 tests with 20 skipped and no failures. CUDA memcheck on the new mechanism
test and racecheck on the transactional test reported no errors (nor race
warnings). These focused checks do not establish absence of every possible bug.
The [verification summary](../data/verification/mechanism-fixes/summary.json)
records the before/after executable hashes and results.

## Recalibration protocol

1. Fit intact heating using only no-cooling, warm-starting from the old training
   profile. Freeze before active-cooling validation.
2. Regenerate event-conditioned initial fields from the fresh heating fit.
   Replay heating with the gas workflow and verify exactly unchanged results.
3. Fit the same six v6 gas/destruction parameters on no-cooling only, warm-start
   from its old training fit; freeze, then replay active-cooling.
4. Check refined steps and a densely sampled early pressure transient.

Measured whole-cell heat capacity (412.87 J/K) and reference conductivity are
not optimized. The casing subtracts its capacity from the contents rather than
adding capacity to the measured total. Old failed-contact and residue factors
remain fixed priors. Validation has been inspected in earlier development but
is not used to select parameters in this search; it is not blind validation.
The optimizer is a local warm-started search, not a global-identifiability proof.

Heating outputs: `data/calibration/upv-physical-casing-v7/heat/`.
Gas/destruction outputs: `data/calibration/upv-gas-energy-v7/`.

## Heating result

| Metric | No-cooling calibration | Active-cooling validation |
|---|---:|---:|
| Witness RMSE | 1.02861 K | 1.31183 K |
| Maximum absolute error | 3.33501 K | 4.74045 K |

Heating converged by ftol (2 optimizer evaluations, 16 actual candidates).
Several connectivity/contact parameters remain at bounds, so these are effective
fit parameters, not uniquely identified material properties. Heating replays in
the gas workflow have zero temperature difference (15,028 / 24,004 steps).
Halving the CFL bound and step/process caps increases these to 29,614 / 47,302
steps; maximum temperature changes are 0.000905 / 0.000677 K. This is the
meaningful refinement check, unlike merely reducing a nonbinding external cap.
See [heating curves](../data/calibration/upv-physical-casing-v7/heat-comparison.png)
and [numerical checks](../data/calibration/upv-physical-casing-v7/numerics/assessment.json).

## Destruction search

The six-parameter local search converged by ftol (2 optimizer evaluations,
14 actual candidates). Its common fit objective changed from 10.696571 for the
old parameters on the repaired model to 10.696234. This is negligible improvement,
not a recovered pressure transient. The late temperature RMSE is 6.42939 K;
the original-scan pressure maximum is 6.24695 bar versus 7.98273 bar measured.

Frozen parameters: local energy 109.6295 kJ, gas chemical energy 213.0765 kJ,
gas yield 0.3914374, wall h 24.60363 W/(m² K), exposed-surface multiplier
2.6330304 and tied solid reference rate 5.197676/s. Their sum is one chemical
reserve, not two independently added energy sources. No gas-fit parameter
reaches its exploratory bound; none is thereby established as a measured property.

On the training case, full step/CFL/process refinement changes temperature by
at most 0.002698 K and pressure by 0.001595 bar (79,830 to 155,968 steps).
This is much smaller than the experimental residual. The targeted bugs were
genuine, but fixing them does not explain that residual in this scenario.

## Frozen validation

| Metric | No-cooling calibration | Active-cooling validation |
|---|---:|---:|
| Late witness RMSE, v7 | 6.42939 K | 13.62750 K |
| Late witness RMSE, previous v6 | 6.42881 K | 13.63093 K |
| Pressure peak on original scans, model | 6.24695 bar | 6.19222 bar |
| Pressure peak on original scans, measured | 7.98273 bar | 7.85628 bar |
| Pressure RMSE, 0–30 s | 0.43602 bar | 0.50540 bar |
| Pressure RMSE, 30–1800 s | 0.05100 bar | 0.09512 bar |

The validation temperature bias is -11.98 K overall; the strongest channel
biases remain C3/TIM (-16.01 K) and C3/ISS (-18.64 K). This is essentially
unchanged from v6. No validation-driven refit was performed, and this profile
is **not promoted as validated destruction/cascade physics**.

Validation refinement changes temperature by at most 0.002583 K and pressure
by 0.001596 bar (78,888 to 155,005 steps). Across the two saved nominal runs,
maximum sampled residuals are 1.20e-7 J thermal, 2.86e-9 J chemical,
4.56e-15 kg total mass and 2.16e-16 kg oxidizer. Conservation verifies accounting,
not physical adequacy or sensor/contact identification.

A dense 0.01 s early replay gives modeled maxima 6.34618 bar at 1.23 s and
6.34596 bar at 1.24 s, still below the measured sampled peaks. Halved-step
dense checks differ by less than 0.00135 bar. A training-only diagnostic with
100 times the opening threshold moves rupture from about 0.0070 to 0.0691 s
but changes the maximum by only 0.00016 bar. This sensitivity check is not a
physically identified casing strength or an extra selected calibration candidate.

Plots: [pressure](../data/calibration/upv-gas-energy-v7/pressure-comparison.png),
[late temperatures](../data/calibration/upv-gas-energy-v7/witness-comparison.png).
The [machine-readable assessment](../data/calibration/upv-gas-energy-v7/assessment.json)
contains the frozen parameters, provenance, individual channels and ledgers.

## Remaining model scope

The source is initialized conditional on the observed primary event; this is
not calibrated prediction of ignition or electrical abuse. Internal gas volume
is a fixed assumed 5 mL. Liquid displacement, evaporation and changing free
volume are not implemented. Gas has a single effective R/cv and no changing
species/molar mass. Finite-oxygen combustion, casing and chamber wall retain
explicit engineering proxies, not identified chemistry or measured casing CAD.
Historical witness-contact numbering remains an unconfirmed hypothesis.
The pressure comparison is for the 120 L chamber, not internal cell pressure.

Mass/energy conservation and time-step convergence cannot by themselves remove
these physical uncertainties. A deficient pressure fit must not be concealed by
fitting measured battery properties or selecting against validation data.

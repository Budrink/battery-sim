# First internal-link calibration: 3×3×3

Status: **exploratory, grid-sensitive; not promoted to a mesh-independent
material calibration**. The original viewer/default and saved external-contact
fits remain unchanged. The requested 3×3×3 battery has one core volume and 26
shell volumes; all 54 internal links are solved on CUDA.

## What was fitted

Two shared coefficients, `a` and `b`, change the **internal numerical-edge**
conductance using the existing hot-donor directed law:

```
Gref = k_ref,axis × face_area / center_distance
theta = (Tdonor - 298.15 K) / 100 K
z = log(0.8 / 2) + a theta + b theta²
L = l0 + width exp(z)
G = Gref × [0.2 + 2.8 sigmoid(z)]
```

There is one conservative flux, not an average of two directed conductances.
The independent `thermal.bulk_response` no longer inherits the fitted external
`surface_connectivity` response. Existing scenes without this optional field
keep their old behavior. The same internal law acts on core–shell and
shell–shell edges; no separate coefficients per edge or region were introduced.

At the reference temperature the factor is exactly one. For any temperature it
is bounded between 0.2 and 3; coefficients are bounded to ±12. These are declared
exploratory bounds, not measured battery-property uncertainty. The law supports
a nonmonotonic response. No damage/destruction coefficients were fitted, and
the intact pre-event cases do not constrain their behavior.

Frozen: both batteries' reference k=(0.9,28.3,28.3) W/m/K, C=412.87 J/K,
all pad/material properties, all external contacts and their directed laws,
boundary forcing/loss coefficients, topology and observation bindings.
Changing reference k or a frozen external parameter through the internal probe
is rejected. Every evaluation audits the actual native parameters and energy
balance. With a=b=0 the old 3³ histories reproduce exactly.

Important distinction: this is an effective **graph-edge** calibration, not an
identified continuum k(T). The half-volume resistances from an exterior battery
voxel center to its contact face are still computed from fixed k_ref; the new
factor applies only between internal nodes. Consequently grid refinement also
changes the share of the path affected by the fitted law. This is one structural
reason to review the boundary-half treatment before interpreting the fit as a
resolution-independent material law; the grid test below does not isolate its
contribution from other discretization effects.

## Result

Fit on no-cooling only, scoring 60–884 s. Both starting points converged to the
same bounded solution:

```
a = -9.374269586
b = +12.0                 # upper exploration bound
```

The local Jacobian condition number is about 18.8. Agreement between two starts
is not proof of a global optimum or unique physical parameters. The bound hit
means the quadratic curvature remains constrained by our chosen search domain.
Bounds were not widened to lower the error.

| Experiment | RMSE before, K | RMSE after, K | Max error before, K | Max error after, K |
|---|---:|---:|---:|---:|
| No cooling: parameter fit | 2.185 | 1.239 | 8.044 | 4.027 |
| Active cooling: frozen check | 2.251 | 1.343 | 9.532 | 5.239 |

All four channels improve in RMSE in both cases; the C1 TIM face remains worst
(RMSE 2.081 / 2.345 K). C3 ISS's maximum error on the second case increases
from 2.073 to 2.681 K despite its lower RMSE. Improvement is not uniform at every
time point. The factor is approximately 0.620 at 35 °C, 0.453 at 45 °C and 0.369
at 65 °C. This supports, but does not prove, the hypothesis of excessive internal
mixing in the baseline; it is not a measurement of the cell's conductivity.

The second dataset did not enter this fit or its parameter selection. It was
previously inspected during debugging, so this is not a pristine blind holdout.
The original C1/C3 acquisition-numbering hypothesis is still unconfirmed.
These measured-boundary pre-event runs do not validate runaway or degradation.

## Numerical qualification

| Check | No cooling, K | Active cooling, K |
|---|---:|---:|
| Maximum change after smaller actual dt | 0.00482 | 0.00390 |
| Maximum 3³ versus 4³ difference | 0.629 | 0.591 |
| Maximum 3³ versus 12³ difference | 2.051 | 1.871 |
| Maximum 8³ versus 12³ difference | 0.383 | 0.349 |

The temporal check passes. The former 0.5 K battery-grid criterion **fails**,
and 12³ is not adequately converged under the former 0.1 K reference criterion.
The previous 0.223 K result applied to constant internal links and does not carry
over to this new fitted law. No mesh change or further fitting on active cooling
was used to conceal this. The retained profile is explicitly a **3³ graph
approximation**, not a claim that 3³ resolves a continuum law to 0.5 K.

Absolute energy residuals are below 6e-10 J in the main fitted runs. Exported
native scenarios exactly reproduce the probe results. Native tests cover
independent surface/bulk responses, hot-donor selection, one-step CPU/CUDA
agreement, conservative heat exchange and the zero-response limit. The full
suite passed: 50 Python tests (with native probes enabled) and 12 CTest tests.

## Artifacts and reproduction

- [Assessment](../data/calibration/upv-internal-3x3x3/assessment.json)
- [Before/after temperature plots](../data/calibration/upv-internal-3x3x3/comparison.png)
- [Frozen profile and optimization history](../data/calibration/upv-internal-3x3x3/profile.json)
- [Fit report](../data/calibration/upv-internal-3x3x3/fit.json)
- [Frozen second-experiment report](../data/calibration/upv-internal-3x3x3/validate.json)

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/calibrate_upv_internal.py fit \
  --probe /tmp/battery-sim-cuda-v1/battery_sim_experiment_probe --numerical-check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/calibrate_upv_internal.py validate \
  --probe /tmp/battery-sim-cuda-v1/battery_sim_experiment_probe --numerical-check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/report_upv_internal.py
./run_viewer.sh data/calibration/upv-internal-3x3x3/no-cooling-fitted-scenario.json
```

The exported scenario runs directly in native CUDA. Do not reapply fitted
external parameters or internal probe overrides to it. Model, binary, protocol,
script, dataset and frozen-profile hashes guard the second-experiment check.

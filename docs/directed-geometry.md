# Directed connectivity v6

The v5 `state_geometry` law imposed symmetry by averaging endpoint log-metrics.
That is not the requested ordered transport operator. It is retained only as an
explicit legacy model for reproducing earlier results. `directed_geometry` is
the new native CUDA law and the default UPV calibration family.

## Ordered state response, no averaging

For one candidate direction, identify **source s** and **receiver r**. Read the
source material's local-axis profile and the two volumes' current states:

```
theta_s = (T_s - Tref_s) / Tscale_s
z_s->r = zref_pair
       + a_s,n theta_s + b_s,n theta_s² + d_s,n D_s
       + c_s,n (T_s - T_r)/Tscale_s + e_s,n (D_s - D_r)
L_s->r = l0 + activation_width exp(z_s->r)
Gcontact_s->r = overlap_weight [Gfloor + Gspan sigmoid(z_s->r)]
G_s->r = 1 / [1/Gcontact_s->r + Rbulk_s + Rbulk_r]
```

The receiver material also determines the material-pair reference contact law;
its local state enters through signed contrasts. On reversal, its own material
profile becomes the source profile. There is **no** arithmetic/geometric average
of either direction, or of the endpoint log responses. The polynomial can be
nonmonotonic. Coefficients `a,b,d,c,e` may depend on material and local axis.
The implementation reads the corresponding axis component because the current
mesher requires aligned local frames; arbitrary nonaligned faces remain out of
scope. Source contrasts are configured as `temperature_contrast` and
`damage_contrast` vectors inside `thermal.surface_connectivity`.

This is a phenomenological closure, not a derivation from microscopic contact
physics. No plasma E/m/charge formula was transferred. Absolute L/l0 remain
fixed order-parameter conventions, not measured gaps or mechanical deformation.
The possible-neighbor graph is unchanged; transfer availability is recomputed
from current states during each CUDA substep. Known bulk k and heat capacity
are not modified by this contact law.

## Deterministic selection and conservation

```
Q_a->b = dt [G_a->b max(Ta-Tb,0) - G_b->a max(Tb-Ta,0)]
Ea -= Q_a->b
Eb += Q_a->b
```

Only the hot-source branch is evaluated in the time-stepping kernel. No MC is
needed. Equal temperatures produce exactly zero flux even if the two candidate
conductances differ. Each physical patch has one flux, gathered with opposite
signs; there are not two independently updated node energy budgets. The existing
CFL bound uses the maximum intact/destroyed G floor+span, which bounds both
directions before transfer, including runtime scaling and series resistance.

Distinguish **physical gradient reversal** from **storage relabeling**. Swapping
hot/cold temperatures while retaining the materials can change the magnitude
of transfer. Swapping a/b indices together with all their properties must leave
the physical solution unchanged and merely interchange the directional records.
Both cases are tested. Symmetric transport remains possible as a special case,
e.g. zero response/contrast coefficients and a reciprocal reference contact.
For the UPV fit, the shared sender polynomial plus the contrast term also gives
the same small-gradient reference response at equal temperatures across material
classes: fitted directional material differences enter through delta T, not two
arbitrary discontinuous directional constants.

Destruction updates a **shared physical contact history once**, then switches
both candidate laws irreversibly to destroyed l0/opening/floor/span. It can
increase or decrease conductance sharply. A shared edge-aging temperature is a
separate damage-kinetics assumption, not an average of transport directions.
The source profile still distinguishes branches when the destroyed active span
is nonzero. The UPV placeholder closes that active span and is not calibrated
from these pre-event observations.

## Calibration and observations

Fitted: five material-pair reference h values, environmental h, shared sender
polynomial a,b, and five source-material contrast coefficients c (battery, ISS,
TIM, aluminium, steel). Thus 13 parameters in this first bounded trial. Damage
coefficients and damage kinetics are not fitted and remain inactive in UPV.
Temperature normalization: Tref = 298.15 K, Tscale = 100 K. Signed coefficients
are bounded by ±12; contact h by 1–10000 W/m²/K, reference h by 2–9900, and
environment h by 0.1–30. These are exploration bounds, not measured uncertainty.

All bulk k, cp, density and layer thickness are fixed during fitting. Battery
k=(0.9,28.3,28.3), cp=1007 J/kg/K, mass=.410 kg. ISS cp=770 and TIM cp=1175
are working composition-dependent assumptions, not newly measured pad values.
The fixed cooling-tip prior and unconfirmed C1/C3 numbering hypothesis remain.

Fit only no-cooling (60–884 s), then freeze parameters before active-cooling
(60–1412 s). The second experiment was previously inspected during debugging;
it is not a pristine blind validation. Neither dataset here calibrates a runaway
source or destruction kinetics: C2 remains an imposed measured surface boundary.
The C2 boundary uses the **battery** source profile when hotter than its adjacent
pad, and the pad source profile when the gradient reverses.

Surface-temperature reconstruction invokes the same host/CUDA directed law on
the current local states with dt=0. It does not average the candidate G values
or reuse stale G. Reports contain actual fixed material properties, both ordered
G/L time series, selected G, and net signed power. The reverse-capacity trace is
a diagnostic at the same states, not a claim of measured reverse heat flow.

Old v5 reports remain intact and are not relabeled as directed calibrations.
New artifacts use `upv-directed-geometry-*`. The model/binary/profile hashes must
match for validation. Native exported scenarios contain the fitted values and
can run without the Python fitting process; an empty probe request also preserves
their reference conductances (covered by an export equivalence test).

## Recorded result (2026-10-13)

Two bounded starts, selected using no-cooling only, gave the following result.
Temperature errors in kelvin are numerically identical to errors in degrees C.

| Model / experiment | RMSE, K | Maximum absolute error, K |
| --- | ---: | ---: |
| Directed v6, no-cooling fit | 2.173154 | 8.114049 |
| Directed v6, active-cooling frozen check | 2.234134 | 9.562498 |
| Static contacts, refitted with identical fixed properties, no-cooling | 2.192746 | 8.215634 |
| Previous symmetric v5, no-cooling | 2.183601 | 8.205063 |
| Previous symmetric v5, active-cooling | 2.230057 | 9.684975 |

The directed implementation is different, but its temperature-fit improvement is
small; validation RMSE is slightly worse than v5. Channel 102 (C1 TIM face)
remains the largest residual: its RMSE is 3.5166 K in fitting and 3.6149 K in
validation. This is not a claim that the earlier discrepancies have been solved.

**The directional material coefficients are not identified by this fit.** Nine
of thirteen parameters are near exploration bounds. The scaled optimizer's local
Jacobian condition number is about 4.59e9. The other start gives RMSE 2.175221 K
while, for example, its battery temperature-contrast coefficient is -11.9857
instead of +11.9993, accompanied by different sender and contact coefficients.
Both optimizers stopped on the relative-cost criterion, not proof of a global
optimum or unique material law. Do not interpret these coefficients as measured
rectification, and do not expand bounds solely to lower the score.

The present records predominantly heat from C2 outward. A matched experiment
with the gradient reversed at the same material pair, mean temperature and
contact pressure would constrain the other direction much more directly.
Independent contact-resistance or fixture-loss measurements would also help
separate correlated parameters. No additional fitting to the second experiment
was performed.

Frozen numerical checks, with no retuning:

| Perturbation | Max temperature change, fit / validation, K |
| --- | ---: |
| Half maximum timestep | 0.003630 / 0.002960 |
| Passive mesh refined twofold per axis (112 to 784 total volumes) | 0.094238 / 0.356826 |

These are single refinement checks, not a full mesh-convergence proof. Absolute
energy residuals across these runs are below 8e-9 J. Tests additionally cover
step-by-step CPU/CUDA agreement, equal-temperature equilibrium despite unequal
candidate conductances, nonconforming patches with opposite local donors on one
macro contact, physical gradient reversal versus index relabeling, measured
boundary reversal, and irreversible destruction without spurious energy.

Artifacts: [fit profile](../data/calibration/upv-directed-geometry-fit.json),
[calibration plot](../data/calibration/upv-directed-geometry-calibration.png),
[validation plot](../data/calibration/upv-directed-geometry-validation.png),
[ordered-connectivity plot](../data/calibration/upv-directed-geometry-connectivity.png),
and [native fitted scenario](../examples/upv-directed-geometry-calibrated.json).
The last plot shows candidate capacities at the same state; it is not a reverse
experiment. The native scenario runs directly in CUDA, without a Python solver.

## Reproduction

```bash
python3 scripts/calibrate_upv.py fit --probe /tmp/battery-sim-cuda-v1/battery_sim_experiment_probe --parameter-set directed_geometry --starts 2 --max-evaluations 120 --profile data/calibration/upv-directed-geometry-fit.json --output data/calibration/upv-directed-geometry-calibration.json
python3 scripts/calibrate_upv.py validate --probe /tmp/battery-sim-cuda-v1/battery_sim_experiment_probe --profile data/calibration/upv-directed-geometry-fit.json --output data/calibration/upv-directed-geometry-validation.json --numerical-check
python3 scripts/check_upv_numerics.py --probe /tmp/battery-sim-cuda-v1/battery_sim_experiment_probe --profile data/calibration/upv-directed-geometry-fit.json --report data/calibration/upv-directed-geometry-calibration.json --output data/calibration/upv-directed-geometry-numerics.json
python3 scripts/export_upv_state_geometry.py --profile data/calibration/upv-directed-geometry-fit.json --report data/calibration/upv-directed-geometry-calibration.json --output examples/upv-directed-geometry-calibrated.json
./run_viewer.sh examples/upv-directed-geometry-calibrated.json
```

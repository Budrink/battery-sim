# UPV physical-casing calibration trial (v5)

This is an **assumption-conditioned trial**, not identification of the actual
pouch packaging or validated gas/combustion physics. Historical profiles and the
default viewer scene are preserved. Outputs: `data/calibration/upv-physical-casing-v5/`.

## Evidence versus assumptions

The [UPV description](https://zenodo.org/records/22273467/files/TR_Experimental_Data_Description.pdf)
specifies a 35 Ah NMC semi-solid-state pouch, 11×87×187 mm, 410 g, cp=1007 J/(kg K),
k_through=0.9 and k_inplane=28.3 W/(m K). It specifies a **120 L sealed safety
chamber**: channel 114 measures chamber pressure, not cell-internal pressure.
Pouch laminate/thickness, internal free volume, opening area and rupture strength
are not specified. Stack clamping pressure is not a rupture threshold.

[DNP's packaging description](https://www.global.dnp/en/our-business/innovation/battery-pouch/)
supports using thin laminate rather than a massive metal can, but does **not**
identify this sample or the following engineering proxy: thickness 150 µm,
density 1600 kg/m³, cp 1250 J/(kg K), normal k=0.3 and tangential k=50 W/(m K).
The tensor rotates with each panel normal; seam contact h=500 W/(m² K) is fixed.
Training-only sensitivity uses 100/200 µm without refitting.

Measured whole-cell capacity is partitioned, **never fitted or counted twice**:
412.87 = 401.351446 contents + 11.518554 casing J/K. Proxy casing mass is
9.2148432 g. Contents dimensions shrink by twice film thickness; outer dimensions,
layer positions and sensor sides remain fixed. Reported whole-cell k values are
retained as contents reference priors, not independently measured contents-only
properties. All ISS/TIM/metal properties remain fixed. Thermocouples now observe
the external casing contact faces, not contents through an invisible casing.

## Heat-transfer stage

Fit no-cooling only; freeze before active-cooling validation. The four witness
channels, forcing-defined pre-event end and scoring from 60 s are unchanged.
C2 remains measured surface forcing here, not a solved electrical overcharge
source. The historical **unconfirmed** swapped C1/C3 numbering is retained.
Validation was seen during previous development; it is not pristine blind data.

Seven variables: three internal log-response knots, directed internal contrast,
casing/ISS and ISS/aluminium contact references, contents–casing h. No capacity
variable is fitted. Objective: four-channel temperature residuals, without an
extra smoothness penalty; the reported RMSE uses the previous scoring window.

| Pre-event RMSE | Previous intact model | With casing |
|---|---:|---:|
| No-cooling fit | 1.003729 K | 1.028606 K |
| Active-cooling validation | 1.272585 K | 1.311822 K |

Insertion without refitting gave 1.213109 K on training. Search converged by ftol:
12 optimizer function evaluations / 96 actual candidates including Jacobians. Contents–case
h reached its exploratory upper bound 2000 W/(m² K); the 70 °C internal knot and
internal contrast approach lower bounds. These are not measured limits or uniquely
identified constants. Adding this casing proxy **did not improve** either metric.

Halving max_dt alone did not change the actual CFL-limited step. Independent
verification halves CFL plus max_dt/source/gas controls: maximum scored changes
0.000904 K (fit), 0.000677 K (validation), steps 15028→29614 / 24004→47302.

## Destruction-stage contract

Heat parameters are frozen. C1/C3 contents and casing fields carry over from the
new warmup; C2 initial fields use the same four pre-event face readings and the
existing bilinear assumption. This is event-conditioned, not an identified
electrical overcharge/ignition clock. The previous four-witness 300–1800 s
objective at original instrument scans is retained. Early/source temperatures
and chamber pressure are explicit diagnostics, not fit targets or offset knobs.

Initial search variables: net chemical energy, tied activation/completion reference
rate, two destroyed exterior contacts and two residue conductivity factors. The
training-only six-parameter trial showed no observable sensitivity to the destroyed
contact references in these perturbations and very weak sensitivity to rate.
Those three values were therefore frozen at their starting priors; the final
search fits **energy and the two residue factors only**. The exploratory trial is
preserved in `failure-six-parameter-trial/`. Starting values come from the v4
training-only fit, not validation. The tied rate is not the same constitutive
parameter as v4's single-stage rate. Lack of local sensitivity is not proof that
these parameters are irrelevant in other regimes.

Unidentified fixed priors: 0.2 kg reactive mass, 30 kJ/kg activation absorption,
60 kJ/mol completion activation energy, gas yield 0.2, gas chemical fraction 0.2,
5 mL internal free volume, 10 mm² opening, 100 kPa reference rupture overpressure.
Species heat capacities determine capacity changes on conversion; intact cp is
not fitted. The gas uses a single ideal pseudo-species.

Gas enters the documented 120 L chamber, without a calibrated wall-loss or flame
model. Measured ch109 remains conditional ambient forcing for solids; modeled
gas is **not** applied again as a duplicate ambient source. This is not a
closed-loop prediction of gas exposure or cascade propagation. Pressure is an
independent model-adequacy diagnostic, not claimed calibrated internal pressure.

Two stiff initial runs were stopped without a result. Diagnosis exposed the old
split reaction integrator: a short-lived intermediate forced global microsteps.
It now integrates the coupled two-stage chain **exactly at frozen T**, including
same-step completion of newly activated material. Limits constrain total converted
mass/temperature, not the fractional turnover of a tiny intermediate. Analytic
constant-T and stiff-rate tests pass; nonlinear feedback still requires refinement.
Orifice transfer also has a conservative pressure-equalizing mass cap.

After that integrator change both heating curves, material audits and step counts
replayed **exactly**, without refitting. Original profile retained as
`heat/profile-before-coupled-conversion.json`; current provenance records the replay.
The first completed failure baseline gave 21.3184 K RMSE using old starting
parameters; therefore v4 parameters cannot just be transplanted to the new model.

Failure search uses gas fraction/temperature controls 0.1 / 5 K. Final runs halve
those, source controls, max_dt and CFL together. Scored-witness and all-channel
changes must be distinguished; gas/early-event accuracy is not implied by a small
late-temperature discrepancy. Machine-readable search/result reports are stored
in `failure-fast/`; `failure/` retains inputs of the stopped slow initial attempt.

## Destruction trial result and adequacy limits

| Late witness RMSE, 300–1800 s | Previous v4 | Casing + conversion + gas |
|---|---:|---:|
| No-cooling fit | 6.860917 K | 6.607308 K |
| Active-cooling validation, no refit | 12.820446 K | 12.459468 K |

The reduced search stopped by xtol after 11 optimizer function evaluations / 29
actual candidates. Effective net chemical energy is 37.246 kJ; residue conductivity
factors are 0.0050006 through-plane and 0.0050017 in-plane, at the exploratory
lower bound. Rate 4.91434/s and destroyed ISS/TIM contact references
0.415861/0.559769 W/(m² K) remain **unidentified fixed priors**, not fitted results.
The original six-parameter trial and all earlier profiles are preserved.

Simultaneously halving CFL, dt, source and gas controls changes scored witness
temperatures at most 0.002333 K (fit) / 0.001944 K (validation); refined RMSE is
6.607146 / 12.460318 K. Thermal/chemical/mass residuals are below
6e-9 J / 2e-10 J / 2e-15 kg at sampled states. These checks support numerical
consistency of the implemented equations, **not physical model adequacy**.

The physical gas prediction fails an independent diagnostic:

| Chamber absolute pressure peak | Model | Measured |
|---|---:|---:|
| No cooling | 1.4805 bar | 7.9827 bar |
| Active cooling | 1.4775 bar | 7.8563 bar |

Both modeled sources fully convert, with sampled maximum volume temperatures
216.36 / 213.05 °C. Predicted rupture is at 0.0100 / 0.0112 s within experimental
scan brackets 0.471 / 0.757 s: the data cannot validate millisecond opening time.
No claim is made that the early source temperature or rupture kinetics is reproduced.
In active-cooling validation the largest late errors are C3/TIM and C3/ISS:
RMSE 17.15 / 16.92 K, with mean underprediction 15.77 / 16.40 K.

The fitted 37.246 kJ is **not a recovered total cell calorimetric energy**. It is
conditional on prescribed measured ambient heating and the unmeasured gas/material
priors. Matching late neighbors lets the optimizer trade heat release against
retention while missing the chamber impulse. A gas energy/production and chamber
heat-loss model constrained by pressure is needed before claiming calibrated
destruction; the present result does not identify which missing contribution
dominates. This does not, by itself, establish a need for CFD.

Figures: `heat-comparison.png`, `failure-comparison.png` (includes unfitted early
transient), `failure-scored-comparison.png` (late window only), and
`gas-diagnostics.png`. Model gas temperature is shown against local ch109 only as
a diagnostic: the thermocouple is not a measurement of chamber-average temperature.
Default viewer scenario is unchanged; this candidate is not promoted as validated.

Verification: all 17 native tests passed; Python suite ran 89 tests with 20 skipped
and no failures when CUDA was accessible. GPU-dependent tests fail in the restricted
sandbox where no CUDA device is exposed; the successful run used GPU access.

## Reproduction

```bash
python3 scripts/calibrate_upv_casing.py heat fit --output NEW_DIRECTORY
python3 scripts/calibrate_upv_casing.py heat validate --output NEW_DIRECTORY
python3 scripts/check_upv_casing.py --output NEW_DIRECTORY
python3 scripts/calibrate_upv_casing_failure.py fit --output NEW_DIRECTORY
python3 scripts/calibrate_upv_casing_failure.py validate --output NEW_DIRECTORY
MPLCONFIGDIR=/tmp/battery-sim-mpl python3 scripts/report_upv_casing.py --output NEW_DIRECTORY
```

Use a fresh search directory. Native `battery_sim_casing_probe` evolves all states
on CUDA and reconstructs current surface readings. Python supplies optimization,
scoring and provenance. Neither prior profiles nor raw observations are overwritten.

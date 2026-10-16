# UPV v6: close gas/solid thermal feedback before fitting destruction

Status: exploratory. This is a conditional event replay, not validated prediction
of ignition or cascade propagation. Results live in `data/calibration/upv-gas-energy-v6/`.
Previous v5 results and parameters are unchanged. No default viewer scene is replaced.

## What changed

- CUDA gas reaction converts a finite chemical reserve to sensible energy, limited
  by a transported finite oxidizer inventory. Mass is not created or removed.
- Gas-to-solid contacts have equal/opposite thermal flux, including physical cell
  casing. Every post-event `ambient-*` prescribed-temperature boundary is removed
  and replaced by contact with the modeled chamber. There is no ch109 heat input.
- A finite-capacity stainless-wall **proxy** absorbs heat and cools to the initial
  chamber temperature through an outer boundary. Measured active-cooling sink ch110
  remains a boundary condition; this is not a chiller/electrical-abuse model.
- Solid chemistry splits **one** net reserve into local heat and gas fuel energy.
  Activation absorption and emitted sensible energy remain explicitly accounted.

The thermal, chemical, total mass and oxidizer ledgers are independent diagnostics.
Constant-temperature gas-reaction analytic tests, oxygen exhaustion, no-oxygen,
ignition-threshold, outflow/reverse tracer transport and coupled refinement are
covered in `native/tests/process_network.cpp`. All 17 native tests passed.

## Frozen evidence and explicit assumptions

The [dataset description](https://zenodo.org/records/22273467/files/TR_Experimental_Data_Description.pdf)
gives chamber volume 120 L and pressure conversion `p[bar] = 2.5*U114[V]`.
Battery and pad reference material properties, casing proxy and prior fitted
intact connectivity remain unchanged. Both previous heating runs replay with
**zero temperature difference**, recorded in `heat-replay.json`.

Gas R=287 J/(kg K), cv=1000 J/(kg K), initial chamber oxidizer mass fraction 0.232,
no emitted oxidizer, effective gas reaction rate 20/s are fixed priors. Gas is
assumed ignited once fuel arrives; its local ignition process is unresolved.
The 13.1 MJ/kg oxygen conversion is a conventional-fuel engineering proxy from
[NIST](https://nvlpubs.nist.gov/nistpubs/sp958-lide/280-282.pdf), not a measured law
for this cell's gas mixture. Gas composition/molar-mass changes, temperature-varying
cv, radiation and pressure waves are not represented. Pressure is therefore a
lumped thermodynamic prediction, not a resolved blast transient.

The wall proxy uses area `6*(0.120 m³)^(2/3)`, 2 mm thickness, rho=8000 kg/m³,
cp=500 J/(kg K), k=15 W/(m K), external h=5 W/(m² K) to initial chamber T.
Only the volume is measured; shape, wall thickness, thermal skin depth and
external losses are assumptions. One aggregate wall node is not a chamber CAD.
No extra gas-to-contents path bypasses the casing. An internal-plenum/casing
thermal film is not identified; emitted gas already carries sensible energy.

## Calibration protocol

Fit no-cooling only, then freeze before replaying active-cooling. Validation was
inspected during previous development: it is held out from this optimizer, not
blind data. Historical unconfirmed C1/C3 numbering and event-conditioned initial
fields are retained; there is no time-offset or sensor-offset optimization.

Six positive log-parameters: net local energy, emitted gas chemical energy,
gas yield, chamber-to-wall h, common exposed-surface h multiplier, tied solid
activation/completion reference rate. Residue factors and failed-contact priors
are frozen at v5 values. Measured battery capacities/conductivities are not fitted.

Joint objective is a sum of squared block-normalized residuals:

- pressure 0–30 s, divided by 0.25 bar and sqrt(sample count);
- pressure 30–1800 s, divided by 0.05 bar and sqrt(sample count);
- each of four witness channels at the previous 300–1800 s window, divided by
  5 K and sqrt(sample count).

Those scales are declared fitting weights, not experimental uncertainty claims.
Reported late temperature RMSE retains the original v5 window. New and old fits
have different objectives and boundary conditions, so an improvement must not
be described as a controlled single-feature ablation.

Initial trial (before fitting) gave 8.232 bar on original scan times vs measured
7.983 bar, but peaked at 0.471 s instead of 0.942 s and decayed too quickly.
Late witness RMSE was 7.972 K. A peak alone is not adequate validation.

## Frozen trial result

The six-parameter search converged by ftol, 10 optimizer function evaluations /
64 actual candidates including Jacobian perturbations. No parameter reached its
exploratory bound. The fitted profile has local energy 109.084 kJ, gas chemical
energy 212.942 kJ (one total reserve 322.026 kJ), gas yield 0.391504, wall h
24.5912 W/(m² K), exposure multiplier 2.61318 and tied solid rate 5.20027/s.
These are effective, assumption-conditioned parameters, not measured properties.

| Metric | No-cooling fit | Active-cooling validation |
|---|---:|---:|
| Pressure peak on original scans, model | 6.24377 bar | 6.18957 bar |
| Pressure peak, measured | 7.98273 bar | 7.85628 bar |
| Pressure RMSE, 0–30 s | 0.43620 bar | 0.50647 bar |
| Pressure RMSE, 30–1800 s | 0.05098 bar | 0.09510 bar |
| Late witness RMSE, v6 | 6.42881 K | 13.63093 K |
| Late witness RMSE, v5 | 6.60731 K | 12.45947 K |

The validation temperatures **worsened**. The strongest underpredictions remain
C3/TIM (mean -16.02 K) and C3/ISS (-18.64 K); C1 is also underpredicted by about
6–7 K. Validation did not trigger a refit. This candidate is not promoted as a
validated destruction/cascade model. V5 used measured hot ambient forcing while
v6 predicts gas feedback, so the comparison is informative but not a controlled
single-parameter ablation. The present residual does not isolate a unique cause
among exposure topology, source kinetics, gas properties, wall assumptions and
unresolved radiation/jet effects.

Simultaneously halving CFL, dt and source/gas step limits changes full witness
curves by at most 0.002665 / 0.002560 K and pressure by 0.001582 / 0.001590 bar.
Step counts are 77207→153845 / 76396→152988. Maximum sampled ledger residuals over
both cases: thermal 8.1e-8 J, chemical 2.8e-9 J, mass 2.8e-15 kg and oxidizer
5.9e-16 kg. These verify implemented conservation, not model adequacy.

The training dense check resolves a 6.34167 bar maximum at 1.23 s; it remains
below the measured maximum, rather than hiding a large inter-scan pressure spike.
The validation dense maximum is 6.34142 bar at 1.24 s. Dense coarse/fine curves
differ by at most 0.001338 / 0.001331 bar. Experimental maxima are observed at
0.942 / 0.757 s, with scan brackets 0.471 / 0.757 s; do not infer millisecond
event timing from these records.
Increasing rupture threshold by 100 moves opening from 0.00703 to 0.06916 s but
changes its dense peak by only about 0.00015 bar against the same coarse control.
Thus that threshold change does not repair the deficit even with reactive gas.
This diagnostic does not establish the actual casing strength or opening time.

All 17 native tests pass. Python ran 91 tests with 20 skipped and no failures.
`jsonschema` is not installed in this environment: the schema was JSON-parsed,
and the new scenario was exercised by the native loader, but a separate full
JSON Schema validation was not run.

## Reproduction

```bash
python3 scripts/calibrate_upv_gas_energy.py verify-heat --output NEW_DIRECTORY
python3 scripts/calibrate_upv_gas_energy.py baseline --output NEW_DIRECTORY
python3 scripts/calibrate_upv_gas_energy.py fit --output NEW_DIRECTORY --max-evaluations 10
python3 scripts/calibrate_upv_gas_energy.py validate --output NEW_DIRECTORY
python3 scripts/check_upv_gas_energy.py --output NEW_DIRECTORY
MPLCONFIGDIR=/tmp/battery-sim-mpl python3 scripts/report_upv_gas_energy.py --output NEW_DIRECTORY
```

Python supplies the optimizer, data alignment and provenance; CUDA performs
thermal/reaction/gas evolution. A profile records numerical convergence/budget
termination, not a declaration of physical validity. Do not tune on validation.
The dense check uses 0.01 s outputs through 5 s and 0.1 s through 30 s, with
simultaneously halved CFL/source/gas limits. This diagnoses missed inter-scan
peaks; it does not synthesize extra experimental observations for fitting.

## V10: aperture-directed transfer with paid opening work

**Historical, superseded geometry:** this trial used a fitted 4.4538 mm² aperture
ceiling. Current progressive rupture derives area from the physical shell panel
and rejects that old ceiling. Results below document the old trial, not the
corrected model; its frozen profiles must be recalibrated before reuse.

The separate trial in `data/calibration/upv-aperture-v10/` enables the
[aperture connection and fracture ledger](physical-shell-gas-network.md#aperture-exposed-surface-connections-and-opening-work).
All evolution still runs in CUDA; Python only orchestrates fitting and reporting.
Neither measured solid capacity/conductivity nor the frozen intact connection
laws are optimized. The six effective energy/gas/contact coefficients from v9
and one aperture heat-transfer coefficient are selected using **no-cooling only**.
Active-cooling is replayed after freezing, without parameter selection on it.
It is previously inspected held-out data, not a blind validation experiment.

The old v9 shell strength/rate/total opening area remain fixed. The total area
is 4.4538055e-6 m² (4.4538055 mm²), distributed over six panels; seam width and
strength ratio remain priors. Thus the fitted surface coefficient is conditional
on this effective area and geometry, not an independently measured jet or
material coefficient. The first receiving solids are TIM2 at +x and ISS2 at -x;
the other four normals have no receiving solid in this geometry. No connection
passes through these first surfaces to a more distant layer.

Training-only screening spans h=100..1e7 W/(m² K) and opening-work coefficient
1..1e6 J/m². These are exploratory sensitivity ranges, **not material-property
measurements or established plausible intervals**. At fixed initial energy and
gas parameters, changing the work coefficient across that whole range changes
the witnesses by less than 0.000236 K and chamber pressure by less than
0.000103 bar. It is therefore not identified from these observations. Its prior
1000 J/m² is retained explicitly; opening work remains debited and audited.
The resulting full-opening cost is 0.0044538055 J in this effective-area model.
This does not establish the real pouch's rupture energy, total crack area,
delamination or deformation work.

The scan also contains an exact legacy-v9 replay with both new mechanisms
disabled, and an enabled ablation at otherwise identical parameters. Existing
heating curves replay with zero difference on both experiments. The reported
thermal objective remains the four witness temperatures at 300–1800 s, while
pressure is scored separately at 0–30 s and 30–1800 s. Burst thermocouples are
not used to fit the temperature peak, and no synthetic samples are introduced
into the experimental objective. No sensor offsets or timing shifts are fitted.

Reproduction in a fresh output directory:

```bash
python3 scripts/calibrate_upv_aperture.py verify-heat --output NEW_DIRECTORY
python3 scripts/calibrate_upv_aperture.py scan --output NEW_DIRECTORY
python3 scripts/calibrate_upv_aperture.py fit --output NEW_DIRECTORY --max-evaluations 12
python3 scripts/calibrate_upv_aperture.py validate --output NEW_DIRECTORY
python3 scripts/calibrate_upv_aperture.py check --output NEW_DIRECTORY
MPLCONFIGDIR=/tmp/battery-sim-mpl python3 scripts/report_upv_aperture.py --output NEW_DIRECTORY --dense --baseline-validation --dense-baseline
```

The profile and scan fingerprints bind native sources, the executable and
calibration dependencies. The report records optimizer termination, coefficients,
per-surface errors, each objective block, channel heat/work, conservation,
time-step refinement, and dense pressure maxima. Internal cell pressure, unlike
chamber pressure, is unmeasured; its predicted peak is exposed as a diagnostic,
not silently treated as validated. The candidate does not replace any accepted
profile or default viewer scenario automatically.

### Frozen v10 fit and held-out replay

The local seven-parameter fit terminated by `ftol` after 4 optimizer evaluations
and 32 native candidates including finite-difference probes. Its parameters are
349.894 kJ local net energy, 200.149 kJ emitted chemical energy, gas yield
0.386836, wall h=26.3739 W/(m² K), exposure multiplier 1.90501, tied solid rate
2.89658/s and aperture h=6.52347e6 W/(m² K). None is at its search bound.
This large aperture h applies to the tiny effective opening area, **not** to the
whole cell face; its uncapped total G is about 29.054 W/K. Actual heat transfer
is further limited by opening and the outgoing gas heat-capacity rate.

| Metric | No-cooling fit | Active-cooling held-out |
|---|---:|---:|
| Previous v9 late witness RMSE, 300–1800 s | 6.86166 K | 14.00918 K |
| V10 late witness RMSE, 300–1800 s | 6.58311 K | 13.82862 K |
| C1/ISS RMSE | 1.45071 K | 5.19358 K |
| C1/TIM RMSE | 4.13447 K | 5.43369 K |
| C3/TIM RMSE | 4.51654 K | 18.97915 K |
| C3/ISS RMSE | 11.56512 K | 18.66056 K |
| Pressure RMSE, 0–30 s | 0.40815 bar | 0.50084 bar |
| Pressure RMSE, 30–1800 s | 0.05158 bar | 0.09606 bar |
| Model pressure peak at original scans | 6.36555 bar | 6.29879 bar |
| Observed pressure peak | 7.98273 bar | 7.85628 bar |

Against the exact v9 training replay, late witness RMSE decreases from 6.86166
to 6.58311 K (about 4.1%), but the pressure peak prediction worsens from
6.75084 to 6.36555 bar. The fitted objective includes the whole pressure curves,
not a separate peak-matching penalty. On training C3/ISS retains mean +10.945 K
error; on held-out C3/TIM and C3/ISS have mean -17.804 and -18.181 K errors.
The held-out parameters are not retuned. This is **not an accepted destruction
or cascade calibration** despite numerical optimizer convergence.
The held-out v9 baseline was freshly replayed only after the v10 coefficients
were frozen; v10 improves its temperature RMSE by about 1.3%, not enough to
resolve the substantial per-face errors. The previous held-out pressure peak
was 6.62877 bar, also closer to the measured peak than v10.

The new connection sends 7.83839 kJ to solids in the training replay: 4.92594 kJ
to the first +x surface (TIM2), 2.91245 kJ to the first -x surface (ISS2).
Meanwhile about 200.140 kJ of emitted chemical reserve reacts in the chamber.
The surface connection transfers sensible heat from the discharge; it does not
relocate chamber chemical reaction onto an impingement surface. This is an
explicit model limitation, not proof that it alone explains the residual.
Likewise, actual fracture direction/geometry and internal pressure are not
identified by this chamber-pressure record. No additional per-face or
per-experiment correction was introduced to hide these limitations.

## Geometric opening: fixed-energy regime screen (v12)

This screen uses the **uncapped physical-panel opening** model, not v10's fitted
aperture ceiling. The interrupted v11 local search had reached 7.64449 K on
training after joint steps from a 21.51404 K starting point. Its first nine
native evaluations were finite-difference probes, not nine failed optimization
iterations. Neither that interruption nor this screen establishes convergence.

The new search holds the total net chemical inventory at **385147.711823 J**,
the v11 checkpoint's sum. This is an exploratory prior, **not independently
measured calorimetry**. Each trial redistributes that inventory without adding
another energy source. Reference solid capacities/conductivities, geometry,
intact-heating laws, sensor bindings, shell geometry/work and gas constants
remain fixed. There are no per-face corrections or fitted time shifts.

Five deliberately large perturbations and one joint proposal were evaluated
with full CUDA replays, using only the no-cooling experiment:

| Training trial | Late witness RMSE, K | Pressure RMSE 0–30 s, bar | Sampled pressure peak, bar | Original joint objective |
|---|---:|---:|---:|---:|
| Frozen v11 checkpoint | 7.64449 | 0.42508 | 6.38476 | 13.35772 |
| Gas fraction 0.8 | 7.12100 | 0.75547 | 7.64375 | 18.32994 |
| Gas fraction 0.3 | 9.52383 | 0.74895 | 4.80233 | 24.85302 |
| Gas-to-assembly exposure ×3 | 7.27586 | 0.46694 | 6.17750 | 13.06997 |
| Gas fraction 0.8, exposure ×3 | 7.81078 | 0.57967 | 7.41599 | 16.27189 |
| Solid reaction rates ×5 | 7.63614 | 0.71249 | 7.01460 | 18.56934 |
| Joint proposal, checked in CUDA | **6.55388** | **0.42662** | **6.46460** | **10.93290** |

The joint proposal comes from a bounded residual interpolant of the large
perturbations (quadratic energy split, linear log exposure/rate, and a split–
exposure cross term). Its predicted objective was 10.78833; the **actual native**
objective was 10.93290. The interpolant is only a proposal mechanism, not a
thermal solver, a global optimum certificate or a confidence estimate.
It does not exhaust other combinations of wall transfer, gas yield, rupture
strength or inventory. The selected candidate was frozen before held-out replay.

Selected parameters: solid energy 161777.371770 J, gas chemical energy
223370.340053 J (fraction 0.57996019), exposure scale 5.03995500, tied solid rate
4.04784003/s. The unchanged seed coefficients are gas yield 0.39020842,
wall h=25.43942540 W/(m² K), aperture h=6258.03167444 W/(m² K), shell pressure
strength 996223.51923 Pa and pressure damage rate 0.20165131/s.

| Metric | No-cooling training | Active-cooling held-out |
|---|---:|---:|
| Late witness RMSE, K | 6.55388 | 13.62616 |
| C1/ISS mean signed error, K | -1.07142 | -7.45943 |
| C1/TIM mean signed error, K | -5.42249 | -7.33507 |
| C3/TIM mean signed error, K | -0.80477 | -15.75259 |
| C3/ISS mean signed error, K | +11.31225 | -18.60768 |
| Pressure RMSE 0–30 s, bar | 0.42662 | 0.51681 |
| Pressure RMSE 30–1800 s, bar | 0.05358 | 0.10270 |
| Predicted / observed sampled peak, bar | 6.46460 / 7.98273 | 6.40273 / 7.85628 |
| Predicted / observed peak time on original scans, s | 1.500 / 0.942 | 1.602 / 0.757 |

This recovers approximately v10's fitting accuracy **without its artificial
rupture ceiling**, but does not resolve transferability. The held-out case was
not used to select this candidate (and is not historically blind).
The profile remains explicitly `calibration_accepted: false`; no viewer/default
or previously accepted profile is replaced.

### What the screen diagnoses, and what it does not

Increasing global gas exposure cools C3/ISS but also makes the other surfaces
too cold. Moving energy into gas improves two witness curves and the pressure
peak amplitude but worsens the full early pressure curve. Faster reaction
changes pressure timing without fixing the late spatial temperature residual.
This is evidence of a persistent spatial/temporal mismatch in the tested
parameter subspace, not proof that every possible parameter set fails.

In the selected training trial only **133.94865 J** reaches solids through the
directional aperture links, versus **223354.05285 J** released chemically in
the chamber. The ratio is about 0.060%; these are distinct energy pathways, not
a measured flame efficiency. The six visible channels carry no chemical
reaction at the impingement surface: chemistry is credited to a single
well-mixed chamber compartment. Gas contacts then use that common temperature.
The side-facing channels have no ray-visible solid target in this geometry.
Thus merely raising aperture h cannot relocate the large chamber reaction
energy into a directed burning jet. Local gas heat release/mixing and changed
post-event contacts are hypotheses worth discriminating, **not proven causes**
of the C3/ISS discrepancy. Burst thermocouples remain excluded from fitting;
their large excursions alone cannot validate a bulk battery temperature.

The geometry is still the previously frozen, explicitly unconfirmed acquisition-
numbering hypothesis. No topology or sensor relabeling was introduced here.
The selected run's sampled maximum thermal ledger residual is 6.52e-8 J;
mass and chemical ledgers also close. Conservation does not imply correct
spatial heat distribution. Numerical refinement results are recorded separately
in `upv-geometric-joint-v12/refinement.json`.

### Artifacts and reproduction

- `scripts/scan_upv_rupture_regimes.py`: explicit fixed-inventory design,
  immutable seed, strict control bounds and fingerprinted resumable reports.
- `scripts/propose_upv_rupture_regime.py`: one bounded joint proposal, native
  verification, and a diagnostic freeze before validation.
- `scripts/report_upv_rupture_regimes.py`: signed residual/pressure comparisons.
- `data/calibration/upv-geometric-regimes-v12/`: five full trials and seed.
- `data/calibration/upv-geometric-joint-v12/`: frozen diagnostic candidate,
  training/held-out results, numerical checks and plots.

Use fresh output directories when changing the design or model. Example:

```bash
PYTHONPATH=src python3 scripts/scan_upv_rupture_regimes.py --output NEW_SCREEN
PYTHONPATH=src python3 scripts/propose_upv_rupture_regime.py --screen NEW_SCREEN --output NEW_JOINT
PYTHONPATH=src python3 scripts/calibrate_upv_geometric_rupture.py validate --output NEW_JOINT
PYTHONPATH=src python3 scripts/calibrate_upv_geometric_rupture.py check --output NEW_JOINT
MPLCONFIGDIR=/tmp/battery-sim-mpl PYTHONPATH=src python3 scripts/report_upv_geometric_rupture.py --output NEW_JOINT
```

Disabling the new links and work at the **same frozen v10 coefficients**, without
refitting, gives training RMSE 6.95585 K versus 6.58311 K with them. The new path
has a real effect, but is not sufficient for the observed distribution. This
controlled ablation is distinct from the before/after-v9 comparison, in which
the effective energy/gas coefficients also change.

### V10 numerical checks and remaining warning

Halving dt/CFL and source/gas/damage step limits changes full witness traces by
at most 0.006469 K (training) and 0.006458 K (held-out), and chamber pressure by
0.001951 and 0.001967 bar. A dense 0.01 s output grid through the first 5 s,
then 0.1 s to 30 s, resolves chamber peaks 6.52592 bar at 1.17 s and 6.52665 bar
at 1.18 s. Their coarse/fine pressure differences remain below 0.001624 bar:
the model is not hiding a narrow peak that matches the observed maximum.
Both model versions are rendered on this same dense grid in the short-time
pressure comparison; experimental scan samples are explicitly marked.

The same dense output exposes internal-cell pressure peaks about **433 and
449 bar**, despite chamber peaks around 6.53 bar. These are modeled internal
states, not measured pressures or validated predictions. The fixed v9 opening
ceiling (4.4538 mm²), lumped gas volume and production kinetics therefore need
independent scrutiny; fitting chamber pressure alone does not constrain all of
them. Neither high-pressure gas-law validity nor real pouch geometry/strength
has been established by this fit. Increasing transfer h does not address that
missing constraint. No new parameter selection follows the held-out replay.

All 21 native tests pass sequentially. Five aperture/calibration Python checks
pass; the optional JSON Schema check is skipped because `jsonschema` is absent.
Conservation audits pass on all calibration and verification runs; for example,
the refined dense thermal residual stays below 1.38e-7 J, chemical below 3.1e-9 J,
mass below 1.43e-14 kg. Conservation and time-step checks validate numerical
bookkeeping, **not** the physical adequacy of the assumed gas/fracture model.

Artifacts: [temperatures](../data/calibration/upv-aperture-v10/witness-comparison.png),
[chamber pressure](../data/calibration/upv-aperture-v10/pressure-comparison.png),
[opening transient](../data/calibration/upv-aperture-v10/opening-transient.png),
[energy routing](../data/calibration/upv-aperture-v10/energy-routing.png),
[profile](../data/calibration/upv-aperture-v10/profile.json) and
[full assessment](../data/calibration/upv-aperture-v10/assessment.json).

# Shape-first gas-access geometry experiments (2026-10-21)

These are inverse **effective-geometry hypotheses**, not measured holder
dimensions or a completed calibration. The aim is to remove false flash
heating from protected faces while retaining a sharp event on the open ISS
side. Peak heights, late cooling and validation are separate later stages.

## Frozen reference

The explicit source is
`data/calibration/upv-separated-chemistry-v6/cooling-fit-20261021-r5/trial-000`.
R1/R2/R3, electrical heating, heat capacities, material conductivities, solid
contacts, fracture strength/work/kinetics, chamber cooling and normalized
directed S(T,D) response tables stay fixed. Thermocouple labels and observation
operators stay fixed. No temperature gain, time shift or prescribed ignition
is introduced. Gas volumes and inventories are unchanged; all transport edges
retain their original neighbouring endpoints. No CUDA solver was rebuilt.

The reference itself inherits an unmeasured ISS1 passage-saturation probe and
the explicitly unconfirmed swapped-witness topology. Thus a good shape fit
must not be represented as an experimentally verified assembly geometry.

## Modular geometric controls

`scripts/upv_gas_access.py` produces a copy of that reference:

- Protected slots: TIM1, TIM2, TIM3, ISS3. Open slots: ISS1, ISS2. This is
  spatial access, not different properties assigned to the same material.
- `protected_fraction`: common accessible fraction of perimeter inlet mouths.
  Passage area is reduced. For a rupture backed by the compressed slot, its
  downstream hydraulic access and diffusive A/L are reduced together using
  Cd and effective path resistance. Merely reducing Cd would leave a mixing
  bypass. Actual fracture area and its energy cost remain in the original
  shell operator.
- `lip_open_fraction`: common exposed fraction of existing rim-gallery surface
  patches on protected slots. A holder lip masks these patches. Gref scales
  with exposed area, while h and all temperature/damage response tables stay
  unchanged. Interior slit/solid contacts are not mistaken for rim exposure.
- `backing_open_fraction`: common holder coverage for **all** Al1/Al2/Al3 gas
  exposures, including the uninstrumented ISS4 side of Al3. Solid Al/TIM links
  and aluminium mass/properties are unchanged. Overlapping lip/backing masks
  use the smaller exposed fraction, not their product.
- `witness_edge_open_fraction`: common holder coverage of narrow y/z casing
  faces on both C1 and C3, leaving the broad open ISS faces and C2 outlets.
- `jet_wetted_fraction`: one common gas-wetted fraction on both hot ISS1/ISS2
  slit faces, applied equally to casing and pad. This is a homogenized area
  approximation, not a resolved jet footprint or a new heat-transfer law.
- Optional `plume_exit_fraction`: apertures in existing shared gas faces around
  the C2–C3 corridor. It does not add a shortcut or a mixed hub. Strong 10%
  exit geometry failed the existing pressure convergence check and was
  rejected, **not** rescued by a smaller timestep or looser tolerance. The
  milder 50% experiment did not give a substantial shape improvement.

Coverage is homogenized on existing patches, not resolved as new holder
solids. It is an approximation to occlusion, not a simulation of holder heat
capacity, decomposition or deformation. No claim about a measured physical
window shape or unique inverse solution is made.

## Staged observations and controls

`scripts/probe_upv_gas_access.py` resumes the same committed, sealed 835 s
state and runs to 950 s with the reference observation schedule. An existing
native contract-checked gas-response capture first rebinds the compatible
reference state; conservative identity remapping then applies a geometry
candidate. There is no restart-identity bypass. These are conditional
diagnostics: a selected candidate still needs a fresh run before acceptance.

The neutral geometry control at
`data/calibration/upv-separated-chemistry-v6/gas-access-neutral-control-20261021`
reproduces all 12 reference temperature histories from 835–850 s **bitwise**.
Hence the geometry comparisons are not an artifact of changed restart state.

The regular timestep floor remains 0.02 s, above the user's 0.005 s floor.
The rolling 0.25x and average 0.5x speed checks remain enabled. Completed
115-second reference continuations take approximately 174–179 wall seconds
with accepted-step heat tracing (~0.64–0.66x real time).

`scripts/review_upv_gas_access.py` recomputes a common shape assessment for
explicitly selected completed trials and plots **raw** temperatures. Own-rise
normalization is used only inside the shape diagnostic, never to transform
the plotted or simulated temperatures. Missing scans are not filled across
gaps and cannot erase a model peak from its normalizer. Non-heating solutions
are rejected. Protected-face rapid rise is penalized; open hot-face comparison
concentrates on the front. Cooling after the fitting window does not affect
the objective, which is dimensionless and must not be called temperature RMSE.

Actual accepted gas-film heat records and independent slit balances are saved
alongside the shape comparison. They distinguish incoming hot gas from heat
subsequently arriving through a pad or backing plate.

## Causal results before the final edge-cover comparison

1. Inlet restriction alone (0.1% or 1%) reduces, but does not remove, the
   false impulses. At 0.1%, C1/TIM still peaks around 215 C and C3/TIM at 235 C.
2. Actual heat tracing finds the remaining bypass: C1/TIM casing receives
   +3.459 kJ through exposed rim contacts over 880–900 s while its slit
   transfers -0.266 kJ to that face. Narrowing the inlet again cannot solve
   a direct rim exposure.
3. Adding 1% exposed rim area removes the large impulses: C1/ISS 88 C,
   C1/TIM 65 C, C3/TIM 116 C, with C3/ISS still showing a sharp 1213 C event.
   These heights are outcomes, not fitted amplitude targets. C3/ISS remains
   late and the protected warming curves still need work.
4. The next path is gas -> Al backing -> TIM -> casing. With rim protection,
   at 895 s Al1 is 213 C and feeds TIM1 at about 937 W while the mean C3 core
   is 78 C. Al3 similarly feeds TIM3. Shielding the Al exposure reduces TIM
   overtemperature, but did not initially improve the total normalized shape
   score; a lower peak alone is not grounds to select it.

Every completed trial retains native mass/chemical/thermal residual checks.
No final calibrated profile or validation result is claimed by this series.

## Edge-cover and gas-footprint continuation

`gas-access-shape-20261021-r8/trial-000` completes with protected entrances
0.1%, exposed protected rims 0.1%, Al exposure 1%, lateral witness exposure
10%, and no additional plume collar. Its 835–950 s continuation takes
178.68 s (0.644x); minimum regular step is 0.02 s. Local slit heat closure is
1.45e-11 J, native thermal residual 1.12e-7 J, mass residual 4.69e-14 kg.

For 880–900 s, net incoming transport energy into the C3/ISS slit changes
from 16.98 to 24.52 kJ. C1/TIM changes from 23.92 to 0.151 kJ, C3/TIM from
23.17 to 0.147 kJ. This is actual redistribution, not extra chemical energy.
Large false witness impulses disappear: C1/TIM maximum rise over 1.5 s is
4.40 K instead of 579.39 K; C3/TIM is 5.47 K instead of 568.97 K.
However their remaining warming is still too fast, C3/ISS is about 2.82 s
late, and inherited pre-burst witness offsets are still present. This is
**not** a claim of complete heating-shape agreement.

The stronger shared 0.01% entrance trial r7 failed the existing pressure
convergence check (512 iterations) and is excluded. No timestep below the
allowed floor, solver-tolerance change or conservation exception was used.

The next remaining route is demonstrably solid-mediated. In r8 at 895 s,
Al1 is 165.46 C; ISS2 feeds it at 730 W and it feeds TIM1 at 667 W. Over
880–900 s gas heats TIM1 directly by only 77 J. Changing TIM material Cp/k or
continuing to close the witness gas inlet would not address the upstream
ISS2 -> Al1 path. Trial r9 tests a common 50% hot-face gas-wetted area,
retaining the same directed response and h on both sides of each slit.
This explicitly tests the old full-face exposure assumption without fitting
individual thermocouple gains or changing material properties.

## Saved outcome of this bounded search

The stable selected **shape candidate** remains
`gas-access-shape-20261021-r8/trial-000`. All seven completed candidates were
reviewed with the same current assessment; the dimensionless objective falls
from 101.494 for the reference to 1.696. This is not temperature RMSE.
The raw curves and independent energy closures are saved under
`data/calibration/upv-separated-chemistry-v6/gas-access-shape-20261021-review-r8`.

The common 50% footprint trial r9 was rejected before the main external burst:
the unchanged pressure solve reached 512 iterations with relative residual
1.04013e-6. It has no completed shape result and is not selected. This does
not prove that the physical footprint hypothesis is wrong. It means this
candidate cannot be used under the current numerical controls. The kernel,
tolerance, performance floors and timestep were not changed to rescue it.

Remaining discrepancies are visible, not hidden by the peak tolerance:

- C1/TIM and C3/TIM still warm faster than the measured curves, although the
  former hundreds-of-degrees impulses are gone.
- C3/ISS first peak remains about 2.82 s late. Its subsequent hot tail, and
  especially the C2/ISS tail, are not reproduced yet.
- The inherited pre-burst C1/ISS and C3/TIM offsets (about -16 and -11 K)
  remain. This conditional gas-access study does not claim to have recalibrated
  the preheating from ambient.
- Hot-face footprint, solid-mediated loading of TIM, and actual probe placement
  remain unresolved hypotheses. No independent validation has been run.

No canonical profile was overwritten. The selected scenario is available for
the next stage, but is not marked as fully calibrated or experimentally
validated. Eleven targeted geometry/shape tests, three accepted-heat-budget
tests and ten existing cooling-fit tests pass. The neutral restart check is
bitwise identical over its 835–850 s control interval.

## Subsequent selected geometry and fresh verification

The historical r8 above is no longer the latest shape candidate. The next
series separates TIM entrance restriction from access to the genuine ISS3
pulse: shared TIM access 0.0001, ISS3 access 0.003, protected rim 0.001,
backing exposure 0.01, witness lateral exposure 0.025, open-ISS footprint 1.
ISS4 remains fully exposed. These are unmeasured effective geometry fractions,
not altered material properties or fitted thermocouple gains.

The curve-only score slightly preferred a smoothed-away real C1/ISS pulse.
Both scores are now retained: `shape_objective` is unchanged, while
`shape_with_front_objective` adds the discrepancy in maximum 1.5 s rise divided
by each curve's own rise. All compared candidates are rescored identically.
No observation-time shift or temperature rescaling is applied to raw plots.

`gas-access-refinement-20261021-r9` verifies the selected geometry from
ambient to 950 s using its own sealed 835 s checkpoint. Total wall time is
537.85 s (1.766x real time); the 835–950 s suffix is 182.95 s (0.629x).
The C1/ISS 1.5 s rise is 14.295 K versus measured 12.605 K. C3/ISS front
10–90% is 0.770 s versus 0.818 s, but about 2.5 s late. C3/TIM still peaks
at 97.52 C and has the wrong warming timescale. Pre-event offsets and
cooling mismatches remain. Neither full calibration nor validation is claimed.
The raw fresh-reference review is `gas-access-refinement-20261022-review-reference`.

## Rejected timestep shortcut and retained performance bound (2026-10-22)

`gas-access-coarse-check-20261021-r1` changes ONLY minimum dt 0.02 -> 0.04 s
from the same sealed r9 state. It finishes in 121.83 s (1.502x suffix speedup),
but fails the 5% numerical-trajectory budget: time-weighted normalized RMS is
6.69% C2/ISS, 9.02% C3/ISS and 8.69% chamber pressure. Some local gas-energy
and deposited-heat histories also fail. This is comparison with the fine
reference, NOT a claim of 7–9% error against the experiment. A roughly one
DAQ-sample phase change causes much larger pointwise front differences; those
are explicitly reported. Conservation was never relaxed.

The user rejected this coarsening. Keep the regular floor 0.02 s, the original
pressure residual acceptance 1e-6, all mixing tolerances and mass/energy checks.
Average runtime must not exceed twice simulated time (speed >=0.5x); the
existing 20 s rolling guard is retained. The existing fine reference already
meets the bound, so a speed requirement is not a reason to distort its dynamics.

`refine_upv_gas_access.py --reuse-seed-geometry --seed ...r9` reconstructs the
entire saved prefix from the frozen physical source before reusing it. Native
fingerprint, source identity and conservative remap checks remain mandatory.
Changed-geometry continuations are explicitly conditional until freshly
verified; unchanged geometry can be identified separately. An explicit small
`--jet-wetted-fractions` grid tests only homogenized exposed area, not a new
heat-transfer law, dt, chemistry or bulk material k/Cp. This avoids repeating
the 835 s warmup while comparing the same pre-event state.

The 0.9/0.8 common-footprint tests both finish above the 0.5x performance
floor (0.562/0.595x), but do not beat the retained fresh-reference curve+front
score: 2.23307/2.23845 versus 2.22395. At 0.9 the gas heat into ISS2 drops
only 1.2%, with C3/TIM peak changing 97.52 -> 96.61 C. Reducing two opposing
wall areas together allows hotter gas to compensate much of the restriction.

An additional optional `open_iss_pad_wetted_fraction` tests one common
pad-side footprint on BOTH open ISS1/ISS2 slots without reducing their facing
casing exposure. It scales the existing area-extensive conductances only;
all normalized S response tables, solid contacts, Cp/k, gas paths/inventories,
rupture laws and dt remain frozen. This is an **unmeasured inverse hypothesis**
of unequal plume footprints across a slit, not evidence of a particular holder.
At 1 it reproduces the earlier scenario exactly (including metadata). New
tests cover the neutral identity, both-pad symmetry, unchanged casing/ISS3
exposure and frozen transport/solid physics. A changed footprint still needs
fresh verification before any acceptance.

## Full verification of the refined footprint (2026-10-22)

Four local suffix trials were completed without timestep coarsening; all run
at 0.562–0.645x real time. The tested pad-side footprint 0.5 plus ISS3 mouth
0.0045 is then independently verified from ambient in
`gas-access-refinement-20261022-r4-fresh`: 950 s / 537.606 wall s (1.767x),
and 835–950 s / 178.387 wall s (0.6447x, only 1.55x slower than real time).
The 0.02 s floor, original numerical residual tolerances and strict ledgers
are unchanged. All 12 temperatures join bitwise at the 835 s checkpoint.
Preheating versus the prior full reference differs by <=0.0511 K on the six
main observation channels (not a comparison against experimental error).

The fresh C3/TIM maximum is 93.15 C instead of 97.52 C, with its heating
timescale still too fast. C3/ISS front 10–90% is 0.7975 s versus measured
0.8175 s; its midpoint remains 2.017 s late. C1/ISS has a clear pulse, but
its 1.5 s rise is 21.05 K versus measured 12.60 K. Curve+front loss improves
2.22395 -> 2.12781, while the original curve-only loss worsens
1.70837 -> 1.78945. Keep both diagnostics; do not claim a lower temperature
RMSE, agreement of every shape or completed calibration. Source cooling and
pre-event offsets remain wrong. The inverse exposed areas are still unmeasured.

`gas-access-refinement-20261022-review-fresh/shape-comparison.png` shows both
full references and experiment without gains or fitted shifts. The review
also independently closes accepted heat and gas-energy balances in each slit.
The old canonical parameters are not overwritten, and validation is not used
for selection. The new scenario is only a fully replayed candidate for the
next stage. Targeted checks: 48 Python tests and two native tests pass.

## Careful local refinement and numerical work ceiling

The next bounded series preserves the r8 incumbent, all materials, solid
contacts, chemistry and rupture physics. r10 changes only the common open-ISS
wetted fraction from 1 to 0.9. It completes at 0.630x for the suffix; the shape
objective changes only 1.69566 -> 1.69374.
Several channels worsen, so this tiny aggregate change is not sufficient
evidence to promote the candidate.

Another real solid-mediated bypass is gas -> exposed ISS4 rim -> Al3 -> TIM3.
ISS4 has no open-slit mouths, but its rim had remained fully gas-exposed. Its
coverage is now an explicit OUTER-holder geometry hypothesis, separate from
the compressed internal slot. It scales extensive contact areas, not ISS Cp/k,
surface response, or thermocouple gains. Overlapping Al/rim masks use their
minimum exposure rather than multiplying the same cover twice.

On the old numerical budget, r11 (outer opening 0.001), r12 (0.5), r13 (0.9)
and r14 (witness-edge opening 0.05 instead of 0.1) all stop at the existing
512-iteration pressure limit before their full heating curves are obtained.
They are numerically unresolved, not evidence of invalid physical geometry.
No failed trajectory is ranked as a successful fit.

A separate build `/tmp/battery-sim-pressure-budget-20261021` permits an
explicit CG work budget up to 2048 instead of 512. The DEFAULT remains 512;
the CUDA transport implementation, residual acceptance 1e-6, time integration,
materials, and minimum regular dt=0.02 s are unchanged. Both parser and mesh
validation use one shared maximum. Old binaries/checkpoints are preserved.
No checkpoint is migrated across the binary fingerprint.

`scripts/refine_upv_gas_access.py` starts a fresh r8 control from ambient,
saves its sealed 835 s state under the new executable, and then compares
bounded geometry continuations. Only the unchanged r8 continuation is a
fresh full 0–950 s trajectory; other geometries remain conditional until
their own fresh-start check. Temperature-divergence pruning has a fourfold
but finite relaxation at the uncertain peak. Its guards and 10/50/90% front
diagnostics are tested; missing scans cannot be bridged by a front estimate.

The first fresh launch r1 was rejected by the second, mesh-side 512 guard
before any physical step. That guard has been synchronized and tested. The
retry is `gas-access-refinement-20261021-r2`.

### Fresh control and first local results

The r2 r8-control prefix takes 343.35 wall seconds for 835 model seconds.
Together with its completed continuation this is a fresh 0–950 s control
in 522.73 s, or 1.817x. Prefix end and suffix start are bitwise equal for
all 12 temperature histories. Its shape objective is 1.72140. It is **not**
interchangeable with the earlier conditional r8: at 835 s C2/ISS is 8.624 K
colder and C3/TIM is 1.762 K colder than that old checkpoint. The earlier
checkpoint came from a different hot-gas-response prefix. Never compare
the two series as if only one newly fitted geometry coefficient changed.

With this same fresh prefix, the ISS4 opening reductions do NOT improve
shape: 50% gives 1.76349 and 25% gives 1.81213. At 50%, actual gas heat to
ISS4 over 880–900 s falls from 4369.7 to 2455.5 J; C1/TIM peak in the fit
window falls from 45.16 to 42.13 C, but its normalized 10–90% rise remains
too fast (12.57 -> 13.33 s, experiment 34.22 s). A cooler curve is not an
improved shape, and this geometry change is not selected.

Rim exposure of the narrow C1/C3 edges is then reduced with ISS4 restored.
5% gives shape 1.70835 and 2.5% gives 1.70641. The latter changes C1/TIM's
largest 1.5 s rise from 3.888 to 3.390 K (experiment 1.029 K), so the gain
is still modest. The series next isolates the TIM slit entrance from the
ISS3 entrance; both are mass-access parameters, not thermocouple gains.
ISS3 can carry the small real C1/ISS impulse and should not automatically
be closed along with TIM. Every such conditional continuation still needs
its own ambient-start check before acceptance.

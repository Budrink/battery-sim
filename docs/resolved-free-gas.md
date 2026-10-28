# Spatial free gas and autonomous external heating

The FSRI 30-cell test charges **no cell during the run**. All cells begin charged
and electrically isolated. One cell is heated externally at 6 K/min until thermal
runaway. The separate nitrogen pressure-vessel characterization is not its room.
Primary sources and the transferred, unvalidated R2 hypothesis are documented in
[fsri30-source-reconstruction.md](fsri30-source-reconstruction.md).

## Separate reusable laws from apparatus geometry

`fsri30_resolved_gas.py` is an apparatus adapter, not a battery material law.
`free_gas_mesh.py` is a geometry-only Cartesian mesher around an envelope. Neither
fits temperatures. All mesh levels retain the same gas volume, mass, oxygen,
chemical reserves, material capacities, internal core mesh and can rupture laws.

External air previously mixed the outgoing fuel into one approximately 1 m³
reservoir. It is now divided spatially, with fine volumes near the enclosure and
coarse far-field volumes. No adjacency crosses the solid envelope. Gas escapes
only through the two reconstructed apertures. ABS panels receive local inner and
outer gas contacts on their own thermal voxels, rather than a full-panel thermal
shortcut. Ordinary thermal contacts still use directed connectivity; pressure
transport and conservative counterexchange remain separate operations.

The current refinement ladder resolves **external air and ABS thermal contacts**.
It does not yet refine each internal Voronoi body/top/base region, or implement a
cylindrical thermal mesher. Those remain explicit independent discretizations.

For distributed open-gas passages:

    resistance_factor = path_length / reference_length
    mass_flow = existing_orifice_flow / sqrt(resistance_factor)

The new dimensionless channel property defaults to 1. Its purpose is to keep
quadratic path resistance additive by length under subdivision. Physical area
remains unchanged for thermal/composition exchange. The .05 m reference length
is an **unmeasured free-space resistance prior**, not a reported experimental
dimension. Physical outlet orifices retain factor 1. Grid-independent steady
resistance does not prove the entire transient grid-independent: test it.

Optional unresolved free-gas stirring uses:

    ell = min(channel_length, mixing_length_cap)
    D_eff = D_molecular + C * ell * sqrt(g * ell * abs(Ta-Tb)/max(Ta,Tb))

`C=0` exactly retains the previous model. The development experiment uses .05,
with a .05 m cap. This is a **constitutive prior**, not calibrated physics or
resolved CFD/buoyant momentum. Internal molecular pores are excluded. The same
conservative exchange packet carries energy, fuel, oxygen and vapor: stirring
neither creates heat nor supplies infinite oxygen, and closed barriers still
block it. It does not make the two directed solid/gas conductances equal.

## Heater

`HeatingProtocol` drives a fixed passive resistor through the existing I²R/UI
ledger. A bounded feedforward + proportional controller follows an apparatus
temperature ramp. Power cannot become negative, so it cannot cool the cell or
clamp its reaction temperature. It latches off when its sensor exceeds 500 °C
while rising faster than 10 K/s (reported operational runaway criterion).

Ramp and cutoff criterion come from the source. Sensor position, proportional
gain, feedforward and maximum power are apparatus priors. The current sensor is
the heated can's -x panel reference-capacity mean, not a claimed point TC.
There is no experimental-time ignition or scheduled heater cutoff. Controller
state is checkpointed and committed only with an accepted simulation step.

## Numerical and spatial checks

The averaged pressure solver checks the direct `b-A*x` residual in pressure-
correction coordinates before accepting convergence. It restarts the search
direction only if apparent recursive convergence fails that check. Fixed
128-iteration restarts were removed: they prevented convergence on the graded
air mesh. The local relative threshold remains 1e-6. This does not relax the
physical law or take smaller physical steps. An independent dense long-double
reference tests the actual CUDA operator on uniform and heterogeneous graphs.

Run `check_fsri30_resolved_gas.py` with a pinned probe. Default levels 40/20/10 mm
contain 396/1590/9400 external volumes. Adjacent levels must agree within 5% on
per-component temperature RMS in Kelvin, gas R3 energy and its cumulative shape,
hottest external-gas waveform, conversion order/count/timing. The interval must
actually exercise R2 and R3; an inert cold smoke test cannot qualify convergence.
Comparison is unshifted in time. The .01 s physical floor and .2× realtime guard
remain fixed. Failure stops the ladder instead of assigning a numerical failure
a physical score. A finite-horizon mesh check is not full-cascade convergence
or independent experimental validation.

## Field output for native 3D visualization

The casing observer can export `gas_frames_path` at its existing observations.
This does not introduce solver time boundaries. `BATTGAS1` is little-endian:
8-byte magic, u64 gas-node count, then frames containing f64 time followed by
N triples of f32 temperature [K], chemical energy [J], oxygen [kg]. IDs and frame
count are in the result `gas_fields`; external boxes are in scenario metadata
`gas_volume_geometry`. Read-only `gas_history_groups` retain global region sums
without multi-gigabyte JSON histories. This is field export, **not a completed
new viewer**. The target viewer remains the native ImGui application.

Remaining limitations include finite air without renewal, radiation, ABS
pyrolysis/combustion, unresolved can contacts and exact sensor locations. None
is repaired by inflating the chemical reserve or claiming more measured data.

# Grid-independent circulation and recycled pressure patterns (2026-10-25)

The completed 40/20/10 mm ladder on the retained-conjugacy core did **not**
establish spatial convergence. R3 total changed 10.09%, then 5.89%; the hottest
external voxel history changed 6.74%, then 7.51%. Battery-mean temperatures
were much closer, but are not sufficient to qualify a reactive gas mesh.
The 10 mm run advanced 1800 s in 3682 s (0.489x real time).

An additional constitutive confound was found: the unresolved stirring prior
used the numerical edge length as its eddy size. On a smooth linear temperature
field its extra diffusivity vanished as h². Refining therefore changed the
model itself. Version 2 uses an explicitly declared physical stirring length
(20 mm exploratory prior, NOT a measured material value), a resolved thermal
gradient and a bounded contrast:

    ell = min(thermal_stirring_length_m, mixing_length_cap_m)
    contrast = min(1, ell * abs(Ta-Tb) / (distance * mean(Ta,Tb)))
    D = D_molecular + C * ell * sqrt(g * ell * contrast)

The same linear field now has the same diffusivity on 40/20/10/5 mm meshes.
Internal molecular pores, directional connectivity, reactions, inventories,
heat capacities and physical opening laws are unchanged. Positive stirring
in JSON now requires the explicit physical length: old unqualified stirring
scenes are not silently interpreted as this revised law.

`pressure_pattern_predictor` is a separate, optional numerical accelerator.
It learns a single spatial correction from the previous accepted pressure
window within one GPU launch and fits its amplitude to the next matrix/rhs
by an energy projection. It is not a neural network or a replacement physical
flux. A worse local residual discards the guess. CG and its direct b-A*x
acceptance at 1e-6 remain mandatory. The cache resets on every launch, including
transaction retries and restored checkpoints. No speculative state is committed.
The test repeats an actual long-path pressure window, then deliberately supplies
a useless pattern to verify the unchanged fallback solver.

Fresh coupled spatial trajectories and pattern-on/off timings must be checked
before promoting either candidate. These changes alone do not prove a cascade,
spatial convergence, or agreement with the experiment.

The sequential controller is `scripts/check_fsri30_grid_compromise.py`. It
pins the physical profile, core and geometry-code hashes and fixes ABS spacing
to 10 mm throughout. It first compares predictor OFF/ON on the same 40 mm gas
grid: every trajectory metric must agree within 0.1%, and measured full-run
speed must improve by more than 2%. Otherwise subsequent meshes run OFF.
It then compares uniform 20/10 mm and aperture-focused 40/10 and 20/5 mm grids.
The latter pairs mean far-field/local spacing, not cell thermal resolution.
A converged uniform 20/10 reference pair is mandatory before choosing a local
mesh. A failed optional predictor preserves its evidence and falls back OFF;
a failed physical baseline cannot qualify. Solver pattern counters describe
the last GPU launch, not cumulative accepted patterns over the whole run.

Current results, comparisons and charts are written under
`data/development/fsri30-grid-pattern-20261025-r1/comparison-r3`.
First fresh baseline completed 1800 s in 575.511 s (3.128xRT), R3 64955.525 J,
one converted cell, minimum step 0.01 s. Thermal/chemical errors remained below
2.14e-7 J, mass/oxygen below 7.10e-14 kg. This is a successful baseline replay,
NOT yet a converged spatial mesh or successful cascade reproduction.

The predictor replay has now FINISHED: 566.434 s vs575.511 s baseline, only
1.60% faster. Battery RMS differs0.211%K and R3 total1.631%; worse, identical
node/time gas fields differ by up to357.254 K (worst local relative22.882%).
Fuel-field L1 difference reaches13.36%, despite close global maxima. It FAILS
qualification and stays OFF for the remaining grids. This does not demonstrate
an energy leak: both runs passed strict ledgers. It demonstrates insufficient
coupled trajectory equivalence under the current pressure solve acceptance.
The mechanism of this sensitivity is not yet isolated.

The controller now checks the complete same-mesh gas temperature/composition
fields, not only aggregate curves. A regression deliberately changes a cold
node while preserving the global maximum and verifies rejection. Both full
40mm trajectories are preserved and reused only on their exact matching core,
profile and grid. The queue has advanced to fresh uniform20mm, predictor OFF.
Spatial convergence and full5945s continuation remain pending.

## Full 20/10 mm check

The completed1800s pilot pair PASSED: component RMS0.425%K, R3 total0.216%,
R3 cumulative shape0.545%, external hottest-gas RMS1.674%; conversion count
and order match. Uniform20 took860.907s, uniform10 3845.158s (4.466x cost).
This selects20mm as a candidate, not a fully qualified mesh or calibration.

On the user's request, a fresh sequential0..5945s20/10 pair is now running at
`data/development/fsri30-full20-10-20261025-r1/checks` on the exact same native
binary and physical profile, with predictor OFF. The remaining focused20/5
diagnostic was stopped gracefully and its partial evidence retained to free
the GPU. No warmup checkpoint or previous score is substituted for either
full trajectory.

`scripts/report_fsri30_full_mesh.py` independently waits for completion and
writes `verification.json` at that run root. It checks provenance, unchanged
physical scenes, repetition of the1800s pilot prefixes, global convergence
and separate impulse/cooling/fixed300s windows. All phase temperatures use
Kelvin and5%; the report never shifts time to align peaks. It explicitly reports
the computed R2 event count without inventing an experimental cascade count.
A numerical pass does not assert experimental validation or realistic cascading.

### Available-prefix result

Full20mm completed5945s in3097.172s (1.919xRT), with model R2 conversion
propagating through all30 cells by2371.435s. Full10mm hit the rolling0.2xRT
guard near2243s; completed snapshots extend through2240s. This stopped run
still supports a common-prefix comparison; it must not be discarded just
because the full requested horizon was not reached.

Run `report_fsri30_full_mesh.py --run RUN --available-prefix --output NEW_DIR`
to produce it. The regular full-pair auditor now attempts the same diagnostic
automatically after an aborted replay. No live-state extrapolation or time
alignment is applied. Existing failed reports are not rewritten.

The saved common-prefix report shows good0..2200s agreement: component RMS
0.493%K, hottest gas1.755%K, total R3 difference0.225%. First-event phase also
passes5%. A small late afterburn increment differs6.50% (64.9J), and the
2200..2240s cascade-onset window does NOT pass: coarse B09 has already reacted,
fine B09 is just below the10% reporting threshold at the cutoff. A separate
live snapshot at2240.97s shows B09 local Tmax1631K. This supports a delayed
fine-grid event, not an absent cascade; it does not supply a precise10%-mass
event time or the missing later fine-grid history.

Graphs and complete phase metrics are under
`data/development/fsri30-full20-10-20261025-r1/common-prefix-20261025-r1`.

### Slow reference: explicit performance waiver

The user has authorized finishing10mm irrespective of speed. The dedicated
`scripts/complete_fsri30_fine_reference.py` replays the exact archived10mm
scene/core to5945s without a speed floor or wall deadline. This exemption is
local to the reference; it does not change solver tolerances or production
guards. Output: `data/development/fsri30-unrestricted10-20261025-r1`.

The old JSON prefix is not a restart state, so warmup is recomputed. A real
committed checkpoint is captured at the existing2200s observation without
stopping the trajectory. Completed20mm evidence is reused unchanged.
After completion the controller automatically compares the full pair and
spatial propagation patterns. Fuel/O2 fields are restricted by physical box
overlap with their extensive totals conserved. Volume-weighted gas temperature
is only a field diagnostic, not an energy reconstruction. Event-relative
shape analysis and a held-out arrival-law hypothesis cannot time-shift the
unshifted convergence tests or modify the running physics.

Current boundary is still finite CLOSED air, unlike the open-air experiment.
It stays frozen to isolate mesh convergence. Experimental cooling/boundary
correction and any learned conservative flux closure are separate subsequent
checks; no pattern has yet been activated.

### Cascade-only stop and higher-order candidate

The user subsequently authorized stopping after the cascade. The10mm reference
is now stopped with observations through2525s, a separate live state2526s,
all30 reported R2-conversion events and a complete2200s committed checkpoint.
Full5945s cooling was NOT completed on10mm. New cascade-only diagnostics and
plots are under `cascade-analysis-20261025-r1` inside that reference folder.

Most battery response difference is onset timing: B09 is~10s later on10mm,
but secondary-to-last propagation spans140.43s vs141.33s. All30 recorded peaks
are within3.93%K, and own-event aligned post-jump RMS is below.61%K. That
alignment does not replace the failing unshifted cascade acceptance check.

Mean+3 directional spatial slopes can represent>99% of resolved subcell field
variance in an a-priori fit, but a frozen R3 source probe still misses its
nonlinear response. The candidate is coarse pressure with conservative moment
transport and local reactive enrichment, not a promoted learned transport law.
Neither moment evolution nor its performance/energy closure is implemented or
validated by the field-compression diagnostic.

## Tested nested refinement, 2026-10-25

Local aperture refinement now preserves all coarse Cartesian planes and divides
selected cells, instead of re-anchoring the mesh at the refinement-window ends.
Equal bulk/near spacing is exact identity. On the20/10mm pair every local box
is an exact union of full10mm reference children; volume and surface-area checks
remain active. This is a reusable GEOMETRY fix, not a learned material law.

The complete fresh0–2525s test with20mm bulk/10mm jets has4048gas nodes and
cost1788.97wall seconds (1.41xRT). All30 model conversion events occur; ledgers
and.01s floor pass. However maximum event difference from full10mm increases
from9.99s to28.43s; mean difference6.05s→18.72s. Better first-burst battery RMS
(.55%→.23%K) does not improve the whole cascade. Do NOT select this mesh as a
better FSRI dynamics profile. Keep the original20mm development compromise.
Strict5% unshifted cascade convergence is still not achieved.

A separate stateless spatial R3 quadrature also failed (14.7% first-burst heat
error). Its prototype code was removed from the working kernel after archival.
No intrinsic chemistry, materials, thermal contacts or boundary were refitted.
See `data/development/fsri30-local-refinement-20261025-r2/decision.json`, its
`comparison.png` and the development journal for provenance and qualifications.


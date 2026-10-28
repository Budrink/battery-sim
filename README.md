# battery-sim

CUDA thermal-runaway simulator with a native ImGui/OpenGL viewer.

Current model:

- continuous R1 aging → R2 exothermic decomposition → oxygen-limited R3 oxidation, with its own Arrhenius law;
- CUDA Kirchhoff current redistribution and local Joule heating, with exact integration of prescribed current profiles;
- one reduced electrical-conductivity coefficient for all condensed decomposition products;
- charge-dependent effective R; ALL charging precursor is paid from measured UI after I²R; U·I is not heat;
- separate healthy charging and heat-neutral R1 degradation; only R1 accumulates defects and reduces healthy charge capacity;
- 3³ internal battery volumes, physically distinct casing, thermal pads and insulation;
- instantaneous directed tensor thermal connectivity, without an independent material l;
- pressure-driven gas mass/enthalpy transfer, finite gas/fuel/oxygen inventories and deterministic progressive casing aperture;
- separate thermal, chemical, electrical and mass ledgers on CUDA.

The old chemistry and old demo scenarios are retired. Legacy calibration results
and the recovery archive were permanently deleted. The sole current demo,
[reaction-chain-demo.json](examples/reaction-chain-demo.json), is synthetic and
**not experimentally calibrated**.

## Build and view

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/battery_sim_viewer examples/reaction-chain-demo.json
```

`./run_viewer.sh` builds and opens this same scenario. CUDA and a working native
OpenGL context are required; no browser frontend is used.

### Cascade presentation

`./run_cascade_viewer.sh` opens the saved FSRI 30-cell trial 23 near the
collective burst. This is **recorded CUDA output**, not a fresh simulation or
independent validation. Space plays/pauses; drag rotates, wheel zooms; the
timeline seeks in either direction. The left map selects a battery.

The native viewer has separate temperature/damage, volumetric hot gas, shell
opening, gas motion, damage hatching and enclosure layers. Temperature runs
from blue through yellow to saturated red. Purple hatching marks damage;
spent cells have a purple outline (and purple fill in the damage layer),
so destruction is not confused with cooling. Battery density and gas opacity
are independent display controls. Distant, coarse chamber gas is hidden by
default so it does not obscure the module; it can be enabled under display
settings. The underlying geometry and physics are unchanged. The FSRI model
uses equal-volume prismatic surrogates, which the viewer labels explicitly.

The archived recording retains 5-second gas-volume samples, solid **mean**
temperatures and exact event times. The new full-flow observer records every
channel, and observes committed substeps about every 0.5 s during the cascade,
without changing the solver's original integration boundaries. Time and gas
temperature interpolation are display-only. Gas gradients are bounded by
connected neighbouring volumes, clipped to free gas geometry and do not cross
closed barriers. A cloud texture and soft moving parcels make redistribution
visible. Their phase follows cumulative **accepted** mass transfer, normalized
by a fixed initial inventory per link (5–500 mg per display cycle); this is not
a CFD velocity field, resolved turbulence or a
measured flame front. Recorded motion uses accepted interval mass / duration,
not interpolated instantaneous rates that could disagree with net transport.
Pause/seek is deterministic. Unknown channels in the
old recording do not invent flow; recorded coverage is shown in the UI.
Live mode displays the actual
CUDA spatial thermal field, including on the dense battery faces. Cutaway
switches back to the optical volume. Cyan marks opened shell boundaries;
hatching indicates damage, not a prediction of individual cracks. The launcher
prefers a completed full-flow recording when present, otherwise the archived
trajectory with limited flow coverage. Partial recordings can be inspected
explicitly, but the launcher never silently substitutes one for a full cascade.

```sh
python3 scripts/export_visual_replay.py \
  data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023 \
  data/visualization/fsri30-best-flow-display
./build/battery_sim_viewer --replay data/visualization/fsri30-best-flow-display/manifest.json --at 2480
./build/battery_sim_viewer examples/reaction-chain-demo.json
# Optional same-physics observer, separate from playback/calibration:
./build/battery_sim_record_visual path/to/original-scenario.json new-recording-directory 3000
```

Replay packages are read-only presentation data, never solver restart states.
Archived export hashes are retained in its manifest. The native observer keeps
source path, sampling policy and completeness in its manifest. The parser
checks IDs, frame sizes, finite values and monotone time. Source provenance is
not a claim of experimental agreement.

## Tests

```sh
ctest --test-dir build -R 'cuda_(joule|reaction_chain|transaction|directed_geometry)' --output-on-failure
PYTHONPATH=src python3 -m unittest discover -s tests -p test_r123_calibration.py -v
```

## Fresh UPV calibration

The original instrument CSV files and ZIP remain in `data/experiments/upv/`.
The current workflow explicitly migrates the LAST full profile into a versioned
uncalibrated seed, never silently fills missing fitted parameters from defaults.
It does not prescribe C2's measured temperature or reset chemistry at the explosion.

```sh
python3 scripts/migrate_upv_chemistry_profile.py --output data/calibration/upv-separated-chemistry-v6/my-seed.json
python3 scripts/fit_upv_kinetics.py --seed data/calibration/upv-separated-chemistry-v6/my-seed.json \
  --output data/calibration/upv-separated-chemistry-v6/my-trials \
  --fit-electrical --fit-charge-formation --duration 626.956
```

All physical time integration is CUDA. Python constructs inputs and proposes
parameters. Only the no-cooling experiment is used for fitting; active cooling
is a held-out replay. Measured C/k and fixed material geometry are not fit
parameters. Resistance and effective chemical kinetics remain unidentified
until this calibration is completed; the electric-log synchronization and
casing/gap/wall proxies are declared assumptions, not measured specifications.
Separated-charge/R1 results use `data/calibration/upv-separated-chemistry-v6/`.
Older RMSE values are not transferred to the revised physics. Warmup-only fits
are not explosion calibration. Freeze parameters before held-out validation.

Historical artifacts are provenance, not executable current profiles. The older
calibration module still provides replay/metric utilities; the active bounded
search is `fit_upv_kinetics.py`, which requires a matching full current-schema seed.

## FSRI thirty-component cascade search

```sh
python3 scripts/fsri30_conditioned_start.py \
  --output data/development/fsri30-conditioned-start-20261024-r1
python3 scripts/search_fsri30_suffix.py \
  --output data/calibration/ul-fsri-30-conditioned-my-search \
  --trials 24 --budget-hours 6
python3 -m unittest discover -s tests -p 'test_fsri30*.py' -v
```

The active diagnostic now starts every transport candidate from the **same
committed state at 1755 s**. The earlier full-from-zero geometry screen changed
cold heat losses, sometimes suppressing the initiating reaction entirely; this
did not isolate post-burst transport. First micro-venting occurs near731s, so
this is explicitly an **already-vented** restart, not a sealed state. Existing
apertures, damage, pressure, composition, chemical energy and history are kept.
No temperature or ignition event is imposed. A same-state restart reproduces
the archived burst through1800s bitwise (6350fields plus all30 mean temperatures).

The diagnostic probe allows only fixed-volume gas contact/passage changes;
chemistry, heater, shell mechanics, solids, inventories and numerical controls
are frozen and checked natively. Upper-sleeve volume repartition is excluded
from these suffix trials: remapping an already-vented mixture silently would
not be the same initial condition. Full gas-energy sums and accepted gas-film
heat are recorded, rather than inferred from sparse temperature samples.
These are **conditioned transport experiments, NOT fresh full-run calibration
or independent validation**. Any useful layout must subsequently be checked
from time zero with a consistent heater boundary. Older geometry-screen and
joint-chemistry drivers remain for reproducibility, not the active strategy.
Details: [conditioned transport](docs/fsri30-conditioned-transport.md).

The earlier r5 from-zero screen targeted neighbour energy delivery, not the
initiating cell's reaction clock. Five full-5945-s sensitivity tests isolated shared facing
gas cavities, upper-cavity shell exposure, local residence time, hot directed
contact response, and their combination. Chemistry (including R3), heater,
shell strength, solid materials and numerical controls are frozen. Exposed
areas, free volume, initial air/O2 and energy are conserved. Upper/body exposure
uses disjoint rectangles on the actual outer steel; no external plume-to-core
shortcut through ABS is introduced. The hot contact range is normalized to
retain the original cold conductance in both directions.

In that earlier driver, only after a meaningful neighbour-heating/conversion
response does constrained GP search start over six shared transport parameters.
If the isolated tests show no response, status is `needs-model-review`, rather than spending hours
repairing source timing. This gate and the smooth neighbour-preheat objective
are declared **search heuristics**, not measured ignition thresholds. The full
experimental envelope/cooling loss remains part of the objective. Source data
mapping and prior R3 numerical sensitivity are still unresolved: results are
provisional, never automatically called calibrated/validated. Existing CUDA
core and all earlier runs remain intact; use a new output directory.

The earlier joint-chemistry driver `calibrate_fsri30_bayes.py` remains available
for reproducing its archived search, but is not the active cascade strategy.
It runs full 5945 s CUDA trajectories, not a warmup-only fit. It first checks
time/near-field resolution, then uses constrained Gaussian-process expected
improvement over nine shared bounded parameters. `--probe` and `--base` select
explicit immutable native/input artifacts. Rerun the same command/output to
resume; incompatible core, source, dataset or search-space hashes are rejected.
`status.json`, `journal.jsonl`, `observations.json` and `best.json` record actual
progress. SIGTERM terminates the active native child and preserves its prefix.
Numerical sensitivity above 5% triggers a bounded refinement ladder BEFORE
optimization; if it cannot converge, the search stops instead of fitting
artifacts. One checked regime is frozen for all physical proposals. An explicit
`--preflight-cache OLD_SEARCH` can reuse only matching complete control runs.
Repeat that option to reuse exact controls from multiple preserved searches.
Every time/space control is checked against schema/native iteration limits
before long launches; gas intervals do not request sub-floor refinement.
An additional declared prior prevents fitting a hot fire by rapid R3 oxidation
of initially cold external gas (298.15 K rate must stay below 1e-6/s).

For explicitly provisional development, `--continue-unverified-search` can
continue parameter exploration after an unresolved R3 branch check, provided
full controls meet the bounded temperature/coupled-heat/timing criteria. It
does NOT mark the failed strict check as passed. Candidates creating a cascade
or improving loss by 20% automatically get time/space checks; a failed candidate
or plot does not kill the worker. Use e.g. `--initial-design 6 --trials 48
--budget-hours 12` to reproduce that earlier joint search. Strict acceptance and
unknown observation geometry remain separate from search permission.

This is currently an **exploratory development search, not validation**: the
nine published thermocouple positions and calorimeter boundary are unresolved.
The loss uses temperature envelopes; it cannot establish individual measured
failure order/count. Chemical inventory, capacities and rupture laws are not
fit variables. No scheduled secondary ignitions, time warping or calorimeter
energy inflation are allowed. A numerically verified candidate still requires
identified observation geometry and an independent experiment. The native
viewer is unchanged by this search.

## Model notes

- [R1/R2/R3 chemistry and connectivity](docs/reaction-chain.md)
- [Current model revision: charging versus degradation](docs/model-revision-v6.md)
- [Joule heating and calibration protocol](docs/joule-heating.md)
- [Continuous gas combustion](docs/continuous-gas-combustion.md)
- [Spatial gas geometry and declared clearance priors](docs/spatial-gas-geometry.md)
- [Committed pre-rupture restart and conservative gas-map rebase](docs/restart-checkpoints.md)
- [Gas-to-casing/pad heat-transfer audit](docs/gas-surface-heat-audit.md)
- [Shared directed connectivity for all UPV thermal contacts](docs/thermal-connectivity.md)
- [Coupled gas integration and hard 0.5x realtime acceptance](docs/coupled-gas-integrator.md)
- [Transactional CUDA updates](docs/transactional-cuda-step.md)
- [Retired artifacts](docs/retired-scenarios.md)

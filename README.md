# battery-sim

CUDA thermal-runaway simulator with a native ImGui/OpenGL viewer.

Current model:

- continuous R1 aging → R2 exothermic decomposition → oxygen-limited R3 gas combustion;
- independent Joule–Lenz heating: I²R(T,D), with exact integration of prescribed current profiles;
- explicit overcharge acceleration of heat-neutral R1; normal charging/discharging is not implicitly abuse;
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

## Tests

```sh
ctest --test-dir build -R 'cuda_(joule|reaction_chain|transaction|directed_geometry)' --output-on-failure
PYTHONPATH=src python3 -m unittest discover -s tests -p test_r123_calibration.py -v
```

## Fresh UPV calibration

The original instrument CSV files and ZIP remain in `data/experiments/upv/`.
The current workflow does not read old profiles, does not prescribe C2's
measured temperature, and does not reset chemistry at the measured explosion.

```sh
python3 scripts/calibrate_upv_r123.py heat
python3 scripts/calibrate_upv_r123.py failure
python3 scripts/calibrate_upv_r123.py validate
```

All physical time integration is CUDA. Python constructs inputs and proposes
parameters. Only the no-cooling experiment is used for fitting; active cooling
is a held-out replay. Measured C/k and fixed material geometry are not fit
parameters. Resistance and effective chemical kinetics remain unidentified
until this calibration is completed; the electric-log synchronization and
casing/gap/wall proxies are declared assumptions, not measured specifications.

Historical scripts/tests still present in the tree are obsolete and are not
part of this workflow; their source-file deletion awaits separate approval.

## Model notes

- [R1/R2/R3 chemistry and connectivity](docs/reaction-chain.md)
- [Joule heating and calibration protocol](docs/joule-heating.md)
- [Continuous gas combustion](docs/continuous-gas-combustion.md)
- [Transactional CUDA updates](docs/transactional-cuda-step.md)
- [Retired artifacts](docs/retired-scenarios.md)

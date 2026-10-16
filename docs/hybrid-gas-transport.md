# CPU local groups with CUDA thermal evolution

## Scope and order

The full solver can select `gas_network.mass_backend = "cpu_local"` with
`transport = "split_local_equilibrium"`. Existing scenarios remain on the
historical global CUDA transport for reproducible comparisons.

One transaction consists of:

1. Leading casing fracture on CUDA, including the energy debit for opening.
2. Unchanged thermal contacts and local condensed-phase chemistry on CUDA.
3. Gas production and local gas/material energy equilibration.
4. Synchronized candidate download; CPU pressure transport in connected groups
   of at most nine volumes, followed by the existing finite-fuel gas reaction.
5. Candidate upload; constitutive cache refresh, mixture aging and validation.
6. Commit, or discard every candidate buffer.

The CPU object retains topology, not accepted state. Repeated preparation and
discard cannot accumulate gas transfers. There are no concurrent CPU/CUDA
writes, separate streams, or asynchronous exchange of stale temperatures.

## Modules and conservation

- `gas_partition.hpp`: deterministic edge-disjoint connected groups; groups
  within a color have no shared volume. An outside sink is not a writable node.
- `local_gas_transport.cpp`: local adaptive substeps, conservative mass,
  enthalpy, chemical-energy and oxidizer packets; cumulative channel ledgers.
- `implicit_gas_transport.cpp`: optional joint backward-Euler solution of a
  group, analytic Jacobian, positivity-checked Newton solve and step doubling.
- `mixture_equilibrium.hpp`: common CPU/CUDA inversion of total material + gas
  energy, including nonlinear capacity and changing condensed composition.
- `mass_step.cuh`: orchestration and transactional synchronization only.

Each gas substep uses the current pressure and gas temperature in the orifice
law. Material heat capacity participates in the pressure-response and thermal
limits. After each local transfer gas and its bound material share one
temperature; this is not the gas-only approximation used by the microbenchmark.
Free volume and condensed inventory remain fixed during the mass operator;
they were already updated in the local chemistry stage.

The old-state gas reaction proposal is preserved: incoming fuel cannot enlarge
its kinetic proposal, exported fuel cannot be spent twice, and reaction heat is
debited from chemical inventory. The post-transfer LTE closure refreshes gas
and material temperatures without creating energy.

## Numerical controls

`local_exchange_interval_s` limits the exchange interval between groups.
Each group is integrated for half that interval in forward color order and
half in reverse order. Every edge acts for the same physical interval, not for
the interval times the number of groups. This symmetric ordering is **not a
claim of second-order accuracy**: local explicit integration remains first order.

Gas pressure/mass stiffness no longer directly reduces the outer CUDA thermal
step: it is handled inside CPU groups. Thermal, source, reaction, fracture-work
and aperture/pressure-feedback bounds remain active. Both the outer step and
the group exchange interval must be refined for coupled convergence. Pressure
tolerances are numerical controls, not additional material-fit degrees of freedom.

The default exchange interval is 10 microseconds. This is a conservative
starting control, **not an established converged value for all networks**.
Near-equilibrium gas remains stiff; changing hardware does not remove it.

### Optional implicit local integration

Set `gas_network.local_integrator = "implicit"` together with
`mass_backend = "cpu_local"`. The default stays `"explicit"`, which is also
the independent fine-step reference. The thermal-contact operator, fracture
law, chemistry and material-fit parameters are unchanged.

Each group jointly solves signed mass packets and final temperatures. The
residual uses the signed square of the original compressible-orifice law,
without its square-root singularity at zero pressure difference. An analytic
Jacobian and small dense partial-pivoting solve replace numerical derivative
sampling. Only the active matrix footprint is cleared and factored.
The zero-flux derivative regularization affects Newton directions only, not
the residual. A limited explicit predictor is a starting guess, not a transfer.

Mass/enthalpy are then committed as identical donor/receiver packets. Fuel
and oxidizer use the same mass flux through an implicit upwind M-matrix;
local gas/material equilibration uses the shared extensive-energy closure.
The solver never repairs failed convergence by silently accepting an invalid
state. Newton failure, negative inventory or excessive local error rejects the
step; exhausting the bounded retry loop aborts the candidate transaction.

One full backward-Euler step is compared with two half steps. The accepted
state is the two-half-step result, without Richardson extrapolation. Controls:

- Existing `pressure_absolute_tolerance_pa` and `pressure_relative_tolerance`.
- `implicit_temperature_tolerance_k` (default 0.02 K).
- `implicit_tracer_relative_tolerance` (default 0.001), normalized by each
  group's chemical or oxidizer inventory.

These are **local numerical error controls**, not guaranteed global errors.
They do not measure cumulative edge-packet error, nor the splitting error
between groups. Refine the exchange interval and outer time step separately.
Both local integrators are first order; the symmetric group sweep does not
make the complete method second order. Implicit is not automatically faster
on small/easy groups, and groups are not yet weighted by exchange time or put
to sleep.

`Simulation::profile()` and replay `solver_profile` report wall time for
preparation, CPU gas integration, downloads and uploads, plus group/substep/
Newton/rejection counters. These count attempted work, including discarded
transactions; they are diagnostics, not physical state. Timings are host wall
times, not CUDA kernel-event timings.

## Benchmark context

The V34 fixed-gas audit contains 676 volumes and 2194 channels. On the shared
Ryzen 5 2600 / RTX 4060 Ti desktop, 30 microseconds of physical evolution took
8.23 seconds with global CPU stepping and 1.82 seconds with CPU groups at a
62.5 ns exchange interval. Its maximum difference from the global CPU trace
was 0.357 K / 0.135% pressure. The corresponding local CUDA run took 10.25 s:
the high-degree common chamber leaves many groups serialized. This is an
isolated gas benchmark, not a full-model performance or validation result.

Artifacts: `data/calibration/upv-local-gas-v34/`. The smaller 15.625 ns
interval took 2.55 s on CPU; refinement does not monotonically improve speed.

## Destruction-only replay

`scripts/calibrate_upv_local_cpu.py` generates the fracture-first 3x3x3 UPV
model, freezes the physical heat profile and chemistry, and changes only the
declared shell strength and opening rate. It supports a separate cooled
experiment replay; do not select parameters from that held-out experiment.

The script records scenario/binary fingerprints, raw results, ledgers and
metrics at original observation times. Long runs save atomic partial snapshots.
A wall-time limit terminates only its own replay child and records an incomplete
result. A partial trajectory is never reported as a calibrated profile.

Example (a trial, not a validated fit):

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/battery-area-plots PYTHONPATH=src:scripts \
python3 scripts/calibrate_upv_local_cpu.py \
  --output data/calibration/upv-hybrid-trial \
  --duration 30 --exchange-interval .001 \
  --pressure-atol 10 --pressure-rtol .001 --timeout 300
```

Before accepting a fit, refine pressure controls, exchange interval and thermal
step; then score the full destruction window and replay the cooled holdout.
No new calibrated destruction profile is implied by successful compilation,
unit tests, or the gas-only benchmark.

## V35 integration results

Artifacts are under `data/calibration/upv-hybrid-v35/`; regenerate the compact
report with `scripts/report_upv_hybrid.py data/calibration/upv-hybrid-v35`.

- Full native build, including viewer: successful. All 30 CTest tests passed,
  including production/benchmark multi-group parity, hybrid LTE blowdown,
  nonlinear capacity, chemical inventories, fracture-first transfer and rollback.
- Both frozen heating replays reproduce **identical** final fields and sample
  temperatures: zero difference for both uncooled and cooled experiments.
- A complete 0.02 s coupled smoke replay takes 28.65 s with the strict pressure
  controls and 5.10 s with numerical tolerances 10 Pa / 0.001. Their sampled
  difference is at most 1.043 Pa / 0.0841 K in gas and 1.43e-7 K at witnesses.
- Reducing the outer cap from 1 ms to 0.25 ms, the group interval from 1 ms to
  0.5 ms, and both pressure tolerances by two takes 9.22 s. Sample differences
  from the relaxed short run are 0.000142 K at witnesses, 0.0393 K in gas,
  269 Pa locally and 0.00495 Pa in the chamber. These are **pre-breakthrough**
  checks, not a convergence certificate for the explosion.
- The requested 30 s destruction baseline was attempted with a 300 s wall-time
  limit. It reached approximately 0.2245 s; the last complete saved sample is
  0.22 s, 1045 outer steps. The first nonzero experimental observation is near
  0.471 s. There is therefore no full-window RMSE or new fitted parameter set.
  `baseline/status.json` explicitly records `calibration_accepted: false`.
- Through the saved partial trajectory, maximum ledger residuals are
  7.69e-8 J thermal, 1.01e-12 kg mass, 1.56e-8 J chemical and 1.68e-12 kg oxidizer.

Timing is from a shared desktop; some tests/replays overlapped. These full-model
wall times diagnose cost, not isolated hardware speedups. The pressure-tolerance
experiment did not use measured temperatures to select physical parameters.

Five targeted Python loader/frozen-physics tests passed. A longer Python
progressive-opening demo exceeded its 60 s timeout for **both** backends while
the full replay was active; it is not counted as passed. The optional JSON Schema
test was skipped because `jsonschema` is not installed. The CPU loader smoke test
is now explicitly short; the native hybrid fracture tests retain their full
mechanical/conservation/rollback checks.

## V36 implicit prototype: measured gain, not a promoted calibration backend

Artifacts: `data/calibration/upv-implicit-v36/`. Run
`scripts/compare_upv_gas_integrators.py reference.json candidate.json` to compare
matching saved samples, channel inventories, opening and balance residuals.
Incomplete replays remain explicitly marked as partial, not fitted profiles.

The final sequential 20 ms replays used the same physical parameters, 1 ms
exchange cap and pressure tolerances 10 Pa / 0.001:

| Measurement | Explicit groups | Implicit groups |
| --- | ---: | ---: |
| Whole replay wall time | 5.040 s | 1.980 s |
| CPU gas wall time | 4.709 s | 1.723 s |
| Accepted internal steps | 3,331,194 | 36,922 |
| CPU/GPU transfer time | 0.012 s | 0.011 s |

This is a **2.55x short-replay speedup**, not a full-explosion speedup. Implicit
work also includes 240,570 Newton iterations and 623 rejected attempts; counting
only accepted steps would exaggerate the gain. The initial finite-difference
Jacobian prototype was slower than explicit (32.28 s); the analytic Jacobian
and active-footprint matrix storage removed that overhead.

At the saved short-run samples, differences are 9.04e-6 K at witnesses,
0.473 K in gas, 1.03 Pa local pressure and 0.00446 Pa chamber pressure. Maximum
thermal and mass ledger residuals are 4.75e-8 J and 7.89e-15 kg. A fourfold
coupled refinement changes gas temperatures by 0.0428 K and witness readings
by 0.000209 K, but local pressures by 1471 Pa; it is not a global convergence
certificate. The independent stricter explicit short reference is also retained.

The 0.5 s attempted implicit replay reached a saved 0.22 s in 178.82 s before
its 180 s wall budget expired. The archived explicit replay reached that same
sample in 277.39 s: an indicative 1.55x ratio under shared-desktop conditions.
Both take 1045 outer steps. However, while witnesses differ by only 0.00389 K,
individual gas cavities differ by up to **207.66 K** and channel mass packets
by 5.75e-6 kg. Chamber pressure differs by 20.42 Pa. These discrepancies cannot
be dismissed using pressure agreement or conservation alone. The implicit
mode therefore remains **opt-in, experimental**, not the production default.
There is no accepted destruction fit or full-window RMSE from these runs.

A new hot, stiff, 12-volume multigroup regression isolates the splitting risk:
with a 10 microsecond exchange interval, temperature error versus a 1 ns
explicit reference is 69.24 K for explicit groups and 68.92 K for implicit
groups. Refining implicit exchange to 1 microsecond then 10 ns reduces it to
39.33 K then 0.0911 K. Pressure can already look equilibrated while advected
enthalpy/composition is not resolved. Local step doubling does not control this
intergroup error. This establishes a failure mode, not proof that it alone
explains every UPV discrepancy. Further work must control or jointly iterate
the intergroup mass/enthalpy coupling, not merely enlarge implicit steps.

The additional coupled UPV refinement (`--refinement .5`, all numerical
controls, not just gas tolerance) completes 0.14 s with implicit in 108.56 s.
Against its coarse trajectory it changes gas temperature by up to 70.40 K,
local pressure by 4.414 MPa, chamber pressure by 465 Pa, and witness temperature
by 0.01694 K at matching samples. Thus even the coupled rupture trajectory is
not numerically converged; the good witness agreement must not be used to
accept it. The matched refined explicit attempt exhausted 120 s near 0.1155 s;
its last checkpoint is 0.08 s, so there is no matched refined post-rupture
comparison. See `rupture-refinement-summary.json` and both run/status files.

Verification: full native/viewer build and all 33 CTests passed; the subsequently
strengthened hot multigroup test also passed. Three targeted JSON loader/error
tests passed; the optional schema test was skipped (jsonschema unavailable).
Both original frozen heating replays retain identical sample temperatures,
final fields and step counts. Physical thermal calibration was not changed.

Example opt-in audit (not a validated parameter fit):

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/battery-area-plots PYTHONPATH=src:scripts \
python3 scripts/calibrate_upv_local_cpu.py \
  --output data/calibration/my-implicit-audit --integrator implicit \
  --duration .02 --exchange-interval .001 \
  --pressure-atol 10 --pressure-rtol .001 --timeout 60
```

Use `--refinement .5` in a fresh output directory to halve the probe's numerical
step/error controls. This scales numerical controls only, never material or
thermal-fit parameters. It does not by itself establish convergence.

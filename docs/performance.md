# CUDA performance — 2026-10-12

Hardware: NVIDIA RTX 4060 Ti 8GB; AMD Ryzen 5 2600; 16GB system RAM.
Driver 595.99.02, CUDA 13.2.51, GCC 11.4, Release, SM89. Physics uses FP64.
These are local measurements with the desktop running and GPU clocks unlocked,
not guaranteed application frame rates or experimentally validated physics.

Raw repeated samples and executable hashes:
[benchmark JSON](../data/benchmarks/2026-10-12-rtx4060ti.json).

## Solver throughput

### Current numerical-reference profile and continuous renderer

The calibrated 34,832-volume default advances 1400 physical seconds in 93,882
substeps / 14.277 s wall without setup or rendering (about 98x realtime).
All 52 batteries are fully spent by 1194.60 s. Halving the effective substep
cap to 0.0075 s takes 187,680 substeps / 28.599 s wall; onset times change by
at most 0.514 s. Reports: 
[default](../data/calibration/2026-10-12-reference-fit-report.json),
[smaller dt](../data/calibration/2026-10-12-reference-fit-small-dt.json).

These are cold-start trajectory runs, not a throughput comparison against a
matching CPU solver. The fitted material parameters alter CFL limits; the old
prior-profile measurements below must not be presented as current-default speed.

The viewer now ray-integrates one continuous volume per macro object (24 optical
samples per fragment), interpolating scalars before palette lookup. It does not
render separate little boxes for the eight battery zones, or read the full field
back through the CPU. Opacity does not depend on thermal mesh resolution.
The GL regression compares identical uniform 1/8/64-zone renders and checks
nonuniform gradients and cutaways.

On the final fitted scene at 1280x720, 60 measured frames after 10 warmup
frames give 1.114 ms scene rendering and 2.968 ms solver work for 16 physical
substeps/frame (245 combined FPS, no UI or VSync). These are microsteps, not
16 outer one-second intervals; the interactive outer-steps/frame slider can
therefore make the solver, rather than rendering, dominate frame time.
Raw: [continuous renderer](../data/benchmarks/2026-10-12-continuous-render.json).
Nine native CTests, eleven Python tests, the GL continuity and palette/sample
tests pass. Compute Sanitizer reports zero errors in CUDA/GL sample export.

### Original uncalibrated multiscale priors (historical measurement)

That prior scenario retains 416 battery zones and refines only its 87 interlayers.
Three repeats of 1000 substeps after 50 warmup substeps; synthetic hot field,
host summaries/events included, no rendering. Raw data:
[multiscale measurements](../data/benchmarks/2026-10-12-multiscale.json).

| Total volumes | Links | Median ms/substep | Simulated seconds / wall second |
| ---: | ---: | ---: | ---: |
| 34,832 | 116,860 | 0.266 | 313 |
| 275,744 | 861,652 | 1.653 | 7.38 |

The physical time advanced per substep differs (about 0.0833 vs 0.0122 s in
these hot-field samples). Refinement's CFL cost therefore matters much more
than steps/s alone. These are NOT measurements against an equivalent CPU
implementation and NOT convergence/validation results. The conservative
global bound includes high-conductivity/destroyed material phases even while
they are inactive. Local stepping or an implicit diffusion solve remain future
optimizations; global host-synchronized subcycling is the current implementation.

A separate complete prior-profile cold-start module run advanced 6000 s in 6000
outer intervals / 72,000 substeps, taking 17.8574 s without rendering/setup.
Energy residual was 1.33e-6 J. Only the initiating battery ran away during this
window under the explicit provisional interlayer properties; final Tmax was
424.738 K. This is a model run, not a containment claim about the public module.
Eight native CTests, ten Python tests, CUDA memcheck and racecheck passed.

### Earlier spatial-battery baseline

Same 52 macro objects, refined internal meshes. Synthetic hot-field workload
exercises conduction, local damage, contacts, and finite reaction sources.
50 warmup steps; median of three runs of 500 measured steps. Setup, mesh
construction and CUDA initialization excluded; macro downloads/events/history
included; no field readback or rendering. Recorded steps here advance 0.2s;
that dt is specific to these material priors, not a universal stable step.

| Thermal volumes | Baseline steps/s | Optimized steps/s | Speedup |
| ---: | ---: | ---: | ---: |
| 13,312 | 6,522 | 13,467 | 2.06x |
| 106,496 | 871 | 2,494 | 2.86x |
| 851,968 | 132 | 336 | 2.55x |

Stress run: 1,664,000 volumes, 4,967,784 links,
165 steps/s
(6.06 ms/step). Three runs of 300 measured
steps. This is about 275
million volume updates/s. Larger meshes were not tried on this 16GB machine:
host meshing still holds verbose geometry/law records and has allocation peaks.

## Scene rendering

Original demonstration initial conditions; 1280x720 hardware OpenGL, 10 warmup
and 60 measured frames, 16 solver steps/frame. Each frame waits for glFinish.
Scene cost includes CUDA/GL synchronization, color updates and all geometry.
No ImGui UI or VSync: the combined FPS column is NOT a measured full-app FPS.

| Thermal volumes | Scene ms/frame, before → after | Solver + scene FPS, before → after |
| ---: | ---: | ---: |
| 13,312 | 1.85 → 0.94 | 231.0 → 483.4 |
| 106,496 | 12.16 → 1.55 | 33.0 → 130.7 |
| 851,968 | 148.92 → 5.89 | 3.7 → 19.6 |

No mesh elements were removed for this comparison. Geometry is cached and
colors are written from CUDA directly into a registered OpenGL buffer. The
temperature field no longer travels GPU → CPU → GPU each frame. CPU and CUDA
palettes are checked against each other, including destroyed-volume colors.

## Retained changes and limits

- Exact constant-state links bypass nonlinear law evaluation.
- CFL upper bound is cached, invalidated on runtime conductance/safety changes.
  Source completion still restricts dt every step.
- Contact summaries and stability minima use block reductions.
- Compact GPU volume/connectivity records omit host geometry; dynamic laws
  are deduplicated and referenced by index.
- CUDA/OpenGL interop and persistent instance geometry remove per-frame field
  downloads and matrix reconstruction.
- FP64 heat transport, event/output boundaries and conservative paired fluxes
  are preserved. No fast-math precision switch was enabled.

A later trial storing reciprocal capacity and source constants per volume
increased the record from 16 to 40 bytes and regressed large-mesh throughput
(roughly 336 → 296 steps/s at 852k in these runs). It was reverted, not retained.

Nsight Systems on the retained implementation at 106k volumes attributes most
kernel time to edge updates (~49%) and volume updates (~36%); reductions no
longer dominate. These profiling runs use 200 measured steps and include
profiling overhead, unlike throughput samples.

Next candidates: separate static/dynamic edge kernels, structured bulk stencils
alongside sparse contacts, and device-side step batching/CUDA Graphs for small
meshes. Reaction-cost growth and refinement's dt restrictions still need their
own measurements. This is a first optimization pass, not a hardware ceiling.

## Reproduction

Build as described in the README, then:

```bash
./build/battery_sim_benchmark examples/narada-52-schematic.json --scale 2 --steps 500 --repeats 3 --output /tmp/battery-benchmark.json
./build/battery_sim_render_benchmark examples/narada-52-schematic.json 2 /tmp/battery-render.json
./build/battery_sim_cuda_visual_test
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/battery_sim_cuda_test
./run_viewer.sh examples/narada-52-schematic.json --mesh-scale 2
```

The viewer exposes FPS, solver time per frame, simulated-time speed, VSync and
steps/frame (1–256). Default resolution is 13,312 volumes; scale 2 gives 106,496,
scale 4 gives 851,968. Camera/viewport and steps/frame change actual FPS.

Checks passed: six native CTests, CUDA/GL palette test, seven historical Python
tests; Compute Sanitizer memcheck reports zero errors. Energy-residual ratios
in these benchmarks are below 1e-13. These are numerical checks, not calibration.

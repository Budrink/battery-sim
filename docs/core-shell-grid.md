# Battery core and shell: 3×3×3

The UPV generator now defaults to **3×3×3** battery volumes. This gives one
interior volume and 26 surface volumes. Every volume keeps its own temperature,
damage and heat-transfer connections; the shell is **not** merged into one node.
The old saved 2×2×2 calibration scenes and the eight-zone Narada reference are
retained unchanged for reproducibility.

Follow-up: the [first internal-law fit](internal-calibration.md) improves the
experimental agreement but **does not pass** this page's grid criterion.
The numerical adequacy below applies to the fixed-internal-link baseline only.

## Geometric meaning

A volume is `Shell` if its index touches any of the six exterior faces. All
other volumes are `Core`. This is one **numerical** layer, not the thickness or
properties of a physical battery casing. Both regions retain the existing
battery material properties. For the 11×87×187 mm UPV battery, a 3×3×3 volume
measures approximately 3.667×29×62.333 mm. There are three divisions along the
length as well as along the width and thickness; physical volumes are not cubes.

| Grid | Core volumes | Shell volumes | Core–core links | Core–shell links | Shell–shell links |
|---|---:|---:|---:|---:|---:|
| 3×3×3 | 1 | 26 | 0 | 6 | 48 |
| 4×4×4 | 8 | 56 | 12 | 24 | 108 |

`ThermalVolume::region` and `ThermalLink::region` label these groups explicitly.
Contacts between different objects and prescribed boundaries are separate link
classes. External contacts attach only to surface volumes, including when the
battery and pad face meshes do not match. Area-overlap weights are conserved.
No transport coefficient or heat capacity is changed by a region label.

The CUDA solver is unchanged: flux is integrated on the existing conservative
finite-volume graph. Region labels are host-side topology metadata, ready for
subsequent law configuration; this change does **not** install or fit distinct
core/shell constitutive laws. Directed external contact laws remain active.

The viewer still renders one continuous translucent battery. Its diagnostics
show the grid and core/shell counts. The experiment probe additionally exports
`battery_regions`: volume counts, heat capacities, and time histories of
capacity-weighted mean/min/max temperatures for each region. An absent core
(e.g. the legacy 2×2×2 mesh) has null temperatures, not fabricated zero values.

## Numerical choice, without refitting

`scripts/check_upv_core_shell.py` carries the saved directed contact fit to
3³, 4³, 8³ and 12³ battery meshes. Passive meshes, boundaries, sensor bindings,
material conductivities and heat capacities are fixed. The old 2³ result must
first reproduce within 1e-7 K. Actual native material parameters are audited.

The provisional mesh criterion is a maximum surface-temperature difference
of 0.5 K against 12³ over the **entire** simulated interval. Reference adequacy
requires 8³ versus 12³ below 0.1 K. Each grid also gets a genuinely smaller
integration step, with a maximum allowed temperature change of 0.05 K.
These are numerical engineering thresholds, not sensor uncertainty estimates.

| Experiment | Max 3³ vs 12³, K | Max 4³ vs 12³, K | Max 8³ vs 12³, K |
|---|---:|---:|---:|
| No cooling, 0–884 s | 0.213 | 0.110 | 0.024 |
| Active cooling, 0–1412 s | 0.223 | 0.184 | 0.072 |

The largest time-step refinement change is below 0.005 K. **3³ passes** and is
selected. The 12³ calculation is a refined numerical reference, not an exact
solution. This checks battery-grid sensitivity only, not full-domain convergence.

Against experimental observations (scoring starts at 60 s), the unchanged
parameter set on 3³ gives RMSE 2.185 K without cooling and 2.251 K with cooling.
These errors are distinct from the mesh differences above. No parameter fitting
was performed, and this is not a new frozen-model calibration/validation claim.
The C1/C3 acquisition-numbering hypothesis is still unconfirmed. Active cooling
was previously inspected and is not a pristine blind holdout.

## Reproduce and inspect

```bash
cmake --build /tmp/battery-sim-cuda-v1 -j 4
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_upv_core_shell.py \
  --probe /tmp/battery-sim-cuda-v1/battery_sim_experiment_probe \
  --output-dir data/calibration/upv-core-shell-check \
  --export examples/upv-core-shell-3x3x3.json
./run_viewer.sh examples/upv-core-shell-3x3x3.json
```

Full results and provenance are in
[`summary.json`](../data/calibration/upv-core-shell-check/summary.json).
The exported viewer scene carries prior contact parameters already applied;
do not apply fit scales again. It is measured-boundary **pre-event** heating,
not an autonomous runaway demonstration.

To generate an uncalibrated 4³ alternative, pass
`--battery-divisions 4 4 4` to `scripts/generate_upv.py`.
The legacy depth-only `battery_sim_profile_probe` requires even depth divisions
for its exact central-face flux; use the experiment probe's `battery_regions`
for this odd-grid core/shell study.

## Limit of this decision

One interior node cannot represent a gradient *inside* the core. It is adequate
for the current surface-temperature test with fixed battery k and C, not proven
adequate for hot spots, runaway or strongly state-dependent internal laws.
After internal-link calibration, repeat this check and upgrade to 4³ if needed.
The physical extent of a one-cell numerical shell changes when refining the
grid, so coarse/fine core means are not temperatures of identical regions.
Any future material-specific casing layer needs a fixed physical definition.

Tests cover region/link counts, total heat capacity, conservative CUDA exchange,
maximum temperature bounds, nonmatching face area and shell-only contacts,
probe readout, unchanged material parameters and continuous 27-volume rendering.

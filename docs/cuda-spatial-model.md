# CUDA finite-volume model (v1)

This supersedes the native lumped-cell solver. Python `src/battery_sim` is the
historical v0 reference, NOT the solver used by the application. Its synthetic
timings can be used as a numerical regression target, not as experimental
calibration evidence. Both the native viewer and headless
runner link the same CUDA implementation. There is no silent CPU fallback.

## Two spatial scales

The default `narada-52-multiscale.json` uses 8 finite-conductivity zones per
battery and 87 separate passive solids between batteries. Battery conductivity
is a provisional fast-equilibration parameter (1000 W/m/K), NOT a measured
material property. Interlayers have independently specified conductivity,
volumetric heat capacity and l/l0/phase/damage laws. `--mesh-scale` refines only
passive objects when present; it does not change the battery's zonal model.
The historical `narada-52-schematic.json` is retained unchanged for regression.

Objects have `role: "battery"` (default) or `"passive"`, and a `material_id`
for calibration grouping. Passive objects retain local thermal damage and can
be destroyed, but cannot have a runaway source or be scheduled for ignition.
Only batteries participate in secondary-runaway counts and all-destroyed pause.
The Python historical lumped loader rejects passive scenarios explicitly.

A battery is a macro object containing a structured grid of thermal control
volumes. Each volume stores temperature, thermal damage, reaction state,
released energy and boundary-energy accounting. Empty space is not meshed.
Passive macro objects disable runaway sources but can retain thermal damage.

The current mesher supports prismatic and pouch solids. Cylinders and contacts
between differently oriented local frames fail explicitly; they need additional
mesh adapters. Objects can share an arbitrary rigid rotation. Surface ports
must name a local face; centroids/normals are in world coordinates. Smaller
ports are square patches; a full-face port uses the actual face rectangle.
`case` cooling is area-distributed over all six external faces. It is an
effective imposed boundary, not automatic exclusion of covered/insulated faces.

```json
"thermal": {
  "heat_capacity_j_per_k": 5500,
  "initial_temperature_k": 298.15,
  "divisions": [8, 4, 8],
  "conductivity_w_per_m_k": [1, 1, 1]
}
```

Conductivity is diagonal in the object's material frame. Omitted resolution
defaults to 4x4x4; omitted conductivity defaults to an UNCALIBRATED 1 W/(m K).
The demo explicitly declares these assumptions. `initial_temperature_field_k`
optionally supplies a temperature for each volume, x-fast, then y, then z.
Reference capacity and reaction energy are distributed by volume; their totals
remain equal to the configured object reference totals at every mesh resolution.
With an optional node capacity law, instantaneous capacity can differ from its
reference value without changing the reaction-energy budget.

The spatial step h is a numerical length, NOT l0. Refining the mesh must not
change the material connectivity threshold. Local interface patches have
independent l, l0, phase and damage. They retain the v0 transition laws.

## Heat transport and reactions

### Separate node storage and edge transport

`thermal.bulk_response.temperature_log_response` is an optional 2..8-knot
table of `[temperature_k, log_response]`, linearly interpolated with constant
endpoint extension. It replaces the linear/quadratic temperature response
(combining them is rejected). Damage and signed sender/recipient contrast
terms remain independent. The active directed branch uses the **donor**
temperature. An empty table preserves old scenes. This modifies graph-edge
connectivity, not the reference bulk tensor or external half-volume resistance.

Node storage is configured independently:

```json
"heat_capacity_law": {
  "reference_temperature_k": 298.15,
  "factor": [[298.15, 1.0], [318.15, 1.1], [343.15, 0.9]],
  "damage_factor_at_one": 1.0
}
```

These example factors are illustrative, not material data. All factors must be
positive, temperatures strictly ordered, and the reference-temperature factor
exactly one. The piecewise-linear factor `f(T)` multiplies the node's reference
capacity. The optional defect multiplier is `a(D)=1+(a1-1)D`, with positive
`a1`. It defaults to one and is **not identified** by the UPV pre-event data.

For enabled laws the GPU stores authoritative energy
`U=C_ref*[T_ref+a(D)*integral(T_ref..T) f(theta)dtheta]`. Thus
`dU/dT|D=C_ref*a(D)*f(T)`. The flux gather adds energy first; the monotone
piecewise-quadratic primitive is inverted analytically to recover temperature.
Damage changes then re-evaluate temperature at the same U. The chosen energy
reference `U(T_ref,D)=C_ref*T_ref` is independent of damage; no hidden chemical
or defect energy is introduced. Existing reaction sources remain separate.
This is a minimal constitutive convention, not thermochemical identification.

The stability bound uses the **global minimum** allowed capacity factor,
including defect and endpoint-extension limits, so cached CFL remains safe.
Macros sum stored energy and instantaneous capacity separately; temperature is
instantaneous-capacity weighted, whereas damage/destruction fractions retain
reference-volume weighting. Probe region `heat_capacity_j_per_k` is reference
capacity, not the instantaneous sum. The old constant-C update remains the
default and is regression-tested against a unit-factor energy law.

Variable capacity currently requires finite-volume mode: combining it with
the algebraic isothermal limit is explicitly rejected, including through the
native C++ API. The historical Python solver rejects these new constitutive
fields instead of silently ignoring them. The viewer still renders a single
continuous object; constitutive changes do not introduce visual zone boxes.

### Optional isothermal internal limit

Set `thermal.internal_heat_transfer` to `"isothermal"` on an object to impose
quasi-instantaneous internal heat redistribution. The default is
`"finite_volume"`, which retains spatial gradients and the configured finite
conductivity. Viewer/headless flag `--isothermal` selects this limit for all
objects without changing the scenario file or the mesh resolution.
For mixed-material scenarios the CLI flag applies to batteries only, keeping
the resolved passive environment intact.

The full volume grid, contact patches, local reaction budgets and CUDA/GL field
remain. Internal finite-diffusion links and the object's center-to-face
resistances are replaced by an exact uniform-temperature constraint. Each CUDA
step sums external heat and released reaction energy across the object and
updates every volume with the same `delta_T = delta_E / C_object`, BEFORE
evaluating temperature-dependent damage. External contact resistance, l/l0,
phase transitions and contact damage stay finite and unchanged. Mixed
isothermal/spatial contacts retain the spatial object's half-volume resistance.

The explicit stability bound uses the isothermal object's total heat capacity
and total external conductance; it does not acquire a stiff internal-diffusion
restriction as the visualization grid is refined. This is not a large numeric
conductivity and is not an implicit spatial diffusion solver. Specified
nonuniform initial fields or finite `bulk_connectivity` on an isothermal object
are rejected rather than silently ignored. Local prescribed ignition still
consumes only the selected volume's budget; uniform mixing alone does not
force the entire object into runaway.

This mode reproduces the historical single-temperature experiment with uniform
initial state and whole-object initiation. It is a controlled model limit, not
a calibration or a claim that real battery interiors equilibrate instantly.

### Finite-volume transport

Internal face conductance is k_axis * area / center_distance. Contact patches
are paired by geometric overlap; configured interface conductance is divided
by overlap area fraction, never copied in full to every patch. Center-to-face
resistances of both adjacent volumes are added in series. Ambient boundaries
also include the local center-to-face resistance.

Optional `thermal.bulk_connectivity` applies the same l/l0/phase/damage law to
every internal face. It uses `background_factor`, `active_factor`,
`destroyed_background_factor`, `destroyed_active_factor` (dimensionless),
multiplied by k_axis*A/d. All other fields match the interface law, including
required `l0_m`. Without it, bulk conductivity stays constant. Thus refinement
changes geometric G, not material thresholds, and internal connectivity can
degrade or switch phase independently at each face.

Optional `endpoint_damage_weight` (0..1, default 0) couples the adjacent
volumes' accumulated damage into a contact or bulk link. Before its ordinary
thermal-damage update, link damage becomes at least this weight times the mean
endpoint damage. This history is irreversible and feeds the existing l/phase/G
law. It is a calibratable phenomenological closure, not a derived material law.
All effective link conductances are recomputed from the same old field on each
substep, before any volume updates. Geometry/connectivity candidates remain
sparse; no dense material-to-material tensor is stored.

Contacts remain zero-capacity surface resistances in v1. Their visualization
is interpolated, not an independently solved spacer temperature. A spacer
with heat capacity/latent heat must be a separate meshed solid, not a hidden
extra energy source in l. Gas flow, ejecta, radiation and electrical physics
are still absent. The heat source/damage kinetics are uncalibrated v0 scaffolding.

Each volume has its own finite reaction-energy budget. Thermal damage triggers
that volume only. A scheduled event can specify `thermal_volume` (local x-fast
index); without it, the entire selected object is ignited, preserving the
explicit whole-object trigger assumption of existing scenes. A power multiplier
changes release speed, not total chemical energy. A zero multiplier pauses
release without destroying the remaining budget.

## CUDA pipeline

### Physical substeps within outer intervals

`simulation.step_interval_s` sets the requested outer interval (1 s in the
multiscale demo). `max_dt_s` caps individual physical substeps. `Simulation::step`
still exposes one adaptive substep; `Simulation::advance` advances an outer
interval and invokes its optional stop observer after every substep.
The viewer counts outer steps per frame and displays both counters and the
actual substep size. Host synchronization is currently once per substep for
dt, macro summaries and event handling; the full spatial field stays on GPU.

Each substep computes `Q_ab = G_ab * (T_a - T_b) * delta_t` once per face,
then subtracts/adds that same energy from/to the two volumes. Updates use full
volume heat capacities `C_i` in J/K, not specific heat alone. A large battery
zone gathers all its small contact-patch fluxes. Each new substep recomputes
connectivity from the newly evolved local state; multiple graph sweeps do not
transfer a whole outer interval's heat repeatedly.

The substep is capped by the outer end time, max_dt, output times, scheduled
ignition, remaining source energy and the conservative thermal bound
`safety * min_i(C_i / sum_j G_ij_max)`. The bound includes both phases and
destruction, so a conductance increase in the current step remains covered.
Source energy is finite; prescribed boundaries are included in the energy
ledger. A pause may stop partway through an outer interval, exactly after the
substep containing destruction. Surviving pads do not prevent an all-battery
pause. Quiet gaps or a finished heat source do not prove containment; reaching
the horizon reports only what happened within that observation window.

This is global explicit subcycling, not local asynchronous time stepping or an
implicit diffusion solver. Nonlinear reaction/event accuracy must still be
checked by reducing max_dt and refining material meshes; thermal stability
alone does not establish convergence or experimental validity.

Topology is built once on the host. Device allocations persist across steps.
The GPU performs:

1. CFL and remaining-source-time restriction of the host event/output dt cap.
2. Local edge-state evolution and one signed heat flux per edge.
3. Race-free CSR gathering of the same flux with opposite signs at its ends;
   double-buffered temperature, local damage and source updates.
4. Parallel macro reductions: capacity-weighted mean temperature/damage,
   Tmin/Tmax, maximum damage, active/spent volume fractions, stored/released
   energy, heat capacity, and boundary energy. Contact conductances are summed;
   l/l0/damage are area-weighted summaries, not inputs replacing local states.

The explicit-step restriction uses an upper bound on each edge's conductance
over both phases and all allowed damage states; runtime scales are included.
It therefore covers an abrupt transition in the same step. This is a thermal
stability bound, not proof of reaction/event convergence: refine time and space.
Pair fluxes and state use double precision. CFL minimum uses a positive-double
atomic CAS; flux gathering has no floating-point atomic adds.

The host downloads macro summaries for event/UI orchestration, not the full
field per solver step. Rendering uses CUDA/OpenGL interop: a CUDA kernel writes
scalar temperature/degradation/destruction samples into a registered GL buffer.
Each physical battery/pad is one continuous translucent volume, NOT separate
voxel cubes. The fragment shader integrates 24 samples along each clipped ray,
trilinearly interpolates the scalar within the same object, then applies the
same palette as the legend. There is no interpolation through empty space or
across material boundaries. Destruction is categorical (nearest volume), not
an interpolated scalar-palette value. Opacity is independent of numerical zone
count; object volumes are composited back-to-front. `Volume cutaway X` exposes
the interior continuously without removing whole numerical zones. Outlines
draw only the physical box edges, not face triangulation or internal zone walls.
Host field snapshots remain available for tests/export, but are not used per frame.
The GL continuity regression checks identical uniform-field renders for 1, 8
and 64 numerical zones, and verifies nonuniform fields and cutaways remain visible.

Constant links bypass nonlinear law evaluation only when their state is
provably constant. Dynamic laws are shared by index; GPU connectivity and
volume records omit host-only geometry. The conservative CFL bound is cached
until conductivity runtime scales/safety change; local source-end restrictions
still run every step. Reductions use blocks, not one thread per contact patch
group. All thermal arithmetic remains double precision; rendering uses floats.
Device-side multi-step scheduling and implicit diffusion remain future work.

## Tests and interpretation

`battery_sim_cuda_test` checks analytical two-volume exchange, insulated-slab
mesh convergence, uniform equilibrium, energy conservation, surface-only
cooling, local ignition/budget exhaustion, partial destruction, contact
destruction and l0 replacement. `compute-sanitizer` can run this executable.
The 52-object case now has 13,312 thermal volumes; its old lumped propagation
times/outcome are not spatial-model reference measurements. Do not retune the
new solver merely to recreate the old animation.

Calibration and measured material tensors remain separate work. Optional node
capacity laws and internal connectivity tables are implemented; the sequential
UPV trial in the README does not identify universal material laws or defect
dependence. SOH/cycle-aging evolution is not added by voxelization.

`battery_sim_constitutive_test` checks energy/temperature inversion and dU/dT,
CPU/CUDA one-step agreement, conservative variable-C exchange, unit-C regression,
defect changes and ignition without artificial energy, separate source budgets,
GPU table-law deduplication and the minimum-capacity CFL bound.

`battery_sim_isothermal_test` additionally checks analytical uniform-body
exchange, grid-independent CFL, mixed spatial/isothermal coupling, mixing before
damage, reset, conflicting inputs, local reaction budgets, and the full 52-object
historical sequence at 13,312 and 106,496 volumes. Its historical timings are
numerical regression data, not measured public-dataset validation data.

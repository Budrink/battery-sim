# Physical casing, finite-rate conversion and lumped gas discharge

## Separate thermal and mass geometries

The opt-in `gas_network.transport = "split_local_equilibrium"` mode separates
two graphs and two operators. Omission selects `legacy_coupled` for exact replay
of saved experiments. No calibrated thermal coefficients or saved UPV profiles
are modified by enabling support for the new solver.

1. **Thermal graph:** existing finite volumes, directed state-dependent
   connectivity, contact resistances, boundaries and finite-capacity cooling
   walls. It transfers heat without moving mass. Local material conversion
   creates a conservative gas-production packet from the condensed inventory.
2. **Mass graph:** explicitly declared gas/pore nodes and pressure channels.
   It transfers mass, donor enthalpy, chemical potential and oxidizer. It does
   not borrow conductance from a thermal edge. Damaged casing determines the
   exposed flow area; pressure relief reduces flow without an imposed lifetime.

The two graphs share geometry only through an explicit local binding:

```json
"local_equilibrium": {
  "cell_id": "membrane",
  "thermal_volume": 0
}
```

This is an optional property of a gas compartment, not a replacement for its
mass, temperature and `volume_m3`. The index is local to the configured material
grid, x fastest, then y, then z. Grids need not be the same size in different
materials. There may be at most one effective mobile phase per material volume.
Unbound compartments remain finite gas cavities. A bound compartment receives
conversion products from its volume; unbound source volumes retain the declared
`conversion.gas_compartment` fallback. Initial gas and matrix temperatures must
agree; inconsistent initialization is rejected, not silently equilibrated.

`volume_m3` is the **initial free gas volume**, not the whole material envelope.
For a reacting volume, optional `reactant_density_kg_m3` and
`residue_density_kg_m3` must be supplied together. Then:

```text
Vfree = Vfree_initial + m0/rho_reactant
        - (m_reactant + m_activated)/rho_reactant - m_residue/rho_residue
P = m_gas R T / Vfree
```

Without densities, pore volume is explicitly fixed. Geometry and inventory must
fit, and invalid/nonpositive free volume rejects the candidate. There is no
inferred density from heat capacity, hidden pore-volume clamp, or mass creation.
The activated material currently uses reactant density. This is an effective
composition closure, not an electrolyte vapor-pressure/boiling model.

### Two stages of one transactional substep

```text
committed state
  -> A: old-state heat fluxes + local conversion/production
  -> local energy-conserving thermal equilibrium
  -> B: new-pressure mass fluxes + transported fuel/oxidizer reaction
  -> local equilibrium, material damage, casing work, final closure
  -> validate budgets/state -> commit
```

Each channel creates one extensive transfer packet, gathered with opposite
signs at its endpoints. For the common effective gas:

```text
delta_U_advected = delta_m [cv (Tdonor - Tref) + R Tdonor]
delta_Echemical = delta_m Echemical_donor / m_donor
delta_m_oxidizer = delta_m m_oxidizer_donor / m_donor
```

Enthalpy includes pressure-flow work; transferring only `m cv T` would miss
blowdown cooling. A pressure gradient can move matter toward a hotter volume:
the temperature gradient is not a mass-flow gate. Temperature still affects gas
density/EOS and compressible orifice flow. An unobstructed non-rupture channel
with zero opening threshold is bidirectional from initialization; a thresholded
barrier must first open. An outside-domain endpoint is a sink, not an inlet.

LTE solves `Usolid(T, composition, D) + Ugas(T, m) = Utotal`. It does **not**
average temperatures or overwrite energy with a guessed temperature. Total
local capacity is `Csolid + m_gas cv`. Gas loss reduces the mobile contribution;
conversion changes the condensed contribution. Gas and solid retain separate
energy ledgers, so macro `stored_energy_j` still reports solid energy only;
`process_diagnostics()` includes both. Macro capacity includes bound gas.
For nonlinear capacity tables, closure uses a fixed-count monotone energy
inversion. Damage of a bound nonreacting matrix is evaluated once, at the
shared temperature after both transport operators, not at a provisional
solid-only temperature. Conversion damage remains inventory-based.

New-pressure fluxes are budgeted again after stage A. If necessary, **both**
stages are retried from the committed state with smaller dt. No partial stage
can leak into committed mass, energy, damage or channel histories. Point kernels
are independent; fixed-order CSR gathers and reductions preserve repeatability
across supported CUDA work tiles. This is first-order operator splitting, not
an implicit CFD solve; tiny pore inventories/large flows can still require many
substeps and host synchronizations.

The pressure-equalization cap uses the mixture's `dP/dm`, including local solid
capacity. A gas-only adiabatic cap can overshoot when a hot matrix keeps incoming
cold gas hot; the near-equilibrium regression specifically covers this case.

The old heat-only `surface_link` shortcut is rejected in split mode. Route mass
to a receiving pore/cavity explicitly. Ordinary gas/solid film contacts remain
available for **different** volumes; a film contact duplicating local LTE is
rejected. The existing shell damage law, evolving solid thermal contacts and
separate fracture-work ledger remain in use. This does not force thermal and
mass conductance to increase together when a barrier fails.

### Modules, verification and scope

| Responsibility | Implementation |
| --- | --- |
| Local phase mapping and geometry checks | `mass_mesh.hpp`, `mass_mesh.cpp` |
| Two-stage orchestration/retry | `mass_step.cuh` |
| Production gather, LTE and mixture aging | `mixture_kernels.cuh` |
| Pressure flow, conservative payloads and gas reaction | `gas_kernels.cuh` |
| Existing thermal operator | `cuda_step.cuh`, thermal kernels in `cuda_solver.cu` |
| Casing integrity and energy-paid opening | `shell_kernels.cuh`, `shell_work_kernels.cuh` |

`examples/split-transport-demo.json` is a minimal one-volume source, physical
casing, gas+membrane volume and chamber. Its parameters are synthetic; it is
not an automatic migration of UPV contact geometry. Native `cuda_split_transport`
checks analytic LTE heating and blowdown, mass-only receiver heating/damage,
backflow, closed barriers, local production, free volume, step refinement and
all conservation ledgers. `cuda_transaction` covers split-step retries and exact
64/256-thread agreement. JSON integration tests exercise progressive rupture,
paid opening and downstream mass/fuel delivery.

The frozen pre-event calibration and cooled-validation scenes replay with
**zero temperature change**, identical final fields and identical step counts
(15462 and 25572). Reproduce using `scripts/check_split_transport_regression.py`.
That regression verifies the historical thermal mode; it is **not validation of
new destruction physics** or of a newly populated pore graph.

Current mobile matter is one effective **gas phase** with chemical/oxidizer
tracers. Solid matrix and condensed reactants remain stationary until converted
to gas. Droplet entrainment, solid fragments, momentum, multi-species transport,
flame radiation and resolved liquid convection are not implemented. New pore
volumes and pressure-channel geometry must be supplied explicitly, not fitted
by silently changing the calibrated thermal graph. The instantaneous local
equilibrium assumption can overestimate very short gas/surface heat exchange;
it is the stated first approximation, not a measured boundary-layer law.

### UPV pressure paths (v15)

Current barrier implementation additionally separates thermal weakening and
mechanical aperture history (see the whole-shell closure below). The initial v17
protocol held the v13 thermal inputs,
chemical energy, reaction kinetics and v15 gas geometry fixed. Its declared
five-regime search changed only reference pressure strength, pressure-damage
rate, aperture exponent and discharge coefficient. Pressure errors at original
measurement times in 0–3 s and 3–30 s select a candidate; dense diagnostic samples
are not additional observations. V17 was rejected: halving the step changed
the -x aperture from 11.50% to 1.09% and fracture work from 0.0462 to 0.0232 J,
despite only 0.043 K witness and 0.007 bar chamber-pressure changes. Full training
completed, but held-out replay was stopped; the v17 profile is not accepted.

V22 (`scripts/calibrate_upv_permeability.py`) replays baseline and that prior
candidate after the prospective-aperture timestep correction. It is a numerical
audit, not another broad search. Its refinement gate additionally checks the
aperture history/final area, fracture work and cumulative exported mass.
Temperature curves remain diagnostic evidence,
not extra adjustable thermal parameters. After a half-step check, full 1800 s
training and held-out replays use the same selected parameters. The held-out
experiment has previously been inspected; it is not a blind validation.

**V22 result: numerical gate failed.** The 30 s coarse/fine runs took 166896 /
358738 steps (189.0 / 396.1 s). Halving numerical controls changes witness
temperatures by at most 0.08495 K and chamber pressure by 0.005289 bar, but the
-x open fraction changes from 0.000409807 to 0.000298715: 37.19% relative to
the finer result. Other faces except +x differ by 24–32%. Fracture work changes
by 0.0003583 J (2.11%); maximum cumulative mass-export difference is 0.07886 g.
Thus the approximate pressure/aperture feedback treatment is **not yet
converged on this network**. Passing isolated shell tests does not override
this failure. No full held-out replay or new fitted profile was accepted.

The finer trial predicts a chamber peak of 6.488 bar at 1.30 s, versus the
measured peak 7.983 bar at about 0.942 s. Its maximum internal pore pressure
is 1871.5 bar, an unvalidated model prediction, not a measurement. The missing
C3/ISS spike persists. In the coarse run, gas reaction releases 215.6 kJ in
the chamber, but only 2.69 J in the ISS3 gap; net advected enthalpy into that
gap is 30.77 J over 30 s. This localizes a transport/observation mismatch, but
does not by itself identify its physical cause. Thermal calibration was not
changed: both pre-event regressions match the frozen reference exactly.

The CUDA equalization regression uses two equal free volumes, initially 1 g and
10^-9 g of gas, with internal conduction to establish a common final temperature.
Both permeabilities converge to 0.5 g transferred (within 10^-7 g), also with a
halved timestep. At 0.2 s, transfer differs: approximately 0.0296 versus 0.1038 g.
An exact vacuum is not yet supported by the gas-energy representation; this
test explicitly uses a positive seed mass. Without eventual thermal equilibrium,
equal pressure alone would not imply equal masses.

`proposed_mass_flow_kg_s` and `mass_flow_kg_s` expose the uncapped and accepted
rates from the same substep. The numerical equalization cap is not a calibrated
barrier law. Pressure gates **new mechanical opening**, never flow through an
already opened area. Tests cover residual strength at full thermal degradation,
irreversibility, reverse flow, work on new area only, energy-limited growth,
conservation, exact prepare/discard replay and different CUDA block widths.

`scripts/upv_mass_paths.py` builds a separate, explicit pressure graph on top of
the frozen v13 thermal scenarios. Original scenes and profiles are not edited.
The [experiment description](https://zenodo.org/records/22273467/files/TR_Experimental_Data_Description.pdf)
specifies a sealed 120 L safety chamber, pouch cells clamped between ISS/TIM and
metal layers, and a bottom cooling plate in the cooled experiment. It does **not**
specify measured rupture positions, open gap dimensions or detailed holder CAD.

The implemented routing hypothesis is:

- 27 local source pore volumes follow the C2 solid grid. Adjacent volumes exchange
  mass through pressure-driven internal paths; chemical products originate locally.
  Each aggregate casing-face opening currently draws from the central boundary
  pore on that face; the other pores feed it through the internal graph. The
  actual multi-patch rupture pattern is not reconstructed from measurements.
- Each broad-face rupture feeds an entrance manifold and eight surface-gap patches
  adjacent to ISS2 or TIM2. These patches exchange gas along the interface and leave
  at the perimeter. They are **not permeability through intact silicone or ISS**.
- Analogous perimeter-accessible gaps at ISS1/TIM1 and ISS3/TIM3 can admit gas
  from the exterior regions, so neighbouring cells are not artificially excluded
  from mass-delivered heat. All six interfaces use the same gap-height prior.
- Four edge ruptures feed four near-field gas regions. Those communicate with the
  chamber. With the cooling plate present, the bottom region connects around its
  lateral edges instead of directly through the plate to the chamber.
- All paths permit pressure-driven backflow. No channel leaves the sealed chamber.
  Near-field/gap volumes, initial gas mass, thermal energy and oxidizer are carved
  out of the original chamber inventory, not added on top of it.

`kind: "internal"` channels join face-adjacent, locally bound volumes of one
material and omit `port_id`. Native validation rejects nonadjacent endpoints,
cross-material internal channels and an area larger than the shared face.
External channels remain explicitly anchored to existing ports. The reduced
perimeter paths use port direction/area bounds, not a resolved external fluid mesh.

Gap height, the 20 mm near-field extent and the effective internal open fraction
are engineering hypotheses. In particular, the bottom plenum is a reduced
perimeter volume, not a claim that gas occupies the solid cooling plate. LTE with
a pad patch is the ideal-contact approximation discussed above; short jet impact,
moving gaps, entrained droplets, liquid mixing and flame radiation are unresolved.

The destruction-only search is `scripts/calibrate_upv_mass_paths.py fit`, followed
by `validate` with the frozen profile. Only total reaction energy, its gas chemical
fraction, reaction rate and effective gap height vary. The 3–30, 30–300 and
300–1800 s witness windows and chamber pressure enter the declared objective;
0–3 s witness temperatures remain explicitly reported diagnostics, not silently
discarded measurements. The resolved pressure peak is still scored. The cooled
experiment is excluded from parameter selection (held out from this search, not
historically blind). A half-step replay must pass before freezing parameters.

Solid properties, thermal contacts, sensor bindings, initial fields and the old
heat profile are hashed and checked. Replaying the frozen **pre-event** cases is
a regression check, not another thermal calibration. Gas-contact coefficients are
unchanged; their gas-side neighbours are now the appropriate local gas regions.
The legacy heat-only aperture shortcut is removed in favour of mass delivery.

`scripts/plot_upv_mass_paths.py` saves temperature/pressure comparisons, a topology
diagram and **net signed** accumulated mass/chemical energy per rupture. Net mass
must not be interpreted as gross outward throughput when backflow occurs.

Status: opt-in CUDA implementation with numerical verification; realistic path
**hypotheses**, not measured jet geometry or validated failure physics.

The subsequent user-directed test is `scripts/calibrate_upv_shell_strength.py`:
`scan` compares casing reference-strength factors 1, 2, 4 and 8 over the first
30 s; `finalize` checks the selected stronger casing over 1800 s, halves the
numerical step limits and then replays the cooled experiment. Only the six
rupture `opening_pressure_pa` values change. Reaction energy, gas fraction,
reaction rate, gap size, seam ratio, thermal softening and damage kinetics stay
at the original v15 baseline. Factors are sensitivity trials, not measured pouch
strengths. Selection among the **stronger** cases uses chamber-pressure RMSE on
original measurement times; 1x is an unchanged control, not an optimized candidate.
Additional 20 ms output during the first five seconds resolves pressure buildup
and blowdown but is not counted as additional experimental observations.
The earlier eight-case energy/gap scan was stopped at the user's request; no
accepted v15 profile or cooled validation is implied by those partial artifacts.

The example is `examples/shell-gas-network-demo.json`; its material constants,
reaction parameters, contact coefficients, cavity and rupture threshold are
illustrative. Do not treat its event times as predictions for a real battery.

## Thermal topology: casing is not an outer layer of battery material

```text
contents (3×3×3 or configured grid)
    ↕ contents/casing contacts
six physical casing panels ↔ panel seams
    ↕ explicit exterior ports
TIM / ISS / cooling boundary / exposed gas cavity
```

`shells` builds six nonoverlapping panels around axis-aligned prismatic/pouch
contents. Each has its own material ID, thickness, density, specific heat,
conductivity, temperature, thermal damage and residual conductivity. Capacity
is rho*cp*volume; corners are counted once. Each panel has one volume through
thickness and inherits the contents' tangential divisions. There are six
contents–panel contacts and twelve panel–panel seams. Their `contents_contact_h`
and `seam_contact_h` are in W/(m² K); solid half-volume resistances are also
included by the usual interface solver. The contact and panel conductivity
change on destruction; the casing remains in the energy ledger.

Generated panel/port names for contents `B`, e.g. its +z face:

- `B:case:+z`: physical panel object;
- `B:case:+z:contents`: contents-side contact patch;
- `B:case:+z:inner`: panel-side contact patch;
- `B:case:+z:outer`: exposed panel face for exterior connections/discharge.

Port `rectangle_m` specifies the two tangential dimensions in axis order
`(normal_axis+1)%3`, `(normal_axis+2)%3`; their product must match `area_m2`.
The mesher conservatively intersects nonmatching patches. For an enclosed
battery, exterior thermal contacts, boundaries and gas exposure cannot bypass
the physical casing. A rupture channel must refer to an actual casing panel.

The historical numerical `core/shell` partition only identifies interior versus
surface voxels of the **same contents material**. It is not a physical casing.

**Avoid double counting:** the contents' geometry excludes casing thickness;
its heat capacity excludes casing and free gas. A measured whole-cell heat
capacity cannot simply be reused unchanged and then have casing capacity added
on top. Migrating a calibrated experiment requires independently justified
inventory partitioning, actual casing geometry/material, and rebinding exterior
ports/thermocouples to the casing. None of those facts is inferred by the builder.

## Local material conversion

The `conversion` block replaces, rather than stacks onto, the old prescribed
runaway budget/kinetics. Combining it with old chemistry, aging, variable-cp law
or timed bulk failure is rejected. Each volume holds original mass `mr`,
activated intermediate `ma`, solid residue `mp`, temperature, damage and a
chemical-energy inventory. Its initial reactive mass is `m0`.

Two irreversible finite-rate stages are used as a small constitutive closure:

```text
original -- absorbs a J/kg --> activated -- releases (a+q) J/kg --> residue + gas
l(T) = l0 * sqrt(1 + max(0, cp_r*(T-Tref))/energy_scale)
ka(T) = activation_rate * max(0, l(T)/lcrit - 1)
kb(T) = reaction_rate * exp[Ea/R * (1/Treaction_ref - 1/T)]
dmr/dt = -ka*mr
dma/dt = ka*mr - kb*ma
dmp/dt = (1-y)*kb*ma
dm_gas/dt = y*kb*ma
D = 1 - mr/m0
```

`a=absorption_j_per_kg`, `q=net_heat_j_per_kg`, `y=gas_yield`. They are effective
energies per initial reactive mass, using the common reference temperature;
do not import calorimetric enthalpy numbers without checking conventions.
The length law is an explicitly **hypothesized, uncalibrated** energy coordinate,
not a recovered microscopic battery law. It is independent of numerical voxel
mass; no `l=lcrit` clamp or artificial temperature plateau is imposed. It is
distinct from existing edge-connectivity `l`. Cooling stops new activation below
threshold and slows intermediate conversion; it never restores original mass.

Heat capacity is a node/composition property, separate from edge conductance:

```text
Cinert = Cref - m0*cp_r > 0
C = Cinert + mr*cp_r + ma*cp_a + mp*cp_p
Usolid = Cref*Tref + C*(T-Tref)
Echemical_solid = q*mr + (q+a)*ma
```

This keeps a fixed energy origin while composition and remaining mass change.
Species cp values are constants in this first closure; the total capacity is
not constant. No old fitted whole-cell cp law is silently superimposed.
Reaction completion changes the three internal conductivity multipliers:
`factor_axis = exp(completed_fraction * log(residue_conductivity_axis))`.
Existing state/temperature-dependent connectivity can still provide its own
response. Forward/reverse edge laws remain ordered; each shared flux has equal
and opposite energy contributions. A separate transient reaction enhancement
law is not introduced in this closure.

Of the second-stage energy, `q*gas_chemical_fraction` per converted kg remains
chemical potential in emitted gas. It is **not** counted as local heat. Gas
sensible energy at the source's old temperature is also debited from the solid.
Generated gas enters one shared internal plenum; local pore pressure and
percolation delays are not resolved. The owner plenum's free volume must fit
within the contents envelope. Gas and reactive inventory must not be added a
second time to the supplied contents capacity.

## Pressure and specified discharge paths, without CFD

Each well-mixed, fixed-volume gas compartment carries mass, sensible energy,
unreleased chemical energy, temperature and pressure. All compartments currently
share one ideal pseudo-species and reference temperature:

```text
Ug = m*cv*(T-Tref)
p = m*Rgas*T/V
hflow = cv*(T-Tref) + Rgas*T
```

Channels declare upstream/downstream compartments (`outside_domain` is a sink),
surface port, effective area, discharge coefficient and opening pressure
**difference**. `vent` uses a fixed threshold. `rupture` uses the physical
panel's old-state maximum temperature and damage:

```text
threshold = threshold0 * max(0.01, 1-Dpanel_max)
            * exp[-max(0, Tpanel_max-Tstrength_ref)/strength_scale]
```

Opening is irreversible. A rupture marks the selected panel destroyed; this is
a coarse panel-level loss of integrity, not a computed crack shape. The panel
is not deleted, no energy disappears, and contact changes affect subsequent
fluxes. Mechanical deformation/fracture work is neglected in this first model.

Flow uses the isentropic compressible-orifice formula with choked/subcritical
branches and `gamma=1+Rgas/cv`. See the primary
[NASA mass-flow derivation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/mass-flow-rate-equations/).
Mass, **enthalpy including flow work**, and the donor's specific chemical
potential cross the same channel together. This is why a gas blowdown cools
even with no thermal contact or chemical reaction.

Once an internal channel has opened, gas can flow in **either direction**, using
the current pressure difference and the actual donor's temperature, enthalpy
and chemical concentration. Reported transfer/flow is signed relative to
`from -> to`. Opening itself represents outward pressure failure, not inward
crushing. `outside_domain` is an outflow-only sink: ambient entrainment is not
modeled. Use an explicit external compartment when backflow matters. Initial
gas mass must be positive; exact vacuum is not supported.

Gas/solid contact `h_w_per_m2_k` is an **effective gas-to-volume-centre** coefficient
including the unresolved boundary layer and half-volume resistance. Each patch
uses one flux `h*A*(Tgas-Tsolid)` with opposite signs on its recipients. Exposed
TIM/ISS elements can accumulate thermal damage and lose conductivity using
their material laws, including while gas keeps discharging after reaction slows.

Gas compartments may now opt into **effective finite-oxidizer heat release**.
The default `reaction_rate_per_s=0` preserves the old unburned-gas behavior.
`initial_oxidizer_mass_fraction` labels a constituent of the initial gas mass;
it is not additional mass. Both this tracer and chemical energy advect with the
donor concentration. Generated gas currently carries no oxidizer. At each step:

```text
available = min(old_gas_chemical, old_oxidizer_mass * oxidizer_heat_j_per_kg)
Qburn = available * (1 - exp(-reaction_rate_per_s * dt))
Qburn = min(Qburn, chemical_after_transport, oxidizer_after_transport * Hoxygen)
gas_chemical -= Qburn; gas_sensible += Qburn
oxidizer -= Qburn/Hoxygen; consumed_oxidizer += Qburn/Hoxygen
```

No heat is released below `ignition_temperature_k`; zero means an explicitly
already-ignited/effective compartment, **not predicted ignition**. Newly arriving
fuel reacts on the next substep. A fixed R/cv pseudo-mixture and unchanged total
gas mass are used; species stoichiometry and changing mole counts are unresolved.
13.1 MJ/kg oxygen is a conventional-fuel proxy supported by
[NIST's oxygen-consumption discussion](https://nvlpubs.nist.gov/nistpubs/sp958-lide/280-282.pdf),
not measured battery-gas calorimetry. Oxygen-limited extinction/flammability
kinetics beyond the finite-reserve and temperature limits are not modeled.

There is no flame geometry, radiation from a flame, droplet/solid ejection,
jet erosion, ablation, variable gas volume or mechanical propagation. These
omissions matter for module propagation. A finite-capacity solid wall connected
through ordinary gas thermal contacts can now absorb and later release heat;
the new UPV trial removes measured hot ambient forcing instead of counting it
again alongside modeled gas exposure. See [the gas-energy trial](gas-energy-calibration.md).

## CUDA substep and conservation

Fluxes/rates are evaluated on old states; shared fluxes are consumed by both
recipients. CUDA advances solids and gas, commits transfer ledgers, applies
panel rupture, swaps buffers and computes macro reductions. The host controls
time/event synchronization, not the per-volume constitutive solve.

Thermal CFL uses a lower bound on composition capacity and an upper conductance
bound. Ordinary contacts remain in the cached static bound. Aperture transport
is proposed first from committed pressure/damage/temperature; its **active**
conductances are summed per receiving volume in the existing stability kernel.
The combined safe limit is
`dt <= 1 / (1/dt_static + sum(G_aperture)/(safety*Cmin))`.
Thus a closed high-conductance aperture does not throttle the solver, while
multiple open patches sharing one solid capacity cannot overspend it. Accepted
mass capping can only lower these proposed conductances. No extra host barrier
or kernel launch is needed. Reaction and gas flow impose additional fraction/temperature limits;
multiple outlets compete against the same gas inventory. The two-stage reaction
now uses its exact coupled solution at frozen temperature, including same-step
conversion of new intermediate. Nonlinear thermal coupling remains first-order
and needs step refinement. Total converted mass and source heating are limited;
a tiny fast intermediate no longer forces a global lifetime-sized step. Channel
mass transfer is capped at pressure equalization for the current step, with a
shared-node degree bound for multiple channels. See the
[casing calibration trial](casing-calibration.md) for the stiffness diagnosis.

With `released_net = gross_completion_heat - absorbed_activation_heat + gas_burn_heat`:

```text
reactive_solid_mass + gas_mass + escaped_mass = initial tracked mass
Usolid + Ugas - Uinitial = released_net - boundary_heat - escaped_enthalpy
solid_chemical + gas_chemical + escaped_chemical + released_net = initial_chemical
remaining_oxidizer + consumed_oxidizer + escaped_oxidizer = initial_oxidizer
```

Fixed casing/inert masses cancel and are not included in the tracked mass sum.
Existing energy-only legacy outlets, if present in other objects, also have
their heat/potential ledgers included. The probe exposes absorption and gross
completion separately. `Simulation::stored_energy_j()` and remaining-chemical
convenience totals are still **solid-only**; `process_diagnostics()` and the
balance residuals include the gas network and external transfers.

The viewer lists casing objects and adds gas pressure, temperature, mass,
channel status and escaped-energy diagnostics. The gas cavities do not fill
empty space with a temperature field; they are lumped states, not CFD voxels.

## Reproduce checks

```bash
ctest --test-dir /tmp/battery-sim-cuda-v1 --output-on-failure
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/check_shell_gas_network.py --output-dir /tmp/battery-shell-gas-check
/tmp/battery-sim-cuda-v1/battery_sim_viewer examples/shell-gas-network-demo.json
```

Use a fresh verification directory; the script refuses to overwrite a prior run.
It saves three step resolutions, conservation residuals, final temperatures,
pressures and rupture times. This is **numerical verification, not experimental
validation or a fit**. Tests also cover exact casing volume, nonmatching face
areas, isolated gas blowdown versus its adiabatic analytic solution, two-stage
kinetic convergence, mesh-independent length, cooling/reheating without healing,
rupture before fuel depletion, receiver damage, reset reproducibility and shared
multi-outlet inventories, and internal flow reversal under receiver heating.
Invalid bypass contacts/overlapping chemistry fail closed.

Historical verification BEFORE the coupled-chain/equalization updates
(2026-10-13, illustrative 6-second example, RTX 4060 Ti):
52,217 / 104,091 / 208,018 substeps. Between the last two resolutions the largest
final solid-temperature difference was 0.00266 K, gas-pressure difference 51.5 Pa,
and rupture-time difference 0.0000989 s. Peak ledger residuals across all runs:
mass 3.8e-17 kg, thermal energy 6.6e-11 J, chemical energy 7e-12 J. These measure
discrete consistency, **not physical prediction accuracy**. Near pressure
equalization the explicit orifice flow can oscillate in sign; final pressure
and small shell-temperature differences are not strictly monotone under step
refinement. The example does not establish accuracy for stiffer/larger networks.
Diagnostic wall times include per-substep downloads and are not a solver benchmark.

The Python suite with all optional GPU probes enabled ran 87 tests: 85 passed;
two schema-library checks skipped because `jsonschema` is not installed. Native
JSON parsing and invalid-contract tests still ran. No dependency was installed.

Next experimental work: identify casing material/thickness and partition measured
capacity without changing the measured total; bind external sensors to actual
surfaces; check pre-failure thermal transport first. Then fit gas generation,
vent/rupture thresholds, channel area and energy partition against suitable
pressure, mass-loss, calorimetry or vent-timing data. Temperature traces alone
do not uniquely determine these parameters.

## Stochastic whole-pixel shell rupture (opt-in)

`examples/stochastic-shell-pixels-demo.json` uses a 3x3x3 battery interior and
24x24 elements on **each** casing face. `shells[].surface_divisions` specifies
counts along the three global axes; the face-normal count is always one. Its
default is to inherit the contents grid, preserving older scenarios. Refinement
does not change casing mass, total heat capacity, thickness or contact area.
Existing nonuniform initial fields must be regenerated on the new grid; there
is no implicit interpolation of a frozen coarse field.

Select `damage_opening.mode = "stochastic_pixels"`. A pixel has continuous
thermal damage `D_T`, but its aperture is strictly closed/open. The minimal law
is:

```text
Pcrit = P0 * max(0.01, 1-D_T) * (1-seam_fraction*(1-seam_strength_factor))
lambda = pixel_rate_per_s * max(0, deltaP/Pcrit-1)^pixel_pressure_exponent
Pr(failure during dt | intact) = 1-exp(-lambda*dt)  [frozen lambda]
H(t) = integral(lambda dt);  E = -log(U), U in (0,1), drawn once per pixel
candidate failure: H >= E and deltaP > Pcrit
Aopen = sum(area of fully failed pixels)
```

The seam overlap weakens the **whole pixel's threshold**; it no longer opens a
fractional seam inside that pixel. Explicit exponential temperature softening,
progressive `damage_onset`, aperture `exponent`, and mechanical damage kinetics
are not used in this mode. Temperature acts through the existing thermal-damage
law. The 1% residual pressure barrier is retained, not newly calibrated.

`seed` plus channel ID and local pixel index produce independent exponential
thresholds via a stateless integer hash. Thresholds are generated once on the
host, not redrawn by GPU steps. The transactional state stores `H`: rejected
steps cannot advance it, reset restores it, and CUDA block size does not alter
the draw. A fixed seed represents one reproducible stochastic realization,
not a deterministic physical failure law. With the same mesh, timestep
refinement compares that same realization. Changing mesh changes the set of
pixels; compare ensembles/statistics, not identical hole locations. A per-pixel
hazard gives a resolution-independent expected failed area at uniform prescribed
pressure, but not resolution-independent variance or first-hole timing.

The step is bounded by the earliest remaining pixel-clock event, using the
committed pressure/damage snapshot. The endpoint rechecks pressure, pays work,
and latches complete pixels. The next transaction sees the new aperture and
recomputes gas outflow and hazards. This is an explicit, event-limited coupling;
changing pressure/temperature still requires a timestep convergence check.
Below threshold clocks pause; holes remain open and pass pressure-driven flow.

New area costs `fracture_energy_j_per_m2 * max(0.01,1-D_T) * pixel_area`.
Work is debited once from gas energy and recorded separately from advected
enthalpy. Simultaneously due pixels sharing a donor form an energy batch. If
the available energy cannot pay the batch, it is deferred rather than partly
opening pixels or selecting them in thread/index order. Such energy-limited
cases require checking the fracture-work/inventory assumptions. The model
opens a gas path but does **not** transport casing fragments: destroyed solid
inventory/energy remains in the existing damaged-material thermal description.
Gas routing still follows the declared panel-to-compartment channels.

`pixel_rupture.hpp` owns sampling/hazard functions; `shell_kernels.cuh` owns
transactional clocks and event-time bounds; `shell_event_kernels.cuh` owns
whole-pixel acceptance and energy accounting. No Monte Carlo sampling was added
to ordinary thermal or gas transport. `mode = "binary_pixels"` remains available
as a non-random threshold comparison; `"progressive"` preserves the older law.

The synthetic CUDA regression checks the probability law, no subthreshold
failures, first-pixel opening, fixed-seed dt refinement, 64/256-thread identity,
discard/retry/reset, damage weakening, contact-area conservation, and energy
limits. In its physical-outflow case, 7 of 576 pixels open before pressure
equalizes; the remainder stay intact. These are implementation checks, **not
experimental calibration**. The previous V22 progressive-law failure is not
evidence for or against convergence of this new stochastic mode.

## Progressive whole-shell opening (opt-in)

`examples/progressive-shell-opening-demo.json` connects **all six casing panels**
to a prescribed receiving cavity. No panel is intrinsically unbreakable. Each
panel has one full-face rupture channel and optional conventional vents. The
channel's `damage_opening` object selects the new closure; omitting it preserves
the historical instantaneous rupture model and existing calibration inputs.
Mixed or multiple rupture laws on the same progressive panel are rejected.

A physical perimeter band `seam_width_m` is weaker by `seam_strength_factor`
(0 < factor <= 1). Its exact intersection area with every thermal patch is
computed on the host. A coarse boundary voxel is NOT entirely weak. Each patch
stores thermal weakening `D_T` and two irreversible mechanical integrity
variables, `D_bulk` and `D_seam`, at a shared temperature. Zero width disables the weak band. This is a generic perimeter
seam assumption, not an assertion that a particular manufactured case has
welds on every edge. Configure no band on panels without such seams.

The phenomenological pressure-damage closure for each sub-area is:

```text
D_effective = D_T + (1-D_T)*D
strength = P0 * strength_factor * max(0.01, 1-D_effective)
           * exp(-max(0, T-Tref)/temperature_scale)
rate     = pressure_damage_rate * max(0, (Pinside-Poutside)/strength - 1)
Dnext    = D + (1-D) * (1-exp(-rate*dt))
f(D)     = clamp((D-damage_onset)/(1-damage_onset), 0, 1)^exponent
Aeff     = Apanel * area_mean((1-seam_fraction)*f(D_bulk) + seam_fraction*f(D_seam))
```

The exponential temperature factor has the same -80 numerical exponent clamp
as the legacy strength law. Thermal degradation advances `D_T`, NOT the
mechanical histories: even `D_T=1` does not make an aperture without pressure.
Only pressure above the residual strength advances mechanical opening. The
thermal node sees `D_T + (1-D_T)*area_mean(D_mechanical)`, so existing conductivity
and capacity closures still see structural loss. After damage, temperature is reclosed
from **unchanged stored energy**; damage is not an added heat source.

`Apanel` is the **physical full panel face**, validated against its geometry.
Progressive rupture has no independent maximum-orifice parameter. Normally omit
`area_m2`; if explicitly supplied it must equal the physical panel face. A small
fitted ceiling is rejected rather than silently replayed as the new model.
Fixed-area vents and legacy instantaneous orifices retain their explicit area.

The feedback is damage → exposed panel area → gas discharge → reduced pressure
load → arrest of further pressure-driven damage. Area is irreversible: pressure
relief stops growth, not closes the hole. Continued gas production may reload
the same panel and restart damage. Material strength and the finite work budget
still govern rupture kinetics; the only geometric upper bound is the panel
itself. This is not a resolved moving crack or detached-flap model.

The isolated pressure-feedback regression starts at 300 kPa against 100 kPa
outside, with physical discharge coefficient 0.7 and no thermal damage. Pressure
falls to 100 kPa and the total opening arrests at approximately 0.594 mm²;
damage/area remain unchanged from 0.05 to 0.1 s. Half-step refinement changes
the final area by less than 2%. This is numerical verification of the feedback,
not an experimental rupture-area measurement or an identified material law.

A larger aperture raises flow **at the same pressures and gas temperature**;
depressurization can reduce actual flow even while damage and aperture increase.
Gas mass, enthalpy, chemical potential and oxidizer use the existing conservative
channel flux. Internal channels can reverse after opening. Negative pressure
difference does not cause new damage: inward buckling is outside this closure.

The old-state pressure proposal also limits dt by `max_damage_step` (default
0.02 absolute increment of D). A nearly destroyed patch cannot consume more
than its remaining integrity, so it does not force arbitrarily short steps.
Thermal degradation retains its separate limiter.
While an aperture grows, `shell_feedback.cuh` also resolves pressure unloading
through its **prospective** area. A Lipschitz bound gives
`Aend <= Aold + area_rate*dt`; the positive root of
`unit_orifice_flow*(Aold + area_rate*dt)*dt = mass_fraction*pressure_equalizing_mass`
provides a safe starting bracket. A bounded monotone solve then uses the actual
frozen-rate `Aend(dt)` instead of this overestimate (32 bracket expansions,
40 bisections). This avoids the overly small v18 steps near a quadratic opening
onset. If the conservative growth bound adds at most 1% to the existing area,
the analytic bound is used directly, avoiding needless iterations. This is only
a performance shortcut to a stricter timestep, not an approximation to fluxes.
The mass fraction is independent of the thermal CFL safety factor: halving
the mass budget halves the allowed pressure relaxation, without an extra
thermal factor accidentally quartering it.
For very slow aperture evolution, an alternative timestep is
`1e-3*mass_fraction*Aold/area_rate`. It bounds relative area growth by 0.01%
at the UPV trial's mass fraction of 0.1 (0.001% at the model default of 0.01),
and halves under mass-step refinement. The solver
may satisfy either this geometric budget or the pressure-unloading budget;
fixed-hole equilibration still uses the conservative pressure/mass cap. This
avoids resolving a fast gas timescale when the barrier is effectively fixed.
The alternative vanishes at zero area, so it cannot skip the initial opening.
This numerical approximation must pass the explicit aperture/work/mass
refinement gate; temperature agreement alone is insufficient.
A fully closed panel can first land on its opening onset.
The bound is checked again after the thermal operator; any stricter budget
retries the complete split transaction. Fixed/unloaded apertures do not incur
this extra bound, so pressure-equilibrium chatter is still handled by the
ordinary conservative mass cap. No constitutive coefficient is fitted here.
Remaining integrity below 1e-10 is closed to exact D=1. This is a numerical
terminal convention, not a fitted fracture parameter: otherwise legacy thermal
connectivity waiting for D=1 can stay intact at D=0.999999999998 indefinitely.
The accepted step updates integrity and aperture, used by gas transport on the
next substep. Repeated prepare/discard does not alter committed shell history;
commit swaps both shell state buffers. There is no additional host synchronization
and no floating-point atomic damage/flux accumulation. Pointwise CUDA work may
be tiled at 64, 128 or 256 threads with the same fixed-order gathers.

Laws are in `shell_opening.hpp`, evolution/area gathering in `shell_kernels.cuh`,
gas transport remains in `gas_kernels.cuh`, and execution ordering remains in
`cuda_step.cuh`. Viewer and reaction probe expose effective area, aperture
fraction and (probe) maximum bulk/seam damage. Area is the **end-state** value;
reported flow is the last accepted substep's rate.

Tests cover six-side damage, weaker seams, area invariance under 3/4/8 subdivision,
pressure and thermal loading, arrest without healing, variable-capacity energy
closure, flow reversal, refinement, conservation and transactional replay.
These are numerical checks, **not experimental validation**. Strength/rate/area
parameters in the new demo are synthetic; no frozen calibration was overwritten.

Still omitted from the original opening-only closure: elastic deformation, fracture surface work, crack interaction
between adjacent panels, mechanical delamination, detached panel motion, changing
gas volume, directed jets and liquid/ejecta transport. The weak band is a local
sub-grid integrity model, not a resolved crack. Gas connectivity to each receiving
cavity must be specified explicitly; it is never inferred through neighbouring
solids. Fit opening/strength parameters to pressure and vent data before claiming
predictive destruction performance.

## Aperture-exposed surface connections and opening work

Implemented as an opt-in extension of existing gas/solid thermal patches. There
is no jet object, timer, empirical heat pulse or extra chemical energy source.
The synthetic example is `examples/aperture-surface-demo.json`; previously fitted
UPV scenes are unchanged. The synthetic example is numerical verification, not
experimental validation. A separate [v10 calibration and held-out replay](gas-energy-calibration.md#v10-aperture-directed-transfer-with-paid-opening-work)
now exercises this extension; it does not replace an accepted profile.

### Where to read the code

- `native/src/aperture_geometry.cpp`: first-surface visibility and exact projected
  rectangle overlap, including partial occlusion and unequal grid resolutions.
- `native/include/battery_native/aperture_transfer.hpp`: pure conductance law.
- `native/src/aperture_kernels.cuh`: local opening, directed flux, channel ledger.
- `native/src/shell_work_kernels.cuh`: work budget and energy-limited opening.
- `native/src/cuda_step.cuh`: proposal/acceptance order only.
- `native/tests/aperture.cpp`: geometry, conservation, work, refinement, tiling,
  repeated prepare/discard, an intentionally energy-starved aperture, closed
  aperture CFL independence and many-to-one active aperture stability.

### Geometry and instantaneous conductance

Add `"surface_link": {"h_w_per_m2_k": ...}` to a progressive gas channel. The
coefficient has the same effective gas-to-solid-centre meaning as an ordinary
gas thermal contact. It is not a measured jet coefficient and must not silently
be treated as calibrated. The outward panel normal defines the direction.
Projection is partitioned onto the first solid surfaces; a nearer obstacle
removes its footprint from consideration for all farther targets. Empty space
does not acquire a solid heat link. Potential geometry is built once, while
local damage and conductance are evaluated on the GPU every substep.

For a visible area fraction `w`, local aperture fraction `a`, panel mean `abar`,
physical panel area `Apanel`, and positive accepted outward gas rate `mdot`:

```
flow_share = w * a / abar
G = min(h * Apanel * w * a, mdot * (cv + R) * flow_share)
Qsurface = dt * G * (Tupstream_gas - Tsolid)
```

Zero opening or nonpositive outward flow gives zero directed enhancement.
Cooling suppresses the temperature driving force without healing the hole.
The local aperture uses the same bulk/seam damage law as gas discharge, rather
than a channel-wide mean damage. Seam location within one shell tile remains
unresolved, consistently with the existing sub-grid seam model.
The host helper `gas_channel_area` supplies the **same** geometric normalization
to CUDA flow/fracture work and to surface-contact construction. No new per-step
kernel, floating-point atomic or host synchronization is introduced.

V9/v10 progressive scenarios with fitted `area_m2` ceilings are historical and
intentionally fail validation in this corrected model. Their coefficients must
not silently be promoted. `scripts/check_upv_pressure_feedback.py` removes only
that ceiling from the saved v10 training scenario and checks the first 30 s plus
time refinement; this is a diagnostic, not a recalibration or validation claim.
The first attempted 30 s / 3 s dense replays were stopped for run time, so no
new UPV calibration or full transient refinement is claimed. A bounded
30000-step progress check reached 0.41583 s, 95.46 mm² total opening and a 3.04 µs
last step. This exposes numerical stiffness with the inherited coefficients;
see `data/diagnostics/upv-geometric-rupture-first-3s/status.json` for the explicitly
incomplete result. The static fully-open aperture CFL artefact was corrected,
but it was not the only source of short explicit steps.

Each accepted channel packet loses `Qsurface` before its energy is credited to
the receiving gas compartment. The solid receives exactly that quantity.
Upstream gas is charged only its existing outgoing enthalpy; it is **not**
charged again for surface heating. Chemical energy and mass are not deposited
on the surface. An ordinary compartment contact, if configured, describes the
subsequent ambient exchange and uses the remaining compartment energy. Do not
keep an old fitted artificial *direct-discharge* contact as a duplicate path.

### Opening is not energetically free

`damage_opening.fracture_energy_j_per_m2` specifies effective irreversible work
per newly opened area, including unresolved fracture/plastic deformation:

```
delta_W = fracture_energy_j_per_m2 * max(0.01, 1-D_T_old) * delta_Aopen
Ugas -= delta_W
Wfracture += delta_W
```

This coefficient is not automatically the material's fracture toughness: the
model uses effective aperture area, not resolved crack area or crack length.
Its value needs material/rupture data. Zero retains legacy numerical scenarios;
it must not be interpreted as a calibrated physical assumption. An explicit
`damage_opening.initial_mechanical_damage` sets pre-existing opening history;
it is not charged again. Ordinary cell `initial_damage` means thermal weakening
and no longer initializes an aperture. This intentionally changes old scenarios
that used thermal damage to stand for a pre-existing hole.

Work is supplied by the internal gas pressure reservoir, separately from the
enthalpy exported through the hole. It remains in the irreversible structural
work ledger, not returned as a new heat pulse. No elastic reservoir, mechanical
motion or plastic-work-to-heat partition is claimed in this closure.

Only pressure-driven opening work is included in the gas timestep budget.
Thermal weakening does not spend the gas mechanical-work budget. After candidate thermal/gas updates, all outgoing
panels share the remaining gas energy budget. If it is insufficient, a fixed
52-iteration monotone solve scales only their proposed mechanical increments, then
recomputes aperture and constitutive temperatures. A 1 K positivity floor is a
numerical guard for pathological inputs, not an inferred material threshold.
Ordinary runs should remain far from that guard.

The audited balance is now:

```
Usolid + Ugas + Wfracture - Uinitial
    = released_net - boundary_heat - escaped_enthalpy
```

`surface_energy_j` records the signed cumulative heat diverted to surfaces;
`delivered_gas_energy_j` is channel enthalpy minus that heat; `fracture_work_j`
records irreversible work. Both native probes expose these separately.
All candidate ledgers are transactional: discard/retry does not spend energy.

### Explicit scope limits

The first version supports axis-aligned rectangular solids and progressive
full-panel openings, matching the physical-shell builder. Rotations, intersecting
aperture geometry, isothermal receiving solids, and implicit outside-domain gas
receivers are rejected instead of guessed. The geometric target is the nearest
remaining solid; damage alone does not delete its mass or make it transparent.
There is no angular spreading, deflection around corners, moving debris, liquid
entrainment or explicit liquid mixing. Reverse gas flow remains supported by
the existing gas network but does not activate an outward surface connection.

Run the synthetic checks with:

```bash
cmake --build /tmp/battery-sim-cuda-v1 --target battery_sim_aperture_test battery_sim_reaction_probe -j 4
ctest --test-dir /tmp/battery-sim-cuda-v1 -R cuda_aperture_connections --output-on-failure
/tmp/battery-sim-cuda-v1/battery_sim_reaction_probe examples/aperture-surface-demo.json
```

## Finite-capacity cooling wall

The UPV cooling plate is now constructed by `scripts/upv_cooling_wall.py` as an
**ordinary passive material**. There is no new wall-specific CUDA kernel or
fluid solver. Its temperature is evolved from stored energy, not clamped to
channel 110. Three directed contacts connect the Al1/Al2/Al3 tips to disjoint
rectangles on the plate. The existing mesher resolves the overlaps even though
the plate and spreaders use different grids. The same local state/geometry law
selects each directed conductance; forward/reverse responses are not averaged.

The plate has D(0)=0, zero damage rate and zero chemical energy. Surface and
bulk thermal responses remain available independently of damage. The default
surface response inherits the scene's aluminium response; the bulk response
defaults to zero structural variation, preserving reference k=235 W/(m K).
Nonconstant, anisotropic bulk response can be supplied explicitly, without
changing the solver. A temperature dependence is not invented as measured data.

The only external sink is on the plate's lower face:

```
dUplate/dt = sum(Qcontacts) + Qgas - Qwater
Qwater = Gwater * (Tface - Twater)
```

The boundary's G is constant; the power is not. The ordinary half-volume
conduction resistance remains in series with that boundary. Setting G=0 closes
the water path but retains heat storage and transfer through the plate. A
larger G represents more effective cooling, not an imposed plate temperature.
The fixed-temperature external reservoir approximation permits reverse heat flow if the
plate becomes colder than the supplied water temperature.

Current UPV construction:

- Thickness 13 mm, aluminium k=235 W/(m K), rho=2700 kg/m³ from the dataset.
- Footprint 70×87 mm, inferred from the stack clamp extent (not measured CAD).
- cp=900 J/(kg K), a fixed prior; resulting capacity 192.3831 J/K.
- Grid 7×3×1: 21 ordinary thermal volumes, one visually continuous object.
- Reference tip/plate h=10000 W/(m² K), explicit uncalibrated prior. The
  directed contact ranges between 0.01× and 2× that reference as its endpoint
  response changes. This is not an independently measured contact law.
- Water-side G=10 W/K, explicit uncalibrated prior. The initial inlet reading
  (channel 111, about 24.5 °C) supplies a constant reservoir temperature.
- No ordinary gas film is added to the new plate. Existing geometric rupture
  rays can now hit it where it is the nearest visible solid; this is handled
  by the same aperture path and energy ledger, not a duplicate heat source.

Channel 110 is now a diagnostic observer on the Al2 tip side of its plate
contact; it is neither imposed on the three contacts nor used to choose the
new generator's valid forcing window. Its precise experimental placement
within a finite contact temperature jump remains an observation-model caveat.

`generate_upv.py` defaults to `--cooling-model thermal-wall`; explicit
`--cooling-model legacy-boundary` reproduces the historical ch110 boundary
construction. Existing saved scenes and calibration results are not overwritten.
The casing builder leaves passive-material observers intact; boundary refresh
preserves the fixed coolant reservoir instead of replacing it with ch110.

`scripts/replay_upv_cooling_wall.py` first reruns the complete intact warmup,
then transfers the wall, witness, passive-solid and shell temperature fields
to the post-event scene. Only the existing event-conditioned C2 initialization
and chamber initial conditions are retained from the prior destruction scene.
Starting the new plate cold at the explosion while keeping old witness fields
would be inconsistent and is deliberately avoided.

The first replay is an **implementation check, not a recalibration**. With
G=10 W/K and the previous destruction coefficients, the plate ends warmup at
301.919–303.366 K; channel 110's model-side contact is 334.546 K versus the
measured 314.657 K. Late witness RMSE is 20.30273 K, compared with 13.62616 K
for the old prescribed-contact construction. This is not an improvement and
does not validate these unmeasured cooling coefficients. No parameter was
selected by minimizing the held-out experiment's temperatures.

Artifacts: `data/calibration/upv-cooling-wall-v1/{warmup-scenario.json,
warmup.json,active-cooling-scenario.json,replay.json,provenance.json}`.

```bash
PYTHONPATH=src python3 scripts/replay_upv_cooling_wall.py --output NEW_OUTPUT --sink-g 10 --contact-h 10000
./run_viewer.sh data/calibration/upv-cooling-wall-v1/active-cooling-scenario.json
cmake --build /tmp/battery-sim-cuda-v1 --target battery_sim_cooling_wall_test
ctest --test-dir /tmp/battery-sim-cuda-v1 -R cuda_cooling_wall --output-on-failure
PYTHONPATH=src python3 -m unittest discover -s tests -p test_cooling_wall.py
```

The native test covers finite heat storage, zero damage, constant-G analytic
cooling, state-dependent conductance, conservative exchange, and identical
committed results after discard/retry and with different CUDA block sizes.

### Training-only recalibration protocol

The no-cooling experiment has natural convection and **no liquid-cooling
plate**. Adding the new plate to that training geometry would be incorrect.
Likewise, fitting its water-side conductance using the active-cooling
temperature traces would make that experiment calibration data, not a holdout.

`scripts/refit_upv_training.py` separates the two stages explicitly:

1. Refit intact links on no-cooling only, using the physical-casing builder.
   All measured capacities and reference conductivities stay fixed.
2. Transfer those links to the event scene, including the C2 source contacts
   and the shared internal response. Recompute the full warmup and transfer
   witness, passive-solid and shell fields. C2 remains event-conditioned.
3. Evaluate five broad destruction regimes and one interpolant-proposed
   candidate with the native solver. Only chemical energy partition, common
   gas exposure and reaction rate vary; total chemical energy is fixed to
   the earlier training profile. This total is not a calorimetry measurement.
   Selection uses the existing joint temperature/pressure objective, not
   temperature RMSE alone. Halve numerical budgets before freezing.
4. Only a separate `validate` invocation loads the active-cooling record.
   It adds the finite-capacity plate and evaluates **all** predefined water
   conductances, 3/10/30 W/K, with the same frozen failure parameters. No
   cooling case is selected. Contact h=10000 W/(m² K), the footprint and cp
   remain explicit fixed priors, not independently measured quantities.

The sensitivity series is not a confidence interval or an identified cooling
law. Flow rate alone does not identify plate/contact thermal resistance.
Previously inspected holdout data also cannot be described as blind validation.
The source temperature forcing and event-conditioned initialization mean this
remains a conditional transport/destruction test, not an ignition prediction.

```bash
PYTHONPATH=src python3 scripts/calibrate_upv_casing.py heat fit \
  --output data/calibration/NEW_TRAINING_RUN \
  --initial-profile data/calibration/upv-physical-casing-v7/heat/profile.json \
  --max-evaluations 6
PYTHONPATH=src python3 scripts/check_upv_training_heat.py data/calibration/NEW_TRAINING_RUN
PYTHONPATH=src python3 scripts/refit_upv_training.py fit --output data/calibration/NEW_TRAINING_RUN
PYTHONPATH=src python3 scripts/refit_upv_training.py validate --output data/calibration/NEW_TRAINING_RUN
PYTHONPATH=src python3 scripts/report_upv_training_refit.py data/calibration/NEW_TRAINING_RUN
```

The new output directory must not contain a previous manifest/profile. Old
results stay immutable. The manifest records source/profile hashes, native
fingerprint, the complete training design and cooling cases before any
held-out replay. `tests/test_training_refit.py` checks training-only loading,
fixed material/geometry/state preservation, and fixed chemical inventory.
The heating numerical check halves both the CFL stability factor and max_dt:
reducing max_dt alone leaves a CFL-limited calculation unchanged and therefore
does not demonstrate convergence. It also checks that the step count increases.

Completed run: `data/calibration/upv-training-refit-v13/`. Intact no-cooling
RMSE is 1.028606 K. The seven full destruction candidates (including the
baseline and interpolant proposal) reduce late witness RMSE only from
6.553869 to 6.544515 K and the joint objective from 10.932876 to 10.924632.
This is not a meaningful improvement or an accepted destruction calibration.
The chosen energy split is 164.763 kJ solid / 220.384 kJ gas, at unchanged
total 385.148 kJ; common exposure is 4.840634 and reaction rate 4.047840/s.
No held-out temperatures selected these values.

| Water-side G, W/K | Held-out warmup RMSE, K | Held-out late witness RMSE, K |
| --- | --- | --- |
| 3 | 1.3023 | 18.9924 |
| 10 | 1.3167 | 20.1965 |
| 30 | 1.3412 | 20.6247 |

All four held-out late witness biases are negative. Varying G tenfold within
this declared range does not remove the large discrepancy. This is a
sensitivity result, not proof that an arbitrarily different G or an uncertain
contact/footprint could never matter. The model is not validated by these runs.
The predicted chamber peak is about 6.344 bar versus 7.856 bar measured;
on training it is 6.420 versus 7.983 bar. Pressure is absolute and evaluated
on acquisition times, not a claimed dense-time peak measurement.

Thermocouple 110 adds a separate warning: at the end of warmup its reading is
41.507 °C, whereas the modeled Al2-side surface is 65.927/61.396/59.523 °C
for G=3/10/30. Its exact experimental side within the contact jump remains
unconfirmed; these residuals do not uniquely identify a contact resistance.

Actual step refinement changes warmup temperatures by at most 0.000905 K
(15028 to 29614 steps), and destruction witness temperatures by 0.04821 K;
the chamber-pressure change is 0.001672 bar. Across training and all held-out
cases maximum thermal ledger error is below 1.35e-7 J and mass ledger error
below 1.9e-14 kg. These checks establish numerical consistency, not completeness
of the physical model. The focused Python regression suite passes 25 tests.
Graphs are `witness-comparison.png`, `pressure-comparison.png` and
`contact-comparison.png`; machine-readable results are in `assessment.json`.

## Stochastic partial-area rupture (experimental, not a calibrated material law)

`damage_opening.mode = "stochastic_area"` replaces whole-pixel rupture with
repeated marked events. The computational patch remains a thermal control
volume; it is not a prescribed physical fragment. Its open fraction `a` is
irreversible and lies in `[0,1]`. The two independent random variables are:

1. An exponential waiting threshold for the integrated event intensity.
2. An exponential fragment area with mean `fragment_mean_area_m2`.

For patch area `A`, mean fragment area `Af`, thermal damage `D`, and the
existing seam-area strength factor `s`, the proposed closure is:

```text
Pcrit = P0 * max(0.01, 1-D) * s
lambda = rate * (A/Af) * (1-a) * max(0, deltaP/Pcrit - 1)^exponent
P(event over dt with frozen state) = 1 - exp(-lambda*dt)
deltaA = min(A*(1-a), -Af*log(Uarea))
a_new = a + deltaA/A
fracture_work = G * max(0.01, 1-D) * deltaA
```

Below the pressure threshold the intensity is exactly zero, even at `D=1`.
As intact area disappears, event intensity decreases. The `A/Af` factor keeps
expected removed physical area approximately mesh-independent when fragments
are much smaller than patches. Clipping at the remaining patch area still
introduces a finite-size effect: spatial convergence must be measured, not
assumed. This distribution and the linear intact-area factor are modeling
assumptions, not measured casing statistics.

Each patch stores a cumulative hazard and committed event counter. Hashing
seed/channel/patch/event/stream supplies separate waiting and area marks.
An attempted or rejected CUDA transaction does not redraw either mark.
The proposal limits the step to the next event for frozen rates; pressure and
damage evolution still require timestep convergence checks. Insufficient gas
energy defers the proposed event batch without redrawing or resizing it.
Fracture work is debited once, only when new area opens; the next transport
transaction sees that aperture. Gas advection and its enthalpy ledger are
unchanged. No ejected solid-fragment mass transport has been introduced: casing
material/energy remain in the local damaged-solid inventory as before.

Implementation is split between `area_rupture.hpp` (stateless distributions and
rates), `shell_kernels.cuh` (transactional clocks), and `shell_event_kernels.cuh`
(events and conservative work accounting). Existing progressive and whole-pixel
models remain available. The public schema accepts fractional
`initial_mechanical_damage` in this mode.

`scripts/calibrate_upv_area.py check` compares temporal refinement and 64/32 mm
casing meshes. `scan` runs a declared broad, single-seed parameter screen;
`finalize` checks the chosen candidate before running additional seeds and a
frozen second experiment. Thermal inputs and chemistry are never fitted by
this script. Its objective includes all four 3–30 s temperature witnesses, so
a chamber-pressure fit cannot hide the large C3/ISS residual. A low external
RMSE alone does not establish correct internal pressure or aperture dynamics.

## Local free-volume routing (geometry diagnostic, not calibrated)

The old four `near-*` reservoirs instantaneously mixed each entire side of the
stack. Each whole-face rupture also used one central core pore. Neither retained
the position of a hole. The optional `upv_local_mass_paths.attach` transformation
replaces those assumptions without changing the thermal calibration or chemistry.

- `gas_void_geometry.py` builds an axis-aligned, nonuniform grid of free volumes.
  Solid faces are grid cuts; occupied boxes are removed. Only shared faces create
  passages. Edge/corner touching is not a passage. Only outer envelope faces
  connect to the mixed chamber; the chamber remains sealed to the outside world.
- `upv_local_mass_paths.py` supplies the frozen UPV geometry, explicitly assumed
  interfacial slits, optional holder obstacles, and local thermal neighbours.
  No new optimizer is involved. It must run after casing-grid selection.
- `shell_thermal_volume` selects one x-fastest casing voxel on a full-face port.
  Its area, pressure endpoints, stochastic clock, damage and fracture-work ledger
  are local. Overlapping rupture channels are rejected. Omitting the selector
  preserves legacy whole-panel behaviour.
- `kind: passage` connects two exterior gas compartments without pretending to
  be a vent through a material port. It cannot connect a battery interior to the
  exterior. The builder supplies geometric adjacency; the native solver consumes
  that explicit graph and performs the existing conservative pressure transport.
- A gas thermal contact may select `thermal_volume`. This partitions its original
  area while keeping the original film coefficient; it does not multiply hA.

At 32 mm casing resolution the uncooled diagnostic has 54 local rupture channels,
676 gas volumes and 2200 pressure connections. Gas in each pad-face slit can move
in-plane and leave at its own perimeter, never through the intact pad. Manifolds
that distributed a rupture across an entire pad face have been removed. A flush
cooling plate blocks a bottom path unless an actual gap is explicitly provided.
Blocked paths are reported in metadata, never silently moved to another side.

The local exterior envelope extends 20 mm from the stack, with two radial layers;
these are reduced-model choices, not measured confinement. The frozen solid stack
uses full-face layers. The source schematic is not a dimensioned CAD drawing and
does not establish actual holder openings or pad-edge clearances. The original
unmeasured 0.2 mm pad-face slit and pad-voxel LTE approximation remain assumptions.
They are not evidence that silicone is gas-permeable. Hole positions within a
casing voxel are not resolved: its centroid selects the containing core/gap
volume. Pressure-only orifices do not resolve jet momentum, buoyancy or entrained
solid/liquid fragments. This geometry must not be presented as reconstructed
experimental ground truth.

Source re-inspected: [UPV experimental description, Tables 2–3 and Figures 1–2](https://zenodo.org/records/22273467/files/TR_Experimental_Data_Description.pdf),
SHA256 `da8192e47841225391fb30ea86f3dc5588ebb27721bafe6885996951fa7bf405`.
It specifies resin holders but not dimensioned free passages.

Generate with `python3 scripts/replay_upv_local_paths.py --output /tmp/upv-local-gas --generate-only`;
omit `--generate-only` and use `--duration 0.2` for a short CUDA conservation check.
Longer runs and temporal/spatial convergence remain necessary before any new
destruction calibration can be accepted. Old v24 calibration artifacts remain
historical and are not silently rerun against this topology.

### What currently starts destruction

`conversion.hpp` uses `l(T)=l0*sqrt(1+cp*max(T-Tref,0)/energy_scale)` and
`ka=activation_rate*max(l/lcrit-1,0)`. Thus activation has a temperature-derived
threshold; it is not yet an independently calibrated electrochemical instability
criterion. The current UPV values imply about 341.845 K (68.695 °C). Reactant is
converted into an activated inventory and then into residue/gas with a separate
temperature-dependent rate and explicit absorption/release ledgers. Cooling can
stop fresh activation, but existing activated material can continue reacting.

Thermal casing damage accumulates under its own temperature/rate law. Pressure
rupture then requires a damage-dependent pressure threshold and, for the current
stochastic-area mode, a sampled event and affordable fracture work. It is not an
instantaneous `T > threshold => open` switch.

The UPV destruction replay is **event-conditioned**: `failure_prepare` starts from
the experimental pre-event state, disables legacy primary-failure forcing, and
enables conversion in C2. The current uncooled core field is already 386–395 K,
above the activation threshold. This replay does not independently predict the
overcharge-to-runaway time. Thermal fit, post-event fit, and prediction of onset
are distinct claims.

## Fixed-fraction stochastic rupture

`damage_opening.mode = "stochastic_fraction"` keeps the random event clock but
replaces the random fragment area with a fixed fraction `q` of the **original**
local shell patch. `fixed_area_fraction` defaults to 0.02; the diagnostic compares
0.02 and 0.05. These are 50/20 aperture increments without adding thermal voxels.
For an already opened fraction `a`:

```
delta_a = min(q, 1-a)
lambda = rate * max(dp/Pcrit - 1, 0)^exponent * (1-a) / q
P(event over frozen-state dt) = 1 - exp(-lambda*dt)
fracture_work = G * remaining_barrier(Dthermal) * patch_area * delta_a
```

Dividing the intensity by q keeps the frozen-state expected aperture drift
`rate * pressure_excess^exponent * (1-a)` the same for the two resolutions,
except for the final clipped fragment. Open area is irreversible; falling
pressure can stop further events. Insufficient energy defers the event, never
shrinks the fixed fragment. Existing seeded clocks, transactional retries and
conservative mass/energy transport remain unchanged. The random-size mode is
still available for comparisons, not silently reinterpreted.

Unlike physical-area fragments, this q is tied to the numerical shell patch.
Refining the mesh changes the physical event size and stochastic variance;
equal mean opening drift is not proof of mesh-independent failure statistics.

`scripts/calibrate_upv_fraction.py` uses the new local gas graph with all thermal,
chemical and geometrical parameters frozen. `check` compares q=2%/5% at two
step controls for the first 0.2 s. `scan` screens the declared reference,
strong/slow, strong/fast and weak/fast pressure regimes on 0–30 s observations.
The short check is necessary but not sufficient: `finalize` requires the full
selected-trajectory refinement test before seeds or the held-out experiment.
A failed refinement is a failed numerical gate, not an accepted calibration.

## Deterministic aperture: fracture before transport

`damage_opening.mode = "deterministic_aperture"` is the new opt-in law. Historical
stochastic runs keep their original semantics. The new UPV builder matches each
casing face to the contents tangential grid (currently 3×3 per face), conservatively
remaps casing temperatures, and regenerates local gas routes. No thermal or
chemical parameters are refitted.

One irreversible opened fraction `a` determines the aperture, with two stages
on patches containing a weak perimeter seam:

1. Fill the seam band under its reduced pressure threshold.
2. Only after the band is completely open, grow rectangular strips from those
   edges into the remaining patch, under the ordinary material threshold.

At corners, the union of the two strips counts their overlap only once. Growth
is clipped to the patch; it does not damage neighbouring patches. Before the
seam fills, its width grows proportionally; afterwards occupied edges advance
by equal distances. Patches without a seam use a centred disk intersected with
the rectangle. An explicitly specified circular valve instead supplies the weak
first stage and its centre; no valve position is invented for UPV. A local patch
cannot currently declare both a valve and a perimeter seam.

For a frozen state:

```
Pcrit = P_reference * max(0.01, 1-Dthermal)
Pstage = Pcrit * seam_strength_factor   # while a < geometric seam fraction
Pstage = Pcrit                          # afterwards
rate = area_rate_per_s * max(dp/Pstage - 1, 0)^pressure_exponent
da/dt = rate * (1-a)
work = fracture_energy_j_per_m2 * max(0.01, 1-Dthermal) * patch_area * delta_a
```

Below threshold there is no growth, but existing holes do not heal. Integration
uses `expm1` and the exact frozen-state time to reach the seam/valve boundary;
the remainder of a step then belongs to the stronger material. Insufficient gas
sensible energy scales the proposed increments consistently across all outgoing
patches of that source node. Work is irreversibly debited, never counted as heat
in the recipient. This is a reduced constitutive law, not a fracture-mechanics
or CFD reconstruction.

The split-step transaction is:

```
committed pressure, thermal defects, energy
  -> aperture growth + fracture-work debit + local thermal equilibrium
  -> heat exchange / chemistry / local equilibrium
  -> pressure-driven mass and enthalpy transfer / local equilibrium
  -> ledger validation -> commit
```

New temperature-induced weakening is available to the *next* fracture step.
Updated aperture and already-debited energy are used by both transfer operators.
If the new aperture or pressure requires a shorter timestep, the complete
candidate is discarded and all three stages repeat from committed inputs.
The leading-step accuracy bound also couples prospective area to the raw
orifice flow and the pressure excess above the active rupture threshold.
An inventory/pressure transfer cap by itself is not a time-accuracy guarantee:
it may hide excessively large opening increments. Alternatively, the existing
slow-aperture bound permits capped pressure relaxation only while the relative
area change is tiny. These are numerical controls, not fitted strengths.
Preparation/discard does not leak damage, energy or aperture into committed
state. One CUDA thread owns each source's outgoing work claims; no RNG or
floating-point atomic accumulation is used in this operator. Legacy trailing
rupture laws cannot be mixed into this mode's rupture network.

Code is separated into:

- `deterministic_aperture.hpp`: area/shape reconstruction and constitutive law;
- `leading_shell_kernels.cuh`: timestep bound and conservative GPU work allocation;
- `leading_shell_step.cuh`: rollbackable prefix buffers and launches;
- `mass_step.cuh`: operator ordering and retry/commit integration.

The transport network still assigns **one source/receiver pair per patch**.
Within-patch circles/strips do not redirect a jet to a different receiver;
transfer uses their open area. Diagnostic output records disk radius or the four
strip widths (`-u,+u,-v,+v`, tangential axes cyclic after the face normal).
Subpatch routing and rendered holes are not implemented by this change.
The percentage-area rate is a reduced patch law. Temporal convergence does not
establish spatial convergence or transferability of its fitted value to a
different shell grid.

Generate/replay without fitting:

```
PYTHONPATH=src:scripts python3 scripts/replay_upv_leading_aperture.py \
  --output /tmp/upv-leading --duration 3
```

Use `--max-dt 0.00005` and a fresh output directory for a temporal audit.
Compare matching replays with `scripts/audit_upv_leading_aperture.py COARSE FINE
--output REPORT.json`; it reports area, individual patch fractions, pressure,
mass, work and temperature differences without declaring a calibration valid.
The inherited 50 MPa / 20000 s⁻¹ trial values are **not calibrated** for this law.
`cuda_leading_aperture` tests weak-stage priority, clipped geometry, frozen-state
time integration, first-step mass transfer/work, conservation, GPU tile size,
rollback, energy-limited opening and a blowdown timestep refinement.

### V31 implementation audit — not calibration-ready

The shell/core matching-grid builder and deterministic leading operator are
implemented, but the coupled UPV gas-network trajectory is **not temporally
converged**. Artifacts are in `data/calibration/upv-leading-aperture-v31/`.

| Same physical inputs, different maximum dt | Adaptive (cap 0.2 s) | Cap 50 µs |
|---|---:|---:|
| Initial implementation, area at 3 s (mm²) | 1172.03 | 164.99 |
| With pressure-excess feedback, area at 1.6 s (mm²) | 1800.41 | 260.86 |
| With feedback, sampled chamber peak through 1.6 s (bar) | 6.6370 | 6.5896 |
| With feedback, exported mass at 1.6 s (g) | 77.2764 | 77.4253 |

Do not compare the two rows as a controlled accuracy improvement: the rows
have different horizons. Each *column pair* uses identical horizons and physical
parameters. The feedback improves an isolated stiff pressure/aperture test
(0.0263% area difference for caps 100 µs and 0.1 µs with a 0.001 pressure-feedback
budget), but that is not evidence that the complete network is accurate.

The full-network thermal/mass residuals are below 2e-7 J / 7e-14 kg in the
feedback runs. Conservation is therefore passing while trajectory convergence
is failing. The mass operator still uses `min(raw_flow*dt, pressure_mass_cap)`;
its cap ensures conservative bounded transfers, not time accuracy of a whole
network. That coupling, together with source production and changing damage
thresholds, needs a separate temporal audit before any destruction fit. This
audit does not establish that the cap is the sole cause.

`numerics-initial.json` and `numerics-feedback.json` retain the failed comparisons.
No strengths, thermal properties or chemistry were retuned to hide them, and
no frozen calibration profile was replaced.

Verification status: 23 selected native tests passed, and `cuda_isothermal`
also passed separately. After the final feedback change, all five directly
affected native tests passed again. The extended native run hit 60 s timeouts
in `cuda_process_network` and `native_reference_fit`. The Python suite ran 166
tests with GPU access: 140 passed, 23 skipped, three legacy native replay
wrappers timed out (`test_native_physical_path`, `test_progressive_whole_shell`,
`test_rupture_mixture_and_chamber`). Thus this is not an all-green whole-project
test claim; some tests ran concurrently with the UPV diagnostic.

### V32 fixed-opening audit — transport error isolated

`battery_sim_gas_transport_audit` runs the existing CUDA mass operator with
fixed passages and no reactions, destruction, solid heat exchange or gas/solid
LTE. Production kernels and calibration parameters are unchanged. Reproduce
with `scripts/audit_fixed_gas_network.py chain|snapshot --output DIRECTORY`;
`scripts/report_fixed_gas_audit.py` collects the saved comparisons. Results and
the plot are in `data/calibration/upv-fixed-gas-audit-v32/`.

The three-volume chain starts at [100100, 100000, 100000] Pa with equal volumes
and temperatures. After the same 20 µs, one 20 µs step leaves pressures
[100075, 100025, 100000] Pa, with exactly zero transfer on the second edge.
At 2 µs the pressure spread is 5.631 Pa; at 0.2 µs and smaller it is below
1e-7 Pa. Mass and energy conservation pass even in the inaccurate coarse run.
Thus changing openings are not required to reproduce the timestep dependency.

The larger audit starts all variants from one stored gas snapshot at 1.1 s
in `upv-leading-aperture-v31/feedback-dt50us`. It preserves gas mass,
temperature, free volume and active route endpoints/areas/discharge coefficients,
then evolves gas only for 2 ms: 676 nodes and 2194 positive-area paths,
including 48 fixed rupture paths. Six zero-area paths are omitted. Since the
production cap counts incident paths even when closed, that omission changes
some cap degrees relative to the original graph. All timestep variants use the
same reduced graph. This is a controlled ablation, not a multiphase restart.

| Maximum step (µs) | Final pressure in `void-6-3-1` (bar) |
|---:|---:|
| 200 | 12.53524 |
| 50 | 11.44981 |
| 0.78125 | 6.54138 |
| 0.1953125 | 6.43522 |
| 0.048828125 | 6.42255 |
| 0.01220703125 | 6.41940 |

The finest run is a comparison reference, not an exact solution. Between the
last two runs the maximum sampled pressure difference is 354 Pa, but the
maximum final gas-temperature difference is still 5.03 K. The largest signed
cumulative channel-transfer difference is 1.139 mg (`void-face-1648`: 5.465
versus 4.326 mg), and this metric is not monotonically converged. Pressure
agreement alone therefore does not establish convergence of advected energy
or individual routes. Mass/thermal residuals stay below approximately 5e-14 kg
and 6e-8 J across the runs.

The identified numerical mechanism is `dm = min(abs(raw_flow)*dt, mass_cap)`
applied once per accepted step using old node states. When the cap is active,
the transferred packet no longer grows with dt, so its effective rate scales
as `mass_cap/dt`. The timestep guard examines capped demands and can accept
a long step when those packets fit the inventory budget, despite unresolved
pressure relaxation through the network. The cap protects boundedness; it is
not a time-accuracy criterion. Raw rejected proposals are not lost mass.

This establishes an independent transport error, not that it explains every
discrepancy in the coupled reactive model. The next correction must resolve
pressure-driven transport in physical time (internal substeps or a coupled
network solve), preserving conservative budgets and damage-before-transfer
ordering. Merely removing the cap or retuning rupture parameters is not a fix.
The three new audit tests pass; one explicitly characterizes the current
coarse-step defect and must be replaced when the operator is corrected.

### V33 adaptive pressure substeps

The gas operator now limits the common accepted physical timestep, on the same
ordered CUDA stream as heat and destruction. No asynchronous clocks, skipped
gas steps or forced instantaneous equilibrium were introduced. The existing
transaction/retry path re-evaluates the limit after leading destruction and
after thermal production/LTE. Destruction therefore still precedes transfers.
Physical orifice laws, material properties and thermal calibration are unchanged.

For a directed edge, let `sd` and `sr` be the positive donor/receiver pressure
response magnitudes per kg transferred, including the local LTE capacity where
bound. Let `nd` and `nr` be their incident degrees. The numerical limits are:

```
B = nd*sd + nr*sr
equalization_mass_cap = abs(Pd-Pr)/B
pressure_tolerance = pressure_absolute_tolerance_pa
                   + pressure_relative_tolerance*max(abs(Pd),abs(Pr))
pressure_dt = pressure_relaxation_fraction
            * max(abs(Pd-Pr), pressure_tolerance)/(B*abs(raw_mass_flow))
```

A zero-flow edge imposes no pressure timestep restriction. The solver takes
the minimum pressure bound over the network, together with outgoing inventory,
temperature, thermal, reaction and fracture bounds. Inventory/temperature
bounds now use raw flow rates, not the smaller capped transfer packets. The
existing conservative packet cap is retained as a safety net. Below the pressure
tolerance it can still activate; the tolerance does not turn mass flow off.
This timescale controller is **not** an embedded local-error estimator or a
guarantee of a global pressure error below the configured tolerance.

The previous degree factor was `max(nd,nr)*(sd+sr)`. That unnecessarily applied
a large chamber's hundreds of connections to a small pore's fast pressure
response. Weighting each degree by its own endpoint sensitivity avoids that
extra restriction without changing physical channel areas or coefficients.
The new CUDA hub regression compares a small source feeding a chamber with
and without 100 additional chamber ports.

Numerical JSON controls in `gas_network` (defaults used unless overridden):

```
"pressure_relaxation_fraction": 0.2,
"pressure_absolute_tolerance_pa": 1.0,
"pressure_relative_tolerance": 0.00001
```

The fraction must be in `(0,0.5]`, absolute tolerance positive, and relative
tolerance in `[0,1]`, all finite. They are numerical controls, not fit parameters.
The small host/device criterion is in `gas_timestep.hpp`; CUDA capacity/degree
evaluation and rate budgets stay in `gas_kernels.cuh`. Scenario loading, native
validation and the JSON schema expose the same controls.

Artifacts: `data/calibration/upv-adaptive-gas-v33/`. The audit builder accepts
`--pressure-fraction`, `--pressure-atol`, `--pressure-rtol`, `--sample-interval`.
`scripts/report_adaptive_gas_audit.py DIRECTORY` collects the comparisons.
The gas-only ablation limitations from V32 (including six omitted closed edges)
still apply; none of these runs is an experimental calibration.

- Three-volume chain over 20 µs: the old single step left a 75 Pa spread.
  The new adaptive run uses 104 steps and leaves 9.91e-6 Pa. Compared with
  the 4000-step run, the largest final pressure difference is 4.96e-6 Pa.
- The 676-node UPV network over the first **30 µs**, with samples every 3 µs:
  default pressure controls and external caps 200 µs / 0.5 ns take 32602 /
  66938 steps. Maximum sampled differences are 29.41 Pa, 0.00798 K,
  and 4.40e-8 kg in the whole-network mass L1 norm. Final maximum pressure
  difference is 5.64 Pa; maximum final path-transfer difference is 6.08e-11 kg.
  Mass/thermal residuals are below 5e-14 kg / 7e-8 J.
- Independently halving all three pressure controls over the same 30 µs takes
  75226 steps. Relative to the default adaptive run, maximum sampled differences
  are 21.32 Pa / 0.00577 K; against the 0.5 ns external-cap run they are
  8.15 Pa / 0.00221 K. This checks numerical-control sensitivity as well as
  external-cap sensitivity, but still only on the short fixed-opening interval.
- A longer **2 ms** trial explicitly relaxes the pressure tolerances to 10 Pa
  and 0.001, retaining fraction 0.2. It takes 164429 substeps and ends at
  6.41855 bar in `void-6-3-1`, versus the old coarse run's 12.53524 bar.
  This is a separate tolerance experiment, not the strict run extended in time.
  Despite close pressure agreement with V32's finest run, final gas temperatures
  differ by up to 18.45 K. Neither the old finest reference nor this relaxed
  trial establishes convergence of all thermal/route histories. Its cumulative
  mass/thermal residuals are about 4.1e-12 kg / 2.8e-6 J.

The default strict controller remains expensive on very small, highly connected
gas volumes: its global timestep is set by the fastest edge. Step counts above
are reproducible workload indicators; wall timings overlap other tests and
are not a controlled speed benchmark. Full-duration coupled UPV destruction
convergence and calibration have **not** been rerun. Do not loosen tolerances
to fit experimental temperatures, or label the full destruction model validated
from these short fixed-opening checks.

Verification: full native build (including viewer) succeeded. Eleven selected
native tests passed: split transport, leading aperture, mechanism consistency,
transaction rollback, thermal solver, aperture connections, shell opening,
stochastic shell, fractional rupture, physical shell, and reaction process.
`cuda_process_network` hit its 90 s timeout while a diagnostic GPU run was also
active; it is **not counted as passed** (this test also timed out in V31).
The Python audit suite passed three tests; its optional JSON Schema test was
skipped because `jsonschema` is not installed. Native invalid-input tests pass.
Earlier trial failures/timeouts were followed by the final targeted test runs;
there is no claim of an all-green full-project suite.

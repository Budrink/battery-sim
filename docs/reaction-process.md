# Process-based failure, without a spatial mass solver

This page describes the retained energy-only kinetic branch. The newer opt-in
[physical casing and gas-network model](physical-shell-gas-network.md) has actual
gas mass/pressure and two-stage material conversion. Its conversion block replaces
this branch's chemistry; the two budgets must not be stacked on one battery.

Status: implemented in the CUDA core and numerically verified. An
[UPV conditional calibration trial](kinetic-calibration.md) now exists, but
chemical reserve and reaction speed remain weakly constrained; this is **not
validated failure physics**. The demo kinetic/structural/outlet numbers remain
illustrative. This is an effective single-reaction closure, not a recovered
electrochemical mechanism. It does not infer chemical reserve from electrical Ah.

## Local chemistry and energy ledgers

Each numerical battery volume has its own temperature, damage, finite chemical
reserve `E`, cumulative chemical release, discharged heat and exported unreacted
chemical potential. The existing 3×3×3 geometry is sufficient to exercise it.
These states and macro reductions run on CUDA, not in Python.

For an activated volume:

```text
k(T,D) = k_ref * exp[Ea/R * (1/T_ref - 1/T) + beta*(D-D_activation)]
P_chemical = k(T,D) * E
dE/dt = -(P_chemical + P_exported_unreacted)
dU/dt = P_chemical - P_vent_heat + sum(incoming link fluxes) - P_boundary_out
```

`k_ref` has units 1/s, `Ea` J/mol, temperature K. `D` and `beta` are dimensionless.
The exponential argument is clamped to ±80 as an overflow guard, not a measured
high-temperature law. Extreme stiffness still reports a timestep failure rather
than silently running an unresolved catastrophic event.

Damage continues accumulating from the existing thermal-aging law during the
reaction. An additional `damage_per_conversion * released/E_initial` increment
represents irreversible reaction damage. Prescribed ignition only raises damage
to `activation_damage` and activates the reaction: no instantaneous heat deposit
or forced D=1. Unactivated volumes reach the same threshold by local aging.
This phenomenological damage threshold is not an identified overcharge mechanism.

Cooling reduces the rate; remaining fuel is not discarded when the reaction is
suppressed. `quenched` is a reporting state for `k < quench_rate_per_s`, not a
hard kinetic cutoff or proof that the cell is safe. Reheating can resume reaction.
`spent` denotes chemical depletion to 1e-12 of the initial local reserve; the
tiny residual remains in the ledger, not silently released or erased. Structural
damage and reserve depletion are separate quantities.

At every step the following identities are checked by the verification tests:

```text
E_initial = E_remaining + E_released + E_exported_unreacted
U - U_initial = E_released - E_vent_heat - E_boundary_out
```

Exported *unreacted* potential is not heat produced in the model. The thermal
ledger never subtracts it from U or counts it as an internal source.
`released_energy_j` is gross chemical release; retained heat is release minus
vent heat. For all legacy scenes vent heat is zero and the old meaning is intact.
Node heat capacity remains a separate constitutive law. Nothing in this change
fits or changes the measured battery capacity or intact conductivity.

## State-dependent anisotropic structure

Optional `thermal.reaction_structure` contains three `active_factor` and three
`residue_factor` components in the object's local x/y/z frame, plus an activity
rate scale. Its deliberately small constitutive closure is:

```text
q = 1 - E_remaining/E_initial
a = min(1, P_chemical / (E_initial * activity_rate_per_s))
f_axis = exp[(1-q)*a*log(active_factor_axis) + q*log(residue_factor_axis)]
```

Here q is **depletion**, including exported unreacted potential, not purely
chemical reaction extent. Depleting material can leave a different structure.
The law permits temporary enhancement followed by weak residue, but neither
enhancement nor weak residue is mandatory: all factors are material parameters.
There is no relaxation timer. Cooling suppresses the active enhancement while
irreversible depletion remains. The pristine, unactivated factor is exactly 1.

Internal links multiply their existing constitutive conductance by the sender's
directional `f_axis`; opposite directions are evaluated separately, never
averaged. This is an effective ordered structural law, not a measurement of a
universal bulk-k tensor. Each shared edge still transfers **one** heat quantity,
added at one endpoint and subtracted at the other. Positive conductance sends
heat down the temperature gradient. Contacts between objects remain separate
laws; their two bulk half-volume resistances use each endpoint's own factor.
Surface thermocouple reconstruction uses the same changed half-volumes.

Do not combine this closure with the old timed bulk transient or a second
damage-driven internal destruction law. Such combinations fail validation.
Finite-volume batteries are supported; kinetic `isothermal` is rejected, not
silently approximated. Passive interlayers and legacy isothermal objects may
still coexist with kinetic spatial batteries.

## Explicit energy-only outlet

`runaway.energy_outlet` requires an existing `kind: vent` port belonging to that
battery. Its face, normal, position and area are validated. The only supported
sink is `outside_domain`; reusing that port as an internal contact or a thermal
boundary is rejected. **No vent was invented or enabled for the UPV pouch.**

The unresolved discharge path is a lumped closure available to each local
reacting volume, not gas transported through the intervening thermal voxels.
The declared port identifies the outlet and its destination; its area currently
does not compute a pressure-flow relation. Opening is a smoothstep in q between
`opening_start_conversion` and `opening_end_conversion`. Optional closure uses
another smoothstep between the closure thresholds; default 1/1 disables closure.
This is a calibration hook for changing channel availability, not a prediction
of valve opening pressure, clogging or pouch-seal rupture.

With opening fraction w, heat fraction h, and exported-potential ratio r:

```text
P_exported_unreacted = r*w*P_chemical
P_vent_heat = h*w*P_chemical
```

With rates frozen for one substep, reserve depletion integrates exactly as
`E * (1-exp(-k*(1+r*w)*dt))`; generated and exported portions split in the ratio
1 : r*w. This prevents releasing an exported reserve again inside the battery.
Channel closure stops this route without resetting fuel, damage or structure.

There is no resolved pressure, mass, gas temperature, jet momentum, gas
convection, external combustion, radiative jet exchange or change of solid
mass/heat capacity by ejection. Only a share of newly released heat leaves via
this closure; extra sensible cooling of the existing solid is **not** implicitly
added. Ordinary boundary cooling continues independently. Significant loss of
solid mass would require extending the capacity/material model.

If a jet heats another battery or a shared cavity, it is **not outside the
thermal domain**. An explicit receiving path/control volume is then required.
The present outlet cannot validly replace that route. Whether a full mass solver
is necessary must be decided from such experiments, not assumed away.

## Numerical resolution and diagnostics

CUDA uses the old-state snapshot for rates and fluxes, then updates energy and
inverts the node capacity law. The local finite-reserve integral is exact only
for the frozen rates; the coupled temperature feedback is time-discretized.
Substeps obey the conductive CFL bound, a maximum consumed fraction, a maximum
source temperature rise, and a damage increment bound. The conductivity CFL
envelope includes the largest active/residue factor on internal edges and
boundary half-volumes, including contacts with isothermal neighboring objects.

The ImGui inspector shows current chemical power, remaining reserve, discharged
heat and exported potential. Thermal and chemical residuals are separate.
The diagnostic executable emits time histories and structural min/max by axis:

```bash
/tmp/battery-sim-cuda-v1/battery_sim_reaction_probe examples/kinetic-process-demo.json
/tmp/battery-sim-cuda-v1/battery_sim_viewer examples/kinetic-process-demo.json
```

For step refinement the probe accepts `--max-dt`, `--fraction` and
`--temperature-step`. Changing only max_dt may not change the actual step if
another limiter is active.

Verification on 2026-10-13:

- Independent constant-rate analytic reserve/temperature solution on CUDA.
- Cooling/reheating: 2677.65 J of 2700 J remains at quench, chemical power
  0.000726 W; subsequent imposed reheating consumes the remaining reserve.
- Feedback refinement: final 524.401 → 524.437 → 524.455 K as the step controls
  halve twice; errors decrease. This is numerical convergence, not validation.
- Full nonlinear 3×3×3 demo: loose controls (0.1 s, 0.01 reserve fraction,
  1 K source rise) shift the fast fronts noticeably despite exact energy balance.
  Five levels of successive halving were checked. The shipped demo uses
  0.0125 s / 0.00125 / 0.125 K. Halving all three once more changes sampled macro
  temperature by at most 0.9833 K and sampled local maximum by 2.2864 K
  (0.3231% of the finer absolute temperature in kelvin). These are differences
  between two numerical resolutions, **not error bounds against the continuum
  or experiment**. Energy residuals remain below 4e-9 J. On the test GPU, the
  shipped controls used 194897 substeps and about 11 s wall time for 180 s of
  simulated time; this is a small-scene diagnostic, not a module benchmark.
- Directed anisotropy, boundary resistance/CFL, non-generating link changes,
  local rather than whole-battery ignition, reset, variable-C energy balance,
  competing heat/export budgets and rejection of invented outlet geometry.
- Saved no-cooling and active-cooling curves for both intact
  `upv-links-contrast-3x3x3` and historical `upv-failure-transient-v3` replay with
  **0 K maximum difference**. Old profile hashes/artifacts were not rewritten.

Next calibration must identify a *process*, preferably constraining total heat,
duration and heat/export partition independently. Existing late witness curves
alone did not identify the early release history even in the smaller pulse
model. Do not label an attractive temperature fit as identified vent chemistry.

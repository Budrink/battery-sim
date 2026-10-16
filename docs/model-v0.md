# Model v0

Historical lumped Python model. The native application now uses the
[CUDA spatial model](cuda-spatial-model.md); these are not interchangeable solvers.

## Scope

The v0 solver is a deterministic reference implementation for checking the
state model, signs, event order, and energy balance. It is not yet a calibrated
predictor. A cell has its real outer shape and named surface ports, but one
uniform temperature. Ports have geometry and topology, not their own heat
capacity.

The thermal graph is bipartite in meaning:

- cell nodes store energy and accumulated cell damage;
- interface edges store connectivity, phase, and interface damage;
- ambient boundaries remove or supply heat through a port.

This separation lets a later version split one cell into thermal zones without
changing the external port and interface contract.

## Cell energy

For cell `i`, with constant lumped heat capacity `C_i`,

```text
C_i dT_i/dt = sum_j G_ij (T_j - T_i) + Q_runaway,i - G_i,amb (T_i - T_amb)
```

Every internal transfer is added to one cell and subtracted from the other in
the same numerical operation. The reported energy residual is therefore a
direct regression check:

```text
Delta U - (E_runaway - E_boundary,out) = 0
```

Runaway v0 is a prescribed finite energy released at constant power. Replacing
that source with ARC-derived heat-release curves does not alter the graph
solver.

## Edge connectivity: `l` and `l0`

`l` is a phenomenological effective channel scale owned by an interface. It is
not necessarily the physical separator thickness. `l0` is the first activation
threshold of the current material structure, not a lower bound for `l` and not
a universal constant.

The thermal conductance used by the solver is

```text
G(l, D_e) = [G_background + G_active sigma((l - l0) / Delta_l)] f_damage(D_e)
```

Consequences:

- heat still leaks for `l < l0` through `G_background` and the smooth tail;
- increasing `l` may open a strong channel;
- a material may instead use a negative damage shift and close the channel;
- after structural destruction, both `l0` and the conductance law can change.

The JSON contract uses metres for `l` so material fits have a declared scale.
Only the mapping from `l` to an observable conductance has physical meaning in
v0; fitted `l` values must not be interpreted independently.

The implemented equilibrium branches are:

- `constant`: no temperature-driven transition;
- `smooth`: logistic interpolation between `l_low` and `l_high`;
- `first_order`: discontinuous low/high branch with hysteresis;
- `second_order`: continuous square-root branch above the critical temperature.

Finite `relaxation_time_s` gives

```text
dl/dt = (l_target(T, phase, D_e) - l) / tau_l
```

An interface is massless in v0, so it cannot store sensible or latent heat.
When data require a finite transition enthalpy, the interface must become an
explicit thermal node rather than hiding that energy in the `l` law.

## Damage and failure

Cell and edge damage are bounded cumulative variables in `[0, 1]`. Their v0
thermal-dose rate is deliberately empirical:

```text
dD/dt = r0 exp((T - T_onset) / T_scale),  T >= T_onset
dD/dt = 0,                                T <  T_onset
```

For cells, the rate is multiplied by
`1 + aging_damage_multiplier * aging_fraction`. `aging_fraction` describes the
condition on entry to a propagation experiment; `damage` evolves during the
experiment. Calendar and cycle-aging kinetics are not yet claimed by v0.

At cell damage `D = 1`, a secondary runaway source starts. At an edge-specific
destruction threshold, the edge changes to its destroyed phase and takes
`destroyed_l0_m`, `destroyed_l_m`, and a separate conductance law. This is the
explicit representation of material restructuring discussed during design.

## Numerical method

The solver uses an explicit conservative Euler step. It lands exactly on
scheduled failures, source endings, output times, and the final time. The step
is reduced according to

```text
dt <= stability_factor * C_i / sum_j G_ij
```

after proposing the new edge states. CPU execution and small time steps are
intentional: this implementation is the correctness oracle for later optimized
or GPU solvers.

## Known omissions

- no spatial temperature field inside a cell;
- no electrical network, SOC evolution, or normal-operation heat generation;
- no gas, ejecta, mass transfer, pressure, or vent-jet edges;
- no radiation view factors;
- no uncertainty propagation or parameter inference;
- no finite thermal mass or latent heat in an interface.

These are extension points, not quantities silently folded into `l`.

# Open external gas boundary

`GasCompartment.ambient_boundary` is an opt-in boundary for **free exterior
gas**, not a wall contact, can rupture, or modification to the internal S law.
It is supported by the split CUDA network backend. Existing closed scenarios
are unchanged when the field is absent; zero area and zero renewal are exact
no-ops. The CUDA checkpoint ABI is now `cuda-state-v2` because gas inventories
also retain cumulative boundary ledgers.

The local boundary operator has two parts:

- Bidirectional compressible pressure flow to an ambient reservoir, limited
  by exact pressure equalisation and solved implicitly without microsteps.
  Exhaust carries donor enthalpy and donor constituent fractions; incoming
  air carries ambient enthalpy and oxygen.
- Optional prescribed renewal flow, representing unresolved hood entrainment.
  Equal gross mass enters and leaves. Composition uses the exact CSTR
  exponential; temperature uses the enthalpy-based exponent `cp/cv` times the
  mass turnover, not a temperature clamp or a solid heat-transfer coefficient.

Pressure half-step, renewal, pressure half-step is applied to the accepted
gas-stage output. The graph pressure/mixing/reaction solve is unchanged. Its
splitting with this boundary still needs representative timestep sensitivity;
positivity and isolated exact identities alone are not experimental validation.

Signed outward mass, sensible-plus-latent energy (using the solver reference
temperature), chemical reserve, and oxygen close the existing global ledgers.
Gross inlet/outlet mass and sensible entropy production are also retained.
Fuel which leaves the domain does **not** magically burn inside it. No oxygen
is reset in internal pores. The bath entropy balance includes the mass-specific
chemical-potential term; local total entropy may fall when entropy leaves.

For FSRI, `fsri30_open_boundary.py` attaches boundaries only to the outermost
faces of the existing resolved air boxes. It does not add nodes, change 20 mm
spacing, change module holes, or move thermal contacts. Local renewal is spread
over external face area. The article describes an open-air test under a hood;
its facility exhaust of 420 m3/min is not a measured flux through this model's
1 m3 far field. Local renewal remains an explicit apparatus uncertainty.

The comparison controller runs fresh full trajectories through 5945 s. It
labels the previous closed-core archive as a reference, never as a current-core
replay. Nine TC channels versus thirty component means are compared as
exploratory envelopes only: no invented channel correspondence, no time shift,
no claim of independent validation.

## FSRI development result (2026-10-25)

The selected **candidate**, not a quantitative fit, is the full 20 mm replay in
`data/development/fsri30-open-boundary-20261025-r2/open-0.01m3s`. It retains the
original module openings and .01 m3/s local renewal. The development registry
scope is `geometry_prior:fsri30-open-exterior`; never transfer its flow or
apertures to another fixture as material constants.

The full cooling tail is monotone, but still too hot. Widening two openings by
four did not improve the computed tail; that negative run was stopped with its
actual prefix retained, not awarded a full-horizon score. The audit found late
vent energy transport tiny relative to total heat export. Explicit radiation
and ABS structural/combustion behaviour are absent; do not compensate by
unbounded aperture or chemical-energy fitting. Black-body estimates in the
audit are potentials only and were **not added** to any saved trajectory.

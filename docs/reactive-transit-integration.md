# Reaction during accepted gas transit

This is an experimental numerical integration option, `gas_network.reactive_transit`.
It defaults to false. It integrates the existing oxygen-limited R3 law during
accepted pressure-driven transport, rather than only reacting the gas resident
before and after a potentially multi-volume transfer. It does not add chemical
energy, change R3 kinetics or alter the accepted carrier mass flux.

## Local problem

For one physical transport window, express fuel F and oxygen O in joules
(oxygen mass times the existing heat-per-oxygen coefficient). With x=t/dt,

    dF/dx = Fin - h F - s min(F,O)
    dO/dx = Oin - h O - s min(F,O)

Here Fin/Oin are integrated arrivals, s=k(T)dt, and h is solved from the
accepted carrier-mass budget. The difference F-O obeys a scalar linear equation
independent of reaction. It has at most one zero in the window. Integrating the
two possible limiting-species segments analytically gives remaining fuel and
oxygen, outgoing constituents and released heat, without physical microsteps.

`native/include/battery_native/reactive_transit.hpp` implements this bounded
local problem. `native/src/reactive_average_gas.cuh` couples it to the accepted
CUDA pressure-flow graph. Algebraic snapshot iterations reconcile integrated
incoming and outgoing payloads; they do not advance the physical clock.
Released heat enters the same window's implicit upwind enthalpy solve, so it
is not trapped in a tiny reactor after its gas has already left.

The Arrhenius coefficient is frozen to the pure-transport temperature for this
window, including arriving sensible enthalpy. Reaction self-heating within that
coefficient and the existing lumped vapor/fuel partition are approximations,
not newly validated chemical laws. Full temperature-dependent trajectory and
step-sensitivity tests remain necessary before enabling this by default.

## Evidence and limits

The CUDA regression `cuda_reactive_transit` checks 5184 analytical local cases,
fuel/oxygen/energy closure, a switch of limiting constituent, exact subdivision
of the frozen local problem, carrier-budget inversion, a closed full solver and
a fast throughflow full solver. In the throughflow test, reacted fraction is
0.561777 at dt=.01s and .566974 at dt=.005s. Resident-only split integration at
dt=.01s gives .00426134. This establishes a numerical failure mode; it is not
evidence that every experimental discrepancy has the same cause.

Mass, chemical, sensible, latent and oxygen ledgers remain strict. Unreactive
transport's entropy check runs before reaction, so reaction heat cannot hide
an invalid pressure/enthalpy update. Negative inventories or unconverged
payload iterations reject the transaction. The physical minimum time step
is unchanged.

## Carrying learning forward

`scripts/model_evidence.py` stores content-addressed, scoped records, separating
fixture-specific geometry, material constitutive laws and numerical methods.
`scripts/search_fsri30_suffix.py` automatically seeds a compatible new search
from the latest retained geometry, including its sensitivity controls and GP
anchor. A resumed search pins its original immutable seed even if the mutable
development pointer has advanced. Solver changes always recompute trajectories
and scores. They never reuse a previous core's checkpoint implicitly.

The FSRI conditioned post-burst profile is a development candidate, not a
validated material law, universal gas geometry or fresh-from-zero experiment.
Its fixed-state evidence must not be used to claim warm-up/cascade validation.

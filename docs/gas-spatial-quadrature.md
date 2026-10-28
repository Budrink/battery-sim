# Conservative spatial gas quadrature (rejected, archived prototype)

This prototype was rejected on the first-burst replay (R3 heat error14.7%).
It has been removed from the working native core/schema; the description below
documents the ARCHIVED implementation, not a current supported option. Exact
sources, CUDA test and executable are recoverable from
`data/development/fsri30-spatial-reactor-20261025-r2/source-evidence.tar.gz`
and that directory's `native-probe`. Do not use current production binaries
with the archived prototype controller. Default physics was never switched.

`gas_network.reactive_reconstruction=true` enables a numerical candidate for
averaged CUDA reactive transit. It is OFF by default and not a material law.
Each eligible free-gas compartment declares an explicit `reconstruction_box`
with `low_m`/`high_m`. The box must match its physical volume and must not bind
to a solid. Geometry metadata is not used by physics or restart identity.

The pressure graph, conductance laws, accepted carrier transfer and enthalpy
operator are unchanged. Current neighbor means reconstruct densities of total
mass, absolute sensible energy, chemical energy, oxygen, vapor and phase-reference
energy with three geometrical slopes. MC/minmod limits centered slopes; a
one-sided slope is used at a boundary. Only open, positive-area, face-adjacent
free-gas channels participate. Non-centred/irregular connections fall back to a
constant reconstruction. No gradient is inferred through a closed barrier.

Eight Gauss points sample each box. Zero-mean convex limiting guarantees
nonnegative inventories and nonnegative inert mass `m - oxygen - vapor`, without
changing coarse totals. Temperature is computed from reconstructed energy and
mass, never reconstructed as a supposedly conservative scalar. The existing R3
Arrhenius law is evaluated at each point for the current physical interval.
Its summed frozen-rate heat defines an effective rate for the existing exact
reactive-transit operator; that operator still owns chemical/oxygen/sensible
ledgers and accepted mass flow. Uniform fields recover the original rate exactly.

This first version does NOT evolve subcell memories, alter transport face fluxes,
resolve CFD or use oracle fine-grid coefficients. In particular, matching a
frozen quadrature does not prove that the effective rate is exact during
throughflow. Fresh cascade replay against a fine reference is required. A
negative result is grounds to reject the candidate, not to refit chemistry.

`battery_sim_gas_reconstruction_test` checks an independent analytic linear-field
quadrature, 2000 positive random mixtures, mean inventories, species bounds,
closed barriers, exact uniform equivalence, host/device agreement and actual
production CUDA reaction/transport ledgers. `check_fsri30_spatial_reactor.py`
archives a new binary/core, starts at zero with frozen physical parameters,
compares the known 0–2525 s cascade without time shifts, then runs an OFF control
through the first burst. No old-core checkpoint is used as a new-core state.

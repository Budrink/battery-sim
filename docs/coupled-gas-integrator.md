# Coupled gas reactor and realtime acceptance

This is a numerical change, not a new chemical energy source, modified material
capacity, fitted resistance or altered gas passage. It is currently opt-in:
`gas_network.coupled_gas_films=true`, CUDA network split transport only.

## Local operator

For a frozen Arrhenius rate k over a local interval, available oxygen-limited
fuel L decays exponentially. The gas equation is integrated analytically:

    C dT/dt = k L exp(-k t) - G (T - Twall)

`Twall` is the conductance-weighted temperature of the actual adjacent solid
patches. `C` is the REAL gas capacity, never a surrogate wall capacity. The
coincident-rate k=G/C limit uses its regular analytical form. In reference mode,
step doubling locally controls
temperature error (0.05 K at unit refinement) and fuel error (0.1% of available
stock with a small absolute roundoff allowance).

Cooling occurs DURING R3, not after an artificial adiabatic flash. Heat, fuel,
oxidizer, condensable-vapor labels and reaction diagnostics retain their existing
ledgers. An equal-and-opposite heat record is generated for every surface patch;
the final record closes the gas's exact heat total. A fixed-order CSR gather
updates each solid once, without floating-point atomic scatter.

Local solid temperatures are frozen during a gas macro transaction. This is a
splitting approximation; finite solid-capacity feedback returns next macro step.
Passive surface aging is recomputed from the old committed state after gas heat,
replacing (not adding to) the earlier disposable aging proposal. Film targets
must currently be passive constant-capacity solids without a separate LTE gas
binding; unsupported combinations fail explicitly.

The outer explicit-film and duplicate R3 bounds are removed ONLY in this mode.
Solid conduction/chemistry, gas generation, fracture work and opening constraints
remain. Transport coupling has a separate gas/receiving-solid heat budget,
bounded by 2% of gas T; it is NOT the local ODE tolerance. Pressure, temperatures,
aperture timing and reaction histories still require temporal convergence tests.
A small-reactor test passing does not validate the full destruction trajectory.

The global predictor uses a cheap conservative k(T+allowed rise) bound. It does
not repeatedly integrate a hypothetical full future ignition merely to choose a
much shorter next interval. Profiles distinguish graph reaction intervals from
local reactor iterations. All adaptive work is candidate-only and reset on retry.

## Coarse 5 ms integration (19.10.2026)

The user stopped the search and disallowed adaptive steps below 0.005 s.
`simulation.minimum_dt_s=0.005` enables a deliberately coarser approximation.
Current coupled-film calibration scenarios explicitly enable it. Zero is
reserved for reference/unit-test scenarios; positive values below 5 ms fail.

- The outer chemistry/source/aperture accuracy estimate cannot shrink the
  transaction below this floor. Frozen-rate triangular R1/charging/R2 integration
  still exactly closes inventories and chemical/electrical energy budgets.
- The explicit conduction CFL remains a HARD constraint. If it requires a step
  below the floor, the solver fails explicitly rather than taking microsteps or
  pretending an unstable explicit update is safe.
- Gas graph intervals are balanced across the remaining physical interval so a
  stiff reaction cannot force a graph-wide microstep or a tiny final remainder.
- R3/film stages use one analytic frozen-rate evaluation rather than adaptive
  local ODE microcycling. Symmetric half-stages are fixed algebraic evaluations,
  not adaptive retries. Conservative pressure/inventory limits remain intact.
- Exact externally requested event/output/end alignment can have a remainder
  below 5 ms; it does not resolve stiff physics and is counted separately as
  `aligned_short_steps`. Regular minimum and floor activations are also recorded.

Retry caps retain the immutable lower bound derived from the ORIGINAL external
interval. Both post-fracture and post-heating mass-budget retries apply their
0.95 safety margin only above that bound; an internal retry may never disguise
itself as an externally requested short alignment. `floor_bounded_retries` and
`split_step_retries` expose this path. `cuda_coarse_aperture_retry` reproduces
the former failure using fracture cooling followed by a changed source budget;
it also verifies rollback, energy/mass and CUDA tiling independence.

This trades fast-peak temporal accuracy for bounded work; it is NOT evidence
of full-experiment accuracy. CUDA tests exercise 5/10 ms floors with a stiff
finite heat source and hot oxygen-limited gas, verify no hidden ODE cycling,
and close heat/chemical/mass/oxygen balances. Full destruction comparison is
still required. The long calibration remains STOPPED, not silently restarted.

## Realtime acceptance

User clarification: at least **0.5 simulated seconds per wall second averaged
over the whole trajectory**, allowing **0.25x over a rolling 20 s window**.
These are separate criteria; neither inflates dt nor waives physics.

`scripts/realtime_guard.py` uses the native progress clock and a rolling 20 s
window after the first progress report, excluding CUDA startup. A fast warmup
cannot mask a slow destruction phase. A missing heartbeat for 20 s is rejected
as well. Completed runs also need a total ratio >=0.5.
Rejections retain their checkpoint and `last-speed-rejected.json`. They are
performance failures, NOT completed physical fits. Earlier r1-r4 rejections
below used the superseded 0.5x rolling floor, not the clarified 0.25x floor.
Numerical unit references
are separate from production replay performance acceptance.

## Evidence so far (not a successful calibration)

- Stopped reference: last complete sample 840 s, advanced 840.642785 s in
  3845.402669 wall s, 111148 steps. Last data, inputs and binary retained under
  `spatial-gas-20261019-r1`. It never reached the principal destruction event.
- First 200 s coupled test: 13.164409 wall s, **15.192x realtime**, 2560 steps.
  Maximum surface-temperature difference from reference: 0.000439 K.
- Coupled r1: automatic speed rejection near 755.317 s, **0.204827x** on the
  rolling window. Retained sample 755 s has max surface error 0.008210 K and
  chamber pressure 1.032726 vs reference 1.032445 bar. Its gas temperatures are
  not identical (max external local void 61.1 vs 45.5 C): full temporal and
  destruction checks are still needed, not inferred from surface agreement.
- CUDA tests cover analytic constant-rate and resonant limits, energy/mass,
  host/device agreement, finite surface heating, aging, retry and refinement.
  Surface refinement errors: 6.99e-5 -> 4.90e-5 K in the isolated fixture.
- Coupled r2 was rejected at ~755 s for **0.3890x** rolling speed. Its last
  gas profile showed 807 graph intervals but only one local reactor iteration:
  speculative arrival bounds, not the local reaction solve, dominated work.
- Coupled r3 reacts actual arrivals locally and caps transport windows at 0.01 s.
  It reached 873.048 s in 130.435 wall s, but was nevertheless rejected for
  **0.449254x** over its final 20 s window. Through 840 s its maximum surface
  history difference from the explicit reference was 0.4812 K (C2/ISS);
  chamber pressure differed by at most 0.003125 bar. This does not establish
  destruction accuracy or sustained competitive speed.

## Fresh precursor timestep bound

The frozen-temperature triangular chemistry is unchanged. Only its timestep
heat bound now distinguishes fresh precursor formation from completed R2.
For R2 rate k, old stock B, bounded precursor feed F and requested horizon H:

    completion_rate_bound = k B + F (1 - exp(-k H))

Every accepted h <= H can react no more fresh precursor than
`F h (1-exp(-k h))`. Charging energy follows the same bound. R1 remains heat
neutral: its fresh products no longer count as instantaneous R2 heat when k is
small. Real heat-capacity changes from R1 and the material-turnover bound remain.
Tests cover zero R2, slow R2 and conservative old/fresh-stock bounds.
Coupled r4 tested this correction with physical parameters frozen. It reached
878.528 s but was rejected for **0.288966x** rolling speed. Last checkpoint:
sample 875 s, advanced 878.526560 s in 140.073739 wall s, 11900 steps.
Maximum thermal/chemical residuals were 5.15e-8 / 8.86e-8 J; mass residual
1.98e-13 kg. The limiter did not force an inaccurate minimum timestep.
Do not mark the approach competitive until the event itself passes the floor.

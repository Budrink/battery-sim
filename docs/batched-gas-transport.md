# CUDA batched pressure transport (experimental)

Select `gas_network.mass_backend: cuda_batched`. The default solver and all
thermal material/calibration parameters are unchanged. No CPU Monte Carlo
implementation exists. This mode reuses the transactional CUDA mixture adapter;
the old `cuda_kawasaki` mode remains an event-by-event comparison reference.

## State and physical clock

At each internal time interval, all edges read the same state. Compressible
orifice flow F, donor pressure, and the conservative pair-equalizing mass M_eq
are evaluated using the existing constitutive functions, current aperture and
current free volumes. Closed barriers transmit nothing. Temperature affects
gas density and enthalpy, but is not a gate on pressure-driven backflow.

For a stochastic edge, choose a fixed quantum q = eta M_eq / 2 and draw
N ~ Poisson(F h / q). Propose dm = q N. The frozen-state expectation is F h;
there is no fitted seconds-per-sweep coefficient. The inverse-CDF sampler is
used only for means below 32. Above the configured threshold, transfer the
deterministic mean F h instead. This explicitly suppresses fluctuations in
large batches. It is NOT an exact compound-Poisson aggregation of the original
uniform-packet process: its finite-packet variance is different.

Near pressure equilibrium, use the existing finite-time square-root relaxation
dm = M_eq r (2-r), r = min(1, F h / (2 M_eq)). It is bounded by both an absolute
and a relative pressure tolerance. It retains aperture and elapsed-time
dependence, rather than immediately mixing connected volumes. Differences below
1e-6 Pa + 1e-10 P are numerical equilibrium and cause no packet.

`batch_mean_threshold` is 32 by default; zero selects deterministic means for
numerical-convergence tests. `batch_pressure_fraction` defaults to 0.2 and
limits the per-interval linearized pressure change. `kawasaki_packet_fraction`
controls stochastic resolution, not physical conductivity. The tail tolerances
keep their existing meaning. `local_exchange_interval_s` sets the physical
network substep.

## Common snapshot, budgets and conservation

1. Compute all proposals in parallel on the shared interval
   `h = min(local_exchange_interval_s, remaining_duration)`. A single stiff
   channel does not shrink every other edge's time step. The tail's instantaneous
   F is not extrapolated beyond its finite equalizing inventory.
2. Sample counts or mean packets. Limit a proposed packet to its pair's M_eq.
3. Each node gathers all incident requests in fixed CSR order. Pressure budgets
   use its own dP/dm and cap the summed linearized pressure change to
   `batch_pressure_fraction` of the incident gaps, not the equalizing capacity
   of the other endpoint. A large chamber therefore does not inherit the
   compressibility of each tiny pore.
4. Reconcile edge requests with both endpoints. Outgoing inventory is bounded;
   newly arriving mass cannot be forwarded within the same snapshot.
5. Sum the accepted outgoing mass D of each donor. Compute ONE energy packet
   Q = U [1 - (1 - cv D / (Cs + cv m))^(1 + R/cv)]. Split Q among outgoing
   channels proportionally to their mass. Fuel and oxidizer follow donor mass
   fractions. This accounts for donor cooling without spending its energy twice.
6. Each node gathers signed edge packets into a separate next-state buffer.
   Every edge has exactly one equal-and-opposite mass/energy/tracer ledger.
   Check positive inventories and total nondecreasing entropy; fail rather than
   add heat or silently truncate time. The graph then advances by h.

No edge writes a node. There are no floating-point state atomics or edge colors.
Up to eight resident 256-thread blocks stride over the graph using cooperative
grid barriers. Hardware support and occupancy are checked before launch. A
common chamber still gathers its adjacency in one thread, but does not impose
500 sequential edge-evolution stages/barriers. GPU block/grid size is covered
by trajectory-reproducibility tests. This implementation requires cooperative
CUDA launch support; it does not silently fall back to CPU.

## Deterministic fixed-sweep variant

Select `mass_backend: cuda_deterministic`. Each accepted thermal step is divided
into exactly `deterministic_sweeps_per_step` equal mass sweeps (1–8). Every
sweep proposes `dm = flow(p_donor, p_receiver, T_donor) * dt` from a common
snapshot, caps it at the pairwise equalizing inventory and donor inventory,
then reconciles simultaneous requests with a per-node pressure budget. The
default `deterministic_pressure_fraction` is 0.75; it can be set in (0, 1].
Accepted transfers carry donor enthalpy and composition, then node states are
gathered before the next sweep. `deterministic_pressure_gap_pa` reports the
largest open-edge pressure difference left after the last sweep. This mode has
no random packet counts and never runs more than eight network passes per
thermal step.

The sweep count bounds the transport distance/work per thermal step: gas can
traverse at most `deterministic_sweeps_per_step` sequential edges. These are
elapsed-time substeps, not nonlinear iterations seeking pressure equilibrium.
A pressure difference across an open edge is therefore expected during a
transient and is not, by itself, a solver error. Compare pressure and
temperature histories at matched observation times for 1/2/4/8 sweeps (and,
when useful, a finer offline reference), alongside mass and energy ledgers and
runtime.

## Simultaneous analytic network variant

Select `mass_backend: cuda_network`. Unlike fixed sweeps, this performs one
whole-graph proposal per accepted thermal interval. Every open edge reads the
same node snapshot and computes its directed nozzle rate. Choked edges integrate
the frozen donor-controlled rate over the interval, limited by a linearized
equalizing inventory. Subcritical edges use the analytic local
`sqrt(pressure-gap)` relaxation integral. The edge proposals are then reduced
by endpoint node to satisfy outgoing-inventory and pressure-change budgets;
the minimum endpoint scale is applied to each edge. Mass, donor enthalpy,
chemical tracer and oxidizer are transferred together, followed by one gather
of all node balances. No edge mutates an endpoint while another edge is being
evaluated, so branches and shared donors do not depend on traversal order.

This is an analytic *edge* closure plus a simultaneous network balance, not an
exact solution of the nonlinear network ODE: conductance, donor temperature
and direction are frozen at the beginning of each thermal interval. Gas remains
non-equilibrium; there is no pressure snap. `network_pressure_fraction`
(default 1.0) is a numerical limiter, not a calibrated material property. The
pairwise equalizing-inventory cap and donor-inventory limiter remain active at
this setting.
The mode avoids the repeated nonlinear equalization-root solve used by fixed
sweeps, but the one-pass frozen-state approximation can shift the pressure
trajectory. Compare against the fixed-sweep baseline and refine the accepted
thermal interval before using it as a production replacement.

The stochastic counters belong to candidate state. Prepare/discard does not
advance committed counters; commit swaps them. A fixed seed and accepted step
sequence reproduce across caller tiling and retries. Changing the physical
step does NOT preserve the same random realization: compare ensembles as well
as deterministic time refinements, not individual same-seed curves.

## Approximation limits and diagnostics

Rates are frozen within an internal interval. Pressure and inventory clipping
can bias mean flow; conservation alone does not prove accuracy. Convergence
must be checked by reducing both `local_exchange_interval_s` and
`batch_pressure_fraction`. `gas_limited_batches` exposes
clipped edge batches. `gas_batches` counts active edge evaluations, not global
timesteps; `kawasaki_events` counts sampled virtual packets (not kernel loops).
For batched mode, `kawasaki_tail_updates` counts deterministic mean/tail updates.
There is a hard 200000 internal-iteration limit per outer call: exceeding it
returns an error. Completion never means that this ceiling silently truncated
transport. Full-case time, tolerance and packet refinements remain necessary.

The adapter supports finite positive-volume receivers, positive initial gas
mass, a single constant-cv gas and temperature-independent material Cp
during the mass stage. Conversion can update material capacity between stages.
The existing limitations on nonlinear Cp, exact vacuum and infinite sinks are
explicit rejections, not silently substituted physics.

## Checks

```
ctest --test-dir /tmp/battery-sim-cuda-v1 -R '^cuda_batched_' --output-on-failure
```

`native/tests/batched_gas.cu` checks the CUDA Poisson mean/variance/zero-count
probability, ledger closure, entropy, a 500-channel hub with simultaneous inflow
and outflow, closed paths, aperture/time scaling, tile reproducibility,
deterministic time refinement and a 128-seed comparison against event CUDA.
No CPU stochastic transport is tested or implemented.
The split and fracture suites additionally check conversion, reaction, local
LTE, pressure-driven reverse flow, finite equilibrium, leading fracture work,
channel opening and transactional retry/reset.

The current explosion screening artifacts are in
`data/calibration/upv-batched-v40/`. They are NOT an accepted calibration:
the short runs only bracket the first chamber-pressure observation, and the
longer run through 0.95 s still underpredicts the second pressure observation
(about 5.35 bar modeled vs 7.98 bar measured at 0.942 s). The witness
temperature RMSE also becomes very large after this point, dominated by one
channel. Thermal material parameters were kept frozen. Timed-out attempts
retain their status and must not be scored as finished experiments.

The 0.472 s pressure-threshold screening (rate fixed at 20,000/s) gave chamber
peaks of 2.988 bar at 50 MPa, 3.066 bar at 25 MPa, 2.873 bar at 100 MPa,
2.296 bar at 350 MPa, 2.132 bar at 390 MPa, and 2.106 bar at 500 MPa; the
observed first peak is 2.2365 bar. The non-monotonicity across nearby
stochastic candidates means these are screening points, not a fitted
constitutive curve. A rate increase from 20,000/s to 100,000/s at 350 MPa
raised the 0.942 s pressure only from 5.35 to 5.59 bar, still far below the
7.983 bar measurement, while the dominant witness-temperature discrepancy
remained about 683 K. In particular, matching the first peak does not repair
the later pressure/temperature trajectory; pressure threshold and aperture
rate alone do not span the observed transient.

The deterministic fixed-sweep screening is in
`data/calibration/upv-deterministic-v41/`. At 500 MPa opening pressure, 20,000/s
area rate, and a common per-node pressure budget of 0.75, the 0.472 s run took
4.62/5.34/6.84 s for 1/2/4 sweeps (RTF 0.102/0.088/0.069). The chamber peak
was 2.375/2.368/2.355 bar versus 2.2365 bar observed. Mass and energy ledgers
closed to numerical precision. The largest instantaneous pressure differences
across open edges were 75.9/77.3/79.3 MPa; these describe the non-equilibrium
state at the end of the run and are not pressure-convergence residuals. This
screen suggests modest sensitivity of the chamber peak to sweep count, but it
does not establish time-resolution convergence of the full transient or a
calibrated explosion prediction. Four sweeps were faster than the Poisson run
in this case, though still below real-time performance. A separate run with
the more restrictive 0.2 pressure budget was both slower and less
representative, which is why the deterministic variant has its own explicit
budget control.

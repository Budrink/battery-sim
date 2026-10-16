# CUDA Kawasaki gas prototype: physical-time audit

## Scope

The standalone library is an experimental reference. The full-model adapter is
now selectable as `mass_backend: cuda_kawasaki`; `cuda_batched` selects the
snapshot/tau-leap alternative described in [batched-gas-transport.md](batched-gas-transport.md).
Neither is the default backend. The full adapter preserves leading fracture,
conversion/free-volume updates, local LTE, finite-fuel chemistry and rollback.
It rejects external sinks and temperature-dependent solid heat capacities.
The existing thermal solver, material calibration and production CPU local-group
transport have not changed. There is no CPU Monte Carlo implementation or test.
Host code prepares topology and checks downloaded results; stochastic evolution
is device-only. The audit uses the existing deterministic CPU solver solely as
an independent fine-step reference.

Supported: closed finite networks, positive gas mass and free volume, a single
constant-cv ideal pseudo-gas, optional immobile constant-capacity material in
instant local thermal equilibrium. Chemical energy and oxidizer are advected
passive inventories. No reactions, rupture, changing aperture, nonlinear heat
capacity, zero-mass vacuum or external sink are integrated in the standalone API. A large finite
receiving reservoir is supported, not an infinite pressure boundary.

## Modules and scheduling

- `kawasaki_gas.hpp`: state, controls and CUDA API.
- `kawasaki_gas.cpp`: input validation and edge coloring only.
- `kawasaki_gas_math.hpp`: constitutive arithmetic and device event loop.
- `kawasaki_gas.cu`: storage, disjoint-pair kernels and ordered launches.
- `kawasaki_gas_audit.cpp`: CUDA ensembles and deterministic reference audit.
- `kawasaki_tail_audit.cpp`: tolerance sweep, repeated timings and grid traces.

Each color is a matching: no two edges write the same node. A forward sweep for
dt/2 and a reverse sweep for dt/2 expose every edge to dt, not dt times the color
count. One thread owns an edge in one replica; state updates need no atomics.
There is no host state synchronization between colors. The only atomic records
an error code. A seed/edge/event/draw counter determines random numbers, not the
thread index, launch size or number of outer timesteps.

Arbitrary graphs can require many colors. In particular, the current production
676-volume network has a chamber node incident to 500 edges. Strict pair
matchings cannot parallelize those 500 writes. The Cartesian benchmark below
does not remove this production-topology limitation.

## Packet and energy balance

At each event the higher-pressure node is the donor. Let M_eq be the mass that
would equalize this pair's pressures without reversing their gradient. Draw
dm uniformly from (0, eta M_eq), where eta is `packet_fraction`.

For donor sensible energy U, mass m, immobile heat capacity Cs and gas cv:

```
C = Cs + cv m
gamma = 1 + R/cv
Q = U [1 - (1 - cv dm/C)^gamma]
```

This integrates adiabatic donor cooling with flow work. Q is removed from the
donor and added to the receiver, along with identical mass, chemical-energy
and oxidizer packets. It is not a transfer of dm cv T alone. Each proposed
packet checks positivity and nondecreasing pair entropy, within roundoff:

```
S = (Cs + cv m) log(T) + m R log(V/m)
```

Reference constants cancel in the closed-system difference. This is a
dissipative bulk-flow approximation, not a detailed-balance molecular MC model.

## Physical clock

The physical compressible-orifice mass flow F sets the event intensity:

```
lambda = 2 F / (eta M_eq)
lambda E[dm] = F
```

Each edge retains an exponentially distributed integrated-hazard threshold.
An interval with no event subtracts lambda dt from that threshold instead of
redrawing it. After an event the state, rate and next threshold are updated;
multiple events can occur in one interval. There is no fitted seconds-per-sweep
factor and no single-Bernoulli saturation at high rates.

This matches instantaneous expected mass drift. It does not make finite packets
identical to deterministic trajectories: nonlinear energy drift and ensemble
bias require eta refinement. Event-only isolated-pair trajectories preserve event clocks
under interval subdivision; a network additionally needs color-splitting window
refinement. One stochastic realization is not the ensemble mean.

In the event-only reference (`tail_atol_pa=0`), as dp approaches zero, the square-root flow law and shrinking packets can
generate infinitely many events in finite time. Below
`1e-6 Pa + 1e-10 * donor_pressure`, the remaining pair equalization is completed
conservatively. Its count and transferred mass are recorded. This is a numerical
tolerance, not a physical conductance or fitted time factor. Exhausting the
200000-event safety limit reports failure instead of silently truncating time.

### Time-limited small-pressure tail (optimized default)

The prototype now defaults to `tail_atol_pa=100`, `tail_rtol=0.001`. When
`0 < dp <= min(tail_atol_pa, tail_rtol * donor_pressure)`, replace the remaining
microscopic random events with a deterministic near-equilibrium approximation:

```
r = min(1, F0 * remaining_time / (2 * M_eq))
dm = M_eq * r * (2 - r)
```

This integrates `dM/dt = -F0 sqrt(M/M_eq)` for the remaining equalizing mass M.
It uses the local small-dp expansions F proportional to sqrt(dp) and dp
proportional to M. The same nonlinear, conservative energy/tracer packet and
entropy check are used afterwards. The relative threshold limits the expansion
to a small pressure difference even at low absolute pressure.

Unlike simply enlarging the old instantaneous-equilibrium tolerance, this
retains aperture and elapsed-time dependence. An arbitrarily weak channel does
not equalize the pair in an arbitrarily short interval. The deterministic
approximation suppresses tail fluctuations; it is not an exact aggregation of
the stochastic process. No hazard is consumed while stochastic evolution is
disabled in this region. The remaining hazard is retained for later re-entry.
Tail subdivisions must converge numerically, rather than reproduce identical
event histories. Counts and transferred mass are recorded separately as
`tails` and `tail_mass`. Set the absolute threshold to zero to recover the
event-only reference; its audit fixtures explicitly do so.

CUDA regressions cover finite-time tails, area/time scaling, blocked and tiny
apertures, constant-capacity material, subdivision, conservation and the
unchanged large-dp event sequence. No CPU Monte Carlo implementation was added.

## Event-only reference results

Build targets `battery_sim_kawasaki_gas_test` and
`battery_sim_kawasaki_gas_audit`. Run only the CUDA MC test:

```
ctest --test-dir /tmp/battery-sim-cuda-v1 -R '^cuda_kawasaki_gas$' --output-on-failure
/tmp/battery-sim-cuda-v1/battery_sim_kawasaki_gas_audit data/calibration/kawasaki-time-v37/cuda-only-8192.json 8192
```

The test checks mass/energy/tracer conservation, entropy, closed barriers,
equilibrium, persistent clocks, area/time and volume/time scaling, material LTE,
topology coverage, GPU block-size reproducibility and reset. It passed after
removal of the CPU MC implementation. Earlier JSON reports in the same directory
are historical runs, not the current CUDA-only benchmark.

The event-only report uses seed 20261015, 8192 replicas, 64 sample intervals and
eta = 0.5, 0.125, 0.03125, 0.0078125. Fine-reference step refinement changes
temperatures by at most 0.00024 K. All cases span about 0.025365 s.

| Case | Max mean-T error, eta=0.5 | Max mean-T error, eta=1/128 | Max standard error of mean, eta=1/128 |
|---|---:|---:|---:|
| Isolated pair | 6.336 K | 0.158 K | 0.0755 K |
| Large finite receiver | 3.335 K | 0.039 K | 0.0611 K |
| Six-node chain | 5.726 K | 0.159 K | 0.0753 K |

Fourfold window refinement at eta=1/128 leaves isolated-pair/reservoir outputs
unchanged; the chain's maximum mean error is 0.160 K. The interpolated time when
node 1's mean reaches 305 K differs from the reference by -0.39% for the pair
and -0.49% for the refined chain. The reservoir never reaches that threshold.
These are diagnostics at one threshold, not a universal timing-error bound.
Individual realizations remain noisy: maximum standard deviations are roughly
5.5–6.8 K, despite the small errors of the ensemble mean. Residual sensible-energy
imbalances are at most 4e-13 J in these tests.

On the RTX 4060 Ti, the separate 16^3 benchmark (4096 nodes, 11520 edges, six
colors, eta=0.1, one realization) takes 0.994 s for 0.008 s of physical time.
It executes 19.14 million events and 105705 near-equilibrium completions.
This is not real time and is not an accuracy-matched production speedup.
This is the pre-optimization measurement. Resolving the chamber hub remains
work before integration. No thermal or explosion calibration was
rerun or changed for this audit.

## Tail tolerance sweep

```
/tmp/battery-sim-cuda-v1/battery_sim_kawasaki_gas_audit --tail-sweep data/calibration/kawasaki-time-v37/tail-sweep-final-8192.json 8192
```

The sweep tests 0, 0.01, 0.1, 1, 10, 100 and 1000 Pa, with a simultaneous 0.1%
relative pressure cap. The 1000 Pa setting therefore does not permit a 1000 Pa
tail everywhere. Physical coefficients and the thermal calibration are unchanged.

For the 16^3 grid, timings use one warm-up followed by three reset repetitions;
the median excludes setup and state download. Spatial traces use eight replicas,
averaging each transverse plane of 256 volumes, sampled at 17 common physical
times. These compare against event-only MC, not against deterministic truth.
Both coarse (16 windows) and refined (64 windows) runs cover 8 ms.

The final sweep measured 0.956 s event-only versus 0.0515 s at 100 Pa, an
18.5x speedup. Events dropped from 19144348 to 153811, with 229551 deterministic
tail updates. At fourfold smaller windows the corresponding times were 2.860 s
and 0.0937 s, a 30.5x speedup. Hardware timings vary between runs.

Maximum observed differences of plane-mean temperature were 0.0444 K (coarse)
and 0.0262 K (refined, compared with event-only at the same window), and pressure
differences were 41.6 and 20.9 Pa respectively. The JSON's direct comparison
fields always use the coarse event-only reference; the matched-window values
here compare the corresponding stored traces and timings. These are sampled ensemble
differences, not pointwise or universal error bounds: the maximum standard error
of plane means is about 0.42 K, and individual trajectories diverge. Grid mass,
energy and oxidizer balances stayed within 3e-14 relative error.

In the six-node chain (8192 replicas, eta=1/128), max mean-temperature error
against the independent deterministic reference changed from 0.1589 to 0.1602 K
at 100 Pa, or 0.1615 K with the refined window. The 305 K mean-arrival diagnostic
remained within 0.5% of the reference. Pair and reservoir ensembles never enter
the tail region during this short experiment and remain unchanged; they alone
would not validate the optimization.

The 100 Pa setting is the chosen prototype default, not an experimentally fitted
material parameter. This is not yet real time: even the optimized coarse grid
takes about 6.5 wall seconds per simulated second. Nor does the grid speedup imply
the same gain for the full battery solver, variable capacities or the chamber hub.

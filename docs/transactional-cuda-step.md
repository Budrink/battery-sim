# Transactional CUDA substeps

This changes solver architecture, not material laws or calibration parameters.
Evaporation and liquid-displaced gas volume are **not implemented by this
refactor**. They can now be added as separately tested process proposals.

## Contract

`CudaSolver::prepare_step(time, cap, safety, runtime)` computes a candidate and
returns its actual dt without publishing physical state. It may be repeated
with the same or different controls. `discard_step()` invalidates the candidate.
`commit_step()` publishes it once; a second commit throws. The original `step()`
API calls prepare followed by commit.

Preparation reads committed thermal nodes, gas, interfaces, channels and macro
aggregates, and overwrites separate scratch/candidate buffers. Downloads and
`device_field()` still expose committed state. Reset and ignition invalidate a
pending candidate. Failed preparation leaves no older candidate committable.
The caller advances its clock by the returned dt only after commit. The solver
object is not a concurrently callable host API.

Candidates are checked for nonfinite/invalid node, gas and contact states before
publication. Failure is reported, not repaired by clipping negative inventories.
Automatic smaller-step retry is not enabled. An unrecoverable CUDA context or
device failure is outside the rollback guarantee.

## Execution graph

1. Compute source/transport stability bounds from committed state.
2. Propose channel opening/flow and gas-to-solid heat rates.
3. Reduce gas stability bounds and freeze dt.
4. Propose interface history/heat rates and local source changes.
5. Build extensive gas transfer packets; resolve competing reaction demands.
6. Each thermal node/gas compartment gathers its own contributions.
7. Apply proposed casing rupture to candidate shell state; validate.
8. Publish state buffers together and refresh derived macro aggregates.

The stages are named methods in `native/src/cuda_step.cuh`; orchestration is
separate from gas laws, local conversion and constitutive closure. After the
mechanism audit, gas transport proposes a raw rate and an extensive equilibrium
mass cap, **neither depending on dt**. `capped_budget_dt` solves the monotone
piecewise-affine shared budget `linear*t + sum(min(rate*t, cap)) <= budget`.
Only after global step selection is the signed transfer packet constructed.
Thus an unrelated process reducing dt cannot leave an obsolete flow limiter.

Discrete casing rupture now calls the same `close_thermal_state` helper as
thermal evolution: a changed damage coordinate cannot leave a stale temperature
when heat capacity depends on damage. Energy itself is not changed by that
constitutive refresh.

Launches use the same ordered CUDA stream, without a host barrier between
mechanisms. Preparation's final status download waits for completion; commit
swaps pointers and queues aggregate refreshes.

## Shared accounting, specialized physics

One gas transfer packet contains signed mass, sensible/flow energy, chemical
energy and oxidizer mass. Both channel endpoints and its cumulative ledger use
the **same packet**. Solid conversion emits a separate packet; gas gather no
longer reads the newly updated solid field. Gas combustion proposes a balanced
chemical-to-thermal/oxidizer change before gathering.

Transport retains priority over gas reaction. The old-state kinetic reaction
proposal is capped by fuel/oxygen remaining after accepted transfers. Incoming
material cannot enlarge that old-state proposal. This makes the existing
first-order closure explicit; it is not symmetric splitting or a claim of
order-independent physical operator splitting.

The exact frozen-temperature two-stage solid conversion remains one coupled
mechanism, rather than independent stages spending the same stock. Legacy and
conversion sources occupy mutually exclusive typed pools. Existing source
limits plus the shared gas bound prevent multiple outlets overdrawing a donor.
There is no generic arbitrary-reaction-network resource optimizer here.

For future evaporation: propose one extent, derive all mass/energy changes from
it, reconcile shared liquid/vapour stocks, then apply once. Phase energy
references, mixture EOS and changing free volume must be defined together.
Adding a vapour mass source alone would not meet this contract.

## Parallel execution

- One owner per pointwise output record; no neighbour scatter.
- Stable host-built incidence lists for thermal links, gas contacts, channels,
  conversion sources and shell channels. Gas gather visits neighbours only.
- Fixed, directed, state-geometry and transition thermal laws have separate
  pools; source dispatch is also outside the pointwise kernel. Physical
  thresholds remain; branchlessness is not a physics requirement.
- No floating-point atomic additions to inventories/energy. Stability uses
  atomic minimum; validation uses integer error flags. Macro sums have a fixed
  reduction tree.
- Pointwise block sizes 64, 128 and 256 are supported; reduction sizes stay
  fixed so work tiling cannot change the summation tree.
- Storage remains predominantly typed records (AoS), not a full SoA rewrite.
  Proposal buffers and extra launches have a cost; speedup is not unconditional.

Tested reproducibility means identical results on the same device/build for
repeated proposals and different pointwise tilings. It is not bitwise
portability across GPUs, compilers, floating-point options, mesh permutations
or different time discretizations. Advancing time itself is not idempotent.

## Checks

`cuda_transaction` compares all public thermal, interface, gas, channel and
macro fields exactly, not padded struct bytes. It tests 64 versus 256-thread
work tiles, repeat/replace/discard, double commit, reset/ignite invalidation,
invalid controls, invalid GPU candidates and recovery. It includes solid/gas
reactions, shared donor outflows, shell rupture, heat contacts, interface history
and a partial final CUDA block. Existing physics tests remain required.

For the four frozen heating/failure replays, save the old probe **before**
rebuilding, then run:

```bash
python3 scripts/check_step_refactor.py \
  --before /tmp/battery_sim_casing_probe_pretransaction \
  --after /tmp/battery-sim-cuda-v1/battery_sim_casing_probe \
  --output /tmp/battery-transaction-regression.json
ctest --test-dir /tmp/battery-sim-cuda-v1 --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  /tmp/battery-sim-cuda-v1/battery_sim_transaction_test
/usr/local/cuda/bin/compute-sanitizer --tool racecheck --error-exitcode 99 \
  /tmp/battery-sim-cuda-v1/battery_sim_transaction_test
```

The replay report stores executable/scenario hashes, step counts and all numeric
differences, ignoring only wall time. It does not overwrite frozen profiles.
Changed source/binary fingerprints still require the explicit calibration
workflow before a newly frozen profile; regression is not new validation.

## Pre-audit refactor verification on 2026-10-13

- Full native suite: 18/18 passed; extended transactional test also passed.
- Python discovery: 91 tests, 20 skipped, no failures.
- CUDA memcheck: zero errors; racecheck: zero errors or warnings, including
  the extended isothermal-source retry fixture.
- Both frozen heating replays: exactly unchanged temperatures and step counts.
- Frozen failure replays: maximum temperature difference 5.23e-12 K and
  pressure difference 2.41e-8 Pa; step counts unchanged (77207 and 76396).

Full replay evidence and hashes:
[verification report](../data/verification/cuda-transaction/replay.json).
Wall times there are individual observations, not controlled benchmarks.
The simple heating cases incurred about 35% extra wall time from the extra
passes; the gas cases were faster in these runs. Do not infer a general speedup.

## Subsequent mechanism fixes

The audit then identified a preliminary-dt gas limiter and stale shell
temperature after damage changes. These are fixed, rather than retained for
bitwise agreement with the previous executable. The native suite now has
19 passing tests, including `cuda_mechanism_consistency`: rejected-cap
independence, pressure-equilibrium transfer, rupture U/T/D closure and 2,000
randomized capped-budget cases. See the [v7 report](mechanism-fixes-calibration-v7.md)
for the new calibration, distinct from the pre-audit replay above.

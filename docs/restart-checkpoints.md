# Committed CUDA restart and gas-map diagnostics

`Simulation::save_checkpoint` saves the complete committed state, not a JSON
observation: solid composition/energy, gas constituents, electrical state and
potential warm start, shell damage, channel aperture/history, transport clocks,
simulation time, events, history and energy ledgers. The local binary format
validates its checksum, typed lengths and ABI sizes. CPU transport is unsupported.

An explicit physics/topology contract includes the canonical scenario, numerical
controls and native binary fingerprint. Duration and read-only observations can
change; physical parameters cannot silently change during ordinary restart.

## Verified baseline

`data/calibration/upv-separated-chemistry-v6/restart-gas-baseline-20261019-r1`:

- Full reference: 0–950 s, 126.46 s wall time.
- Last completed sealed observation: **750 s**, `sealed-state.bin`.
- First actual shell opening: **754.682771249402 s**, not the later chamber peak.
- Resumed 750–950 s: 52.56 s wall time.
- Comparison: **20,225 fields bitwise equal**, maximum difference zero.

This is a frozen diagnostic seed, not an accepted destruction calibration.

## Explicit rebase before rupture

A changed gas graph deliberately fails an ordinary restart contract. The
diagnostic protocol can instead load the saved source scenario and explicitly
rebase a CLOSED state:

- Solid physics, subdivision, contact laws, electrical drive, gas EOS and
  reaction laws must remain unchanged.
- Pores and interface slits keep their identity, inventory and binding.
- Spatial external volumes map by box overlap. The former mixed chamber maps
  into the newly resolved complement of those boxes.
- All extensive gas quantities transfer with the same volume fractions: mass,
  sensible/latent energy, chemical reserve, oxygen, vapor and reaction ledgers.
- Every donor must transfer 100% of its inventory; every target volume must be
  covered. Negative fractions, incomplete maps and already vented sources fail.
- New deterministic external edges start fresh diagnostic counters/caches;
  existing pore and rupture histories are retained. No stochastic RNG is reset.

This preserves the actual baseline prehistory; it does NOT claim that a full
warm-up on a changed geometry would produce exactly that prehistory. Re-run the
full trajectory after selecting a promising geometry.

Run a frozen suffix comparison with:

```bash
python3 scripts/check_upv_gas_routes.py \
  --baseline data/calibration/upv-separated-chemistry-v6/restart-gas-baseline-20261019-r1 \
  --output data/calibration/upv-separated-chemistry-v6/gas-routes-new
```

The driver keeps chemistry fixed, checks conservation/speed, saves both physical
inventories and route ledgers, and plots temperatures and a real gas-grid slice.
The pressure observer is volume-mean external pressure, not a second gas volume;
it does not duplicate any inventory. Transducer position is not published, so
this observer is an explicit measurement proxy, not a claim of local uniformity.

# FSRI: separating heat-up from post-burst gas transfer

The r5 full-from-zero geometry screen was confounded: changing gas access also
changed heat losses before runaway. Two variants held B10 near104°C with the
same assumed6W heater. Those cases did not test the propagation of an existing
runaway. The other variants only improved neighbour peaks by about2K.

## Common physical starting state

The original simulation first opens a small channel at730.8s. Main R2 conversion
accelerates much later, around1755s. A last-sealed checkpoint therefore cannot
isolate the process of interest. A new read-only replay control saves a committed
state at an existing1755s observation, without altering integration boundaries.
It retains all previous venting, fracture work, material damage, reaction products,
temperatures, gas pressures, electrical state and physical ledgers.

The saved state already has rupture-B10-4 opening fraction0.005448171733805239.
It is never relabelled sealed, closed again, or initialized from a temperature
array. No scheduled ignition or additional heat is introduced. The unchanged
prefix and resumed burst match the archived native calculation bitwise through
1800s:6350 compared fields and all30 battery mean temperature curves.

## Narrow, native-enforced changes

`restart_transport_scenario` explicitly opts into a conditioned diagnostic.
It permits only:

- repartitioning existing lateral shell exposure between facing neighbouring
  body-gas compartments, conserving each face's cold conductance and area;
- scaling existing passage areas within the declared bounds;
- changing the shared hot gas-contact active range with cold normalization;
- changing the existing flow-associated mixing fraction.

Gas compartments, their volumes, inventory and reaction/EOS laws must be exactly
unchanged. Solid geometry/materials, heater drive, all pores, rupture laws and
numerics must also be unchanged. No new gas-to-core or external-plume-through-ABS
shortcut is permitted. The restored complete serialized physical state must
equal the captured bytes before any step; otherwise execution is rejected.

The existing unvented geometry-remap API retains its prohibition on remapping
after rupture. It has not been weakened. CUDA thermal/mass/reaction kernels and
checkpoint ABI have not changed; only the opt-in replay observer was rebuilt.
The archived0180466 executable remains untouched.

## Measurements and interpretation

Every suffix advances from1755s to5945s, including cooling. Its rate is reported
as4190 physical seconds divided by suffix wall time, not5945 divided by that
time. Shared prefix samples may be combined with suffix temperatures/chemistry
only after checking continuity at the join. Such a composite is explicitly
marked `conditioned_transport_diagnostic:true`, `full_warmup_validation:false`.

Accepted gas-film heat is accumulated from the actual CUDA step per receiving
shell, with positive and negative contributions separated. These integrals are
suffix-only; no heat prehistory is invented for a new contact. `process_totals`
records sums over ALL gas compartments, including the volumes omitted from
individual gas histories. This corrects the earlier observation limitation:
91 selected gas histories were not the complete R3 energy budget.

Five fixed-state hypotheses precede GP search. Chemistry and initial reserve
are not optimization variables. A useful conditioned geometry is only a lead:
it still needs a fresh from-zero run with a heater boundary consistent with the
experiment, numerical qualification and independent validation. Sensor mapping,
ABS burning, radiation and reaction/throughflow numerical sensitivity remain
unresolved; this change does not claim to solve those physical uncertainties.

## Checks

Pure-Python FSRI suite:47 tests pass;6 optional CUDA test methods are skipped in
the normal sandboxed invocation. Explicit CUDA invocation passes6 integration
test methods, including7 permitted reference/hypothesis/boundary continuations.
Negative tests reject altered chemical heat, gas volume/inventory, shell pressure
strength, heater drive and a hood-to-shell shortcut. Physical state equality and
strict mass/chemical/thermal conservation are not relaxed.

```sh
PYTHONDONTWRITEBYTECODE=1 BATTERY_SIM_TEST_CUDA=1 \
  python3 -m unittest discover -s tests -p 'test_fsri30_suffix_cuda.py' -v
```

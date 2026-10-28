# Gas → solid heat audit, 20 October 2026

Historical audit of the **old scalar-film implementation**. That bypass has
since been removed; see [Shared thermal connectivity](thermal-connectivity.md).
The numbers below remain a record of that old binary, not current predictions.

No chemistry, material, shell, transport or heat-transfer law was changed.
The pad-return series `pad-ablation-20261020-r1` was stopped on request after
three complete cases. Case 03 is incomplete, not a completed experiment.

## Former branch (before the connectivity correction)

`GasThermalPatch` holds a scalar `G = h*A`. External gas films are built from
`environment_h`, approximately 1.5634 W/(m² K), except the separately configured
chamber wall. This is a previously shared effective film prior, NOT a measured
hot-jet property. Surface connectivity tensors, damage response and mass-flow
strength do not modify this G. Temperature affects the driving difference and
R3 kinetics, but not the film coefficient itself.

`frozen_gas_reactor` integrates reaction and cooling together using actual gas
capacity `m*cv`; it accumulates heat lost to surfaces. `film_heat_records`
distributes the SAME heat to physical patches. `gather_film_heat` adds it to
each receiver's stored energy and computes its temperature with its own C.
The receiving-capacity average is an accuracy-control input, not a replacement
gas heat capacity. Old explicit gas-film fluxes are disabled in coupled mode;
the audit found no second application of the same heat.

Compiled full-face values:

| Receiver | Gas-film G, W/K | Solid heat capacity, J/K |
| --- | ---: | ---: |
| C3 ISS-side pouch face | 0.02543505424 | 4.8807 |
| ISS1 | 0.02543505424 | 5.010852 |
| TIM1 | 0.02543505424 | 126.166095 |

Patch intersections preserve the full 0.087 × 0.187 m² face area: no duplicate
division by eight/nine was found. The casing material remains the explicitly
unmeasured pouch-laminate proxy; this audit does not validate its properties.
On the bare-pad replay, C3 face→contents conductance at 750 s is 2.39204 W/K,
about 94 times gas→face G. A 400 K gas difference supplies only about 10.17 W
to the whole face, which also exchanges heat with the much larger cold core.

## Direct accepted-step trace

`GasFilmTrace` reads actual GPU patch-heat records after each committed step.
The observer adds no device ledger, does not modify checkpoint ABI, and does
not change the evolution. Positive/negative totals are split by accepted
physical step (not claimed as substep-resolved gross transfer).

Frozen 750–950 s replays, `gas-film-audit-20261020-r2`:

| Configuration / receiver | Heating, J | Return to gas, J | Net into solid, J |
| --- | ---: | ---: | ---: |
| No pads → C3 ISS face | 73.72056 | −6.69014 | 67.03042 |
| All pads → C3 ISS face | 5.46402 | −7.11260 | −1.64857 |
| All pads → ISS1 | 17.81745 | 0 | 17.81745 |

Thus the intact slit delivers net 16.16888 J to its two surfaces, mostly to
the colder ISS. The casing itself is slightly cooled overall by this gas
contact; its other solid/gas paths still exist. In the no-pad case, all six C3
outer casing faces receive about 704.37 J from gas, while the chamber wall
receives about 316.15 kJ. These are signed net exchanges, not total R3 release.

The direct patch sum matches the independent gas-inventory/transport/reaction
balance within 2.85e-14 J. All thermocouple histories reproduce their respective
untraced replays bitwise (maximum difference zero). Removing pads changes the
subsequent reaction dose through thermal feedback; the two configurations are
not equal-power tests and neither is a newly fitted experiment.

## Numerical checks and interpretation

The isolated CUDA fixture uses actual compiled casing/ISS/TIM capacities and
conductance. Nine closed two-capacity tests compare to the finite-reservoir
analytical solution. Maximum surface error is 0.004653 K (50 ms maximum step;
also checked at 5 ms); maximum energy residual is 2.77e-10 J. Patch heat equals
the receiving solid's energy gain. Existing `cuda_gas_local_reactor` and
`cuda_transaction` also pass.

The full traced suffixes take 79.85 s without pads and 51.90 s with all pads
(2.50× / 3.85× realtime); minimum regular physical step stays 0.005 s.

The supported finding is NOT a missing CUDA heat packet or wrong receiver C.
Current bottlenecks are replenishment of hot gas at the face and a weak,
state-independent gas/solid contact. This branch does not yet implement the
general state-dependent surface connectivity used by solid contacts. Deciding
how to describe hot flowing gas contact is a constitutive-model task, not a
reason to refit battery heat capacity or artificially increase R2.

Run the audit with `scripts/check_upv_gas_films.py --output <new-directory>`.
Its generated `gas-film-audit.png` separates gas temperature, casing surface
temperature, actual integrated heat and instantaneous contact power.

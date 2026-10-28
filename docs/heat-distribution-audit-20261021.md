# Heat-distribution audit: latest r5, not a new calibration

## Scope and verification

Source: `data/calibration/upv-separated-chemistry-v6/cooling-fit-20261021-r5/trial-000`.
Diagnostic: `data/calibration/upv-separated-chemistry-v6/heat-distribution-20261021-r1`.

The diagnostic resumes the same sealed 835 s checkpoint and uses the exact r5
observation prefix through 950 s. Only accepted gas-film heat tracing and full
gas/channel history serialization were enabled; duration was shortened.
No material, chemistry, contact, transport, shell or numerical parameter changed.
All 12 reported temperature histories are bitwise identical to saved r5.
The continuation took 177.331 s for 115 simulated seconds (0.6485x).

`heat-routes.json` contains cumulative **accepted-step** contact energy and
independent local gas-energy closures, not an integral of sparsely sampled
instantaneous power. Maximum local closure discrepancy is 4.37e-11 J.
The existing `test_gas_heat_budget.py` suite passes all three tests.

## What the saved curves actually say

Do not conflate a missing experimental impulse with a missing calculated impulse:
the r5 calculation has false sharp impulses on both C1 faces and C3/TIM.
Calculated maxima during 880–915 s: C1/ISS 804.15 C, C1/TIM 618.30 C,
C3/TIM 627.53 C. These are not present on their experimental counterparts.
At C3/ISS the experiment has a prolonged hot tail which the model loses too fast.
Peak-amplitude tolerance cannot excuse a qualitatively false impulse on a
protected face or the wrong cooling shape.

## Exact energy paths

Net gas-to-solid heat over 880–900 s:

| Receiving side | Physical casing face, kJ | Pad, kJ |
| --- | ---: | ---: |
| C3/ISS (ISS1) | 9.902 | 8.299 |
| C3/TIM (TIM1) | 6.595 | 20.716 |
| C1/ISS (ISS3) | 8.275 | 4.741 |
| C1/TIM (TIM3) | 6.632 | 21.585 |

This does **not** support the claim that C3/TIM receives no heat while C1/TIM
takes its share. Their received energy is nearly equal in this specific run.
Instead, both faces are directly exposed to the same short hot-gas event.

The actual TIM gas routes are:

```
C2 rupture outlets -> external void network (fuel + oxygen, R3)
  |-> void-5-6-4 -> void-5-5-4 -> escape-TIM1-3-+y -> gap-TIM1-3
  |                                                         |-> C3:case:+x
  |                                                         `-> TIM1
  `-> void-13-6-4 -> void-13-5-4 -> escape-TIM3-3-+y -> gap-TIM3-3
                                                            |-> C1:case:+x
                                                            `-> TIM3
```

For these representative paths, accepted net sensible energy into the slit
through the listed escape edge is 2.364 kJ for TIM1 and 2.438 kJ for TIM3.
Across all boundary channels of each complete slit, it is 23.175 and 23.922 kJ.
Gas-film heat into the casing through the full-face slit alone is 5.063 and
5.066 kJ, respectively; additional heat arrives through exposed edge contacts.
Local R3 inside the slit supplies only 6.52 J for TIM1 and 5.95 J for TIM3:
the dominant source is incoming hot gas, not local fuel burning beside the TC.

The C1 pulse is not heat conducted through the battery from C2. During the pulse,
the C1 casing is much hotter than its core; the core/casing power flows *into*
the core. At 890.956 s, its mean core is 49.0 C, while its ISS/TIM casing probes
are 804.2/618.3 C. Meanwhile the TIM3 pad is still heating: its mean temperature
reaches about 208 C at 900 s. Later this excessively heated pad returns energy
to gas and adjacent solids. This sequence explains the wrong curve shape.

## Geometry and mixing findings; limits of the conclusion

1. Every pad/battery interface has an unresolved 0.2 mm gas slit spanning the
   full face, connected to perimeter galleries. The slit heats both the casing
   and the pad directly. A nominal solid contact also exists in parallel.
   These slit dimensions and their full-face accessibility are **not measured**.
   Parallel microcontacts can be an effective model, but a full unobstructed
   gas-exposed face is not established by the source experiment.
2. The temperature maps show near-simultaneous hot gas all around the stack,
   not a localized directed jet. In r5 the TIM1 and TIM3 gas means reach about
   810/802 C at 890 s. Their observed thermal histories are not equivalent to
   such a common flash exposure.
3. An existing one-axis comparison is available: r3/trial-000 has the same
   parameters except the added global free-gas response factor of ten in r5.
   It changes C1/ISS maximum 363 -> 804 C, C1/TIM 228 -> 618 C,
   C3/TIM 258 -> 628 C, and delays C3/ISS maximum 887.668 -> 890.486 s.
   Improving a cooling scalar by global mixing worsens spatial selectivity.
4. About 348.82 kJ of R3 heat is released in external gas over 880–900 s;
   341.94 kJ is in wall-touching grid nodes. Those nodes also contain 98.28%
   of external reference volume. This percentage alone is **not** evidence of
   a wrong reaction location or of combustion literally on the chamber metal.

Confirmed: the actual bypass route to C1 and the nearly equal gas loading of
both TIM faces. Not yet established: whether incorrect access/exposed area,
over-fast mixing, missing jet direction, or some combination is responsible
for the discrepancy with the physical installation. Do not silently remove
selected contacts, swap thermocouple labels, or compensate with per-channel
thermal properties. Geometry/access and directional transport need controlled
checks before another chemical-energy calibration.

## Artifacts

- `heat-locations.png`: measured TC, simulated casing, pad, core and adjacent gas.
- `energy-distribution.png`: accepted gas-to-casing/pad energy in three windows.
- `gas-spatial-map.png`: actual resolved external-gas cross-section; solids grey,
  solid temperatures and unresolved face slits are not painted on this map.
- `heat-routes.json`: per-face budgets, local closures and ranked boundary paths.
- `status.json`: source, binary SHA and exact replay comparison.

No physical correction or successful recalibration is claimed by this audit.

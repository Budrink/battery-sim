# Thermal radiation on the shared heat graph

Radiation is an optional nonlinear boundary edge, evaluated by
`evaluate_directed_geometry` in the existing CUDA heat pass. The ordinary
edge gather debits the finite body and records the SAME energy in
`boundary_energy_out_j`. No extra energy sink, mass transfer, reaction,
condensed residue, or shell-opening rule is introduced.

For a grey surface facing a large isothermal reservoir:

```
B = sigma * effective_emissivity * visible_area
Qout = B * (Ts^4 - Tr^4)
Tnode - Ts = Rmaterial * Qout
```

All temperatures are Kelvin; sigma = 5.670374419e-8 W/(m² K⁴).
`Rmaterial` is the existing center-to-surface half-cell resistance, including
the material's current conductivity. The positive monotone surface equation
is solved with a fixed 20 Newton iterations on the GPU. The equivalent
nonnegative edge conductance multiplies `Tnode - Tr`; a hotter reservoir
therefore heats a colder body conservatively as well. Using the voxel-center
temperature directly in the fourth-power flux would skip this resistance.

Example boundary (the port belongs to the actual emitting solid):

```json
{
  "id": "outer-radiation",
  "port_id": "solid:outside",
  "temperature_k": 298.15,
  "radiation": {
    "effective_emissivity": 0.1,
    "area_fraction": 1.0,
    "temperature_limit_k": 4000.0
  }
}
```

Effective emissivity is a grey-body exchange parameter in [0,1], including
unresolved visibility/shielding. It is NOT an arbitrary convection multiplier.
`area_fraction` selects (0,1] of the port: it scales both emitting area and
half-cell resistance consistently. A radiative edge cannot also specify a
conductive `connectivity` or nonzero `conductance_w_per_k`. Existing gas-film
convection remains a separate, unchanged parallel physical contact.

The declared temperature domain supplies a conservative derivative bound
`4 B Tmax³` to the existing CFL calculation, with the same material resistance.
Going outside it rejects the numerical state; it does NOT clamp temperature or
discard excess heat. The production minimum-step rule is unchanged. No
radiation is added implicitly to old scenarios.

## FSRI30 exterior candidate, 2026-10-25

`scripts/fsri30_radiation.py` adds radiation only on actual external ABS patches.
It removes the two reconstructed vent openings from visible area, for a total
0.0739384056 m². Internal can-to-wall radiation and gas/flame radiation are not
invented by this exterior-boundary adapter. The exterior reservoir is 298.15 K.
Effective emissivity 0.1 is an explicit UNMEASURED diagnostic prior, not a
measured ABS property or a calibrated coefficient. All initial states,
chemistry, condensed inventories, gas paths, solid contacts and step controls
are copied unchanged from the retained ABS-seam candidate.

The production seed is
`data/development/fsri30-radiation-20261025-r1/scenario.json`.
It has NOT yet been replayed against the full experimental cooling trajectory.
Do not load an older-core checkpoint as a fresh trajectory with radiation.

## Verification

`cuda_radiation` compares 525 host/device constitutive cases against independent
bisection, and actual CUDA finite-wall cooling against a fine RK4 oracle.
It also checks heating by a hotter reservoir, zero emissivity, passivity,
the material-resistance bound, out-of-domain rejection and energy closure.

`scripts/check_fsri30_radiation.py` runs two ten-second manufactured hot-ABS
fixtures on the entire 20 mm, 30-cell gas geometry. This is NOT an experimental
fit or a resumed cooling tail. With ABS initially 900 K, adding emissivity 0.1
exported 2214.021 J and lowered the final capacity-weighted ABS temperature by
4.48432 K versus zero emissivity. Wall times were 5.14 s versus 5.25 s; this
short test establishes no noticeable overhead here, not cascade performance.
Maximum thermal residual was below 3.1e-8 J, mass residual below 5.1e-14 kg,
and the accepted regular step remained 0.1 s in both fixtures.

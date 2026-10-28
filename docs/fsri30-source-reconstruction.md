# FSRI geometry and UPV R2 transfer

This is a fresh-from-zero diagnostic, not a fitted or independently validated
material model. All old experiments, checkpoints, profiles and cores remain intact.

## Source constraints

[McKinnon et al., section 3.2 and Figure 2](https://doi.org/10.1016/j.dib.2022.108712)
describe 30 electrically isolated 18650 cells, 100% SOC, ABS packaging and an
open-air test under an exhaust hood. An 82 L nitrogen pressure vessel is a
**separate** gas-characterisation experiment. Section 3.3.3 describes central
front/back holes in cell and module housings; it does not give their dimensions.

The [installation report](https://fsri.org/sites/default/files/2021-07/UL9540AInstallationDemo_Report_Final_4-12-21.pdf),
printed pp. 13–15, supplies 18 × 65 mm / 45 g components, a 172 × 92 × 92 mm
assembly, and the staggered 7+8+8+7 layout. Figure 17's 3.1 inch width conflicts
with Table 3's 92 mm; the tabulated exterior envelope is retained. Coordinates
digitized from the drawing are approximate, not measured CAD.

The new graph partitions actual cylinder-free space into bounded nearest-cell
polygons, with top/body/bottom compartments. Upper/lower cross-sections are not
limited to tiny lateral gaps between cylinders. Gas-to-ABS contacts exist only
at actual boundary regions; central gas volumes no longer directly touch all
six walls. Two central end apertures replace 30 fictitious overhead exits.

Vent sizes are **not known**: retain the old best r8 equivalent total Cd·A as
an explicitly unmeasured prior. Square openings at mid-height are a shape prior,
not a discovered drawing dimension. Perforated panels lose the corresponding
effective heat capacity and exposed area. Near-field and far-field partitioning
preserves total gas volume, mass and oxygen. No infinite oxygen source is added.
This finite far-field approximation still needs a renewal/size sensitivity check.

The thermal core still uses equal-volume prismatic surrogates and separate steel
shells because the native cylinder/shell mesher is not implemented. Lateral gas
exposure is corrected to the real cylindrical perimeter. Can-to-can contacts,
heater feedback, radiation and ABS combustion remain unresolved, not silently
replaced by fitted conductances. The old 6 W heater is not a reproduction of
the reported closed-loop 6 K/min protocol.

Update: [resolved-free-gas.md](resolved-free-gas.md) documents the subsequent
spatial external atmosphere, local ABS contacts and autonomous controlled
heater. The limitations above describe the original reconstruction, not an
assertion that the newer controlled heater has already been validated.

## Transferred chemistry

Use the actual complete, accepted training scene
`reverse-cooling-fresh-20261023-r1/trial-000`, SHA256
`747c5aefc5932b1759ee48f6042692907f3992d0719190ba44c3135404b3b210`.
Only three common fields transfer: k(450 K) = 0.01999639111839507 /s,
Tref = 450 K, Ea = 100000 J/mol. This was the last fully accepted training
fit; the subsequent working compromise used 0.03499368445719137 /s but did not
pass the old strict shape gates. It is not silently substituted for the accepted fit.
The R2 law itself has unchanged semantics in the current core; only the later
electrical constitutive closure changed. Independent cooled validation failed,
so this is a cross-family kinetic hypothesis, not a validated chemistry transfer.

R1, charge state, reaction energy, mass, gas yield, R3, rupture strength and
transport algorithms are not copied from the prismatic UPV batteries. Identical
electrolyte chemistry is an assumption, not an established fact about the FSRI cells.

## Check

`check_fsri30_source_reconstruction.py` runs a fresh 2×2 comparison: retained
reference, geometry only, R2 only, combined. There is no old-checkpoint loading,
time alignment, target-temperature forcing, chemical reserve inflation or trigger.
The .01 s physical floor and strict thermal/chemical/mass/O2/UI checks remain.
Speed is guarded at .2× realtime. Temperature envelopes are diagnostic only,
because the nine TC locations are not mapped to the thirty components.
Conversion counts use the R2 gas ledger / constant gas yield, **not** inert R1
residue. Model conversion events are not claimed as measured failure times.

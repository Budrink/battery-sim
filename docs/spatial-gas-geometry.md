# Spatial external gas geometry — 19 October 2026

## Optional retained-pad perimeter openings — 20 October 2026

`gas_geometry.pad_edge_clearance_m` enables explicit open perimeter bands.
Zero is the archived full-face control. The first diagnostic uses **2 mm on
each transverse edge**, giving ISS/TIM footprints 83 × 183 mm instead of the
old 87 × 187 mm proxy. These margins are an UNMEASURED geometric hypothesis,
not dimensions read from the publication. Layer thickness remains 2 mm.
Battery geometry, chemistry, casing mechanics, material density, specific heat
and conductivity are unchanged. Pad total capacity and contact conductance
scales follow their reduced physical volume and contact area; no missing mass
is retained as an artificial thermal escrow.

Open-edge geometry uses individual physical occupied boxes, not one opaque
bounding box for the entire stack. Free bands form 112 resolved gas gallery
volumes, with ordinary shared-face passages around the retained pad centres.
Slit admissions reach the ACTUAL pad edge, then the neighbouring free gallery;
their recorded piecewise path and length do not shortcut through solid pads.
Newly exposed casing, pad side and back-plate areas have cropped ports and the
same directed heat connectivity as every other gas surface. Areas are not
duplicated over a full casing face. The surface observer reconstructs interface
temperature with the mesher's actual half resistances, including partial overlap.

There are 1401 gas volumes / 3592 channels without cooling, or 1493 / 3767 with
the occupied cooling plate. Exterior volume stays **120 L**, initial oxygen
inventory stays unchanged, and there is NO inlet from ambient or an infinite
oxygen supply. Fresh oxygen here means redistribution of chamber air.
Retained-pad openings and full pad-ablation diagnostics are currently mutually
exclusive, rather than silently double-counting newly exposed contacts.

Oxygen is advected with donor concentration and counter-mixed at zero net mass
flow. R3 consumes it at its finite Arrhenius rate; oxygen is not annihilated
when fuel and air meet. `cuda_gas_combustion` explicitly tests oxygen surviving
reaction and reaching a third compartment. Local diagnostic balances now
separate initial inventory, net transfer, consumption and remaining oxygen.

The paired check is `scripts/check_upv_pad_openings.py --output <new-directory>
--gas-parameters <parameter-json>`. It requires an explicit admissible gas law
and uses the SAME parameters in both geometries and the frozen underlying
chemistry/material parameters. Falling temperature laws are now rejected by
the fitter; the original archived run used a recovery law with a falling branch
and is NOT an accepted calibration. Both cases start at t=0 and cover the
observed event +300 s; the old sealed checkpoint is deliberately not reused
after changing pad volume/capacity. This is a geometry diagnostic, NOT a fit.

`tests/test_upv_pad_openings.py` checks volume/oxygen conservation, retained
material properties, physically exposed areas, connectivity without crossing
pads, and the blocked cooling-plate outlet. The first 1 s CUDA smoke check
passes with thermal residual <2.6e-8 J and oxygen residual <8.4e-17 kg.

## Current revision: closed spatial chamber v2

The text below records the original local-grid v1. **v2 removes its mixed
chamber hub completely.** A graded free-volume grid now extends to the sealed
walls, using the published 120 L total external volume. Equal outer clearances
define an explicit, unmeasured box-shape prior; they are not facility dimensions.
The old 20 mm extent now controls a near-grid cut, NOT an exit to a reservoir.
There are 601 gas volumes / 1620 channels in the no-cooling case.

The same single 5000 J/K wall object and same total wall `h*A` are retained;
derived, nonoverlapping wall-contact patches distribute that existing contact
only to gas volumes touching the envelope. No outlet removes mass. Pressure is
a separate volume-weighted observer across external volumes, never the pressure
of one corner cell. Shared-face geometry, total volume, wall contact and blocked
cooling-plate checks are covered by `test_upv_spatial_gas.py`.

See [committed restart and rebase](restart-checkpoints.md) for frozen suffix
comparisons. In `gas-routes-20261019-r1`, 750–950 s takes 52.69 s wall time
(3.80x realtime), with minimum regular step 0.005 s. All conservation residuals
remain below 7e-8 J / 3e-14 kg. **The geometry correction alone does not fix
C3/ISS.** Its slit receives about 16.3 J net transport energy and releases
1.19 J by R3, versus about 102.3 kJ R3 in the external gas. At 860 s, gas near
the slit entrances averages about 477 C while slit gas remains about 34 C.
This isolates the remaining bottleneck to slit admission/mixing/surface cooling,
not absence of any path from C2 to the C3 exterior. No chemistry was refitted.

Final naming/binding verification `gas-routes-20261019-r2` takes 52.75 s and
reproduces r1 temperatures with maximum difference exactly zero. There is now
NO physical node named `chamber`, so an obsolete pressure consumer fails loudly
instead of mistaking a corner pressure for the chamber observer.

Figure 2 qualitatively depicts pads recessed from cell ends, whereas the current
geometry retains full-face pads. Recess dimensions are not specified; do not
scale them from the schematic. Exposed end regions are a candidate geometry
hypothesis to examine separately, not an implemented or fitted correction.

## Historical local-grid v1

This replaces the four shared `near-*` hubs and the pad inlet stars. It is an
explicit geometry correction, NOT an accepted calibration or a new reaction
law. The CUDA transport, R1/R2/R3, phase law, electrical model, solid properties,
solid interfaces and shell opening/fracture laws are unchanged.

## Representation

`scripts/upv_gas_geometry.py` constructs the network; `gas_void_geometry.py`
provides axis-aligned occupied/free boxes. Only shared faces conduct mass.

- The occupied full-face stack remains solid. No gas crosses an intact pad,
  pressure plate or casing. Published component thicknesses are unchanged.
- A spatial void envelope surrounds the stack, including end regions. It is
  split along battery positions and into transverse patches. A rupture targets
  the local receiver at its casing-tile centroid, never a whole-side reservoir.
- Only envelope boundary faces reach the mixed remainder of the sealed chamber.
  Their areas come from actual free faces, not a battery side-face area proxy.
- Face slits have eight patches matching the pad grid, no common inlet node.
  Their perimeter exits go only to adjacent exterior voids. Ruptures on broad
  battery faces enter the adjacent slit patch, not a remote pad.
- Exterior gas is NOT in LTE with an entire pad element. It has its own mass,
  temperature, oxygen and chemical reserve. Each slit patch has finite contacts
  to the pad and physical casing. Cropped casing ports are intersected with the
  unchanged casing grid; no new contact can reach bare enclosed contents.
- The existing shared `environment_h` is retained for these effective gas films
  as a diagnostic prior, including unresolved boundary/half-volume resistance.
  It is not claimed to be a measured jet coefficient or electrolyte property.
- The cooling plate is an occupied box: bottom passages are blocked and gas
  must go around its physical edges. There is no hand-written bottom bypass.

There are 290 nodes / 1000 channels without cooling (160 exterior void cells),
312 / 1031 with the plate (182 exterior void cells). Interior gas still has
27 LTE pore nodes per battery. No CFD or new native evolution kernel is added.

## Known dimensions versus priors

Primary source: [UPV description, Tables 2–3 and Figures 1–2](https://zenodo.org/records/22273467/files/TR_Experimental_Data_Description.pdf?download=1).

The 120 L chamber, 0.02 MPa stack pressure, cell/material dimensions and layer
thicknesses are published. Exact holder openings, exposed face margins and
interfacial gas clearances are not. The schematic is not a scale drawing.

Defaults, explicitly recorded in each scenario:

- Outer local-void extent: **20 mm** in each direction from the stack envelope.
  This selects how much surrounding space is spatially resolved, not an opening
  cut through a solid. Its gas volume is deducted from the chamber reservoir.
- Face slit: **0.2 mm**, retained from the previous unmeasured clearance prior.
  Exterior bypasses are no longer forced through this thin interfacial slit.
- External gas volume accounting remains **120 L** as in the previous model.
  Repartitioning does not add gas, oxygen, fuel or chemical energy.

Do not enlarge a slit merely to fit a thermocouple. Unresolved holders,
deformation, jet momentum, radiation and temperature-dependent gas-film transfer
remain limitations. A shell tile routes to one receiver by centroid; it does
not resolve sub-tile jet directions or duplicate a hole across several channels.

## Verification and comparison

`tests/test_upv_spatial_gas.py` checks shared-face areas, access to the reservoir,
blocked cooling-plate outlets, finite two-sided gap contacts, unique closed
ruptures, volume/EOS inventory and preservation of saved solid/chemical inputs.
Historical contact-class tests retain their solid-port/interface assertions;
they no longer require the intentionally replaced gas graph to stay identical.

Frozen diagnostic: use the SAME `parameters.json` and phase/mixing configuration
as `phase-mixing-20261019-r2/combined`; overlay its saved trajectory with
`check_upv_phase_mixing.py --reference .../trial-run.json.gz`. No optimizer,
validation-data fitting, thermal refitting or accepted-profile overwrite.

Initial checks: 10 geometry tests pass; 34 R123/contact tests, 13 kinetic-fit
contracts, 12 search contracts, 2 phase/mixing tests pass. The optional schema
test is skipped because jsonschema is absent. The historical LocalPaths suite
cannot start because its retired `upv-training-refit-v13` fixture was previously
removed; no archived scenario was recreated to make it pass.

CUDA smoke: both geometries advance .2 s in 7 steps. A separate explicit hot-gas
fixture advances 1 s in 90 steps, burns 0.4171 J of a declared 1 J fuel stock,
with thermal/chemical/mass residuals 2.22e-8 J / 3.96e-9 J / 4.66e-15 kg.

Frozen comparison was launched in `spatial-gas-20261019-r1` (925 s target,
7200 s wall timeout). At its 490 s checkpoint, the largest difference over
all six surface-temperature histories from the saved reference is 0.04741 K
(C2/ISS). This only checks preservation of pre-event heating, NOT runaway fit.

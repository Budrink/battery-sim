# Validation data contract

Public-data adapters must translate source-specific tables into the canonical
scenario JSON accepted by the solver. The solver must never contain column
names, filename conventions, or undocumented assumptions from one dataset.

## Required provenance

Every imported scenario should retain, under `metadata`:

- dataset and experiment identifiers;
- direct source URL and retrieval date;
- raw-file checksum and license or access terms;
- cell identity, chemistry, form factor, SOC, SOH, and preconditioning when known;
- trigger method and trigger location;
- sensor locations and timing convention;
- every unit conversion and imputation;
- adapter name and version;
- calibration/validation split assignment.

Unknown values remain explicitly unknown in the adapter report. They must not
be converted into confident defaults without recording the imputation.

## Canonical levels

1. **Single cell:** source energy or power history, case temperature, ejected
   energy when available, and variability across nominal repeats.
2. **Cell stack:** geometry, gap or interlayer material, SOC, initiating cell,
   propagation label, runaway order, and event times.
3. **Module:** the same outputs plus enclosure boundaries and cooling. Gas and
   pressure observations are retained for a later solver even though v0 cannot
   predict them.

The optional `observations` object currently supports:

```json
{
  "propagated": true,
  "runaway_order": ["cell-1", "cell-2"],
  "runaway_times_s": {"cell-1": 0.0, "cell-2": 47.2}
}
```

The runner compares these fields with the simulation and writes the result to
`summary.json`. Timing residuals are reported without inventing a pass tolerance;
the dataset evaluation configuration must declare that tolerance.

## Initial adapter targets

The source review supplied with the project identifies these validation tiers:

- NASA/NREL Battery Failure Databank and open ARC measurements for single-cell
  heat-release distributions;
- Sandia/FAA stack experiments across SOC and metal interlayers for the binary
  propagation decision;
- Sandia/NITE inter-laboratory experiments for repeatability bounds;
- Gao et al. packs and the TUM five-prismatic-cell stack for event order and
  timing;
- public UL 9540A reports only after enclosure and module boundaries exist.

This repository does not redistribute those files. Manifests under
`data/manifests/` describe externally acquired immutable raw inputs.

## Multiscale material inputs

The native scenario distinguishes `role: "battery"` and `role: "passive"`
objects, grouped by `material_id`. Store geometry/zone resolution separately
from material l0, conductivity and degradation priors. An explicit interlayer
must replace, not duplicate, the direct battery-to-battery path. Its two surface
contacts and its internal resistance/heat capacity are distinct inputs; an old
fitted effective contact G is not automatically a measured material conductivity.
Battery failure observables exclude passive material destruction. When only a
finite horizon was simulated, report no secondary failure *observed by that
time*, not a proof of containment. The Python v0 solver rejects passive scenes.

A passive material wall may use `kind: "rupture"` with the existing
`damage_opening.mode: "deterministic_aperture"` law. It needs a thin panel,
a local `shell_thermal_volume`, a full-face aperture port, and unique gas-film
contacts on opposite faces of that same voxel. Both gas endpoints must be
explicit cavities, outside battery contents. Mixed or partially exposed wall
patches need geometric subdivision before assigning a full-area aperture.
Thermal damage lowers the remaining pressure barrier; pressure opens area and
pays fracture work from the gas energy ledger. This geometric opening does not
remove the wall's mass/Cp or introduce an ABS combustion model. FSRI ABS
strength and weakening coefficients are diagnostic priors until calibrated.

## Leakage control

Splits are assigned by experiment family before fitting. Repeats of the same
physical setup, lab campaign, or publication stay in one split. Parameters are
fit on `calibration`; numerical choices are checked on `development`; headline
errors are produced once on `validation`. A result is not called predictive if
the tested experiment supplied the fitted interface constants.

## UL FSRI 30-component development case

The immutable v1.1 archive is pinned to commit
`8d140bc02f2e2cd04c2ca95f1d925ec4c24365b3`; archive, member and report hashes
are retained under `data/experiments/ul-fsri-30/manifest.json`. Raw source files
are read-only. This is an unfitted DEVELOPMENT reconstruction, not validation.
The report distinguishes one 18650 component, a thirty-component mock-up
"Cell", a 270-component module and a 2430-component initiating rack. Never
confuse these levels or apply the 213 L mock-up gas volume to each 18650.

The selected CSV has physical timestamps 0..5945 s and another 136 rows without
timestamps. Untimed temperatures are excluded, not assigned row-index times.
Nine temperature channels are retained. Their exact mapping to components is
unconfirmed: plot the measurements and predicted ranges separately; do not
invent per-channel correspondence or a calibration RMSE. The calorimeter HRR
integrates to about128 MJ; its boundary/scale must be independently resolved
before using it as the thirty-component chemical-energy calibration target.

`generate_fsri30.py` records the 7+8+8+7 staggered drawing, Table3 envelope,
equal-volume prismatic cylinder approximation, separate steel/ABS materials,
assumed open-loop heater and all kinetic/gas priors. It does not copy UPV
fitted contact or chemical coefficients. The first run leaves initial failure
times unconstrained, has no prescribed primary ignition and performs no fit.

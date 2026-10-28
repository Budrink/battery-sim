# Solid cooling-path audit — 2026-10-25

Read-only diagnosis of the retained complete 20 mm open-boundary replay.
No solver, chemistry, material coefficients, mesh or candidate pointers changed.
This is not another calibration or a reconstructed restart.

## Inputs and reproduction

Scene: `../fsri30-open-boundary-20261025-r2/open-0.01m3s/scenario.json`,
SHA256 `6b19c879af5d1602f58fab20f569cf7d73474a0bd816931a99e558c3e6b9747d`.
Complete replay: same directory `replay/run.json.gz`,
SHA256 `e652bd5dc3754dcf1f70e1685da756fd21f5f614c0ec5ec9c562f59f70ce61a6`.
Static topology built by the same native library as the replay:
`build/calibration-probes/libbattery_native_core-ambient-20261025-r1.a`,
SHA256 `e3293e7a4d1ff5a6f4aaaf61525f4203452913bb754c782fbcfa114b13c1e0aa`.

The included C++ observer calls `load_scenario`, `build_thermal_mesh` and
`gas_contact_conductance`, not a copied Python heat-transfer law. Build from
the repository root (adjust dependency include path if required):

```sh
g++ -O2 -std=c++20 -I native/include \
  -I /tmp/battery-sim-cuda-build/_deps/nlohmann_json-src/include \
  data/development/fsri30-solid-cooling-audit-20261025-r1/solid_path_audit.cpp \
  build/calibration-probes/libbattery_native_core-ambient-20261025-r1.a \
  -o /tmp/fsri-solid-path-audit
/tmp/fsri-solid-path-audit \
  data/development/fsri30-open-boundary-20261025-r2/open-0.01m3s/scenario.json \
  /tmp/fsri-solid-path-audit.json \
  < <(gzip -dc data/development/fsri30-open-boundary-20261025-r2/open-0.01m3s/replay/run.json.gz)
```

`native-audit.json` SHA256:
`1007a542b352142f58c3687dd9b7db25ff8495d83eeb23bb11e853cbb96b215f`.
Observer source SHA256:
`591ac9e3b1be10053c03cecca5001f03b587486abef5948aa6ba33c3dd75787d`.

## Confirmed geometry and state

- Native graph: 217 solid objects, **36 solid-only connected components**:
  30 independent core/can assemblies (one also includes the heater), plus six
  entirely separate ABS panels. Contact through gas is intentionally excluded
  from this solid-only graph.
- Twelve ABS panel pairs actually abut over positive-area rectangles, but no
  solid links join them. Areas: four pairs at 414.586 mm², four at 734.481 mm²,
  four at 373.971 mm². There is thus an explicit artificial break in the
  assembled outer casing. Its effect on the cooling curve is not yet measured.
- No solid can-to-ABS/support/spacer or can-to-can contacts exist. The nearest
  physical cylinder spacing is 19.3 mm vs 18 mm diameter, so direct lateral
  can contact must NOT be invented merely to improve cooling. Internal support
  geometry is unresolved in the reconstruction/source drawing. This scene has
  **no separate TIM/ISS pads** from the older three-cell UPV experiment.
- ABS mass 0.35 kg is inferred from assembly minus component masses, all placed
  in six external panels. Their inferred thickness is 4.50637 mm and combined
  heat capacity 489.79472 J/K. This allocation is a fixture prior, not measured
  wall thickness. Internal supports, if present, would change that allocation.
- ABS is a passive, non-damaging material here, with constant cp and k=0.18 W/mK.
  It has no pyrolysis, structural openings or material-loss process. Nevertheless
  its final voxel temperatures are 487.62–706.42 °C. Structural survival at those
  temperatures is an unresolved constitutive assumption, not established physics.

## Measured thermal-path evidence

At the complete final observation, t=5945 s:

- Median temperature difference between each battery's mean contents and its
  capacity-weighted steel can: **0.59151 K**. Median can-to-local-body-gas
  difference: **6.81634 K**. These mean differences do not describe every local
  gradient or the earlier burst, but do not support a large *late* core/can
  bottleneck.
- Applying the native directed law to observed local gas and ABS temperatures:
  inside gas-to-ABS net heat **31.55658 W**, summed conductance **0.902013 W/K**.
  Film conductances are identical at solid D=0 and D=1 for these configured
  laws: maximum difference exactly zero. Missing solid damage snapshots do not
  invalidate those ordinary-film fluxes in this particular scene.
- External local gas temperatures were NOT recorded individually, only group
  min/max/inventories. No actual external film flux has been fabricated.
  A counterfactual using the *existing* native contact law with cold 298.15 K
  air gives **298.48145 W** from ABS. This is a cold-air capacity bound, NOT
  actual exported heat, NOT evidence that such a boundary is experimentally
  justified and NOT a predicted trajectory after changing it.
- The actual whole-domain ambient thermal export averaged over 5895–5945 s
  is **46.45775 W**; R3 release is zero. Total gas sensible storage changes by
  **−0.29816 W** (module −0.00274 W, outside −0.29469 W). Thus gas is not a large
  late *energy reservoir*, but gas films and near-wall transport can still
  control heat removal from solids. Whole export and instantaneous cold-air
  bound are different quantities and timescales, not directly interchangeable.
- Earlier accepted 3300–3500 s channel ledgers showed only 0.02266 W net thermal
  export through module vents vs 48.0989 W to ambient. Their fourfold enlargement
  did not improve cooling. Do not resume vent/chemistry fitting as a substitute
  for this solid/boundary audit.

## Resistance bookkeeping and next diagnostic

The exterior ABS film includes half-slab resistance t/(2k); the interior film
currently does not. For the static reference h=8 W/m²K this omits
0.012518 m²K/W from the two-sided series path. It makes cooling slightly
**too strong**, not too weak; it cannot explain the excessively hot tail.

The next controlled geometry correction is to connect the twelve *actually
touching* ABS seams through the existing solid connectivity law and material
half-cell resistances, without adding fictitious can/pad contacts. Its effect
must be replayed before calling it the cause. Separately inspect actual exterior
film flux/local air temperature and missing hot-wall losses/ABS structural
evolution. Radiation is already flagged in the preceding cooling audit; do not
hide it in arbitrary k or h. Retain 20 mm and fixed chemistry. Source support
contacts cannot be uniquely inferred from the available schematic alone.

No new simulation or physical correction was performed by this audit. Topology
component count, full observation coverage and D=0/D=1 film-independence checks
passed against the archived complete replay.

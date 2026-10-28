#!/usr/bin/env python3
"""One-factor, fresh 20 mm vent-retention diagnostic; never a fitted geometry.

Restrict the TWO existing module exits, not battery valves or ambient renewal.
Reuse the learned primary-aligned chemistry and unchanged CUDA/sequence guards.
Net vent mass includes displaced air: it is NOT a tagged source-gas inventory.
"""
import argparse
from dataclasses import asdict
import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import tarfile

import numpy as np

import calibrate_upv_r123 as calibration
from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import (
    BASE, BASE_HASH, CORE, PROBE, MINIMUM_SPEED, read_observed_trial, trial,
)
from fsri30_data import ROOT, load, save, sha
from fsri30_multilevel import remesh
from fsri30_wave_fit import POLICY, reference
from fsri30_vent_geometry import module_exits, restrict_exits

SEED = ROOT/'data/calibration/fsri30-all-waves-20mm-20261026-r1/h-20mm/trial-0001'
SCALES = (.25, .1)


def read_run(row):
    path = Path(row['run_path'])
    if sha(path) != row['run_sha256']:
        raise ValueError('Native history changed')
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def snapshot(scene, run, start, end):
    """Accepted ledgers on actual samples only; no future/peak extrapolation."""
    times = np.asarray(run['time_s'])
    if not times[0] <= start < end <= times[-1]+1e-7:
        raise ValueError('Requested interval is outside observed history')
    i = max(0, int(np.searchsorted(times, start+1e-7, side='right'))-1)
    j = max(0, int(np.searchsorted(times, end+1e-7, side='right'))-1)
    if j <= i:
        raise ValueError('Interval has fewer than two observations')
    def change(values):
        return float(values[j]-values[i])
    totals = run['process_totals']
    outside = run['gas_groups']['outside']
    if scene['gas_history_groups']['outside'] != ['air-']:
        raise ValueError('Outside budget must cover every resolved air volume')
    cut = module_exits(scene,allow_closed=bool(scene.get('metadata',{}).get('sealed_module_diagnostic')))
    vent = {field: sum(change(run['channels'][key][field]) for key in cut)
            for field in ('transferred_mass_kg', 'transferred_chemical_j',
                          'delivered_gas_energy_j', 'transferred_oxidizer_kg')}
    source = scene['metadata']['heated_component']
    film = {}
    for value in run['gas_film_trace'].values():
        name = value['solid'].split(':case:')[0]
        film[name] = film.get(name, 0.)+change(value['heat_to_solid_j'])
    other_cans = {key: value for key, value in film.items()
                  if key in run['cell_mean_temperature_k'] and key != source}
    # Global minus exterior includes unobserved INSIDE-battery pores. The saved
    # group named 'module' contains only body/top/base cavities, not those pores.
    burned_outside = change(outside['released_energy_j'])
    burned_inside = change(totals['gas_released_energy_j'])-burned_outside
    neighbors = {key: float(values[j]-273.15)
                 for key, values in run['cell_mean_temperature_k'].items() if key != source}
    hottest = max(neighbors, key=neighbors.get)
    return dict(
        interval_s=[float(times[i]), float(times[j])],
        vent_net=vent,
        escaped_thermal_j=change(totals['escaped_energy_j']),
        escaped_chemical_j=change(totals['escaped_chemical_j']),
        escaped_net_mass_kg=change(totals['escaped_mass_kg']),
        r3_inside_including_battery_pores_j=burned_inside,
        r3_outside_j=burned_outside,
        remaining_fuel_inside_j=float(totals['gas_chemical_j'][j]-outside['chemical_j'][j]),
        remaining_fuel_outside_j=float(outside['chemical_j'][j]),
        oxygen_inside_kg=float(totals['gas_oxidizer_kg'][j]-outside['oxidizer_kg'][j]),
        gas_sensible_inside_change_j=change(totals['gas_sensible_energy_j'])-change(outside['sensible_energy_j']),
        gas_sensible_outside_change_j=change(outside['sensible_energy_j']),
        film_to_source_can_j=film.get(source, 0.),
        film_to_other_cans_j=sum(other_cans.values()),
        film_to_other_cans_by_id_j=other_cans,
        film_to_abs_j=sum(value for key, value in film.items() if key.startswith('ABS:')),
        solid_boundary_out_j=change(totals['thermal_boundary_energy_out_j']),
        hottest_neighbor=dict(id=hottest, mean_temperature_c=neighbors[hottest]),
        maximum_gas_pressure_pa=float(max(v['maximum_pressure_pa'][j]
                                          for v in run['gas_groups'].values())),
        pressure_scope='Observed exterior and body/top/base cavities; not closed battery pores',
        mass_scope='Signed net gas including displaced air; not tagged battery effluent',
    )


def budget_report(scene, run, row):
    source = scene['metadata']['heated_component']
    pulses = row['assessment'].get('model', {}).get('pulses', [])
    primary = min((p['time_s'] for p in pulses if p['channel'] == source), default=None)
    last = float(run['time_s'][-1])
    report = dict(last_observed_s=last, primary_s=primary, partial=bool(run.get('partial')),
                  assessment=row['assessment'], physical_error=row.get('physical_error'),
                  calibrated=False, validation_passed=False,
                  native_sha256=row['native_sha256'], run_sha256=row['run_sha256'],
                  since_start=snapshot(scene, run, run['time_s'][0], last), post_primary=[])
    if primary is not None:
        start = max(float(run['time_s'][0]), primary-11.)
        for requested in sorted(set([primary+4., primary+39., primary+99., 2135., last])):
            if start < requested <= last+1e-7:
                report['post_primary'].append(snapshot(scene, run, start, min(requested, last)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(PROBE) != CORE or sha(BASE) != BASE_HASH:
        raise ValueError('Pinned core or retained physical base changed')
    data, manifest = load()
    ref = reference(data)
    mesh = remesh(json.loads(BASE.read_text()), 20)
    seed = read_observed_trial(SEED, 20, mesh, ref)
    if not seed or seed.get('physical_error'):
        raise ValueError('Need the actual primary-aligned, conservation-passing seed')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'diagnostic.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'contract.json').exists():
        raise ValueError('Preserve previous diagnostic; choose a fresh output directory')
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'scripts').glob('*.py')}
    contract = dict(native_sha256=CORE, base_sha256=BASE_HASH, seed_trial=str(SEED),
                    seed_run_sha256=seed['run_sha256'], seed_status_sha256=sha(SEED/'status.json'),
                    seed_scenario_sha256=sha(SEED/'scenario.json'), parameters=seed['parameters'],
                    outlet_access_scales=SCALES, affected_channels=module_exits(mesh),
                    csv_sha256=manifest['csv_sha256'], sources=sources,
                    policy=asdict(POLICY), minimum_regular_step_s=.01,
                    minimum_realtime_ratio=MINIMUM_SPEED, mesh_mm=20,
                    fresh_from_zero=True, checkpoints_reused=False,
                    ambient_renewal_changed=False, solid_geometry_changed=False,
                    diagnostic_only=True, calibrated=False, validation_passed=False)
    save(out/'contract.json', contract)
    with tarfile.open(out/'source-policy.tar.gz', 'w:gz') as archive:
        for path in sources:
            archive.add(ROOT/path, arcname=path)
    save(out/'reference-waves.json', ref)
    baseline_scene = json.loads((SEED/'scenario.json').read_text())
    baseline = budget_report(baseline_scene, read_run(seed), seed)
    save(out/'baseline-budget.json', baseline)
    rows = []
    state = dict(status='running', pid=os.getpid(), active_trial=None,
                 diagnostic_only=True, calibrated=False, validation_passed=False,
                 native_sha256=CORE, reference_primary_s=baseline['primary_s'])
    calibration.PROBE = PROBE

    def stop(*unused):
        raise KeyboardInterrupt

    def update(row):
        state.update(active_trial=row['trial'], progress={key: row[key] for key in
                     ('assessment', 'physical_error', 'wall_s', 'live_advanced_to_s',
                      'minimum_regular_step_s') if key in row})
        atomic_save(out/'status.json', state)

    signal.signal(signal.SIGTERM, stop)
    atomic_save(out/'status.json', state)
    try:
        for scale in SCALES:
            folder = out/('exits-'+str(scale))
            state.update(outlet_access_scale=scale, active_trial=str(folder), progress={})
            atomic_save(out/'status.json', state)
            # A fresh run per case. Area is the sole changed input field; even
            # observation landing points and the source heater remain unchanged.
            row = trial(folder, restrict_exits(mesh, scale), seed['parameters'],
                        20, ref, data, update)
            row['outlet_access_scale'] = scale
            if row.get('run_path'):
                scene = json.loads((folder/'scenario.json').read_text())
                report = budget_report(scene, read_run(row), row)
                save(folder/'gas-budget.json', report)
            else:
                report = None
            rows.append(dict(outlet_access_scale=scale, trial=row, budget=report))
            save(out/'comparison.json', dict(baseline=baseline, variants=rows,
                                           diagnostic_only=True, validation_passed=False))
            state.update(completed_variants=len(rows), last_result=row)
            atomic_save(out/'status.json', state)
            print(json.dumps(dict(scale=scale, status=row['status'],
                                  assessment=row['assessment'])), flush=True)
        state.update(status='complete', result=str(out/'comparison.json'),
                     note='Two diagnostic sensitivities completed, not a calibrated/validated profile')
    except KeyboardInterrupt:
        state['status'] = 'stopped'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        atomic_save(out/'status.json', state)


if __name__ == '__main__':
    main()

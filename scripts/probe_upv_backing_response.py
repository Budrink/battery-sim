#!/usr/bin/env python3
"""Contract-checked common Al exposure contrasts on the current gas-response law.

First capture the response change at the sealed checkpoint without advancing.
Then use native conservative geometry remapping. No checkpoint identity bypass.
"""
import argparse
import gzip
import hashlib
import json
import itertools
import os
from pathlib import Path
import signal
import subprocess

import calibrate_upv_r123 as cal
from check_upv_restart import native, compare
from probe_upv_gas_access import plot
from refine_upv_gas_access import temperature_pruner
from upv_backing_response import candidate
from upv_cooling_fit import rejection_reason
from upv_cooling_review import phase_report, plot_cooling
from upv_gas_access import received_gas_heat
from upv_gas_response_refinement import assess, front_prune_reason, CRITERIA_VERSION
import upv_r123_data as dataio


def load(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def capture(reference, response, out, contract):
    """Only the native response contract may authorize the zero-step change."""
    scene = json.loads((response / 'scenario.json').read_text())
    scene['simulation']['duration_s'] = 835.
    scene['failure_sample_times_s'] = [0., 835.]
    cal.save(out / 'capture-scenario.json', scene)
    request = dict(restart_contract=contract, restart_from=str(reference / 'sealed-state.bin'),
        restart_gas_response_scenario=str(reference / 'prefix-scenario.json'),
        save_unvented_restart=str(out / 'sealed-state.bin'), stop_after_first_rupture=False)
    with (out / 'capture.json').open('w') as dst, (out / 'capture.log').open('w') as log:
        result = subprocess.run([str(cal.PROBE), str(out / 'capture-scenario.json')],
            input=json.dumps(request), text=True, stdout=dst, stderr=log, timeout=60.)
    if result.returncode:
        raise RuntimeError((out / 'capture.log').read_text())
    run = json.loads((out / 'capture.json').read_text())
    if 'error' in run:
        raise RuntimeError(run['error'])
    if run['started_at_s'] != 835. or run['advanced_to_s'] != 835. or run.get('first_rupture_s') is not None:
        raise ValueError('Capture must stay at the same sealed 835 s state')
    baseline = load(response / 'replay/run.json.gz')
    if any(run['temperature_k'][k][0] != values[0] for k, values in baseline['temperature_k'].items()):
        raise ValueError('Zero-step capture changed temperatures')
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--response', type=Path, required=True, help='Completed current response trial')
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fractions', nargs='+', type=float, default=[.03, .1, .3])
    ap.add_argument('--lip-fractions', nargs='+', type=float, help='Common holder rim exposure; default unchanged')
    args = ap.parse_args()
    ref, response, out = args.reference.resolve(), args.response.resolve(), args.output.resolve()
    cal.PROBE = args.probe.resolve()
    core = hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    ref_state = json.loads((ref / 'status.json').read_text())
    response_state = json.loads((response.parent / 'status.json').read_text())
    if ref_state['status'] != 'complete' or any(s['native_sha256'] != core for s in (ref_state, response_state)):
        raise ValueError('Require completed exact-core reference and matching response core')
    completed = [r for r in response_state['records'] if r['trial'] == response.name and r['status'] == 'complete']
    if not completed or not completed[0]['preserved_front']['accepted']:
        raise ValueError('Response seed must retain the experimental pulse')
    scene = json.loads((response / 'scenario.json').read_text())
    end = scene['simulation']['duration_s']
    if not 1100. <= end <= 1800.:
        raise ValueError('Require resolved cooling interval')
    grid = list(itertools.product(args.fractions, args.lip_fractions or [None]))
    for fraction, lip in grid:
        candidate(scene, fraction, lip)
    out.mkdir()
    data, record = dataio.load('no-cooling')
    event = dataio.event(data)
    strict_prune = temperature_pruner(data)
    def prune(run):
        return strict_prune(run) or rejection_reason(run, data, event) or front_prune_reason(run, data)
    state = dict(status='running', pid=os.getpid(), reference=str(ref), response=str(response),
        native_sha256=core, response_sha256=hashlib.sha256((response / 'scenario.json').read_bytes()).hexdigest(),
        experimental_sha256=record['canonical_sha256'], criteria_version=CRITERIA_VERSION,
        end_s=end, minimum_regular_dt_s=scene['simulation']['minimum_dt_s'],
        chemistry_frozen=True, materials_frozen=True, solid_contacts_frozen=True,
        numerical_tolerances_frozen=True, geometry_is_unmeasured_hypothesis=True,
        accepted_calibration=False, records=[], best=None)
    for filename in ('probe_upv_backing_response.py', 'upv_backing_response.py', 'upv_gas_response_refinement.py'):
        state[filename + '_sha256'] = hashlib.sha256((Path(__file__).parent / filename).read_bytes()).hexdigest()
    cal.save(out / 'status.json', state)
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    baseline = load(response / 'replay/run.json.gz')
    runs = {'Контроль': baseline}
    try:
        capture(ref, response, out, core)
        mapping = [dict(to=i, fraction=1., **{'from': i})
                   for i in range(len(scene['gas_network']['compartments']))]
        for index, (fraction, lip) in enumerate(grid):
            folder = out / f'trial-{index:03d}'
            folder.mkdir()
            cal.save(folder / 'scenario.json', candidate(scene, fraction, lip))
            state['active_trial'] = folder.name
            cal.save(out / 'status.json', state)
            print('START', folder.name, 'common backing exposure', fraction, 'rim', lip, flush=True)
            row = dict(trial=folder.name, backing_open_fraction=fraction, lip_open_fraction=lip, conditional_restart=True)
            try:
                if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest() != core:
                    raise RuntimeError('Native binary changed')
                run = native(folder / 'scenario.json', folder / 'replay', dict(restart_contract=core,
                    restart_from=str(out / 'sealed-state.bin'), restart_source_scenario=str(out / 'capture-scenario.json'),
                    unvented_gas_remap=mapping, stop_after_first_rupture=False),
                    max(900., 2.2 * (end - 835.)), prune_fn=prune)
                original_geometry = scene['metadata']['gas_access_geometry']
                neutral = fraction == original_geometry['backing_open_fraction'] and (
                    lip is None or lip == original_geometry['holder_lip_open_fraction'])
                if neutral:
                    control = compare(baseline, run)
                    cal.save(folder / 'neutral-restart-comparison.json', control)
                    if control['max_absolute_difference'] > 1e-7:
                        raise ValueError('Neutral two-stage restart changed the trajectory')
                result = assess(run, data, end)
                cal.save(folder / 'assessment.json', result)
                cal.save(folder / 'phases.json', phase_report(run, data, end))
                cal.save(folder / 'gas-heat.json', dict(burst=received_gas_heat(run),
                    cooling=received_gas_heat(run, 950., end)))
                row.update(status='complete', **result, realtime_ratio=(end - 835.) / run['wall_s'],
                    initial_conservation=run['rebase_initial_residuals'],
                    neutral_control=neutral,
                    balances={k: v for k, v in run.items() if k.startswith('max_') and 'residual' in k})
                runs[f'Доступ Al {100*fraction:g}%; край {lip}'] = run
                plot_cooling(out, runs, data, {k: k for k in runs})
                plot(out, runs, data, filename='burst-comparison.png',
                    title='Общая доступная газу площадь Al; прочие законы фиксированы')
                if result['preserved_front']['accepted'] and (state['best'] is None or row['objective'] < state['best']['objective']):
                    state['best'] = dict(row)
            except (RuntimeError, TimeoutError) as exc:
                row.update(status='rejected', error=str(exc))
            state['records'].append(row)
            cal.save(folder / 'result.json', row)
            cal.save(out / 'status.json', state)
            print('RESULT', json.dumps({k: row[k] for k in ('trial', 'status', 'objective', 'realtime_ratio', 'error') if k in row}), flush=True)
        state['status'] = 'contrasts_complete_not_calibrated'
    except KeyboardInterrupt:
        state['status'] = 'stopped'
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        state.pop('active_trial', None)
        cal.save(out / 'status.json', state)


if __name__ == '__main__':
    main()

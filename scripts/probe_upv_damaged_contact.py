#!/usr/bin/env python3
"""Fresh-from-ambient contrasts of the existing damaged battery/TIM contact.

No checkpoint can authorize a changed solid contact. Never reuse a sealed gas
prefix here, even if it appears inexpensive. Every trial runs its own warmup.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as cal
from check_upv_restart import native
from probe_upv_gas_access import plot
from refine_upv_gas_access import temperature_pruner
from upv_cooling_fit import rejection_reason
from upv_cooling_review import phase_report, plot_cooling
from upv_damaged_contact import candidate
from upv_gas_access import received_gas_heat
from upv_gas_response_refinement import assess, front_prune_reason, CRITERIA_VERSION
import upv_r123_data as dataio


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True, help='Explicit current scenario, never legacy parameter reconstruction')
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fractions', nargs='+', type=float, default=[.02, .1, .5, 1.])
    args = ap.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    scene = json.loads(source.read_text())
    end = scene['simulation']['duration_s']
    if not 1100. <= end <= 1800.:
        raise ValueError('Require resolved cooling tail')
    cal.PROBE = args.probe.resolve()
    core = hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    for fraction in args.fractions:
        candidate(scene, fraction)
    data, record = dataio.load('no-cooling')
    event = dataio.event(data)
    strict_prune = temperature_pruner(data)
    def prune(run):
        return strict_prune(run) or rejection_reason(run, data, event) or front_prune_reason(run, data)
    out.mkdir()
    state = dict(status='running', pid=os.getpid(), source=str(source), native_sha256=core,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), end_s=end,
        experimental_sha256=record['canonical_sha256'], criteria_version=CRITERIA_VERSION,
        minimum_regular_dt_s=scene['simulation']['minimum_dt_s'], fresh_from_ambient=True,
        chemistry_frozen=True, materials_frozen=True, shell_frozen=True, gas_geometry_frozen=True,
        numerical_tolerances_frozen=True, accepted_calibration=False, records=[], best=None)
    for filename in ('probe_upv_damaged_contact.py', 'upv_damaged_contact.py', 'upv_gas_response_refinement.py'):
        state[filename + '_sha256'] = hashlib.sha256((Path(__file__).parent / filename).read_bytes()).hexdigest()
    cal.save(out / 'status.json', state)
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    runs = {}
    try:
        for index, fraction in enumerate(args.fractions):
            folder = out / f'trial-{index:03d}'
            folder.mkdir()
            cal.save(folder / 'scenario.json', candidate(scene, fraction))
            state['active_trial'] = folder.name
            cal.save(out / 'status.json', state)
            print('START FRESH', folder.name, 'damaged battery/TIM retention', fraction, flush=True)
            row = dict(trial=folder.name, retained_fraction=fraction, fresh_from_ambient=True)
            try:
                if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest() != core:
                    raise RuntimeError('Native binary changed')
                run = native(folder / 'scenario.json', folder / 'replay',
                    dict(restart_contract=core, stop_after_first_rupture=False),
                    max(2300., 2.2 * end), prune_fn=prune)
                if run['started_at_s'] != 0.:
                    raise ValueError('Solid-contact trial must start from ambient')
                if run['solver_profile']['minimum_regular_step_s'] < scene['simulation']['minimum_dt_s'] - 1e-10:
                    raise ValueError('Regular timestep below fixed floor')
                result = assess(run, data, end)
                cal.save(folder / 'assessment.json', result)
                cal.save(folder / 'phases.json', phase_report(run, data, end))
                cal.save(folder / 'gas-heat.json', dict(burst=received_gas_heat(run),
                    cooling=received_gas_heat(run, 950., end)))
                row.update(status='complete', **result, realtime_ratio=end / run['wall_s'],
                    minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
                    balances={k: v for k, v in run.items() if k.startswith('max_') and 'residual' in k})
                label = f'Остаточный контакт {100*fraction:g}%'
                runs[label] = run
                plot_cooling(out, runs, data, {k: k for k in runs})
                plot(out, runs, data, filename='burst-comparison.png',
                    title='Изменяется только остаточная связь батарея/TIM; каждый расчёт с окружающей T')
                if result['preserved_front']['accepted'] and (state['best'] is None or row['objective'] < state['best']['objective']):
                    state['best'] = dict(row)
                if result['accepted']:
                    state['status'] = 'cooling_tolerances_met_pending_full_shape_and_validation'
            except (RuntimeError, TimeoutError) as exc:
                row.update(status='rejected', error=str(exc))
            state['records'].append(row)
            cal.save(folder / 'result.json', row)
            cal.save(out / 'status.json', state)
            print('RESULT', json.dumps({k: row[k] for k in ('trial', 'status', 'objective', 'realtime_ratio', 'error') if k in row}), flush=True)
            if state['status'] != 'running':
                break
        else:
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

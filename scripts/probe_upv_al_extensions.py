#!/usr/bin/env python3
"""Fresh-from-ambient common Al extension geometry contrasts."""
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
from upv_al_extensions import candidate
from upv_cooling_fit import rejection_reason
from upv_cooling_review import phase_report, plot_cooling
from upv_gas_access import received_gas_heat
from upv_gas_response_refinement import front_prune_reason, CRITERIA_VERSION
from upv_gas_route_refinement import assessment
import upv_r123_data as dataio


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True, help='Explicit completed current trial directory')
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--lengths-mm', type=float, nargs='+', default=[5., 15.])
    args = ap.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    scene = json.loads((source/'scenario.json').read_text())
    end = scene['simulation']['duration_s']
    if not 1100. <= end <= 1800.:
        raise ValueError('Require resolved cooling tail')
    for length in args.lengths_mm:
        candidate(scene, length/1000.)
    with gzip.open(source/'replay/run.json.gz', 'rt') as f:
        baseline = json.load(f)
    if baseline['advanced_to_s'] != end:
        raise ValueError('Reference must be completed')
    cal.PROBE = args.probe.resolve()
    core = hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    data, record = dataio.load('no-cooling')
    event, strict_prune = dataio.event(data), temperature_pruner(data)
    def prune(run):
        return strict_prune(run) or rejection_reason(run, data, event) or front_prune_reason(run, data)
    out.mkdir()
    state = dict(status='running', pid=os.getpid(), source=str(source), native_sha256=core,
        source_sha256=hashlib.sha256((source/'scenario.json').read_bytes()).hexdigest(), end_s=end,
        experimental_sha256=record['canonical_sha256'], criteria_version=CRITERIA_VERSION,
        minimum_regular_dt_s=scene['simulation']['minimum_dt_s'], fresh_from_ambient=True,
        chemistry_frozen=True, material_properties_frozen=True, shell_frozen=True,
        numerical_tolerances_frozen=True, geometry_is_unmeasured_hypothesis=True,
        accepted_calibration=False, records=[], best=None)
    for filename in ('probe_upv_al_extensions.py', 'upv_al_extensions.py',
                     'upv_gas_response_refinement.py', 'upv_gas_route_refinement.py'):
        state[filename+'_sha256'] = hashlib.sha256((Path(__file__).parent/filename).read_bytes()).hexdigest()
    cal.save(out/'status.json', state)
    cal.save(out/'reference-assessment.json', assessment(baseline, data, end))
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    runs = {'Исходная геометрия': baseline}
    try:
        for index, length in enumerate(args.lengths_mm):
            folder = out/f'trial-{index:03d}'
            folder.mkdir()
            cal.save(folder/'scenario.json', candidate(scene, length/1000.))
            state['active_trial'] = folder.name
            cal.save(out/'status.json', state)
            print('START FRESH', folder.name, 'Al bottom extension mm', length, flush=True)
            row = dict(trial=folder.name, extension_mm=length, fresh_from_ambient=True)
            try:
                if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest() != core:
                    raise RuntimeError('Native binary changed')
                run = native(folder/'scenario.json', folder/'replay',
                    dict(restart_contract=core, stop_after_first_rupture=False),
                    max(2300., 2.2*end), prune_fn=prune)
                if run['started_at_s'] != 0.:
                    raise ValueError('Solid geometry trial must start from ambient')
                if run['solver_profile']['minimum_regular_step_s'] < scene['simulation']['minimum_dt_s']-1e-10:
                    raise ValueError('Regular timestep below fixed floor')
                result = assessment(run, data, end)
                cal.save(folder/'assessment.json', result)
                cal.save(folder/'phases.json', phase_report(run, data, end))
                cal.save(folder/'gas-heat.json', dict(burst=received_gas_heat(run),
                    cooling=received_gas_heat(run, 950., end)))
                row.update(status='complete', **result, realtime_ratio=end/run['wall_s'],
                    minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
                    balances={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k})
                runs[f'Выступы Al {length:g} мм'] = run
                plot_cooling(out, runs, data, {k:k for k in runs})
                plot(out, runs, data, filename='burst-comparison.png',
                    title='Геометрия выступов Al; одинаковые материалы, контакты и химия')
                # Also retain explicitly per-trial pictures: latest is not best.
                selected = {'Исходная геометрия':baseline, f'Выступы Al {length:g} мм':run}
                plot(folder, selected, data, filename='burst-comparison.png')
                plot_cooling(folder, selected, data, {k:k for k in selected})
                if result['preserved_front']['accepted'] and (state['best'] is None or row['objective'] < state['best']['objective']):
                    state['best'] = dict(row)
            except (RuntimeError, TimeoutError) as exc:
                row.update(status='rejected', error=str(exc))
            state['records'].append(row)
            cal.save(folder/'result.json', row)
            cal.save(out/'status.json', state)
            print('RESULT', json.dumps({k:row[k] for k in ('trial','status','objective','realtime_ratio','error') if k in row}), flush=True)
        state['status'] = 'contrasts_complete_not_calibrated'
    except KeyboardInterrupt:
        state['status'] = 'stopped'
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        state.pop('active_trial', None)
        cal.save(out/'status.json', state)


if __name__ == '__main__':
    main()

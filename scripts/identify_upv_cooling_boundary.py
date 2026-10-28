#!/usr/bin/env python3
"""Sequential native identification of cooling only, then conditioned holdout."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as cal
from check_upv_restart import native
import upv_r123_data as dataio
from upv_cooling_boundary import candidate, surrogate_seed, boundary_score, plot_boundary
from upv_cooling_review import phase_report, plot_cooling
from upv_frozen_validation import digest
from report_upv_frozen_comparison import plot as plot_process


def read_run(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def preserve_conditioned_check(folder, state, data):
    """A failed check must retain honest known-prefix metrics and plots too."""
    paths = [folder/'replay'/name for name in ('run.json.gz', 'incomplete-run.json.gz')]
    existing = next((path for path in paths if path.exists()), None)
    if existing is None:
        return
    run = read_run(existing)
    scene = json.loads((folder/'scenario.json').read_text())
    state.update(known_until_s=run['time_s'][-1], advanced_to_s=run['advanced_to_s'],
        wall_s=run['wall_s'], realtime_ratio=(run['advanced_to_s']-run['started_at_s'])/run['wall_s'],
        refitted=True, accepted_validation=False, fully_blind_validation=False,
        balances={k: v for k, v in run.items() if k.startswith('max_') and 'residual' in k})
    cal.save(folder/'phases.json', phase_report(run, data, scene['simulation']['duration_s']))
    if run['time_s'][-1] >= 650.:
        plot_boundary(folder, scene, run, data)
    label = 'Проверка с охлаждением по 110–112; химия заморожена'
    if run.get('partial'):
        label += ' — НЕПОЛНЫЙ ПРОГОН'
    plot_process(run, data, run['time_s'][-1], label, folder/'checking-only-known-prefix.png', 'overview')
    plot_cooling(folder, {'conditioned': run}, data,
        {'conditioned': 'Расчёт с определённой границей охлаждения'}, pressure_panel=True)
    cal.save(folder/'status.json', state)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-cooled', type=Path, required=True)
    ap.add_argument('--reference-run', type=Path, required=True)
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--iterations', type=int, default=2)
    ap.add_argument('--full-after', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.iterations <= 4:
        raise ValueError('Bounded search: one to four native prefixes')
    out = args.output.resolve(); out.mkdir()
    source = json.loads(args.source_cooled.read_text()); source_hash = digest(source)
    data, record = dataio.load('active-cooling')
    cal.PROBE = args.probe.resolve(); core = hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    reference = read_run(args.reference_run)
    parameters, inference = surrogate_seed(source, reference, data)
    state = dict(status='running', pid=os.getpid(), native_sha256=core,
        source_cooled=str(args.source_cooled.resolve()), source_cooled_sha256=source_hash,
        data_sha256=record['canonical_sha256'], fitted_channels=['110', '111', '112'],
        chemistry_refitted=False, battery_temperatures_or_pressure_used_for_selection=False,
        validation_claim='Cooling-boundary-conditioned check, NOT fully blind validation', trials=[])
    def save(): cal.save(out/'status.json', state)
    def stop(*unused): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop); save()
    best = None
    try:
        for index in range(args.iterations):
            folder = out/f'prefix-{index:02d}'; folder.mkdir()
            scene = candidate(source, data, parameters)
            scene['simulation']['duration_s'] = 1250.
            scene['failure_sample_times_s'] = [t for t in scene['failure_sample_times_s'] if t <= 1250.]
            cal.save(folder/'scenario.json', scene)
            state.update(phase='native_cooling_identification', current_trial=index, parameters=parameters, initializer=inference); save()
            print('START boundary prefix', index, json.dumps(parameters), flush=True)
            run = native(folder/'scenario.json', folder/'replay', dict(restart_contract=core,
                stop_after_first_rupture=False), 1250.*5.5, minimum_speed=.2, rolling_minimum=.2)
            score = boundary_score(scene, run, data)
            entry = dict(parameters=parameters, initializer=inference, metrics=score,
                         run=str(folder), wall_s=run['wall_s'])
            cal.save(folder/'boundary-fit.json', entry); plot_boundary(folder, scene, run, data)
            state['trials'].append(entry); save(); print('BOUNDARY', json.dumps(score), flush=True)
            if best is None or score['score'] < best['metrics']['score']: best = entry
            proposed, inference = surrogate_seed(scene, run, data)
            # Once both independent signals are inside the engineering envelope,
            # do not optimize battery observations as an undeclared extra target.
            if score['score'] <= 1.: break
            change = max(abs(proposed[k]/parameters[k]-1.) for k in parameters)
            if change < .03: break
            parameters = proposed
        state.update(status='boundary_identified', best=best)
        cal.save(out/'identified-boundary.json', best); save()
        if args.full_after:
            folder = out/'conditioned-replay'; folder.mkdir()
            scene = candidate(source, data, best['parameters'])
            cal.save(folder/'scenario.json', scene)
            state.update(status='running', phase='conditioned_check', requested_end_s=scene['simulation']['duration_s']); save()
            print('START full boundary-conditioned check', flush=True)
            run = native(folder/'scenario.json', folder/'replay', dict(restart_contract=core,
                stop_after_first_rupture=False), 5.5*scene['simulation']['duration_s'], minimum_speed=.2, rolling_minimum=.2)
            state.update(status='complete', phase='conditioned_check', wall_s=run['wall_s'],
                         known_until_s=run['time_s'][-1], balances={k: v for k, v in run.items() if k.startswith('max_') and 'residual' in k})
    except KeyboardInterrupt: state['status'] = 'stopped'
    except (RuntimeError, TimeoutError, ValueError) as exc:
        state.update(status='rejected', error=str(exc)); print('REJECTED', str(exc), flush=True)
    finally:
        if digest(json.loads(args.source_cooled.read_text())) != source_hash:
            raise RuntimeError('Original frozen attempt changed during identification')
        preserve_conditioned_check(out/'conditioned-replay', state, data)
        save(); print('RESULT', state['status'], flush=True)


if __name__ == '__main__': main()

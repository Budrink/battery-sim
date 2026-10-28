#!/usr/bin/env python3
"""Fresh sealed prefix, then bounded shape/arrival-aware gas-route search."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as cal
from check_upv_restart import native, compare
from probe_upv_gas_access import plot
from refine_upv_gas_access import temperature_pruner
from upv_cooling_fit import rejection_reason
from upv_cooling_review import phase_report, plot_cooling
from upv_gas_access import received_gas_heat
from upv_gas_response_refinement import front_prune_reason, CRITERIA_VERSION
from upv_gas_route_refinement import candidate, assessment
from upv_tim_gas_access import candidate as tim_candidate
import upv_r123_data as dataio
from upv_observation_policy import metadata as observation_policy


def load(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True, help='Explicit completed current full trial directory')
    ap.add_argument('--probe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-trials', type=int, default=12)
    ap.add_argument('--resume', action='store_true', help='Reuse this exact source/core prefix, preserving prior trials')
    ap.add_argument('--tim-mouth', type=float, help='One common TIM gas-access contrast; all other geometry at source values')
    args = ap.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    cal.PROBE = args.probe.resolve()
    core = hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    original = json.loads((source / 'scenario.json').read_text())
    baseline = load(source / 'replay/run.json.gz')
    end = original['simulation']['duration_s']
    provenance = json.loads((source.parent / 'status.json').read_text())
    if provenance['native_sha256'] != core or baseline['started_at_s'] != 0. or baseline['advanced_to_s'] != end:
        raise ValueError('Require completed exact-core full current profile')
    if baseline.get('partial') or end < 1100.:
        raise ValueError('Require resolved full cooling reference')
    data, record = dataio.load('no-cooling')
    event = dataio.event(data)
    strict_prune = temperature_pruner(data)
    def prune(run):
        return strict_prune(run) or rejection_reason(run, data, event) or front_prune_reason(run, data)
    g = original['metadata']['gas_access_geometry']
    mouth, pad = g['iss3_accessible_fraction'], g.get('open_iss_pad_wetted_fraction', 1.)
    old_tim = g['protected_accessible_fraction']
    tim_mouth = old_tim if args.tim_mouth is None else args.tim_mouth
    def make_candidate(scale, m, p):
        scene = candidate(original, scale, m, p)
        return scene if args.tim_mouth is None else tim_candidate(scene, tim_mouth)
    queue = [(1., mouth, pad, 'control'), (.6, mouth*3., .25, 'moderate-source-confinement'),
             (.3, mouth*3., .25, 'confined-source-contrast'),
             (1., mouth*3., .25, 'pad-and-witness-contrast'), (3., mouth*3., .25, 'directed-route-contrast'),
             (10., mouth*3., .25, 'wide-source-exhaust'), (3., mouth, .25, 'exhaust-only'),
             (1., mouth*3., pad, 'witness-mouth-only'), (3., mouth*6., .25, 'witness-bracket'),
             (.3, mouth*3., .25, 'opposite-source-exhaust')]
    if args.tim_mouth is not None:
        queue = [(1., mouth, pad, 'common-tim-access')]
    for scale, m, p, _ in queue:
        make_candidate(scale, m, p)
    out.mkdir(exist_ok=args.resume)
    prefix = json.loads(json.dumps(original))
    prefix['simulation']['duration_s'] = 835.
    prefix['failure_sample_times_s'] = [t for t in original['failure_sample_times_s'] if t <= 835.]
    if prefix['failure_sample_times_s'][-1] != 835.:
        prefix['failure_sample_times_s'].append(835.)
    if args.resume:
        if json.loads((out / 'prefix-scenario.json').read_text()) != prefix:
            raise ValueError('Resume changes prefix physics or observations')
    else:
        cal.save(out / 'prefix-scenario.json', prefix)
    state = dict(status='running', phase='fresh_prefix', pid=os.getpid(), source=str(source),
        native_sha256=core, source_sha256=hashlib.sha256((source / 'scenario.json').read_bytes()).hexdigest(),
        experimental_sha256=record['canonical_sha256'], criteria_version=CRITERIA_VERSION,
        end_s=end, minimum_regular_dt_s=original['simulation']['minimum_dt_s'],
        chemistry_frozen=True, materials_frozen=True, solid_contacts_frozen=True,
        numerical_tolerances_frozen=True, accepted_calibration=False,
        geometry_is_unmeasured_hypothesis=True, records=[], best=None,observation_policy=observation_policy())
    for name in ('refine_upv_gas_routes.py', 'upv_gas_route_refinement.py', 'upv_gas_response_refinement.py', 'upv_tim_gas_access.py'):
        state[name + '_sha256'] = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    if args.resume:
        previous = json.loads((out / 'status.json').read_text())
        if previous['status'] == 'running':
            raise ValueError('Stop the old controller before resuming')
        for key in ('source','native_sha256','source_sha256','experimental_sha256','criteria_version',
                    'end_s','minimum_regular_dt_s','upv_gas_route_refinement.py_sha256'):
            if previous[key] != state[key]:
                raise ValueError('Resume changes frozen source/objective: ' + key)
        audits = sorted(out.glob('status-before-resume-*.json'))
        cal.save(out / f'status-before-resume-{len(audits):03d}.json', previous)
        state['records'] = previous['records']
        eligible = [r for r in state['records'] if r['status'] == 'complete' and r['preserved_front']['accepted']]
        state['best'] = min(eligible, key=lambda r: r['objective']) if eligible else None
        recorded = {r['trial'] for r in state['records']}
        for folder in sorted(out.glob('trial-*')):
            if not folder.is_dir() or folder.name in recorded:
                continue
            if not (folder / 'scenario.json').exists():
                # SIGTERM can arrive between mkdir and writing the scenario.
                # Preserve the slot; no parameters or result were evaluated.
                state['records'].append(dict(trial=folder.name, status='interrupted',
                    error='Stopped before scenario write; empty slot preserved, not scored'))
                continue
            params = json.loads((folder / 'scenario.json').read_text())['metadata']['gas_route_refinement']
            state['records'].append(dict(trial=folder.name, status='interrupted',
                source_exhaust=params['source_exhaust'], witness_mouth=params['witness_mouth'],
                pad_footprint=params['pad_footprint'],
                tim_mouth=json.loads((folder / 'scenario.json').read_text())['metadata']['gas_access_geometry']['protected_accessible_fraction'],
                error='Partial retained; not scored'))
    cal.save(out / 'status.json', state)
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    seen = {(*tuple(round(r[k],9) for k in ('source_exhaust','witness_mouth','pad_footprint')),
             round(r.get('tim_mouth',old_tim),9))
            for r in state['records'] if r['status'] in ('complete','rejected')}
    first = max([int(p.name.split('-')[1]) for p in out.glob('trial-*') if p.is_dir()] + [-1]) + 1
    runs = {'Полный исходный': baseline}
    try:
        if args.resume:
            print('REUSE verified exact-core current-profile prefix; prior suffixes unchanged', flush=True)
            warm = load(out / 'prefix/run.json.gz')
        else:
            print('FRESH exact current-profile prefix 0..835 s', flush=True)
            warm = native(out / 'prefix-scenario.json', out / 'prefix', dict(restart_contract=core,
                save_unvented_restart=str(out / 'sealed-state.bin'), stop_after_first_rupture=False), 1900.)
        if warm['started_at_s'] != 0. or warm['advanced_to_s'] != 835. or warm.get('first_rupture_s') is not None:
            raise ValueError('Need sealed fresh 835 s checkpoint')
        # Cold history must be identical to the completed source profile.
        import numpy as np
        for key, values in warm['temperature_k'].items():
            if not np.array_equal(np.asarray(values), np.asarray(baseline['temperature_k'][key])[:len(values)]):
                raise ValueError('Fresh prefix disagrees with explicit current source')
        state.update(phase='conditional_geometry_suffix', fresh_prefix_wall_s=warm['wall_s'])
        mapping = [dict(to=i, fraction=1., **{'from': i})
                   for i in range(len(original['gas_network']['compartments']))]
        for index in range(first, args.max_trials):
            while queue and (*tuple(round(v, 9) for v in queue[0][:3]),round(tim_mouth,9)) in seen:
                queue.pop(0)
            if not queue:
                state['status'] = 'geometry_bracket_exhausted_not_calibrated'
                break
            scale, m, p, origin = queue.pop(0)
            seen.add(tuple(round(v, 9) for v in (scale, m, p, tim_mouth)))
            folder = out / f'trial-{index:03d}'
            folder.mkdir()
            cal.save(folder / 'scenario.json', make_candidate(scale, m, p))
            state['active_trial'] = folder.name
            cal.save(out / 'status.json', state)
            print('START', folder.name, 'exhaust', scale, 'ISS3 mouth', m, 'ISS pads', p, flush=True)
            row = dict(trial=folder.name, source_exhaust=scale, witness_mouth=m, pad_footprint=p, tim_mouth=tim_mouth,
                       origin=origin, conditional_restart=True)
            try:
                if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest() != core:
                    raise RuntimeError('Native binary changed')
                run = native(folder / 'scenario.json', folder / 'replay', dict(restart_contract=core,
                    restart_from=str(out / 'sealed-state.bin'), restart_source_scenario=str(out / 'prefix-scenario.json'),
                    unvented_gas_remap=mapping, stop_after_first_rupture=False), 900., prune_fn=prune)
                if (scale, m, p, tim_mouth) == (1., mouth, pad, old_tim):
                    control = compare(baseline, run)
                    cal.save(folder / 'neutral-restart-comparison.json', control)
                    if not control['bitwise_equal']:
                        raise ValueError('Neutral route restart changed the reference trajectory')
                result = assessment(run, data, end)
                cal.save(folder / 'assessment.json', result)
                cal.save(folder / 'phases.json', phase_report(run, data, end))
                cal.save(folder / 'gas-heat.json', dict(burst=received_gas_heat(run),
                    cooling=received_gas_heat(run, 950., end)))
                row.update(status='complete', **result, realtime_ratio=(end-835.)/run['wall_s'],
                    balances={k: v for k, v in run.items() if k.startswith('max_') and 'residual' in k})
                eligible = result['preserved_front']['accepted']
                old = state['best']
                if eligible and (old is None or row['objective'] < .99*old['objective']):
                    state['best'] = dict(row)
                    cal.save(out / 'best-candidate.json', row)
                    runs = {'Полный исходный': baseline, 'Кандидат': run}
                    plot_cooling(out, runs, data, {k: k for k in runs})
                    plot(out, runs, data, filename='burst-comparison.png',
                         title='Геометрия газовых маршрутов: форма и физическое время прихода; без сдвига')
                    if old is not None and args.tim_mouth is None:
                        for value, previous, axis in [(scale, old['source_exhaust'], 0),
                                (m, old['witness_mouth'], 1), (p, old['pad_footprint'], 2)]:
                            if value == previous:
                                continue
                            proposal = [scale, m, p]
                            lo, hi = [(.3, 10.), (.001, .05), (.2, .75)][axis]
                            proposal[axis] = min(hi, max(lo, value * math.sqrt(value / previous)))
                            queue.insert(0, (*proposal, 'follow-improving-geometry'))
                    if result['accepted']:
                        state['status'] = 'tolerances_met_pending_fresh_geometry_verification'
            except (RuntimeError, TimeoutError) as exc:
                row.update(status='rejected', error=str(exc))
            state['records'].append(row)
            cal.save(folder / 'result.json', row)
            cal.save(out / 'status.json', state)
            print('RESULT', json.dumps({k: row[k] for k in ('trial','status','objective','arrival_times','error') if k in row}), flush=True)
            if state['status'] != 'running':
                break
        else:
            state['status'] = 'geometry_trial_budget_not_calibrated'
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

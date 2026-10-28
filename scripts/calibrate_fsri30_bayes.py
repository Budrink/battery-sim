#!/usr/bin/env python3
"""Sequential CUDA Bayesian development search; no hidden validation claims.

Resume is deterministic. Immutable source scene, raw-data/core/source hashes,
physical bounds, all proposals, full trajectories and failed prefixes persist.
SIGTERM stops the active native child through the existing replay supervisor.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from bounded_bayes import propose
from fsri30_data import load, save, sha
from fsri30_objective import cascade, data_audit, loss, model_fields, numerical_comparison, reject_reason
from fsri30_early_stop import rejection as cooling_rejection
from fsri30_search_model import (SPACE, COLD_R3_PRIOR, SCHEMA, admissible, candidate, conserved_fingerprint,
                                 decode, initial_point, informed_points, iteration_bounds, specification, validate_controls)
from fsri30_numerical_policy import check_pair, exploration_policy, select as select_numerics
import calibrate_upv_r123 as launcher_settings
from check_upv_restart import native

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE = ROOT/'build/calibration-probes/battery_sim_casing_probe-fsri30-initial-soc-20261023-r2'
DEFAULT_BASE = ROOT/'data/development/ul-fsri-30-20261023-r1/source-scenario.json'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_run(folder):
    with gzip.open(folder/'replay/run.json.gz', 'rt') as f:
        return json.load(f)


def event(out, kind, **fields):
    with (out/'journal.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), kind=kind, **fields), allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())


def atomic_save(path, value):
    temp = path.with_name(path.name+'.pending')
    save(temp, value)
    temp.replace(path)


def cached_control(roots, level, layers, scene, probe):
    """Read-only exact-input lookup across explicitly supplied search lineages."""
    names = [f'control-time{level}-space{layers}']
    legacy = {(0, 1): 'preflight-nearfield', (1, 1): 'preflight-time-refinement', (1, 2): 'preflight-space-refinement'}
    if (level, layers) in legacy:
        names.append(legacy[(level, layers)])
    expected, core = canonical_hash(scene), sha(probe)
    for root in roots:
        for name in names:
            folder = root/name
            if not (folder/'replay/run.json.gz').exists() or not (folder/'evaluation.json').exists():
                continue
            score = json.loads((folder/'evaluation.json').read_text())
            if not score.get('complete') or score.get('constraint', 1.) > 0. or score.get('native_sha256') != core:
                continue
            if score.get('scene_hash') != expected:
                continue
            if canonical_hash(json.loads((folder/'scenario.json').read_text())) != expected:
                raise ValueError('Cached scenario disagrees with its result fingerprint: '+str(folder))
            return folder.resolve()
    return None


def evaluate(scene, folder, data, probe, timeout, *, prune=False, incumbent=None, provisional=False):
    """Completed numerics and a partial/pruned experiment are different states."""
    validate_controls(scene)
    folder.mkdir(exist_ok=True)
    expected = canonical_hash(scene)
    if (folder/'evaluation.json').exists():
        previous = json.loads((folder/'evaluation.json').read_text())
        if previous['scene_hash'] != expected or previous['native_sha256'] != sha(probe):
            raise ValueError('Refusing incompatible cached evaluation: '+str(folder))
        return previous
    # A diagnostic produced by run_fsri30 can be reused ONLY by matching the
    # complete scenario/core and full horizon, not just its directory name.
    cached = folder/'replay/run.json.gz'
    cached_status = folder/'status.json'
    if cached.exists():
        actual_scene = json.loads((folder/'scenario.json').read_text())
        actual_status = json.loads(cached_status.read_text())
        if canonical_hash(actual_scene) != expected or actual_status['native_sha256'] != sha(probe):
            raise ValueError('Existing diagnostic has different inputs: '+str(folder))
        run = read_run(folder)
        if run.get('partial') or run['advanced_to_s'] < scene['simulation']['duration_s']-1e-7:
            raise ValueError('Cannot reuse partial diagnostic as complete')
    else:
        if (folder/'replay').exists():
            # An interrupted transaction stays intact; do not overwrite its files.
            index = 1
            while (folder/f'replay-interrupted-{index}').exists():
                index += 1
            (folder/'replay').rename(folder/f'replay-interrupted-{index}')
        save(folder/'scenario.json', scene)
        save(folder/'status.json', dict(status='running', pid=os.getpid(), scene_hash=expected, native_sha256=sha(probe)))
        launcher_settings.PROBE = probe
        started = time.monotonic()
        def temperature_prune(prefix):
            decision=cooling_rejection(scene,prefix,data)
            if decision:
                save(folder/'temperature-early-stop.json',decision)
                return decision['reason']
            return reject_reason(scene,prefix,data,incumbent,incumbent_relative=provisional)
        try:
            run = native(folder/'scenario.json', folder/'replay', dict(restart_contract=sha(probe),
                stop_after_first_rupture=False), timeout,
                prune_fn=temperature_prune if prune else None,
                minimum_speed=.2, rolling_minimum=.2)
        except KeyboardInterrupt:
            save(folder/'status.json', dict(status='interrupted', scene_hash=expected, native_sha256=sha(probe)))
            raise
        except Exception as exc:
            result = dict(complete=False, objective=None, constraint=1., error=str(exc),
                wall_s=time.monotonic()-started, scene_hash=expected, native_sha256=sha(probe))
            partial = folder/'replay/incomplete-run.json.gz'
            if partial.exists():
                with gzip.open(partial, 'rt') as f:
                    run = json.load(f)
                if len(run.get('time_s', [])) >= 2:
                    result.update(known_until_s=float(run['time_s'][-1]),
                                  prefix_lower_bound=loss(scene, run, data, include_features=False)['objective'])
            save(folder/'evaluation.json', result)
            save(folder/'status.json', dict(status='rejected', **result))
            return result
    score = loss(scene, run, data)
    if not score['complete']:
        raise ValueError('Native returned incomplete trajectory without failure')
    minimum = run['solver_profile']['minimum_regular_step_s']
    limits = dict(max_thermal_residual_j=1e-4, max_chemical_residual_j=1e-4,
                  max_mass_residual_kg=1e-10, max_oxidizer_residual_kg=1e-10)
    violations = [k for k, limit in limits.items() if not np.isfinite(run[k]) or run[k] > limit]
    if minimum < .005-1e-10:
        violations.append('minimum_step_below_0.005s')
    result = dict(score, constraint=1. if violations else -1., numerical_violations=violations,
        wall_s=run['wall_s'], realtime_ratio=(run['advanced_to_s']-run['started_at_s'])/run['wall_s'],
        scene_hash=expected, native_sha256=sha(probe), minimum_regular_step_s=minimum,
        balances={k: run[k] for k in limits}, calibrated=False, validation_passed=False)
    save(folder/'evaluation.json', result)
    save(folder/'status.json', dict(status='complete' if not violations else 'rejected', **result))
    return result


def report_best(out, folder, scene, run, data, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_fsri30 import plot
    # The existing plot explicitly says unfitted/no sensor correspondence.
    # It remains a development comparison, never an accepted validation plot.
    plot(folder, scene, run, data)
    t, ids, values = model_fields(run)
    observed = np.array([data[f'T{i}_k'] for i in range(1, 10)]).T
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for quantile, label in ((.5, 'медиана'), (.9, '90%'), (1., 'максимум')):
        line, = axes[0].plot(data['time_s'], np.quantile(observed, quantile, axis=1)-273.15,
                            label='Опыт: '+label, lw=1.)
        axes[0].plot(t, np.quantile(values, quantile, axis=1)-273.15, '--', color=line.get_color(),
                     label='Модель: '+label, lw=1.)
    axes[0].set(ylabel='Температура, °C', xlabel='Время, с',
        title='Форма распределения: эвристика без подтверждённой привязки термопар')
    axes[0].legend(fontsize=8, ncol=2)
    for i, key in enumerate(ids):
        axes[1].plot(t, values[:, i]-273.15, lw=.7, alpha=.7)
    for item in result['cascade']['events']:
        axes[1].axvline(item['time_s'], lw=.6, alpha=.3, color='red')
        axes[1].text(item['time_s'], .95, item['component'], transform=axes[1].get_xaxis_transform(),
                     fontsize=6, rotation=90, va='top')
    axes[1].set(xlabel='Время, с', ylabel='Средняя T батареи, °C',
        title=f"Автономно прореагировало {result['cascade']['reacted_components']}/30; вертикали — 10% превращения R2")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(0., data['time_s'][-1])
    fig.suptitle('Лучшая поисковая проба — НЕ независимая валидация')
    fig.savefig(folder/'cascade-comparison.png', dpi=140)
    plt.close(fig)
    atomic_save(out/'best.json', {**result, 'trial': str(folder), 'scenario': str(folder/'scenario.json'),
        'plot': str(folder/'cascade-comparison.png'), 'parameters': scene['metadata']['search_parameters'],
        'sensor_mapping_confirmed': False, 'calorimeter_hrr_used': False})


def safe_report(out, folder, scene, run, data, result):
    """An image/report failure must not kill a persisted scientific search."""
    try:
        report_best(out, folder, scene, run, data, result)
    except Exception as exc:
        event(out, 'report-error', trial=str(folder), error=str(exc))
        atomic_save(out/'best.json', {**result, 'trial': str(folder), 'scenario': str(folder/'scenario.json'),
            'parameters': scene['metadata']['search_parameters'], 'report_error': str(exc),
            'calibrated': False, 'validation_passed': False})


def verify_candidate(base, row, out, data, args, accuracy, layers, number, on_stage=None):
    """Checks reject/qualify a candidate, not the entire optimization process."""
    reference = read_run(Path(row['trial']))
    checks = []
    for label, check_layers, check_accuracy in (('time', layers, accuracy+1), ('space', layers*2, accuracy)):
        folder = out/f'verify-{number:04d}-{label}'
        if on_stage is not None:
            on_stage(label, folder)
        scene = candidate(base, row['x'], layers=check_layers, accuracy_level=check_accuracy)
        result = evaluate(scene, folder, data, args.probe, args.trial_timeout)
        if result['complete'] and result['constraint'] <= 0.:
            check = check_pair(dict(run=reference, score=row, folder=row['trial']),
                               dict(run=read_run(folder), score=result, folder=str(folder)))
            check['objective'] = result['objective']
        else:
            check = dict(accepted=False, error=result.get('error', 'Failed numerical constraints'))
        checks.append(dict(kind=label, **check))
    value = dict(accepted=all(c['accepted'] for c in checks), checks=checks,
        trial=row['trial'], autonomous_cascade=row['cascade']['autonomous_cascade'],
        calibration_accepted=False, validation_passed=False,
        note='Numerical qualification of a provisional candidate; unknown sensor mapping still excludes independent validation')
    atomic_save(out/f'verification-{number:04d}.json', value)
    return value


def run_search(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.base.read_text())
    data, manifest = load()
    if abs(base['simulation']['duration_s']-data['time_s'][-1]) > 1e-8:
        raise ValueError('Cascade search must cover the FULL published time axis, including cooling')
    source_names = ('bounded_bayes.py', 'calibrate_fsri30_bayes.py', 'fsri30_objective.py', 'fsri30_search_model.py', 'fsri30_numerical_policy.py',
                    'fsri30_data.py', 'generate_fsri30.py', 'check_upv_restart.py', 'realtime_guard.py')
    continue_unverified = getattr(args, 'continue_unverified_search', False)
    contract = dict(version=4, seed=args.seed, dimension=len(SPACE), parameters=specification(),
        continue_unverified_search=continue_unverified,
        informed_initial_points=[x.tolist() for x in informed_points()] if continue_unverified else [],
        cold_r3_stability_prior=COLD_R3_PRIOR,
        scenario_schema_sha256=sha(SCHEMA), native_iteration_bounds={k: list(v) for k, v in iteration_bounds().items()},
        base_path=str(args.base.resolve()), base_sha256=sha(args.base), native_path=str(args.probe.resolve()),
        native_sha256=sha(args.probe), raw_csv_sha256=manifest['csv_sha256'],
        source_sha256={n: sha(Path(__file__).with_name(n)) for n in source_names},
        horizon_s=float(data['time_s'][-1]), initial_design=args.initial_design,
        minimum_realtime_ratio=.2, minimum_dt_s=.005, source_unknowns=data_audit(data, manifest))
    plan_path = out/'contract.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != contract:
            raise ValueError('Search contract changed; use a NEW output directory rather than resetting history')
    else:
        save(plan_path, contract)
        save(out/'base-scenario.json', base)
        save(out/'data-audit.json', data_audit(data, manifest))
    # Lock is held for the process lifetime, including while a CUDA child runs.
    import fcntl
    lock = (out/'search.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    journal_path = out/'observations.json'
    observations = json.loads(journal_path.read_text()) if journal_path.exists() else []
    state = dict(status='running', phase='numerical-preflight', pid=os.getpid(),
                 completed_trials=len(observations), calibrated=False, validation_passed=False)
    atomic_save(out/'status.json', state)
    event(out, 'search-started', pid=os.getpid(), previous_trials=len(observations), contract=canonical_hash(contract))
    start = time.monotonic()
    try:
        controls = {}
        cache_roots = getattr(args, 'preflight_cache', None) or []
        if isinstance(cache_roots, Path):
            cache_roots = [cache_roots]  # programmatic compatibility, not physics migration
        # Validate the WHOLE bounded ladder up front, not forty minutes later.
        for level in range(4):
            for layer_count in (1, 2, 4):
                validate_controls(candidate(base, initial_point(), layers=layer_count, accuracy_level=level))
        event(out, 'control-ladder-validated', cases=12, iteration_bounds=iteration_bounds(),
              minimum_dt_s=base['simulation']['minimum_dt_s'])
        def evaluate_control(level, layers):
            scene = candidate(base, initial_point(), layers=layers, accuracy_level=level)
            if conserved_fingerprint(scene) != conserved_fingerprint(base):
                raise ValueError('Forbidden change to baseline physics/material inventory')
            name = f'control-time{level}-space{layers}'
            folder = out/name
            cached = cached_control(cache_roots, level, layers, scene, args.probe)
            if cached is not None:
                # Explicit read-only reuse: evaluate verifies native+whole-scene
                # hash and full horizon. No prefix or merely similar fit reused.
                folder = cached.resolve()
            state.update(phase=name)
            atomic_save(out/'status.json', state)
            result = evaluate(scene, folder, data, args.probe, args.trial_timeout)
            event(out, 'preflight-finished', name=name, result=result)
            value = dict(score=result, folder=str(folder), run=read_run(folder)) if result['complete'] else dict(score=result, folder=str(folder))
            controls[(level, layers)] = value
            return value
        def record_comparison(comparisons):
            atomic_save(out/'preflight-comparison.json', dict(accepted=False, partial=True, comparisons=comparisons))
            event(out, 'numerical-comparison', comparison=comparisons[-1])
        numerical = select_numerics(evaluate_control, on_comparison=record_comparison)
        save(out/'preflight-comparison.json', numerical)
        if not numerical['accepted'] and continue_unverified:
            numerical = exploration_policy(controls, numerical)
            save(out/'preflight-comparison.json', numerical)
            event(out, 'provisional-search-policy', policy=numerical)
        if not numerical['accepted'] and not numerical.get('search_allowed', False):
            raise RuntimeError('Preflight exceeds 5%/cascade-count tolerance. Do not fit numerical artifacts; inspect preflight-comparison.json')
        accuracy, layers = numerical['accuracy_level'], numerical['layers']
        selected = controls[(accuracy, layers)]
        state['numerical_policy'] = dict(accuracy_level=accuracy, layers=layers,
            provisional=numerical.get('provisional', False), numerically_verified=numerical['accepted'])
        save(out/'selected-numerics.json', state['numerical_policy'])
        if not observations:
            row = dict(selected['score'], x=initial_point().tolist(), trial=selected['folder'],
                       proposal=dict(method='declared-initial-point'), numerical_policy=state['numerical_policy'])
            observations.append(row)
            atomic_save(journal_path, observations)
            # Cached original plots are not overwritten by this new search.
            reference_folder = out/'initial-reference'
            reference_folder.mkdir(exist_ok=True)
            safe_report(out, reference_folder, candidate(base, initial_point(), layers=layers, accuracy_level=accuracy), selected['run'], data, row)
            best_info = json.loads((out/'best.json').read_text()) if (out/'best.json').exists() else None
            if best_info is not None:
                best_info.update(trial=selected['folder'], scenario=str(Path(selected['folder'])/'scenario.json'))
                atomic_save(out/'best.json', best_info)
        while len(observations) < args.trials and time.monotonic()-start < args.budget_hours*3600.:
            complete = [r for r in observations if r.get('complete') and r.get('constraint', 1.) <= 0.]
            incumbent = min((r['objective'] for r in complete), default=None)
            number = len(observations)
            if continue_unverified and number <= len(informed_points()):
                point = informed_points()[number-1]
                acquisition = dict(method='declared-physical-hypothesis',
                    rationale='Test gas retention, oxygen-bearing mixing and hot R3 kinetics without changing heater, battery properties or energy reserve')
            else:
                try:
                    point, acquisition = propose(observations, len(SPACE), initial_point(), seed=args.seed, initial_design=args.initial_design, admissible=admissible)
                except Exception as exc:
                    event(out, 'acquisition-fallback', error=str(exc))
                    point, acquisition = propose(observations, len(SPACE), initial_point(), seed=args.seed+1,
                        initial_design=len(observations)+1, admissible=admissible)
            folder = out/f'trial-{number:04d}'
            scene = candidate(base, point, layers=layers, accuracy_level=accuracy)
            if conserved_fingerprint(scene) != conserved_fingerprint(base):
                raise ValueError('Forbidden physics change in optimization proposal')
            state.update(phase='evaluating', active_trial=number, active_folder=str(folder),
                         parameters=decode(point), acquisition=acquisition)
            atomic_save(out/'status.json', state)
            event(out, 'proposal', trial=number, x=point.tolist(), parameters=decode(point), acquisition=acquisition)
            result = evaluate(scene, folder, data, args.probe, args.trial_timeout, prune=True,
                              incumbent=incumbent, provisional=continue_unverified)
            row = dict(result, x=point.tolist(), trial=str(folder), proposal=acquisition)
            observations.append(row)
            atomic_save(journal_path, observations)
            event(out, 'trial-finished', trial=number, result=row)
            if row['complete'] and row['constraint'] <= 0. and (incumbent is None or row['objective'] < incumbent):
                safe_report(out, folder, scene, read_run(folder), data, row)
                event(out, 'incumbent-improved', trial=number, objective=row['objective'], cascade=row['cascade'])
                if row['cascade']['autonomous_cascade'] or (incumbent is not None and row['objective'] < .8*incumbent):
                    def verify_stage(label, path):
                        state.update(phase='verifying-'+label, active_folder=str(path))
                        atomic_save(out/'status.json', state)
                    try:
                        row['verification'] = verify_candidate(base, row, out, data, args, accuracy, layers, number, verify_stage)
                        atomic_save(journal_path, observations)
                        safe_report(out, folder, scene, read_run(folder), data, row)
                        event(out, 'candidate-verified', trial=number, verification=row['verification'])
                    except Exception as exc:
                        event(out, 'verification-error', trial=number, error=str(exc))
                        row['verification'] = dict(accepted=False, error=str(exc))
                        atomic_save(journal_path, observations)
            state.update(completed_trials=len(observations), last_result=row,
                         successful_full_runs=sum(r['complete'] for r in observations))
            atomic_save(out/'status.json', state)
        state.update(phase='verify-best')
        atomic_save(out/'status.json', state)
        complete = [r for r in observations if r.get('complete') and r.get('constraint', 1.) <= 0.]
        if complete:
            best = min(complete, key=lambda r: r['objective'])
            number = observations.index(best)
            verification = best.get('verification') or verify_candidate(base, best, out, data, args, accuracy, layers, number)
            verified = verification['accepted']
            save(out/'best-numerical-verification.json', verification)
            state.update(status='complete', phase='development-search-complete', numerically_verified=verified,
                         autonomous_cascade=best['cascade']['autonomous_cascade'], best_trial=best['trial'], best_objective=best['objective'])
        else:
            state.update(status='needs-attention', phase='no-feasible-full-run')
    except KeyboardInterrupt:
        state.update(status='stopped', phase='interrupted')
        event(out, 'search-stopped', reason='signal/user')
    except Exception as exc:
        state.update(status='needs-attention', error=str(exc))
        event(out, 'search-error', error=str(exc))
    finally:
        state['completed_trials'] = len(observations)
        atomic_save(out/'status.json', state)
        print(json.dumps(state, indent=2), flush=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return 0 if state['status'] == 'complete' else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--base', type=Path, default=DEFAULT_BASE)
    ap.add_argument('--probe', type=Path, default=DEFAULT_PROBE)
    ap.add_argument('--seed', type=int, default=20261023)
    ap.add_argument('--trials', type=int, default=32)
    ap.add_argument('--initial-design', type=int, default=12)
    ap.add_argument('--trial-timeout', type=float, default=7200., help='Wall-time budget per native run, separate from realtime floor')
    ap.add_argument('--budget-hours', type=float, default=6.)
    ap.add_argument('--preflight-cache', type=Path, action='append', default=[],
        help='Explicit old controls; repeat for multiple preserved searches; only full exact-input matches reused')
    ap.add_argument('--continue-unverified-search', action='store_true',
        help='Explicitly permit bounded provisional parameter search; NEVER marks failed R3 convergence as passed')
    args = ap.parse_args()
    if args.trials < 2 or args.initial_design < 2 or args.trial_timeout <= 0. or args.budget_hours <= 0.:
        ap.error('Invalid search budget/design')
    def stop(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    raise SystemExit(run_search(args))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Staged gas-geometry search. Chemistry/drive frozen; negative tests are useful.

First prove neighbour-heat sensitivity with isolated full-horizon experiments.
Only then use GP optimization. An insensitive geometry must not silently spend
another 48 runs adjusting the source reaction's clock. This is development,
not independent validation or numerically qualified calibration.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from bounded_bayes import propose
from calibrate_fsri30_bayes import (DEFAULT_PROBE, atomic_save, canonical_hash,
    evaluate, event, read_run)
from fsri30_data import load, save, sha
from fsri30_numerical_policy import check_pair
from fsri30_search_model import validate_controls
import fsri30_transport_model as model

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'data/calibration/ul-fsri-30-bayes-20261023-r1/preflight-time-refinement'


def evaluate_transport(scene, folder, data, args):
    result = evaluate(scene, folder, data, args.probe, args.trial_timeout, prune=False)
    if result['complete'] and result['constraint'] <= 0.:
        result = dict(result, **model.transport_loss(scene, read_run(folder), data))
    save(folder/'transport-evaluation.json', result)
    return result


def diagnostic_report(out, row, data):
    """Never label surrogate shaping, reference timing or hot peaks as validated."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        run = read_run(Path(row['trial']))
        t, ids, temperatures = model.model_fields(run)
        source = row['transfer']['source_component']
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
        axes[0].plot(t, temperatures[:, ids.index(source)]-273.15, label=source+' (источник)')
        for item in row['transfer']['hottest_neighbours']:
            k = item['component']
            axes[0].plot(t, temperatures[:, ids.index(k)]-273.15, label=k, lw=.8)
        axes[0].set(ylabel='Средняя T батареи, °C', title='Источник и наиболее нагретые соседи')
        observed = np.array([data[f'T{i}_k'] for i in range(1, 10)]).T
        for q, label in ((.5, 'медиана'), (.9, '90%')):
            line, = axes[1].plot(data['time_s'], np.quantile(observed, q, axis=1)-273.15,
                label='Опыт: '+label, lw=.9)
            axes[1].plot(t, np.quantile(temperatures, q, axis=1)-273.15, '--',
                color=line.get_color(), label='Модель: '+label)
        axes[1].set(xlabel='Время, с', ylabel='Температура, °C',
            title='Поисковая эвристика: соответствие термопар батареям НЕ установлено')
        for ax in axes:
            ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=3)
        fig.suptitle('Передача энергии соседям; химия фиксирована; НЕ валидация')
        path = Path(row['trial'])/'neighbour-delivery.png'
        fig.savefig(path, dpi=130); plt.close(fig)
        row = dict(row, plot=str(path))
    except Exception as exc:
        event(out, 'plot-failed', error=str(exc))
    atomic_save(out/'best.json', dict(row, calibrated=False, validation_passed=False))


def run(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    base_path = args.reference/'scenario.json'
    base = json.loads(base_path.read_text()); data, manifest = load()
    old = json.loads((args.reference/'evaluation.json').read_text())
    if not old['complete'] or old['constraint'] > 0. or old['native_sha256'] != sha(args.probe):
        raise ValueError('Reference is not a complete conservative run of this exact native core')
    if old['scene_hash'] != canonical_hash(base):
        raise ValueError('Reference scene/hash mismatch')
    if base['simulation']['duration_s'] != float(data['time_s'][-1]):
        raise ValueError('A short warmup cannot replace the full cascade/cooling reference')
    validate_controls(base)
    # Proposed geometry is screened separately, using the declared level1
    # controls. The old failed strict R3/space checks are NOT waived or erased.
    source_files = ('search_fsri30_transport.py', 'fsri30_transport_model.py',
        'bounded_bayes.py', 'calibrate_fsri30_bayes.py', 'fsri30_objective.py',
        'fsri30_search_model.py', 'fsri30_numerical_policy.py', 'check_upv_restart.py')
    contract = dict(version=1, direction='neighbour-energy-delivery', seed=args.seed,
        parameters=model.specification(), reference=str(args.reference.resolve()),
        reference_scene_sha256=sha(base_path), native_sha256=sha(args.probe),
        raw_csv_sha256=manifest['csv_sha256'], frozen_sha256=canonical_hash(model.frozen(base)),
        full_horizon_s=base['simulation']['duration_s'],
        source_sha256={n: sha(Path(__file__).with_name(n)) for n in source_files},
        diagnostic_gate='source R2 plus >=10K gain in top-three neighbour peaks OR >=1% neighbour conversion',
        temperature_gate_is_search_shaping_not_measurement=True,
        numerical_policy=dict(provisional=True, numerically_verified=False,
            minimum_dt_s=base['simulation']['minimum_dt_s'], max_dt_s=base['simulation']['max_dt_s'],
            gas_interval_s=base['gas_network']['coupled_transport_interval_s']),
        minimum_realtime_ratio=.2, calibrated=False, validation_passed=False)
    contract_path = out/'contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Changed transport contract: use a new output, do not reset history')
    else:
        save(contract_path, contract); save(out/'base-scenario.json', base)
    import fcntl
    lock = (out/'search.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    history_path = out/'observations.json'
    rows = json.loads(history_path.read_text()) if history_path.exists() else []
    state = dict(status='running', phase='geometry-sensitivity', pid=os.getpid(),
        completed_trials=len(rows), chemistry_frozen=True, heater_frozen=True,
        calibrated=False, validation_passed=False, numerically_verified=False)
    atomic_save(out/'status.json', state); event(out, 'transport-search-started', contract=canonical_hash(contract))
    started = time.monotonic()
    try:
        if not rows:
            reference = dict(old, **model.transport_loss(base, read_run(args.reference), data),
                x=model.initial_point().tolist(), trial=str(args.reference.resolve()),
                parameters=model.decode(model.initial_point()), proposal=dict(method='immutable-reference'))
            rows.append(reference); atomic_save(history_path, rows)
            diagnostic_report(out, reference, data)
        experiments = model.experiments()
        while len(rows) < args.trials and time.monotonic()-started < args.budget_hours*3600.:
            number = len(rows)
            if number <= len(experiments):
                label, point = experiments[number-1]
                proposal = dict(method='isolated-geometry-sensitivity', hypothesis=label)
                phase = 'geometry-sensitivity'
            else:
                responsive = [r for r in rows[1:] if r.get('complete') and r.get('constraint', 1.) <= 0.
                              and model.useful_response(rows[0], r)]
                if not responsive:
                    state.update(status='needs-model-review', phase='no-neighbour-energy-response',
                        reason='Isolated transport tests did not materially heat/react a neighbour; GP intentionally NOT started')
                    event(out, 'sensitivity-gate-failed', reason=state['reason'])
                    break
                phase = 'transport-bayesian-search'
                try:
                    point, proposal = propose(rows, len(model.SPACE), model.initial_point(),
                        seed=args.seed, initial_design=1+len(experiments))
                except Exception as exc:
                    event(out, 'acquisition-fallback', error=str(exc))
                    point, proposal = propose(rows, len(model.SPACE), model.initial_point(),
                        seed=args.seed+1, initial_design=len(rows)+1)
            folder = out/f'trial-{number:04d}'
            scene = model.candidate(base, point)
            state.update(phase=phase, active_trial=number, active_folder=str(folder),
                parameters=model.decode(point), proposal=proposal, completed_trials=len(rows))
            atomic_save(out/'status.json', state)
            event(out, 'transport-proposal', number=number, parameters=model.decode(point), proposal=proposal)
            result = evaluate_transport(scene, folder, data, args)
            row = dict(result, x=point.tolist(), parameters=model.decode(point), trial=str(folder), proposal=proposal)
            rows.append(row); atomic_save(history_path, rows)
            event(out, 'transport-result', number=number, result=row)
            feasible = [r for r in rows if r.get('complete') and r.get('constraint', 1.) <= 0.]
            best = min(feasible, key=lambda r: r['objective'])
            if best is row:
                diagnostic_report(out, row, data)
            state.update(completed_trials=len(rows), last_result=row,
                best_trial=best['trial'], best_objective=best['objective'],
                useful_geometry_responses=sum(model.useful_response(rows[0], r) for r in feasible[1:]))
            atomic_save(out/'status.json', state)
        else:
            state.update(status='complete', phase='development-budget-complete')
        # Recheck time/payload sensitivity of any actual cascade, not its fit.
        # No check is called a passed independent experimental validation.
        feasible = [r for r in rows if r.get('complete') and r.get('constraint', 1.) <= 0.]
        best = min(feasible, key=lambda r: r['objective'])
        if best['cascade']['autonomous_cascade'] and state['status'] == 'complete':
            state.update(phase='time-qualification'); atomic_save(out/'status.json', state)
            scene = json.loads((Path(best['trial'])/'scenario.json').read_text())
            finer = deepcopy(scene); finer['simulation']['max_dt_s'] /= 2
            g = finer['gas_network']; g['coupled_transport_interval_s'] = max(.01, g['coupled_transport_interval_s']/2)
            g['averaged_pressure_iterations'] = min(2048, 2*g['averaged_pressure_iterations'])
            g['averaged_payload_iterations'] = min(128, 2*g['averaged_payload_iterations'])
            folder = out/'best-time-check'; folder.mkdir(exist_ok=True)
            score = evaluate_transport(finer, folder, data, args)
            check = check_pair(dict(run=read_run(Path(best['trial'])), score=best, folder=best['trial']),
                dict(run=read_run(folder), score=score, folder=str(folder))) if score['complete'] else dict(accepted=False, error=score.get('error'))
            save(out/'best-time-check.json', check)
            state.update(phase='development-budget-complete', time_check_passed=check['accepted'])
    except KeyboardInterrupt:
        state.update(status='stopped', phase='interrupted')
        event(out, 'transport-search-stopped', reason='signal/user')
    except Exception as exc:
        state.update(status='needs-attention', error=str(exc))
        event(out, 'transport-search-error', error=str(exc))
    finally:
        state['completed_trials'] = len(rows); atomic_save(out/'status.json', state)
        print(json.dumps(state, indent=2), flush=True)
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
    return 0 if state['status'] == 'complete' else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reference', type=Path, default=REFERENCE)
    ap.add_argument('--probe', type=Path, default=DEFAULT_PROBE)
    ap.add_argument('--trials', type=int, default=24)
    ap.add_argument('--budget-hours', type=float, default=6.)
    ap.add_argument('--trial-timeout', type=float, default=7200.)
    ap.add_argument('--seed', type=int, default=20261023)
    args = ap.parse_args()
    if args.trials < 1+len(model.experiments()) or args.budget_hours <= 0. or args.trial_timeout <= 0.:
        ap.error('Need the reference plus isolated transport checks, and positive time budgets')
    def stop(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()

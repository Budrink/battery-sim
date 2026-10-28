#!/usr/bin/env python3
"""Conditioned gas-transfer search, never passed off as full calibration."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import shutil
import time

from bounded_bayes import propose
from calibrate_fsri30_bayes import atomic_save, canonical_hash, event, read_run
import calibrate_upv_r123 as settings
from check_upv_restart import native
from fsri30_data import load, save, sha
from fsri30_conditioned_start import PROBE, ROOT, join_observations, prepare
import fsri30_suffix_model as model
from fsri30_transport_model import transport_loss, useful_response
from search_fsri30_transport import diagnostic_report
from model_evidence import seed,fsri30_compatibility,retain_fsri30_best

START = ROOT/'data/development/fsri30-conditioned-start-20261024-r1'


def carried_profile(out,base,probe,ignore=False):
    path=out/'contract.json'
    previous=json.loads(path.read_text()) if path.exists() else None
    if previous is not None:
        ident=previous.get('carried_seed_record')
        if ident is None:return None  # An explicitly unseeded search stays unseeded.
        if ignore:raise ValueError('Cannot discard the original seed of an existing search')
        return seed('geometry_prior','fsri30-conditioned-gas',
            compatibility=fsri30_compatibility(base),consumer_core=sha(probe),record_id=ident)
    return None if ignore else seed('geometry_prior','fsri30-conditioned-gas',
        compatibility=fsri30_compatibility(base),consumer_core=sha(probe))


def reuse_native_trial(reference,folder,scene,checkpoint,probe):
    """Same-core, identical-scene/state raw evidence only; ALWAYS rescore it."""
    contract=json.loads((reference.parent/'contract.json').read_text())
    if (contract['native_sha256']!=sha(probe) or contract['checkpoint_sha256']!=sha(checkpoint)
            or canonical_hash(json.loads((reference/'scenario.json').read_text()))!=canonical_hash(scene)):
        raise ValueError('Cannot reuse native evidence from a different core, state or scene')
    result=json.loads((reference/'evaluation.json').read_text())
    if not result.get('complete') or not result.get('initial_state_bitwise_equal'):
        raise ValueError('Cannot reuse an incomplete or unverified initial state')
    if result.get('native_sha256')!=sha(probe) or result.get('scene_hash')!=canonical_hash(scene):
        raise ValueError('Native evidence evaluation disagrees with its core or scene')
    source=reference/'replay/run.json.gz';destination=folder/'replay'
    destination.mkdir();shutil.copyfile(source,destination/'run.json.gz')
    save(folder/'reused-native-evidence.json',dict(source=str(source.resolve()),
        source_sha256=sha(source),same_core=True,same_scene=True,same_checkpoint=True,
        reuse_scores=False,reason='Same-core accepted native trajectory, independently rescored'))


def supported_search(rows,has_retained_profile):
    responsive=any(r.get('complete') and r.get('constraint',1.)<=0. and useful_response(rows[0],r)
        for r in rows[1:])
    # Ten kelvin was a first-discovery heuristic, not an obligation for EACH
    # subsequent iteration to add another ten kelvin to an already fitted seed.
    rechecked=bool(rows and rows[0].get('complete') and rows[0].get('constraint',1.)<=0.)
    return responsive or (has_retained_profile and rechecked)


def evaluate(scene, folder, prefix, checkpoint, source, data, probe, timeout,reactive=False):
    folder.mkdir(exist_ok=True); scenario = folder/'scenario.json'
    if scenario.exists() and json.loads(scenario.read_text()) != scene:
        raise ValueError('Cannot reuse a conditioned trial with different physics')
    save(scenario, scene)
    score_path = folder/'evaluation.json'
    if score_path.exists():
        old = json.loads(score_path.read_text())
        if old['native_sha256'] != sha(probe) or old['scene_hash'] != canonical_hash(scene):
            raise ValueError('Conditioned evaluation contract mismatch')
        return old
    settings.PROBE = probe; started = time.monotonic()
    try:
        request=dict(restart_contract=sha(probe),restart_from=str(checkpoint),
            restart_transport_scenario=str(source),stop_after_first_rupture=False)
        if reactive:request['diagnostic_reactive_transit']=True
        run=read_run(folder) if (folder/'replay/run.json.gz').exists() else native(
            scenario,folder/'replay',request,timeout,minimum_speed=.2,rolling_minimum=.2)
        if not run.get('suffix_initial_state_bitwise_equal'):
            raise ValueError('Conditioned initial state is not bitwise identical')
        if run.get('partial') or run['advanced_to_s'] < scene['simulation']['duration_s']-1e-8:
            raise ValueError('Incomplete suffix is not a completed experiment')
        if run['solver_profile']['minimum_regular_step_s'] < .005-1e-10:
            raise ValueError('Physical time step floor violated')
        if run.get('max_oxidizer_residual_kg', 0.)>1e-10:
            raise ValueError('Oxidizer conservation failed')
        composite = join_observations(prefix, run)
        result = dict(transport_loss(scene, composite, data), constraint=-1.,
            complete=True, conditioned_transport_diagnostic=True, full_warmup_validation=False,
            numerically_verified=False, calibrated=False, validation_passed=False,
            suffix_started_at_s=run['started_at_s'], suffix_duration_s=run['simulated_duration_s'],
            wall_s=run['wall_s'], realtime_ratio=run['simulated_duration_s']/run['wall_s'],
            minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
            initial_state_bitwise_equal=True, energy_delivery=model.energy_delivery(run),
            balances={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k})
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        result = dict(complete=False, constraint=1., objective=None, error=str(exc),
            wall_s=time.monotonic()-started, calibrated=False, validation_passed=False)
    result.update(scene_hash=canonical_hash(scene), native_sha256=sha(probe))
    save(score_path, result)
    return result


def run(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    data, _ = load()
    if not (args.start/'status.json').exists() or json.loads((args.start/'status.json').read_text())['status']!='ready':
        raise ValueError('Prepare and verify the shared committed start before launching this search')
    base, checkpoint, prefix_folder = prepare(args.start, probe=args.probe)
    prefix = read_run(prefix_folder)
    reactive=getattr(args,'reactive_transit',False)
    profile=carried_profile(out,base,args.probe,getattr(args,'ignore_profile',False))
    initial=model.point(**profile['parameters']) if profile else model.initial_point()
    files = ('search_fsri30_suffix.py', 'fsri30_suffix_model.py', 'fsri30_conditioned_start.py',
        'fsri30_transport_model.py', 'fsri30_objective.py', 'bounded_bayes.py', 'check_upv_restart.py','model_evidence.py')
    contract = dict(version=1, seed=args.seed, parameters=[asdict(p) for p in model.SPACE],
        native_sha256=sha(args.probe), checkpoint_sha256=sha(checkpoint), scene_hash=canonical_hash(base),
        sources={n:sha(Path(__file__).with_name(n)) for n in files},
        conditioned_transport_diagnostic=True, full_warmup_validation=False,
        carried_seed_record=profile['record'] if profile else None,reactive_transit=reactive,
        continuation_policy='retained-seed-recheck-then-bounded-refinement-v2',
        reuse_first_trial=str(args.reuse_first_trial.resolve()) if getattr(args,'reuse_first_trial',None) else None,
        physically_forced_ignition=False, min_dt_s=base['simulation']['minimum_dt_s'])
    path = out/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:
        raise ValueError('Changed suffix search contract: use a NEW output directory')
    save(path, contract)
    import fcntl
    lock = (out/'search.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    observations = out/'observations.json'
    rows = json.loads(observations.read_text()) if observations.exists() else []
    state = dict(status='running', phase='conditioned-transfer-controls', pid=os.getpid(),
        conditioned_transport_diagnostic=True, full_warmup_validation=False,
        calibrated=False, validation_passed=False, checkpoint=str(checkpoint),
        completed_trials=len(rows), chemistry_frozen=True, heater_frozen=True,
        carried_seed_record=profile['record'] if profile else None,reactive_transit=reactive)
    started = time.monotonic(); atomic_save(out/'status.json', state)
    event(out, 'conditioned-search-started', checkpoint_sha256=sha(checkpoint))
    try:
        experiments = [('carried-forward-profile' if profile else 'unchanged-suffix-reference',initial)]+model.experiments(initial)
        while len(rows)<args.trials and time.monotonic()-started<args.budget_hours*3600:
            number = len(rows)
            if number < len(experiments):
                name, x = experiments[number]; proposal = dict(method='fixed-state-sensitivity', hypothesis=name)
            else:
                if not supported_search(rows,profile is not None):
                    state.update(status='needs-model-review', phase='fixed-state-transfer-insensitive',
                        reason='Common-state controls did not deliver significant neighbour heat; no source-clock fitting')
                    break
                try:
                    x, proposal = propose(rows, len(model.SPACE), initial,
                        seed=args.seed, initial_design=len(experiments))
                except Exception as exc:
                    event(out, 'acquisition-fallback', error=str(exc))
                    x, proposal = propose(rows, len(model.SPACE), initial,
                        seed=args.seed+1, initial_design=len(rows)+1)
            scene = model.candidate(base,x); folder = out/f'trial-{number:04d}'
            if reactive:scene['gas_network']['reactive_transit']=True
            state.update(active_trial=number, active_folder=str(folder), parameters=model.decode(x), proposal=proposal)
            atomic_save(out/'status.json',state); event(out,'conditioned-proposal',trial=number,parameters=model.decode(x))
            if number==0 and getattr(args,'reuse_first_trial',None) and not (folder/'replay/run.json.gz').exists():
                folder.mkdir(exist_ok=True)
                reuse_native_trial(args.reuse_first_trial,folder,scene,checkpoint,args.probe)
                event(out,'same-core-native-reused',trial=number,source=str(args.reuse_first_trial),scores_reused=False)
            score = evaluate(scene,folder,prefix,checkpoint,prefix_folder/'scenario.json',data,args.probe,args.trial_timeout,reactive)
            row = dict(score,x=x.tolist(),parameters=model.decode(x),trial=str(folder),proposal=proposal)
            rows.append(row); atomic_save(observations,rows); event(out,'conditioned-result',trial=number,result=row)
            if number==0 and not row['complete']:
                raise ValueError('Unchanged full suffix failed; do not interpret geometry experiments')
            feasible = [r for r in rows if r.get('complete') and r.get('constraint',1.)<=0.]
            best = min(feasible,key=lambda r:r['objective'])
            if best is row:
                # Existing report reads a suffix; its plotted time axis therefore
                # explicitly starts at the common checkpoint, not time zero.
                diagnostic_report(out,row,data)
                # Persist immutable evidence, not just a mutable best.json.
                try:event(out,'model-evidence-retained',record=retain_fsri30_best(out))
                except Exception as exc:event(out,'model-evidence-error',error=str(exc))
            state.update(completed_trials=len(rows),last_result=row,best_trial=best['trial'],
                best_objective=best['objective'],
                useful_geometry_responses=sum(useful_response(rows[0],r) for r in feasible[1:]))
            atomic_save(out/'status.json',state)
        else:
            state.update(status='complete',phase='conditioned-development-budget-complete')
    except KeyboardInterrupt:
        state.update(status='stopped',phase='interrupted'); event(out,'conditioned-search-stopped')
    except Exception as exc:
        state.update(status='needs-attention',error=str(exc)); event(out,'conditioned-search-error',error=str(exc))
    finally:
        state['completed_trials']=len(rows); atomic_save(out/'status.json',state)
        print(json.dumps(state,indent=2),flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
    return 0 if state['status']=='complete' else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--start',type=Path,default=START)
    ap.add_argument('--probe',type=Path,default=PROBE); ap.add_argument('--trials',type=int,default=24)
    ap.add_argument('--budget-hours',type=float,default=6.); ap.add_argument('--trial-timeout',type=float,default=7200.)
    ap.add_argument('--seed',type=int,default=20261024)
    ap.add_argument('--reactive-transit',action='store_true',help='Explicit experimental reaction-during-transit method')
    ap.add_argument('--ignore-profile',action='store_true',help='Explicitly start a NEW search without a learned geometry seed')
    ap.add_argument('--reuse-first-trial',type=Path,help='Reuse only raw identical-state/scene/core evidence; score is recomputed')
    args=ap.parse_args()
    if args.trials<6 or args.budget_hours<=0 or args.trial_timeout<=0: ap.error('Invalid conditioned-search budget')
    def stop(*unused): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    raise SystemExit(run(args))


if __name__=='__main__': main()

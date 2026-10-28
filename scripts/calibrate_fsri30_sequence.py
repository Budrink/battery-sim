#!/usr/bin/env python3
"""Fresh CUDA all-wave screening: coarse gas AND heat, then 40 and 20 mm.

Only parameters cross mesh levels. Every mesh has independent observations,
scores and fresh initial conditions. Passing a wave gate is not validation.
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
import time
import numpy as np

import calibrate_upv_r123 as calibration
from calibrate_fsri30_bayes import atomic_save, event
from check_upv_restart import native
from fsri30_data import ROOT, load, save, sha
from fsri30_multilevel import LEVELS, remesh, inventory
from fsri30_objective import model_fields
from fsri30_early_stop import rejection
from fsri30_sequence_model import SPACE, candidate, initial, suggest, sequence_rank
from fsri30_wave_fit import POLICY, reference, assess_fields
from fsri30_mesh_screen import assess as mesh_assess, primary_bias_from_pair, uncertain_temperature_window
from fsri30_vent_geometry import restrict_exits

BASE=ROOT/'data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023/scenario.json'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-radiation-20261025-r1'
CORE='3147360f49c85a456e290478ae0cc30c9d63a460e5c3c44109a02bea2015625f'
BASE_HASH='90697fc50e20f1eaa6c5670aee96d4c14ce8a7d7f7bd2073b3575ebc8d192103'
MINIMUM_SPEED=.2
PAIRED_COARSE=ROOT/'data/calibration/fsri30-all-waves-multilevel-20261026-r1/h-80mm/trial-0000'
PAIRED_FINE=ROOT/'data/development/fsri30-wave-policy-20261026-r1/reassessment/reassessment.json'


def mesh_evidence(base):
    coarse=json.loads((PAIRED_COARSE/'status.json').read_text())
    fine=next(r for r in json.loads(PAIRED_FINE.read_text()) if r['trial']==str(BASE.parent))
    if coarse['native_sha256']!=CORE or fine['native_sha256']!=CORE:
        raise ValueError('Paired numerical evidence must use the identical core')
    if json.loads((PAIRED_COARSE/'scenario.json').read_text())!=candidate(remesh(base,80),initial(base)):
        raise ValueError('Coarse numerical anchor changed physical parameters')
    if sha(Path(coarse['run_path']))!=coarse['run_sha256'] or sha(Path(fine['history']))!=fine['history_sha256']:
        raise ValueError('Paired numerical histories changed')
    result=primary_bias_from_pair(coarse['assessment'],fine['assessment'])
    result.update(coarse_history_sha256=coarse['run_sha256'],fine_history_sha256=fine['history_sha256'],
                  coarse_trial=str(PAIRED_COARSE),fine_trial=str(BASE.parent),
                  base_sha256=BASE_HASH,native_sha256=CORE)
    return result


def physical_error(run):
    for key,limit in [('max_thermal_residual_j',1e-4),('max_chemical_residual_j',1e-4),
                      ('max_mass_residual_kg',1e-10),('max_oxidizer_residual_kg',1e-10)]:
        value=run.get(key)
        if value is None or not 0.<=value<=limit:
            return 'Missing or failed conservation ledger: '+key
    floor=run.get('solver_profile',{}).get('minimum_regular_step_s')
    if floor is None or floor<.01-1e-9:return 'Regular physical step below 0.01 s'
    return None


def assess_run(run,ref,level=20,evidence=None):
    t=run.get('time_s',[])
    if len(t)<2:return dict(evaluated=False,accepted=None,primary_offset_s=None,
                           last_observed_s=float(t[-1]) if len(t) else 0.,constraint=4.)
    t,ids,values=model_fields(run)
    return mesh_assess(t,values,ids,ref,level,evidence)


def promotable(row):
    a=row.get('assessment',{})
    return bool(row.get('status')=='complete' and not row.get('physical_error') and
                a.get('accepted') and a.get('phase_window_complete'))


def trial(folder,base,parameters,level,ref,data,update,duration=3600.,evidence=None,probe=None,extra_guard=None,
          committed_restart_at=None):
    folder.mkdir()
    native_core=CORE if probe is None else sha(probe)
    scene=candidate(base,parameters,duration)
    save(folder/'scenario.json',scene)
    row=dict(trial=str(folder),parameters=parameters,level_mm=level,status='running',
             native_sha256=native_core,scenario_sha256=sha(folder/'scenario.json'),
             screening_only=level!=20,calibrated=False,validation_passed=False,
             assessment=dict(primary_offset_s=None,last_observed_s=0.,constraint=4.))
    latest=None
    started=time.monotonic()

    def inspect(run):
        nonlocal latest
        latest=assess_run(run,ref,level,evidence)
        error=physical_error(run)
        temperature_stop=rejection(scene,run,data) if len(run.get('time_s',[]))>1 else None
        deferred=None
        if (temperature_stop and 'above 3500 K' not in temperature_stop['reason'] and
                uncertain_temperature_window(latest['last_observed_s'],ref,level,evidence)):
            deferred=temperature_stop;temperature_stop=None
        row.update(assessment=latest,physical_error=error,
                   delivery=delivery_signal(run,scene),
                   temperature_rejection=temperature_stop,
                   temperature_guard_deferred_for_mesh_timing=deferred,
                   wall_s=time.monotonic()-started,
                   live_advanced_to_s=run.get('live_advanced_to_s',run.get('advanced_to_s')),
                   minimum_regular_step_s=run.get('solver_profile',{}).get('minimum_regular_step_s'))
        atomic_save(folder/'status.json',row)
        update(row)
        if error:return error
        if extra_guard is not None:
            extra_reason=extra_guard(run)
            if extra_reason:return extra_reason
        if latest.get('accepted') is False:return latest.get('reason')
        return temperature_stop['reason'] if temperature_stop else None

    atomic_save(folder/'status.json',row)
    run=None
    try:
        request=dict(restart_contract=native_core,stop_after_first_rupture=False)
        if committed_restart_at is not None:
            request.update(save_committed_restart=str(folder/'committed-state.bin'),
                           committed_restart_at_s=committed_restart_at)
        run=native(folder/'scenario.json',folder/'replay',request,
                   timeout=duration/MINIMUM_SPEED+120.,prune_fn=inspect,
                   minimum_speed=MINIMUM_SPEED,rolling_minimum=MINIMUM_SPEED,probe=probe)
        reason=inspect(run)
        row['status']='complete' if not reason and promotable(dict(row,status='complete')) else 'rejected'
    except KeyboardInterrupt:
        row['status']='interrupted'
        raise
    except (RuntimeError,TimeoutError) as exc:
        row.update(status='pruned' if str(exc).startswith('Pruned:') else 'numerical-failed',error=str(exc))
    finally:
        row['wall_s']=time.monotonic()-started
        for name in ('run.json.gz','speed-rejected-run.json.gz','incomplete-run.json.gz'):
            path=folder/'replay'/name
            if not path.exists():continue
            if run is None:
                with gzip.open(path,'rt') as stream:run=json.load(stream)
                row['assessment']=assess_run(run,ref,level,evidence)
                row['physical_error']=physical_error(run)
            row.update(run_sha256=sha(path),run_path=str(path))
            row['balances']={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k}
            break
        if row.get('physical_error'):
            row['status']='numerical-failed'
            row['invalid_physics_observations']=row['assessment']
            row['assessment']=dict(accepted=False,evaluated=True,phase_window_complete=False,
                constraint=4.,primary_offset_s=None,
                last_observed_s=row['invalid_physics_observations'].get('last_observed_s',0.),
                reason=row['physical_error'])
        atomic_save(folder/'status.json',row)
    return row


def delivery_signal(run,scene):
    """Tie-break for EXPLORATION, never a substitute for a measured second wave.

    Compare actual neighbor heating at one fixed physical time. A hot source,
    a longer simulated prefix or a cooling-tail match cannot raise this signal.
    """
    times=np.asarray(run.get('time_s',[]),float);target=2135.
    if len(times)<2 or times[0]>target or times[-1]<target-1e-7:return None
    i=int(np.searchsorted(times,target+1e-7,side='right'))-1
    source=scene['metadata']['heated_component']
    neighbors=sorted(((float(v[i]),k) for k,v in run.get('cell_mean_temperature_k',{}).items()
                      if k!=source),reverse=True)
    if len(neighbors)<4:return None
    return dict(observed_time_s=float(times[i]),target_time_s=target,
                top4_neighbor_mean_k=float(np.mean([v for v,k in neighbors[:4]])),
                maximum_neighbor_k=neighbors[0][0],neighbors=[k for v,k in neighbors[:4]],
                source_excluded=source,exploration_only=True,qualifies_cascade=False)


def read_observed_trial(folder,level,mesh,ref,evidence=None,native_core=CORE):
    """Read actual SAME-mesh evidence; no score or state transfer from coarse runs."""
    old=json.loads((folder/'status.json').read_text())
    if old['native_sha256']!=native_core or old['level_mm']!=level:
        raise ValueError('Seed belongs to another core or mesh level')
    if old['status']=='running':raise ValueError('Stop previous search before importing mutable observations')
    if not old.get('run_path'):return None
    native_path=Path(old['run_path'])
    if sha(native_path)!=old['run_sha256']:raise ValueError('Historical native observations changed')
    scene=json.loads((folder/'scenario.json').read_text())
    if scene!=candidate(mesh,old['parameters']):raise ValueError('History mesh/physics differs from requested candidate')
    with gzip.open(native_path,'rt') as stream:run=json.load(stream)
    row=dict(old,assessment=assess_run(run,ref,level,evidence),physical_error=physical_error(run),
             delivery=delivery_signal(run,scene),
             historical_policy_reassessment=True,old_assessment=old['assessment'],
             status='complete' if not run.get('partial') and run['time_s'][-1]>=3600.-1e-7 else 'reassessed-prefix')
    if row['physical_error']:row['status']='numerical-failed'
    return row


def recheck_history(history,level,mesh,ref,evidence,native_core=CORE):
    """Same-mesh native observations rescored under the new screening policy."""
    result=[]
    for source in history:
        contract=json.loads((source/'contract.json').read_text())
        if contract['native_sha256']!=native_core or contract['base_sha256']!=BASE_HASH:
            raise ValueError('Cannot import another physical baseline/core as numerical evidence')
        for path in sorted((source/f'h-{level}mm').glob('trial-*/status.json')):
            row=read_observed_trial(path.parent,level,mesh,ref,evidence,native_core=native_core)
            if row is not None:result.append(row)
    return result


def search_levels(start,seed):
    if start not in LEVELS:raise ValueError('Unknown search mesh level')
    if start!=80 and seed is None:
        raise ValueError('Direct fine-grid search requires an explicit measured seed; defaults must not reset learning')
    return [level for level in LEVELS if level<=start]


def replay_candidate(rows):
    from fsri30_sequence_model import primary_aligned
    pending=[r for r in rows if r.get('historical_policy_reassessment') and
             r['status']=='reassessed-prefix' and r['assessment'].get('accepted') is None and
             primary_aligned(r['assessment']) and not any(
                 not q.get('historical_policy_reassessment') and q['parameters']==r['parameters'] for q in rows)]
    return max(pending,key=sequence_rank) if pending else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-trials-per-level',type=int,default=32)
    parser.add_argument('--history',type=Path,nargs='*',default=[])
    parser.add_argument('--start-level',type=int,choices=tuple(LEVELS),default=80)
    parser.add_argument('--seed-trial',type=Path)
    parser.add_argument('--probe',type=Path,help='Explicit newly verified core; requires a same-core 20mm observed seed')
    parser.add_argument('--outlet-access-scale',type=float,default=1.,help='Retain a separately tested outlet-access geometry, not refit it silently')
    args=parser.parse_args()
    if args.max_trials_per_level<1:raise ValueError('Positive search budget required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'search.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    probe=args.probe.resolve() if args.probe else PROBE
    native_core=sha(probe)
    if (not args.probe and native_core!=CORE) or sha(BASE)!=BASE_HASH:raise ValueError('Pinned physical baseline changed')
    if args.probe and (args.start_level!=20 or args.seed_trial is None):
        raise ValueError('A new core needs a same-core fine-grid seed; old mesh-error evidence cannot transfer')
    if not .1<=args.outlet_access_scale<=1. or (args.outlet_access_scale!=1. and args.start_level!=20):
        raise ValueError('Outlet sensitivity must be within the tested .1..1 range on20mm')
    calibration.PROBE=probe
    data,manifest=load();ref=reference(data)
    base=restrict_exits(json.loads(BASE.read_text()),args.outlet_access_scale)
    levels=search_levels(args.start_level,args.seed_trial)
    evidence=mesh_evidence(base) if any(level!=20 for level in levels) else None
    seed=None
    if args.seed_trial:
        seed=read_observed_trial(args.seed_trial,args.start_level,remesh(base,args.start_level),ref,evidence,native_core=native_core)
        if seed is None or seed.get('physical_error'):
            raise ValueError('Seed must have real observations and passing physical ledgers')
    # Pin helper code as well as top-level policy: mesh builders are part of
    # the numerical experiment, and silently changing one invalidates resume.
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('*.py')}
    contract=dict(native_sha256=native_core,base_sha256=BASE_HASH,csv_sha256=manifest['csv_sha256'],
                  sources=sources,levels={str(k):LEVELS[k] for k in levels},
                  outlet_access_scale=args.outlet_access_scale,
                  start_level_mm=args.start_level,seed_trial=str(args.seed_trial.resolve()) if args.seed_trial else None,
                  seed_status_sha256=sha(args.seed_trial/'status.json') if args.seed_trial else None,
                  seed_run_sha256=seed['run_sha256'] if seed else None,
                  policy=asdict(POLICY),parameter_space=[asdict(p) for p in SPACE],
                  minimum_step_s=.01,minimum_realtime_ratio=MINIMUM_SPEED,
                  max_trials_per_level=args.max_trials_per_level,
                  mesh_discrepancy=evidence,
                  history_status_sha256={str(p.resolve()):sha(p) for source in args.history
                                         for p in source.glob('h-*/trial-*/status.json')},
                  scores_transfer=False,checkpoints_transfer=False,
                  acceptance='Coarse mesh uses explicit numerical search windows; final 20 mm keeps raw 100 s gate and 10 s target. No time-shifted validation.')
    if (out/'contract.json').exists():
        if json.loads((out/'contract.json').read_text())!=contract:
            raise ValueError('Changed search contract: start a separate directory, retain parameters explicitly')
    else:
        save(out/'contract.json',contract)
        with tarfile.open(out/'source-policy.tar.gz','w:gz') as archive:
            for path in sources:archive.add(ROOT/path,arcname=path)
        save(out/'reference-waves.json',ref)
        save(out/'mesh-discrepancy.json',evidence)
    state=dict(status='running',pid=os.getpid(),native_sha256=native_core,
               calibrated=False,validation_passed=False,active_trial=None)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    anchor=dict(seed['parameters']) if seed else initial(base)
    def update(row):
        state.update(active_trial=row['trial'],level_mm=row['level_mm'],parameters=row['parameters'],
                     progress={k:row[k] for k in ('assessment','wall_s','live_advanced_to_s',
                                'minimum_regular_step_s') if k in row})
        atomic_save(out/'status.json',state)
    try:
        for level in levels:
            stage=out/f'h-{level}mm';stage.mkdir(exist_ok=True)
            mesh=remesh(base,level)
            save(stage/'mesh-inventory.json',inventory(mesh))
            if (stage/'anchor.json').exists():
                if json.loads((stage/'anchor.json').read_text())!=anchor:raise ValueError('Mesh anchor changed')
            else:save(stage/'anchor.json',anchor)
            # Reconstruct completed/rejected prefixes from immutable trial states,
            # so a controller crash cannot silently lose calibration evidence.
            rows=recheck_history(args.history,level,mesh,ref,evidence,native_core=native_core)
            if seed is not None and level==args.start_level and not any(r['trial']==seed['trial'] for r in rows):
                rows.append(seed)
            for path in sorted(stage.glob('trial-*/status.json')):
                old=json.loads(path.read_text())
                if old['status'] not in ('running','interrupted'):rows.append(old)
            failures=0
            atomic_save(stage/'observations.json',rows)
            while not any(promotable(r) for r in rows):
                if len(rows)>=args.max_trials_per_level:
                    state.update(status='needs-model-review',level_mm=level,
                                 reason='Search budget exhausted without all-wave agreement; finer grid NOT launched')
                    return
                pending=replay_candidate(rows)
                if pending:
                    parameters=dict(pending['parameters'])
                    proposal=dict(method='replay-prefix-admitted-by-mesh-uncertainty',source_trial=pending['trial'])
                else:parameters,proposal=suggest(rows,base,anchor=anchor)
                n=len(list(stage.glob('trial-*')));folder=stage/f'trial-{n:04d}'
                while folder.exists():n+=1;folder=stage/f'trial-{n:04d}'
                state.update(level_mm=level,active_trial=str(folder),parameters=parameters,proposal=proposal)
                atomic_save(out/'status.json',state)
                event(out,'proposal',level_mm=level,trial=str(folder),parameters=parameters,acquisition=proposal)
                if sha(probe)!=native_core:raise ValueError('Native core changed during search')
                row=trial(folder,mesh,parameters,level,ref,data,update,evidence=evidence,probe=probe)
                rows.append(row);atomic_save(stage/'observations.json',rows)
                event(out,'trial-result',**row)
                usable=[r for r in rows if r['status']!='numerical-failed']
                if usable:
                    best=max(usable,key=sequence_rank)
                    atomic_save(stage/'best.json',best)
                    state.update(best_trial=best['trial'],best_assessment=best['assessment'])
                failures=failures+1 if row['status']=='numerical-failed' else 0
                if failures>=3:
                    state.update(status='needs-numerical-review',reason='Three consecutive numerical/speed failures')
                    return
            selected=max((r for r in rows if promotable(r)),key=sequence_rank)
            anchor=dict(selected['parameters'])
            save(stage/'selected-for-refinement.json',selected)
            event(out,'mesh-screen-passed',level_mm=level,trial=selected['trial'],
                  parameters_only_transfer=True,assessment=selected['assessment'])
        # Cooling cannot rescue missing waves. Only a fine-grid sequence that
        # passed the gate receives a fresh full-horizon trajectory.
        folder=out/'full-tail'
        if folder.exists():raise ValueError('Full-tail attempt already exists; inspect it before resuming')
        tail=trial(folder,base,anchor,20,ref,data,update,duration=5945.,evidence=evidence,probe=probe)
        from fsri30_cooling_fit import converged
        if promotable(tail):
            with gzip.open(tail['run_path'],'rt') as stream:run=json.load(stream)
            from check_fsri30_source_reconstruction import metrics
            result=metrics(candidate(base,anchor,5945.),run,data)
            tail['cooling_diagnostic']=converged(run,data,converted_components=result['r2_converted_components'])
            tail['model_conversion_events']=result['model_conversion_events']
            atomic_save(out/'full-tail-summary.json',tail)
        state.update(status='sequence-screen-finished' if promotable(tail) else 'full-tail-rejected',
                     selected_trial=tail['trial'],assessment=tail['assessment'],
                     note='All-wave agreement is not independent validation or an identified TC-to-cell mapping')
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:
        state.update(status='failed',error=str(exc))
        raise
    finally:
        atomic_save(out/'status.json',state)
        print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Persistent bounded cooling search: real CUDA, fresh state, early rejection.

One failed trial does not stop the search. Trials, failures, proposals, best
parameters and fixture-scoped learning persist; SIGTERM preserves the child.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from calibrate_fsri30_bayes import atomic_save,event
from check_fsri30_radiation_experiment import BASE,CORE,PROBE,read_run
from fsri30_cooling_fit import SPACE,observation,converged,rank,suggest,burst_rank,burst_suggest
from fsri30_burst_fit import assess as assess_burst, POLICY as BURST_POLICY
from fsri30_data import ROOT,load,save,sha
from model_evidence import record

SEED=ROOT/'data/development/fsri30-radiation-experiment-20261025-r1'
ENGINE=Path(__file__).with_name('check_fsri30_radiation_experiment.py')


def initial_row(data):
    status=json.loads((SEED/'status.json').read_text())
    if status['native_sha256']!=CORE:raise ValueError('Seed belongs to another core')
    path=SEED/'replay/incomplete-run.json.gz'
    run=read_run(path);row=observation(run,data)
    stop=json.loads((SEED/'early-stop.json').read_text())
    row.update(parameters=dict(effective_emissivity=.1,thermal_stirring_coefficient=.05),
        trial=str(SEED),native_sha256=CORE,run_sha256=sha(path),wall_s=run['wall_s'],
        status='pruned',constraint=max(.001,stop['rmse_k']/stop['limit_k']-1.),
        provenance='Existing exact-core observed prefix; not a full score')
    return row


def analyze(folder,data,explosion_priority=False):
    state=json.loads((folder/'status.json').read_text())
    if state['native_sha256']!=CORE:raise ValueError('Cannot reuse another physical core')
    scene=json.loads((folder/'scenario.json').read_text())
    parameters=state.get('parameters',dict(
        effective_emissivity=scene['metadata']['exterior_radiation']['effective_emissivity'],
        thermal_stirring_coefficient=scene['gas_network']['thermal_stirring_coefficient']))
    complete=state['status']=='complete'
    path=folder/'replay'/('run.json.gz'if complete else'incomplete-run.json.gz')
    row=dict(parameters=parameters,trial=str(folder),status=state['status'],
        native_sha256=CORE,constraint=4.,last_observed_s=0.,full_trajectory=False)
    if not path.exists():return row
    run=read_run(path);row.update(observation(run,data),run_sha256=sha(path),wall_s=run['wall_s'])
    ledgers={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k}
    row['balances']=ledgers
    for key,limit in (('max_thermal_residual_j',1e-4),('max_chemical_residual_j',1e-4),
        ('max_mass_residual_kg',1e-10),('max_oxidizer_residual_kg',1e-10)):
        if run.get(key,0.)>limit:
            row.update(status='ledger-failed',full_trajectory=False);return row
    if run['solver_profile']['minimum_regular_step_s']<.01-1e-9:
        row.update(status='floor-failed',full_trajectory=False);return row
    if explosion_priority:
        row['burst']=assess_burst(scene,run,data)
    stop=folder/'early-stop.json'
    if stop.exists():
        reason=json.loads(stop.read_text());row['rejection']=reason
        if 'rmse_k'in reason:row['constraint']=max(.001,reason['rmse_k']/reason['limit_k']-1.)
    if complete:
        summary=json.loads((folder/'summary.json').read_text())
        check=converged(run,data,converted_components=summary['r2_converted_components'])
        if explosion_priority:
            burst=row['burst'];check['cooling_accepted']=check['accepted']
            check['burst_accepted']=bool(burst['accepted']and burst['phase_window_complete'])
            check['burst_timing_target_met']=burst.get('timing_target_met',False)
            check['accepted']=bool(check['accepted']and check['burst_accepted']and check['burst_timing_target_met'])
        row.update(acceptance=check,full_cooling_rmse_k=check.get('cooling_rmse_k'),
            converted_components=summary['r2_converted_components'])
        if check.get('accepted'):row['constraint']=0.
        else:
            row['constraint']=max(.001,check.get('cooling_rmse_k',1e3)/check.get('cooling_limit_k',80.)-1.,
                check.get('median_rms_relative_kelvin',1.)/.05-1.)
    return row


def publish_best(out,row,explosion_priority=False):
    atomic_save(out/'best.json',row)
    return record('geometry_prior','fsri30-explosion-priority-boundary'if explosion_priority else'fsri30-cooling-boundary',row['parameters'],
        dict(result_path=str(out/'best.json'),trial=row['trial'],
            result_run_sha256=row['run_sha256'],computed_until_s=row['last_observed_s'],
            full_fresh_trajectory=row.get('full_trajectory',False),
            anchor_rmse_k=row.get('anchor_rmse_k'),
            burst=row.get('burst'),explosion_priority=explosion_priority,
            interpretation='Fixture-level development seed, not a material property or independent validation'),
        compatibility=dict(fixture='ulfsri-30-ABS-open-air',frozen_base_sha256=sha(BASE),
            gas_spacing_m=.02,thermal_stirring_length_m=.02),producer_core=CORE,activate_development=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--initial-emissivity',type=float,default=.2174011870653867)
    ap.add_argument('--explosion-priority',action='store_true')
    ap.add_argument('--history',type=Path,nargs='*',default=[])
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'search.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if sha(PROBE)!=CORE:raise ValueError('Frozen CUDA core changed')
    data,manifest=load()
    contract=dict(native_sha256=CORE,base_sha256=sha(BASE),csv_sha256=manifest['csv_sha256'],
        engine_sha256=sha(ENGINE),fit_model_sha256=sha(Path(__file__).with_name('fsri30_cooling_fit.py')),
        controller_sha256=sha(Path(__file__)),parameters=[asdict(p)for p in SPACE],
        explosion_priority=args.explosion_priority,burst_policy=asdict(BURST_POLICY)if args.explosion_priority else None,
        burst_policy_sha256=sha(Path(__file__).with_name('fsri30_burst_fit.py')),
        history_sha256={str(p.resolve()):sha(p/'observations.json')for p in args.history},
        initial_emissivity=args.initial_emissivity,minimum_dt_s=.01,minimum_realtime_ratio=.2,
        chemistry_residue_materials_contacts_gas_topology_frozen=True,
        acceptance=('Collective burst/front timing <=100s mandatory, <=10s target; no missing cascade; cooling is secondary; 'if args.explosion_priority else'')+
            'Full 5945s; cooling RMSE <= max(80K,10% Kelvin), median RMS <=5% Kelvin; autonomous cascade')
    if (out/'contract.json').exists():
        if json.loads((out/'contract.json').read_text())!=contract:raise ValueError('Search contract changed; use a new directory')
    else:save(out/'contract.json',contract)
    if(out/'observations.json').exists():rows=json.loads((out/'observations.json').read_text())
    elif args.history:
        rows=[];seen=set()
        for source in args.history:
            old=json.loads((source/'observations.json').read_text())
            previous=json.loads((source/'status.json').read_text())
            folders=[Path(r['trial'])for r in old]+[Path(previous['active_trial'])]
            for folder in folders:
                if str(folder)in seen:continue
                seen.add(str(folder));row=analyze(folder,data,args.explosion_priority)
                row['historical_policy_reassessment']=True;rows.append(row)
                event(out,'history-reassessed',trial=str(folder),burst=row.get('burst'),
                    source_policy_reused=False,source_core=CORE)
    else:rows=[initial_row(data)]
    atomic_save(out/'observations.json',rows)
    state=dict(status='running',pid=os.getpid(),native_sha256=CORE,trials=len(rows)-1,
        calibrated=False,validation_passed=False,engineering_cooling_converged=False,
        explosion_priority=args.explosion_priority)
    ranking=burst_rank if args.explosion_priority else rank
    if args.explosion_priority:
        best=max(rows,key=ranking)
        if not best.get('burst',{}).get('accepted'):raise ValueError('No admissible explosion seed; inspect the reassessed history')
        state.update(best_trial=best['trial'],best_burst=best['burst'],
            development_record=publish_best(out,best,True))
        event(out,'explosion-first-seed-retained',trial=best['trial'],burst=best['burst'])
    child=None
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        while True:
            if args.explosion_priority and not any(r.get('new_policy_run')for r in rows):
                parameters=dict(max(rows,key=ranking)['parameters'])
                how=dict(method='full-tail-recheck-of-best-admissible-burst',
                    scope='Exact retained parameters; old truncated tail is not extrapolated')
            elif args.explosion_priority:parameters,how=burst_suggest(rows)
            else:parameters,how=suggest(rows,args.initial_emissivity)
            n=len(rows);folder=out/f'trial-{n:04d}'
            # Existing attempts are immutable, including an interrupted prefix.
            while folder.exists():n+=1;folder=out/f'trial-{n:04d}'
            state.update(active_trial=str(folder),parameters=parameters,proposal=how,progress={})
            atomic_save(out/'status.json',state);event(out,'proposal',trial=str(folder),parameters=parameters,acquisition=how)
            command=[sys.executable,str(ENGINE),'--output',str(folder),
                '--emissivity',str(parameters['effective_emissivity']),
                '--stirring',str(parameters['thermal_stirring_coefficient'])]
            if args.explosion_priority:command.append('--explosion-priority')
            with (out/f'trial-{n:04d}-controller.log').open('w')as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                while child.poll()is None:
                    try:child.wait(timeout=10.)
                    except subprocess.TimeoutExpired:pass
                    if (folder/'status.json').exists():
                        progress=json.loads((folder/'status.json').read_text())
                        state['progress']={k:progress[k]for k in ('status','last_completed_observation_s',
                            'live_advanced_to_s','wall_s','minimum_regular_step_s','burst')if k in progress}
                        atomic_save(out/'status.json',state)
                child=None
            if not(folder/'status.json').exists():raise RuntimeError('Trial controller failed before recording its inputs; inspect controller log')
            row=analyze(folder,data,args.explosion_priority);row['new_policy_run']=True;rows.append(row)
            atomic_save(out/'observations.json',rows);event(out,'trial-result',**row)
            best=max(rows,key=ranking);state.update(trials=len(rows)-1,best_trial=best['trial'],
                best_last_observed_s=best['last_observed_s'],best_anchor_rmse_k=best.get('anchor_rmse_k'))
            if args.explosion_priority:state['best_burst']=best.get('burst')
            if best is row:
                state['development_record']=publish_best(out,row,args.explosion_priority);event(out,'best-updated',trial=row['trial'],parameters=row['parameters'])
            if row.get('acceptance',{}).get('accepted'):
                state.update(status='engineering-cooling-converged',engineering_cooling_converged=True,
                    acceptance=row['acceptance'],selected_trial=row['trial']);break
            atomic_save(out/'status.json',state)
            # A rejected temperature prefix is not a reason to stop calibrating.
    except KeyboardInterrupt:
        if child is not None:
            child.terminate()
            try:child.wait(timeout=30.)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

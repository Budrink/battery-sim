#!/usr/bin/env python3
"""Fresh, bounded direct-can-radiation diagnostic and first/second-wave bracket.

The geometry is computed once from the retained layout. Only a shared effective
emissivity and shared R2 rate vary; each geometry is re-centered on the primary.
All-wave/temperature/ledger guards still reject bad prefixes. Matching the first
two events is explicitly NOT full-cascade calibration or independent validation.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import tarfile
import time

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import BASE, BASE_HASH, MINIMUM_SPEED, trial
from check_fsri30_vent_retention import SEED, budget_report, read_run
from fsri30_data import ROOT, load, save, sha
from fsri30_internal_radiation import add_internal_radiation
from fsri30_wave_fit import POLICY, reference


def second_arrival(row, source='B10'):
    events=row.get('assessment',{}).get('model',{}).get('arrivals',[])
    return min((e['time_s'] for e in events if e['channel']!=source),default=None)


def secondary_chemistry(scene,run,arrival):
    """A hot thermocouple proxy is not proof of a new cell's own R2 reaction."""
    source=scene['metadata']['heated_component'];events=[]
    if arrival is None:return dict(events=[],reacted_in_second_wave=0)
    for c in scene['cells']:
        if c.get('role')!='battery' or c['id']==source:continue
        spec=c['conversion'];denominator=spec['mass_kg']*spec['gas_yield']
        values=run['chemistry'][c['id']]['r2_generated_gas_kg']
        hits=[t for t,value in zip(run['time_s'],values) if value>=.5*denominator]
        if hits:events.append(dict(id=c['id'],half_R2_conversion_s=hits[0]))
    return dict(events=events,reacted_in_second_wave=sum(abs(e['half_R2_conversion_s']-arrival)<=50. for e in events),
                interpretation='Actual model conversion, not inferred experimental cell identities')


def propose(rows, anchor):
    if not rows:return .4,dict(anchor),'retained-parameters-plus-direct-radiation'
    last=rows[-1];p=dict(last['parameters']);eps=last['internal_emissivity']
    offset=last['assessment'].get('primary_offset_s')
    if offset is None or abs(offset)>10.:
        family=[r for r in rows if r['internal_emissivity']==eps]
        early=[r for r in family if (r['assessment'].get('primary_offset_s') or 0)<0]
        late=[r for r in family if (r['assessment'].get('primary_offset_s') or 0)>0]
        if early and late:
            pairs=[(a,b) for a in early for b in late if a['parameters']['r2_rate_scale']>b['parameters']['r2_rate_scale']]
            if pairs:
                a,b=min(pairs,key=lambda pair:pair[0]['parameters']['r2_rate_scale']/pair[1]['parameters']['r2_rate_scale'])
                da,db=a['assessment']['primary_offset_s'],b['assessment']['primary_offset_s']
                f=min(.85,max(.15,-da/(db-da)))
                rate=math.exp((1-f)*math.log(a['parameters']['r2_rate_scale'])+f*math.log(b['parameters']['r2_rate_scale']))
            else:rate=p['r2_rate_scale']*math.exp(max(-1.,min(1.,.006*(offset or 140.))))
        else:rate=p['r2_rate_scale']*math.exp(max(-1.,min(1.,.006*(offset or 140.))))
        p['r2_rate_scale']=max(.001,min(2.,rate))
        return eps,p,'recenter-primary-with-shared-R2-no-time-shift'
    aligned=[r for r in rows if r['assessment'].get('primary_offset_s') is not None and abs(r['assessment']['primary_offset_s'])<=10.]
    if second_arrival(last) is None and last['assessment']['last_observed_s']<2275.:
        return None,None,'incomplete-second-window-not-evidence-of-missing-cascade'
    early=[r for r in aligned if second_arrival(r) is not None and second_arrival(r)<2126.]
    late=[r for r in aligned if (second_arrival(r) is None and r['assessment']['last_observed_s']>=2275.) or
          (second_arrival(r) is not None and second_arrival(r)>2146.)]
    brackets=[(a,b) for a in late for b in early if a['internal_emissivity']<b['internal_emissivity']]
    if brackets:
        a,b=min(brackets,key=lambda pair:pair[1]['internal_emissivity']/pair[0]['internal_emissivity'])
        eps=math.sqrt(a['internal_emissivity']*b['internal_emissivity'])
    elif second_arrival(last) is None or second_arrival(last)>2146.:eps=min(1.,eps*2.)
    elif second_arrival(last)<2126.:eps=max(.025,eps/2.)
    elif last.get('secondary_chemistry',{}).get('reacted_in_second_wave',0)>=2:
        return None,None,'first-two-arrivals-and-secondary-R2-target-met-not-full-validation'
    else:return None,None,'hot-arrival-is-not-yet-a-collective-R2-cascade'
    if any(r['internal_emissivity']==eps and r['parameters']==p for r in rows):
        return None,None,'bounded-radiation-bracket-exhausted-model-review-needed'
    return eps,p,'direct-radiation-second-wave-bracket'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--max-trials',type=int,default=16)
    parser.add_argument('--after-status',type=Path,help='Queue behind an existing GPU diagnostic, never overlap full runs')
    args=parser.parse_args();out=args.output.resolve();probe=args.probe.resolve();core=sha(probe)
    if sha(BASE)!=BASE_HASH:raise ValueError('Retained base changed')
    seed=json.loads((SEED/'status.json').read_text());anchor=seed['parameters']
    if sha(Path(seed['run_path']))!=seed['run_sha256']:raise ValueError('Learned parameter history changed')
    base=json.loads(BASE.read_text());data,manifest=load();ref=reference(data)
    out.mkdir(parents=True,exist_ok=False);lock=(out/'search.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths=sorted(p for directory in ('scripts','native') for p in (ROOT/directory).rglob('*') if p.suffix in ('.py','.cpp','.cu','.hpp','.cuh'))+[ROOT/'CMakeLists.txt']
    contract=dict(native_sha256=core,probe=str(probe),seed_trial=str(SEED),seed_parameters=anchor,
        base_sha256=BASE_HASH,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        csv_sha256=manifest['csv_sha256'],fresh_from_zero=True,checkpoints_reused=False,mesh_mm=20,
        minimum_regular_step_s=.01,minimum_realtime_ratio=MINIMUM_SPEED,policy=asdict(POLICY),
        effective_emissivity_bounds=[.025,1.],shared_R2_bounds=[.001,2.],gas_geometry_changed=False,
        chemical_inventory_changed=False,diagnostic_only=True,calibrated=False,validation_passed=False)
    save(out/'contract.json',contract)
    with tarfile.open(out/'source.tar.gz','w:gz') as archive:
        for p in paths:archive.add(p,arcname=str(p.relative_to(ROOT)))
    os.environ['BATTERY_AVERAGED_FAILURE_SNAPSHOT']=str(out/'numerical-failure.json')
    state=dict(status='running',pid=os.getpid(),native_sha256=core,completed_trials=0,
               calibrated=False,validation_passed=False)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    def update(row):
        state.update(active_trial=row['trial'],progress=row);atomic_save(out/'status.json',state)
    rows=[];atomic_save(out/'status.json',state)
    try:
        if args.after_status:
            state.update(status='waiting-for-gpu',predecessor_status=str(args.after_status.resolve()))
            atomic_save(out/'status.json',state)
            while True:
                previous=json.loads(args.after_status.read_text())
                if previous['status'] in ('complete','stopped'):break
                if previous['status']!='running':raise ValueError('Predecessor requires review: '+previous['status'])
                os.kill(previous['pid'],0) # A stale running file must not leave an immortal queue.
                time.sleep(5.)
            state['status']='running';atomic_save(out/'status.json',state)
        for number in range(args.max_trials):
            eps,parameters,method=propose(rows,anchor)
            if eps is None:state.update(status=method);break
            state.update(internal_emissivity=eps,proposal=method);atomic_save(out/'status.json',state)
            if sha(probe)!=core:raise ValueError('Core changed during diagnostic')
            scene=add_internal_radiation(base,eps)
            row=trial(out/f'trial-{number:04d}',scene,parameters,20,ref,data,update,probe=probe)
            row['internal_emissivity']=eps;row['second_arrival_s']=second_arrival(row)
            if row.get('run_path'):
                run=read_run(row)
                save(Path(row['trial'])/'gas-budget.json',budget_report(scene,run,row))
                row['secondary_chemistry']=secondary_chemistry(scene,run,row['second_arrival_s'])
            rows.append(row);save(out/'observations.json',rows)
            atomic_save(Path(row['trial'])/'status.json',row)
            state.update(completed_trials=len(rows),last_result=row)
            atomic_save(out/'status.json',state)
            print(json.dumps(dict(trial=number,emissivity=eps,status=row['status'],
                primary_offset_s=row['assessment'].get('primary_offset_s'),second_arrival_s=row['second_arrival_s'])),flush=True)
            if row['status']=='numerical-failed':state['status']='numerical-review-needed';break
        else:state['status']='bounded-search-exhausted'
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc));raise
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

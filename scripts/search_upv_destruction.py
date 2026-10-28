#!/usr/bin/env python3
"""Persistent bounded GPU destruction search: lattice exploration + trust steps.

No material/contact/electrical fitting. Full trajectories only qualify as fits.
Pruned prefixes can guide recovery, but can never become an accepted full fit.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from scipy.stats import qmc
import fit_upv_kinetics as fit
import upv_destruction_objective as objective
from realtime_guard import AVERAGE_MINIMUM, ROLLING_MINIMUM, WINDOW_S

# Explicit engineering boxes, not measured material confidence intervals.
AXES = {
    'k1_75':(1e-10,1e-4,'log'), 'ea1':(120000.,280000.,'linear'),
    'k2_75':(1e-8,1e-3,'log'), 'ea2':(90000.,220000.,'linear'),
    'energy_j':(250000.,600000.,'linear'), 'gas_fraction':(.30,.85,'linear'),
    'gas_yield':(.20,.65,'linear'), 'gas_rate':(2.,200.,'log'),
    'gas_activation':(40000.,110000.,'linear'),
    'shell_strength_pa':(3e5,1e7,'log'), 'opening_rate':(.05,3.,'log'),
    'chamber_gas_h':(1.,500.,'log')}
KEYS=tuple(AXES)


def encode(p):
    x=[]
    for k,(lo,hi,kind) in AXES.items():
        v=p[k]
        if kind=='log':v,lo,hi=map(math.log,(v,lo,hi))
        x.append((v-lo)/(hi-lo))
    return np.asarray(x)


def decode(x):
    if len(x)!=len(KEYS) or not np.isfinite(x).all() or np.any(np.asarray(x)<0) or np.any(np.asarray(x)>1):
        raise ValueError('Outside hard search box')
    result={}
    for value,(k,(lo,hi,kind)) in zip(x,AXES.items()):
        physical_lo,physical_hi=lo,hi
        if kind=='log':lo,hi=map(math.log,(lo,hi))
        v=lo+float(value)*(hi-lo)
        result[k]=min(physical_hi,max(physical_lo,math.exp(v) if kind=='log' else v))
    return result


def anchor_coordinates(parameters):
    p={k:parameters[k]for k in KEYS if k not in ('k1_75','ea1','k2_75','ea2')}
    p.update(k1_75=fit.rate_at(parameters['r1_rate'],parameters['r1_activation'],400.,fit.ANCHOR_K),
             ea1=parameters['r1_activation'],
             k2_75=fit.rate_at(parameters['r2_rate'],parameters['r2_activation'],450.,fit.ANCHOR_K),
             ea2=parameters['r2_activation'])
    x=encode(p)
    if np.any(x< -1e-10) or np.any(x>1+1e-10):raise ValueError('Seed outside declared search box')
    return np.clip(x,0.,1.)


def make_plan(base,levels=7,global_points=256,seed=20261018):
    anchor=anchor_coordinates(base);coarse=[]
    # Known contrasting regimes first; none overwrites the seed.
    for changes in ({}, {'k2_75':max(AXES['k2_75'][0],decode(anchor)['k2_75']*.5)},
                    {'ea1':max(AXES['ea1'][0],decode(anchor)['ea1']-40000.)},
                    {'k1_75':max(AXES['k1_75'][0],decode(anchor)['k1_75']*.1)},
                    {'chamber_gas_h':80.},{'chamber_gas_h':200.},
                    {'chamber_gas_h':80.,'gas_fraction':.4,'k2_75':decode(anchor)['k2_75']*.5},
                    {'chamber_gas_h':80.,'energy_j':600000.,'gas_fraction':.55},
                    {'chamber_gas_h':80.,'shell_strength_pa':3e6,'opening_rate':1.}):
        p=decode(anchor);p.update(changes);coarse.append(dict(x=encode(p).tolist(),origin='contrast',axis=None))
    # Axial lattice points attribute single-coordinate gains unambiguously.
    for i,k in enumerate(KEYS):
        for v in np.linspace(0.,1.,levels):
            x=anchor.copy();x[i]=v;coarse.append(dict(x=x.tolist(),origin='axial-grid',axis=k))
    count=2**math.ceil(math.log2(max(1,global_points)))
    for x in qmc.Sobol(len(KEYS),scramble=True,seed=seed).random_base2(int(math.log2(count)))[:global_points]:
        x=np.rint(x*(levels-1))/(levels-1)
        coarse.append(dict(x=x.tolist(),origin='space-filling-grid',axis=None))
    return dict(coarse=coarse,local=[],seen=[],radii=[1/(levels-1)]*len(KEYS),
                gains=[0.]*len(KEYS),incumbent=None,trials=0,invalid=0,records=[],since_improvement=0)


def next_point(state):
    # Keep global exploration alive even after local improvements.
    for _ in range(10000):
        local_turn=state['trials']%4!=0
        if state.get('since_improvement',0)>=12:local_turn=state['trials']%4==1
        use_local=bool(state['local']) and state['trials']>=6 and (local_turn or not state['coarse'])
        queue=state['local'] if use_local else state['coarse']
        if not queue:
            if state['local']:queue=state['local']
            else:return None
        item=queue.pop(0);key=','.join(f'{v:.11f}'for v in item['x'])
        if key in state['seen']:continue
        state['seen'].append(key);return item
    return None


def feedback(state,item,row,rank):
    old=state['incumbent'];improved=rank is not None and (old is None or tuple(rank)<tuple(old['rank']))
    if improved and old is not None and rank[0]==old['rank'][0]:
        same_stage=rank[0]==0 or rank[1]==old['rank'][1]
        index=1 if rank[0]==0 else 2
        if same_stage and old['rank'][index]-rank[index]<max(.01,.002*abs(old['rank'][index])):
            improved=False # numerical noise must not create a new trust centre
    axis=item.get('axis')
    single=old is not None and np.count_nonzero(np.abs(np.asarray(item['x'])-np.asarray(old['x']))>1e-9)==1
    if improved:
        state['since_improvement']=0
        old_x=np.asarray(old['x']) if old else None
        state['incumbent']=dict(label=row['label'],rank=list(rank),x=item['x'],
            feasible=row['feasible'],objective=row.get('objective'),parameters=row['parameters'])
        if axis:
            i=KEYS.index(axis)
            # Credit only a tested SINGLE-coordinate move, not a joint proposal.
            if single:
                state['gains'][i]+=1.;state['radii'][i]=min(1/3,state['radii'][i]*1.2)
        order=sorted(range(len(KEYS)),key=lambda i:state['gains'][i],reverse=True)
        candidates=[];center=np.asarray(item['x'])
        # Try the successful direction first, then bracket it and other axes.
        if axis and old_x is not None:
            i=KEYS.index(axis);direction=np.sign(center[i]-old_x[i])
            if direction:
                x=center.copy();x[i]=np.clip(x[i]+direction*state['radii'][i],0.,1.)
                candidates.append(dict(x=x.tolist(),origin='continue-improving-axis',axis=axis))
        for i in order:
            for sign in (1.,-1.):
                x=center.copy();x[i]=np.clip(x[i]+sign*state['radii'][i],0.,1.)
                candidates.append(dict(x=x.tolist(),origin='local-bracket',axis=KEYS[i]))
        state['local']=candidates # stale brackets around old centres are discarded
    else:state['since_improvement']=state.get('since_improvement',0)+1
    if not improved and axis and single and row.get('status')!='failed':
        i=KEYS.index(axis)
        state['radii'][i]=max(1/128,state['radii'][i]*(.5 if not row['feasible'] else .8))
        if state['radii'][i]>1/128:
            for sign in (1.,-1.):
                x=np.asarray(old['x']).copy();x[i]=np.clip(x[i]+sign*state['radii'][i],0.,1.)
                state['local'].append(dict(x=x.tolist(),origin='shrink-critical-bracket',axis=axis))
    return improved


def first_violation(run,meta):
    t=np.asarray(run['time_s'])
    for at in np.arange(60.,t[-1]+5.,5.):
        n=int(np.searchsorted(t,min(at,t[-1]),side='right'))
        slim=dict(time_s=t[:n],temperature_k={k:np.asarray(v)[:n]for k,v in run['temperature_k'].items()},
                  gas={'chamber':{'pressure_pa':fit.dataio.chamber_pressure_pa(run)[:n]}})
        reason=objective.reject_reason(slim,meta)
        if reason:return float(t[n-1]),reason
    return None


def code_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in
            [Path(__file__),Path(fit.__file__),Path(objective.__file__)]}


def run(args):
    out=args.output.resolve()
    if args.resume:
        config=json.loads((out/'search-config.json').read_text());state=json.loads((out/'search-state.json').read_text())
        if config['fingerprint']!=fit.cal.fingerprint() or config['script_hashes']!=code_hashes():
            raise ValueError('Cannot resume against changed code/core')
        base=config['parameters'];started=time.monotonic();used=float(state.get('wall_s',0.))
    else:
        if out.exists():raise ValueError('Output already exists; use explicit --resume')
        seed=json.loads(args.seed.read_text());base=seed['parameters'];fp=fit.cal.fingerprint()
        if seed.get('model_fingerprint')!=fp or set(base)!=set(fit.model.DEFAULT):raise ValueError('Explicit complete current seed required')
        state=make_plan(base,args.levels,args.global_points,args.random_seed);started=time.monotonic();used=0.
        config=dict(parameters=base,fingerprint=fp,script_hashes=code_hashes(),axes=AXES,
            levels=args.levels,global_points=args.global_points,random_seed=args.random_seed,
            max_trials=args.max_trials,max_wall_hours=args.max_hours,trial_timeout_s=args.timeout,
            full_horizon='observed event + 300 s, one continuous trajectory',temperature_policy=objective.POLICY,
            source_seed=str(args.seed.resolve()),source_seed_sha256=hashlib.sha256(args.seed.read_bytes()).hexdigest(),
            scheduling='Contrasting kinetic/destruction starts; then 3 local : 1 global; after 12 non-improvements 1 local : 3 global; meaningful gain >=0.2% or 0.01 loss',
            fixed='all measured materials, geometry, solid contacts, external cooling, phi and electrical heating',
            gas_extensions=seed.get('gas_extensions'),numerical_controls=seed.get('numerical_controls'),
            performance_policy=dict(whole_run_minimum=AVERAGE_MINIMUM,rolling_minimum=ROLLING_MINIMUM,window_s=WINDOW_S),
            validation_used_for_selection=False,peaks_ignored=False,max_output_gib=args.max_output_gib)
        out.mkdir(parents=True);fit.cal.save(out/'search-config.json',config)
    import shutil
    lock=open('/tmp/battery-upv-destruction-search.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fit.cal.save(out/'process.json',dict(pid=os.getpid(),command=sys.argv,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    status='running';current=None
    def terminate(_sig,_frame):raise KeyboardInterrupt
    previous_handler=signal.signal(signal.SIGTERM,terminate)
    try:
        while state['trials']<config['max_trials']:
            state['wall_s']=used+time.monotonic()-started
            if state['wall_s']>=config['max_wall_hours']*3600:status='wall_budget';break
            if fit.cal.fingerprint()!=config['fingerprint'] or code_hashes()!=config['script_hashes']:
                raise RuntimeError('Code changed during search; refusing mixed-core calibration')
            size=sum(p.stat().st_size for p in out.rglob('*')if p.is_file())
            if size>config['max_output_gib']*1024**3 or shutil.disk_usage(out).free<5*1024**3:
                status='disk_budget';break
            item=next_point(state)
            if item is None:status='grid_exhausted';break
            proposal=decode(item['x']);proposal['label']=f"trial-{state['trials']:04d}"
            try:fit.parameters(base,proposal,allow_destruction=True)
            except ValueError:
                state['invalid']+=1;continue
            current=proposal['label'];state['trials']+=1;state['active_trial']=current
            state['status']='running';fit.cal.save(out/'search-state.json',state)
            def prune(r,m):
                reason=objective.reject_reason(r,m)
                if reason:return reason
                incumbent=state['incumbent']
                if incumbent and incumbent['feasible'] and r['time_s'][-1]>m['event']['time_s']+20.:
                    if objective.loss(r,m)['objective']>incumbent['objective']*1.15:
                        return 'objective lower bound exceeds feasible incumbent by 15 percent'
                return None
            remaining=config['max_wall_hours']*3600-state['wall_s']
            row=fit.evaluate(proposal,base,out,config['fingerprint'],min(config['trial_timeout_s'],remaining),
                explosion=True,objective_fn=objective.score,prune_fn=prune,
                gas_extensions=config.get('gas_extensions'),numerical_controls=config.get('numerical_controls'))
            folder=out/current;path=folder/'trial-run.json.gz';rank=None;improved=False
            if path.exists():
                with gzip.open(path,'rt')as stream:r=json.load(stream)
                data,_=fit.dataio.load('no-cooling');event=fit.dataio.event(data)
                meta=dict(data=data,event=event,fit_end_s=event['time_s']+300.)
                violation=first_violation(r,meta)
                row['critical_violation']=violation;row['feasible']=row['status']=='complete' and violation is None
                known=objective.loss(r,meta);row['known_loss']=known
                if row['feasible']:rank=(0,row['objective'],0.)
                elif violation is not None and row['status'] in ('complete','pruned'):
                    rank=(1,-math.floor(violation[0]/10.),known['objective'])
                elif row['status']=='pruned' and 'objective lower bound' not in str(row['error']):
                    rank=(1,-math.floor(row['last_sample_s']/10.),known['objective'])
                row.update(search_origin=item['origin'],changed_axis=item.get('axis'),rank=rank)
                improved=feedback(state,item,row,rank)
                # Retain every candidate's observations/parameters; large full
                # gas fields only for improvements, never touch prior searches.
                compact=dict(time_s=r['time_s'],temperature_k=r['temperature_k'],
                    pressure_pa=fit.dataio.chamber_pressure_pa(r),status=row['status'],feasible=row['feasible'])
                with gzip.open(folder/'observations.json.gz','wt')as f:json.dump(compact,f,allow_nan=False)
                if not improved:
                    for name in ('trial-run.json.gz','last-incomplete-run.json.gz'):
                        (folder/name).unlink(missing_ok=True)
            else:
                row['feasible']=False
                feedback(state,item,row,None)
            row['improved']=improved;fit.cal.save(folder/'search-result.json',row)
            state['records'].append(dict(label=current,status=row['status'],feasible=row['feasible'],
                objective=row.get('objective'),improved=improved,rank=rank,origin=item['origin'],axis=item.get('axis')))
            if improved:
                fit.cal.save(out/'search-incumbent.json',dict(**state['incumbent'],
                    status='feasible candidate, NOT validated' if row['feasible'] else 'RECOVERY PREFIX ONLY, NOT a full calibration'))
                if row['status']=='pruned':
                    import subprocess
                    subprocess.run([sys.executable,str(fit.ROOT/'scripts/report_upv_r123.py'),str(folder),
                                    '--stage','trial','--partial'],check=True)
            if row['feasible']:
                feasible=[v for v in state['records']if v['feasible']]
                if row['objective']==min(v['objective']for v in feasible):
                    fit.cal.save(out/'best-feasible.json',row)
            state['wall_s']=used+time.monotonic()-started;state['status']='running';current=None;state['active_trial']=None
            fit.cal.save(out/'search-state.json',state)
            print('SEARCH',json.dumps(state['records'][-1]),flush=True)
        else:status='trial_budget'
    except KeyboardInterrupt:
        status='stopped'
        if current:fit.cal.save(out/current/'interrupted.json',dict(status='interrupted',full_fit=False))
    except BaseException as exc:
        status='error';state['error']=repr(exc);raise
    finally:
        state['status']=status;state['wall_s']=used+time.monotonic()-started
        fit.cal.save(out/'search-state.json',state)
        print('SEARCH_STOP',status,flush=True)
        lock.close()
        signal.signal(signal.SIGTERM,previous_handler)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--levels',type=int,default=7)
    ap.add_argument('--global-points',type=int,default=256);ap.add_argument('--max-trials',type=int,default=512)
    ap.add_argument('--max-hours',type=float,default=24.);ap.add_argument('--timeout',type=float,default=2400.)
    ap.add_argument('--max-output-gib',type=float,default=10.);ap.add_argument('--random-seed',type=int,default=20261018)
    args=ap.parse_args()
    if not (3<=args.levels<=21 and 1<=args.global_points<=4096 and 1<=args.max_trials<=10000 and
            0<args.max_hours<=168 and args.timeout>=1. and args.max_output_gib>0):raise ValueError('Invalid search budget')
    run(args)


if __name__=='__main__':main()

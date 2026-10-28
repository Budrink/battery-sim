#!/usr/bin/env python3
"""Bounded checkpoint search of shared gas heat connectivity, then full replay."""
import argparse
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from scipy.stats import qmc
import calibrate_upv_r123 as cal
import upv_gas_contact_fit as fit
import upv_r123_data as dataio
import upv_destruction_objective as objective
from check_upv_restart import native
from upv_gas_remap import unvented_remap


def plot(out,run,meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.asarray(run['time_s']);obs=dataio.observed(meta['data'],t)
    fig,axes=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
    for ax,key in zip(axes.flat,dataio.CHANNELS):
        ax.plot(t,obs[key]-273.15,label='Эксперимент',c='#222222')
        ax.plot(t,np.asarray(run['temperature_k'][key])-273.15,label='Расчёт',c='#df6a32')
        ax.set(title=key,ylabel='°C');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=axes[3,0];ax.plot(t,obs['pressure_bar'],label='Эксперимент',c='#222222')
    ax.plot(t,np.asarray(dataio.chamber_pressure_pa(run))/1e5,label='Расчёт',c='#df6a32')
    ax.set(title='Давление камеры',ylabel='bar');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[3,1].axis('off');fig.suptitle('Подбор только газовой связности; полный хвост после разрушения')
    fig.savefig(out/'comparison.png',dpi=130);plt.close(fig)


def plan():
    queue=[]
    for changes in ({},{'gas_ceiling':30.,'metric_450':6.,'metric_900':8.},
        {'gas_ceiling':100.,'metric_450':8.,'metric_900':10.},
        {'gas_ceiling':300.,'metric_450':10.,'metric_900':14.},
        {'gas_ceiling':100.,'metric_450':8.,'metric_900':10.,'wall_h':20.},
        {'gas_ceiling':100.,'metric_450':8.,'metric_900':10.,'wall_h':200.},
        {'gas_ceiling':30.,'metric_450':0.,'metric_900':10.,'wall_h':20.},
        {'gas_ceiling':300.,'metric_450':0.,'metric_900':14.,'wall_h':20.}):
        queue.append(dict(x=fit.encode(dict(fit.BASE,**changes)).tolist(),axis=None,origin='contrast'))
    for x in qmc.Sobol(len(fit.KEYS),scramble=True,seed=20261020).random_base2(4):
        queue.append(dict(x=x.tolist(),axis=None,origin='space-filling'))
    return queue


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-trials',type=int,default=24);ap.add_argument('--max-hours',type=float,default=2.)
    ap.add_argument('--timeout',type=float,default=900.);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    out=a.output.resolve()
    if not a.resume:out.mkdir(exist_ok=False)
    lock=open('/tmp/battery-upv-gas-contacts.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=dataio.ROOT/'data/calibration/upv-separated-chemistry-v6'
    source=root/'restart-gas-baseline-20261019-r1';source_scene=json.loads((source/'scenario.json').read_text())
    template=json.loads((root/'gas-connectivity-20261020-r3/case-1/scenario.json').read_text())
    data,_=dataio.load('no-cooling');event=dataio.event(data);end=event['time_s']+300.
    template['simulation']['duration_s']=end
    template['failure_sample_times_s']=dataio.observation_times(data,event,end)
    template['metadata']['zone_diagnostics']=False
    meta=dict(data=data,event=event,fit_end_s=end)
    request=dict(restart_from=str(source/'sealed-state.bin'),restart_source_scenario=str(source/'scenario.json'),
        unvented_gas_remap=unvented_remap(source_scene,template),stop_after_first_rupture=False,
        restart_contract=json.loads((source/'status.json').read_text())['native_sha256'])
    core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    config=dict(axes=fit.AXES,coordinates=fit.COORDINATES,native_sha256=core,parameters_frozen=template['metadata']['parameters'],
        source_checkpoint=str(source/'sealed-state.bin'),source_state_s=750.,horizon_s=end,
        max_trials=a.max_trials,max_hours=a.max_hours,temperature_policy=objective.POLICY,
        validation_used_for_selection=False,scope='gas contact laws only; fixed chemistry, Cp, k, shell mechanics, topology and electrical input',
        reference_preserved='gas/solid G at 298.15 K; neutral profile through 350 K; wall reference is independently fitted',
        conditional_restart_search=True,full_from_ambient_required=True,
        script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in
            [Path(__file__),Path(fit.__file__),Path(objective.__file__)]})
    used=0.
    state=dict(status='running',pid=os.getpid(),records=[],best=None,best_feasible=None)
    if a.resume:
        previous=json.loads((out/'search-config.json').read_text())
        for k in ('axes','coordinates','native_sha256','parameters_frozen','source_checkpoint','horizon_s'):
            if json.dumps(previous.get(k),sort_keys=True)!=json.dumps(config[k],sort_keys=True):
                raise ValueError('Cannot resume against changed physics: '+k)
        state=json.loads((out/'search-state.json').read_text());used=state.get('wall_s',0.)
        state.update(status='running',pid=os.getpid(),best=None,best_feasible=None)
        # Rescore saved curves after a search-policy correction; never rerun or
        # reset their physical parameters, and never treat an interrupted case as fit.
        for row in state['records']:
            folder=out/row['label']/'replay';path=folder/('run.json.gz'if row['status']=='complete'else'incomplete-run.json.gz')
            if not path.exists():continue
            run=json.load(gzip.open(path,'rt'));rank=fit.ranking(row,run,meta)
            if rank is not None:
                row['rank']=list(rank)
                if fit.meaningful_gain(rank,state['best']):
                    state['best']=dict(row,x=fit.encode(row['parameters']).tolist())
                    if row['feasible']:state['best_feasible']=state['best']
        if state['best']:cal.save(out/'best-recovery.json',state['best'])
        config['previous_search_script_sha256']=previous['script_sha256']
    cal.save(out/'search-config.json',config)
    cal.save(out/'search-state.json',state)
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    queue=plan();local=[]
    seen={tuple(round(v,10)for v in fit.encode(r['parameters']))for r in state['records']}
    radii=np.asarray(state.get('radii',[.18]*len(fit.KEYS)));started=time.monotonic()
    try:
        for number in range(len(state['records']),a.max_trials):
            if used+time.monotonic()-started>a.max_hours*3600.:state['status']='wall_budget';break
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise RuntimeError('CUDA binary changed during search')
            item=None
            while item is None:
                items=local if local and number>=6 and (number%4 or not queue) else queue
                if not items:items=local
                if not items:break
                proposal=items.pop(0);key=tuple(round(v,10)for v in proposal['x'])
                if key not in seen:seen.add(key);item=proposal
            if item is None:state['status']='grid_exhausted';break
            p=fit.decode(item['x']);serial=number
            while (out/f'trial-{serial:03d}').exists():serial+=1
            label=f'trial-{serial:03d}';folder=out/label;folder.mkdir()
            scene=fit.candidate(template,p);cal.save(folder/'scenario.json',scene);cal.save(folder/'parameters.json',p)
            state.update(active_trial=label,wall_s=used+time.monotonic()-started,radii=radii.tolist())
            cal.save(out/'search-state.json',state);print('START',label,json.dumps(p),flush=True)
            def prune(r):
                reason=objective.reject_reason(r,meta)
                if reason:return reason
                best=state['best_feasible']
                if best and r['time_s'][-1]>event['time_s']+20. and fit.suffix_loss(r,meta)['objective']>1.15*best['objective']:
                    return 'known objective lower bound exceeds completed feasible incumbent'
                return None
            row=dict(label=label,parameters=p,origin=item['origin'],axis=item['axis'])
            try:
                run=native(folder/'scenario.json',folder/'replay',request,a.timeout,prune_fn=prune)
                row.update(status='complete',wall_s=run['wall_s'])
            except RuntimeError as exc:
                partial=folder/'replay/incomplete-run.json.gz'
                row.update(status='pruned'if str(exc).startswith('Pruned:')else'failed',reason=str(exc))
                run=json.load(gzip.open(partial,'rt'))if partial.exists()else None
            except TimeoutError as exc:
                row.update(status='failed',reason=str(exc));run=None
            rank=None
            if run is not None:
                score=fit.suffix_loss(run,meta);row.update(score)
                critical=fit.critical_history(run,meta)
                feasible=row['status']=='complete'and critical is None and run['time_s'][-1]>=end-1e-7
                row.update(feasible=feasible,critical_violation=critical)
                # Completed feasible trajectories outrank all recovery prefixes.
                rank=fit.ranking(row,run,meta)
                if rank is not None:row['rank']=list(rank)
            old=state['best'];improved=fit.meaningful_gain(rank,old)
            if improved:
                best=dict(row,x=item['x']);state['best']=best
                cal.save(out/'best-recovery.json',best)
                plot(folder,run,meta)
                if row['feasible']:
                    state['best_feasible']=best;cal.save(out/'best-feasible.json',best)
                center=np.asarray(item['x']);local=[];axis=item['axis']
                if axis is not None and old is not None:
                    i=fit.KEYS.index(axis);sign=np.sign(center[i]-old['x'][i])
                    x=center.copy();x[i]=np.clip(x[i]+sign*radii[i],0.,1.)
                    local.append(dict(x=x.tolist(),axis=axis,origin='continue-improving-axis'))
                for i,k in enumerate(fit.KEYS):
                    for sign in (1.,-1.):
                        x=center.copy();x[i]=np.clip(x[i]+sign*radii[i],0.,1.)
                        local.append(dict(x=x.tolist(),axis=k,origin='local-bracket'))
            elif item['axis']is not None and old:
                i=fit.KEYS.index(item['axis']);radii[i]=max(1/128,radii[i]*.5)
            row['improved']=improved;cal.save(folder/'result.json',row);state['records'].append(row)
            cal.save(out/'search-state.json',state)
            print('RESULT',json.dumps({k:row[k]for k in ('label','status','objective','feasible','improved','critical_violation')if k in row}),flush=True)
        else:state['status']='trial_budget'
        # Never promote a conditional checkpoint fit without its full trajectory.
        if state['best_feasible']:
            state['phase']='full_from_ambient';cal.save(out/'search-state.json',state)
            best=state['best_feasible'];folder=out/'full-from-ambient';folder.mkdir()
            scene=fit.candidate(template,best['parameters']);cal.save(folder/'scenario.json',scene)
            run=native(folder/'scenario.json',folder/'replay',dict(stop_after_first_rupture=False),a.timeout,prune_fn=lambda r:objective.reject_reason(r,meta))
            plot(folder,run,meta);cal.save(folder/'score.json',objective.score(run,meta))
            state['full_from_ambient']='complete'
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        state['wall_s']=used+time.monotonic()-started;state['radii']=radii.tolist();state.pop('active_trial',None)
        cal.save(out/'search-state.json',state);print('SEARCH_STATUS',state['status'],flush=True)
    return 1 if state['status']=='failed'else 0


if __name__=='__main__':raise SystemExit(main())

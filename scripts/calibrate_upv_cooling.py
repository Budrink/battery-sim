#!/usr/bin/env python3
"""Cooling search from the explicit latest profile, with a hard sharpness gate.

One sealed checkpoint per gas-yield value. Gas-boundary continuations are
conditional; an accepted candidate MUST be rerun fresh into the long tail.
"""
import argparse
import copy
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
import upv_r123_data as dataio
import upv_cooling_fit as fit
import upv_destruction_objective as guards
from upv_cooling_review import plot_cooling
from resolve_upv_rupture_impulse import compact,plot_impulse
from check_upv_restart import native


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-trials',type=int,default=24);ap.add_argument('--end',type=float,default=1500.)
    ap.add_argument('--extended-end',type=float,default=3600.)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--prefix-reference',type=Path,help='Previously captured sealed prefix at the same gas yield; native identity is checked')
    ap.add_argument('--known-control',type=Path,help='Completed exact physical control; audited before reuse')
    ap.add_argument('--incumbent',type=Path,help='Completed candidate from previous search, audited before reuse')
    ap.add_argument('--no-objective-prune',action='store_true',help='Diagnostic: keep physical/temperature/front/speed guards but evaluate the whole cooling window')
    a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=a.resume)
    original=json.loads((a.reference/'scenario.json').read_text());data,_=dataio.load('no-cooling');event=dataio.event(data)
    core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest();started=time.monotonic()
    config=dict(source=str(a.reference.resolve()),source_scene_sha256=hashlib.sha256((a.reference/'scenario.json').read_bytes()).hexdigest(),
        native_sha256=core,axes=fit.AXES,end_s=a.end,extended_end_s=a.extended_end,
        conditional_gas_response_only=True,full_fresh_extended_required=True,
        geometry_remains_diagnostic=True,validation_used_for_selection=False,
        hot_source_abort_policy=dict(threshold_k=423.15,margin_k=500.,relative=.9,cold_margin_k=75.,cold_relative=.5,
            unchanged='Witness/pressure/physical guards and objective; source raw errors remain reported'),
        unchanged='R1/R2 kinetics, chemical inventory/energy, Cp/k, solid contacts, shell mechanics, current/UI budget; forward gas heating law and warm reverse-law endpoints',
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__),Path(fit.__file__)]})
    if a.resume:
        previous=json.loads((out/'search-config.json').read_text())
        for key in ('source_scene_sha256','native_sha256','end_s','extended_end_s','axes'):
            if json.loads(json.dumps(config[key]))!=previous[key]:raise ValueError('Resume contract changed: '+key)
        state=json.loads((out/'search-state.json').read_text())
        state.setdefault('resumes',[]).append(dict(previous_status=state['status'],scripts=config['scripts']))
        state.update(status='running',pid=os.getpid())
    else:
        cal.save(out/'search-config.json',config)
        state=dict(status='running',pid=os.getpid(),records=[],best=None,accepted_calibration=False)
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    queue=[dict(p=dict(gas_yield_scale=g,casing_damage_metric=d,warm_mixing_scale=m,warm_surface_scale=w,gas_transport_response_scale=1.),origin='contrast',axis=None)
           for g,d,m,w in [(.6,4.,10.,10.),(.6,4.,10.,1.),(.6,4.,10.,3.),(.6,4.,10.,30.),(.6,4.,20.,10.)]]
    for x in qmc.Sobol(len(fit.AXES),scramble=True,seed=20261021).random_base2(4):
        queue.append(dict(p=fit.decode(x),origin='space-filling',axis=None))
    local=[];seen=set();prefixes={};radii=np.array([.12,.20,.20,.20,.20]);curves={}
    for control_arg in (a.known_control,a.incumbent):
        if control_arg is None:continue
        control=control_arg.resolve()
        p=(json.loads((control/'parameters.json').read_text())if control_arg==a.incumbent else dict(gas_yield_scale=.6,casing_damage_metric=0.,warm_mixing_scale=10.))
        p.setdefault('warm_surface_scale',1.)
        p.setdefault('gas_transport_response_scale',1.)
        def physical(s):
            s=copy.deepcopy(s)
            for k in ('metadata','name','source','failure_sample_times_s','measurement_probes','thermal_path_trace','surface_probe_trace','gas_film_trace','process_history_selection'):
                s.pop(k,None)
            def normalize(v):
                if isinstance(v,dict):
                    if v.get('damage')==[0.,0.,0.]:v.pop('damage')
                    for value in v.values():normalize(value)
                elif isinstance(v,list):
                    for value in v:normalize(value)
            normalize(s);return s
        if physical(json.loads((control/'scenario.json').read_text()))!=physical(fit.candidate(original,p,a.end)):
            raise ValueError('Known control differs from current physical candidate')
        with gzip.open(control/'replay/run.json.gz','rt')as stream:control_run=json.load(stream)
        row=dict(fit.score(control_run,data,a.end),parameters=p,label='known-control',path=str(control),conditional_restart=False)
        if not row['complete']or not row['front']['accepted']:raise ValueError('Known control is incomplete or not sharp')
        seen.add(tuple(np.round(fit.encode(p),8)))
        state['known_control']=row
        if state['best']is None or row['objective']<state['best']['objective']:state['best']=row
    if a.incumbent and not a.resume:
        base=state['best']['parameters']
        queue=[dict(p=dict(base,gas_transport_response_scale=g),origin='free-gas-response',axis='gas_transport_response_scale')for g in (10.,3.,30.,1.,100.)]+queue
    if a.prefix_reference:
        prefix=a.prefix_reference.resolve()
        if not (prefix/'sealed-state.bin').exists():raise ValueError('Sealed prefix missing')
        params=json.loads((prefix/'parameters.json').read_text())
        prefixes[float(params['gas_yield_scale']).hex()]=prefix
        state['external_conditional_prefix']=str(prefix)
    if a.resume:
        for row in state['records']:
            seen.add(tuple(np.round(fit.encode(row['parameters']),8)))
            folder=out/row['label']
            # A valid binary state exists even when a later suffix was pruned.
            if (folder/'sealed-state.bin').exists():prefixes[float(row['parameters']['gas_yield_scale']).hex()]=folder
        if state['best']:
            center=fit.encode(state['best']['parameters'])
            for k in ('gas_transport_response_scale','warm_surface_scale','warm_mixing_scale','casing_damage_metric','gas_yield_scale'):
                i=list(fit.AXES).index(k)
                for sign in (1.,-1.):
                    x=center.copy();x[i]=np.clip(x[i]+sign*radii[i],0.,1.)
                    local.append(dict(p=fit.decode(x),axis=k,origin='resumed-local-bracket'))
    first=max([int(p.name.split('-')[1])for p in out.glob('trial-*')if p.is_dir()]+[-1])+1
    try:
        for number in range(first,a.max_trials):
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise RuntimeError('Native binary changed')
            item=None
            while item is None:
                items=local if local and number%4!=3 else queue
                if not items:items=queue if queue else local
                if not items:break
                v=items.pop(0);key=tuple(np.round(fit.encode(v['p']),8))
                if key not in seen:seen.add(key);item=v
            if item is None:state['status']='grid_exhausted';break
            p=item['p'];label=f'trial-{number:03d}';folder=out/label;folder.mkdir()
            scene=fit.candidate(original,p,a.end);scene['failure_sample_times_s']=dataio.observation_times(data,event,a.end)
            cal.save(folder/'scenario.json',scene);cal.save(folder/'parameters.json',p)
            gas_key=float(p['gas_yield_scale']).hex();prefix=prefixes.get(gas_key)
            request=dict(stop_after_first_rupture=False,restart_contract=core)
            if prefix:
                request.update(restart_from=str(prefix/'sealed-state.bin'),restart_gas_response_scenario=str(prefix/'scenario.json'))
            else:request['save_unvented_restart']=str(folder/'sealed-state.bin')
            state.update(active_trial=label,wall_s=time.monotonic()-started);cal.save(out/'search-state.json',state)
            print('START',label,json.dumps(p),'conditional'if prefix else'fresh',flush=True)
            def prune(run):
                why=fit.rejection_reason(run,data,event)
                if why:return why
                if run['time_s'][-1]>=event['time_s']+20.:
                    f=fit.front(run,data)
                    if not f['accepted']:return 'sharpness: '+', '.join(f['reasons'])
                    if not a.no_objective_prune and state['best'] and fit.score(run,data,a.end)['objective']>1.15*state['best']['objective']:
                        return 'cooling lower bound exceeds completed sharp incumbent'
            run=None;row=dict(label=label,parameters=p,conditional_restart=bool(prefix),origin=item['origin'],axis=item['axis'])
            try:
                run=native(folder/'scenario.json',folder/'replay',request,max(2000.,a.end*2.2),prune_fn=prune)
                row['status']='complete'
            except (RuntimeError,TimeoutError)as exc:
                row.update(status='rejected',reason=str(exc))
                path=folder/'replay/incomplete-run.json.gz'
                if path.exists():run=json.load(gzip.open(path,'rt'))
            if run is not None:
                row.update(fit.score(run,data,a.end))
                row['balance']={k:run[k]for k in ('max_thermal_residual_j','max_chemical_residual_j','max_mass_residual_kg')}
                if 'unvented_restart'in run and(folder/'sealed-state.bin').exists():prefixes[gas_key]=folder
                if row['complete']and row['front']['accepted']:
                    row['realtime_factor']=(run['time_s'][-1]-run['started_at_s'])/run['wall_s']
                    improved=state['best']is None or row['objective']<state['best']['objective']-.01
                    if improved:
                        old=state['best'];state['best']=dict(row);cal.save(out/'best-candidate.json',row)
                        curves={'best':compact(scene,run)};cal.save(folder/'curves.json',curves['best'])
                        plot_cooling(out,curves,data,{'best':'Кандидат охлаждения'})
                        plot_impulse(out,curves,data,{'best':'Кандидат охлаждения'})
                        center=fit.encode(p);local=[]
                        if item['axis']is not None and old:
                            i=list(fit.AXES).index(item['axis']);sign=np.sign(center[i]-fit.encode(old['parameters'])[i]);x=center.copy();x[i]=np.clip(x[i]+sign*radii[i],0.,1.)
                            local.append(dict(p=fit.decode(x),axis=item['axis'],origin='continue-improving-axis'))
                        for k in ('gas_transport_response_scale','warm_surface_scale','warm_mixing_scale','casing_damage_metric','gas_yield_scale'):
                            i=list(fit.AXES).index(k)
                            for sign in (1.,-1.):
                                x=center.copy();x[i]=np.clip(x[i]+sign*radii[i],0.,1.)
                                local.append(dict(p=fit.decode(x),axis=k,origin='local-bracket'))
                    elif item['axis']is not None:
                        i=list(fit.AXES).index(item['axis']);radii[i]=max(.02,radii[i]/2.)
                cal.save(folder/'result.json',row)
            state['records'].append(row);cal.save(out/'search-state.json',state)
            print('RESULT',json.dumps({k:row[k]for k in ('label','status','objective','accepted','front','channels','reason')if k in row}),flush=True)
            del run
            if row.get('accepted'):
                state['phase']='fresh_extended_verification';cal.save(out/'search-state.json',state)
                fresh=fit.candidate(original,p,a.extended_end);fresh['failure_sample_times_s']=dataio.observation_times(data,event,a.extended_end)
                dest=out/('fresh-extended-'+label);dest.mkdir();cal.save(dest/'scenario.json',fresh)
                check=native(dest/'scenario.json',dest/'replay',dict(stop_after_first_rupture=False),a.extended_end*2.2,
                    prune_fn=lambda r:fit.rejection_reason(r,data,event))
                review=fit.score(check,data,a.extended_end);cal.save(dest/'result.json',review)
                plot_cooling(dest,{'fresh':check},data,{'fresh':'Полный свежий прогон'})
                if review['accepted']:
                    state.update(status='cooling_tolerances_met',fresh_extended=str(dest),
                        accepted_calibration=False,note='Numerical fit only; diagnostic geometry must still be physically verified')
                    break
        else:state['status']='trial_budget'
    except KeyboardInterrupt:state['status']='stopped';raise
    except Exception as exc:state.update(status='failed',error=str(exc));raise
    finally:
        state.pop('active_trial',None);state['wall_s']=time.monotonic()-started;cal.save(out/'search-state.json',state)


if __name__=='__main__':main()

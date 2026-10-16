#!/usr/bin/env python3
"""Transient rupture: enhanced internal transfer -> residual structure; no mass solver."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import calibrate_upv_failure as v1
from calibrate_upv import Worker, save

OUTPUT=v1.ROOT/'data/calibration/upv-failure-transient-v3'
NAMES=v1.NAMES+('fast_fraction','tail_time_s','destroyed_bulk_factor','bulk_peak_factor','bulk_relaxation_time_s')
FIT_NAMES=('energy_j','destroyed_iss_h','destroyed_tim_h','destroyed_bulk_factor')
WITNESSES=('101','102','107','108')
PULSE_PRIOR=dict(duration_s=.5,fast_fraction=.8,tail_time_s=30.)
STRUCTURE_PRIOR=dict(bulk_peak_factor=3.,bulk_relaxation_time_s=5.)
BOUNDS=np.array([(1000.,1000000.),(.1,5.),(1.,10000.),(1.,10000.),(0.,1.),(5.,120.),(.005,30.),(.1,100.),(.05,120.)]).T
PROTOCOL=dict(v1.PROTOCOL,revision='transient-medium-late-witness-v3',fitted=list(FIT_NAMES),
    duration_s=1800.,fit_start_s=300.,pulse_prior=PULSE_PRIOR,structure_prior=STRUCTURE_PRIOR,
    score='Four witness temperatures only, original scan times from 300 to 1800 s after primary event. Source thermocouples and all burst-window temperatures diagnostic only. Same fixed window on both cases.',
    source='Finite chemical reserve; normalized exponential burst and tail, each truncated at 6 tau. duration_s is FAST TIME CONSTANT, not rectangular duration.',
    structure='D=1 irreversibly switches l0; internal factor jumps to a peak then exponentially relaxes to residual, normalized/truncated at 6 tau. Physical time since local rupture; independent of chemical reaction clock. Same phase in boundary half-volumes and observers; ordered failed donor, no averaging.',
    fixed='All capacities, reference k, intact directional laws, geometry, initial fields, sensors, event time, measured boundaries; no secondary kinetics',
    limits='No mass solver. This trial changes medium state and heat release only; no additional nonlocal vent channels. Not a total-chemistry identification.',
    comparison='Old v1 parameters replayed unchanged and rescored on IDENTICAL late-witness observations; legacy 300 s curves independently reproduced; no validation data used in fit')

def fingerprint():return v1.hashlib.sha256((v1.fingerprint()+v1.digest(Path(__file__))).encode()).hexdigest()

def check(p):
    if set(p)!=set(NAMES):raise ValueError('Only v2 failure parameters allowed')
    for k,lo,hi in zip(NAMES,*BOUNDS):
        if not np.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError('v2 bounds: '+k)

def activate(scene):
    s=copy.deepcopy(scene);s['metadata'].update(failure_model='medium_transition_v3',failure_protocol=PROTOCOL,
        calibration_status='Exploratory irreversible-medium/burst-tail calibration',
        target_channels=list(WITNESSES),diagnostic_channels=['C2-ISS','C2-TIM'],
        window_rule='Simulate 1800 s after last pre-event scan; score witnesses only from 300 s after first event scan to the end')
    c=next(c for c in s['cells'] if c['id']=='C2');c['runaway'].update(profile='burst_tail',fast_fraction=.8,tail_time_s=30.,duration_s=.5)
    c['thermal']['bulk_destruction_on_boundary']=True
    c['thermal']['bulk_failure_transient']=dict(peak_factor=STRUCTURE_PRIOR['bulk_peak_factor'],relaxation_time_s=STRUCTURE_PRIOR['bulk_relaxation_time_s'])
    c['thermal']['bulk_connectivity'].update(endpoint_damage_weight=1.,damage_rate_at_onset_per_s=0.,destruction_damage=1.,
        destroyed_background_factor=1.,destroyed_active_factor=0.)
    return s

def bake(scene,p):
    check(p);s=copy.deepcopy(scene)
    c=next(c for c in s['cells'] if c['id']=='C2')
    c['runaway'].update(profile='burst_tail',energy_j=p['energy_j'],duration_s=p['duration_s'],fast_fraction=p['fast_fraction'],tail_time_s=p['tail_time_s'])
    c['thermal']['bulk_connectivity']['destroyed_background_factor']=p['destroyed_bulk_factor']
    c['thermal']['bulk_failure_transient']=dict(peak_factor=p['bulk_peak_factor'],relaxation_time_s=p['bulk_relaxation_time_s'])
    ports={q['id']:q for q in s['ports']}
    for e in s['interfaces']:
        if e['id'].startswith('failure-C2-'):
            key='destroyed_iss_h' if e['id'].endswith('ISS') else 'destroyed_tim_h'
            e['law']['destroyed_background_conductance_w_per_k']=p[key]*ports[e['port_a']]['area_m2']
    return s

def audit(run,s,p):
    v1.audit(run,s,p)
    for a in run['applied_objects']:
        c=next(c for c in s['cells'] if c['id']==a['id']);f=a['failure_state']
        if a['id']=='C2':
            expected=(p['fast_fraction'],p['tail_time_s'],p['destroyed_bulk_factor'])
            if f['profile']!='burst_tail' or not f['bulk_destruction_on_boundary'] or f['bulk_endpoint_damage_weight']!=1:raise ValueError('Missing structure switch')
            if (f['fast_fraction'],f['tail_time_s'],f['destroyed_bulk_factor'])!=expected:raise ValueError('Wrong native failure law')
            if (f['bulk_peak_factor'],f['bulk_relaxation_time_s'])!=(p['bulk_peak_factor'],p['bulk_relaxation_time_s']):raise ValueError('Wrong native transient law')
        elif f['profile']!='constant_power' or f['bulk_destruction_on_boundary']:raise ValueError('Unexpected change in witness material')
    t=np.array(run['time_s']);d=np.array(run['destroyed_internal_fraction']);onset=s['primary_failures'][0]['time_s']
    if not (np.all(d[t<onset-1e-9]==0) and np.all(d[t>=onset-1e-9]==1)):raise ValueError('Internal destruction did not switch irreversibly')
    released=np.array(run['released_energy_j_history'])
    if np.any(np.diff(released)<-1e-8) or released[-1]>p['energy_j']*(1+1e-12):raise ValueError('Chemical reserve violated')
    age=np.maximum(t-onset,0)
    def F(tau):return np.where(age>=6*tau,1.,-np.expm1(-age/tau)/(-np.expm1(-6.)))
    expected=p['energy_j']*(p['fast_fraction']*F(p['duration_s'])+(1-p['fast_fraction'])*F(p['tail_time_s']))
    if np.max(np.abs(expected-released))>1e-6:raise ValueError('Wrong CUDA energy integral')
    transient=np.where(t<onset-1e-9,1.,p['destroyed_bulk_factor']+(p['bulk_peak_factor']-p['destroyed_bulk_factor'])*(1-F(p['bulk_relaxation_time_s'])))
    for key in ('bulk_factor_min','bulk_factor_max'):
        if np.max(np.abs(np.asarray(run[key])-transient))>1e-10:raise ValueError('Wrong structural age/relaxation')

def pack(p):return np.log([p[k] for k in FIT_NAMES])
def unpack(x):return dict(PULSE_PRIOR,**STRUCTURE_PRIOR,**dict(zip(FIT_NAMES,map(float,np.exp(x)))))

def errors(run,observed,event):
    t=np.array(run['time_s'])-event['bracket_width_s']
    time_mask=(t>=PROTOCOL['fit_start_s'])&(t<=PROTOCOL['duration_s'])
    result={}
    for k in WITNESSES:
        y=np.asarray(observed[k]);pred=np.asarray(run['temperature_k'][k])
        if y.shape!=t.shape or pred.shape!=t.shape:raise ValueError('Misaligned witness observations')
        mask=time_mask & np.isfinite(y)
        if not mask.any() or not np.isfinite(pred[mask]).all():raise ValueError('Empty or nonfinite late-witness series: '+k)
        result[k]=pred[mask]-y[mask]
    return result

def metrics(run,observed,event):
    es=errors(run,observed,event)
    def score(e):return dict(rmse_k=float(np.sqrt(np.mean(e*e))),bias_k=float(e.mean()),max_abs_error_k=float(np.max(np.abs(e))),count=len(e))
    return dict(**score(np.concatenate(list(es.values()))),per_channel={k:score(e) for k,e in es.items()},scope=PROTOCOL['score'])

def replay(name,probe,path,scene,observed,event):
    old=json.loads((v1.OUTPUT/'profile.json').read_text())['parameters'];w=Worker(probe,v1.OUTPUT/(name+'-trial.json'))
    try:r=w.run(old)
    finally:w.close()
    prior=json.loads((v1.OUTPUT/('fit.json' if name=='no-cooling' else 'validate.json')).read_text())
    delta=max(float(np.max(np.abs(np.array(r['temperature_k'][k])-prior['result']['temperature_k'][k]))) for k in v1.CHANNELS)
    if delta>1e-8:raise ValueError('Legacy model reproduction failed: '+str(delta))
    w=Worker(probe,path)
    try:r=w.run(old);v1.audit(r,scene,old)
    finally:w.close()
    return dict(metrics=metrics(r,observed,event),max_change_k=delta,result=r)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('fit','validate'))
    p.add_argument('--probe',required=True,type=Path);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--max-evaluations',type=int,default=70)
    a=p.parse_args();name='no-cooling' if a.mode=='fit' else 'active-cooling';out=a.output;profile_path=out/'profile.json'
    if a.mode=='validate':
        profile=json.loads(profile_path.read_text())
        if profile['model_sha256']!=fingerprint() or profile['probe_sha256']!=v1.digest(a.probe):raise ValueError('Changed model/probe')
        params=profile['parameters'];check(params)
    base,path,observed,meta=v1.prepare(name,a.probe,out,duration=PROTOCOL['duration_s'])
    comparison=replay(name,a.probe,path,base,observed,meta['event']);s=activate(base);path=out/(name+'-transition-trial.json');save(path,s)
    w=Worker(a.probe,path)
    try:
        if a.mode=='fit':
            from scipy.optimize import least_squares
            starts=[];count=0
            def residual(x):
                nonlocal count
                q=unpack(x);r=w.run(q);audit(r,s,q);count+=1
                e=np.concatenate(list(errors(r,observed,meta['event']).values()))
                if count%20==0:print(f'transient candidate {count}: RMSE {np.sqrt(np.mean(e*e)):.3f} K',flush=True)
                return e
            for initial in ((70000.,30.,500.,.1),(120000.,200.,1000.,3.)):
                f=least_squares(residual,pack(dict(zip(FIT_NAMES,initial))),bounds=(pack(dict(zip(NAMES,BOUNDS[0]))),pack(dict(zip(NAMES,BOUNDS[1])))),
                    diff_step=.003,x_scale='jac',max_nfev=a.max_evaluations,ftol=1e-5,xtol=1e-5,gtol=1e-5)
                singular=np.linalg.svd(f.jac,compute_uv=False)
                starts.append(dict(parameters=unpack(f.x),cost=float(f.cost),nfev=f.nfev,success=bool(f.success),message=f.message,
                    jacobian_singular_values=singular.tolist()))
            params=min(starts,key=lambda r:r['cost'])['parameters']
            profile=dict(status='exploratory_not_promoted',parameters=params,protocol=PROTOCOL,model_sha256=fingerprint(),probe_sha256=v1.digest(a.probe),
                calibration_data=meta,starts=starts,evaluations=count,validation_refitted=False,bounds=dict(zip(NAMES,BOUNDS.T.tolist())),
                near_bounds=[k for k,lo,hi in zip(NAMES,*BOUNDS) if k in FIT_NAMES and
                    min(np.log(params[k]/lo),np.log(hi/params[k]))<.001*np.log(hi/lo)])
            save(profile_path,profile)
        r=w.run(params);audit(r,s,params);fine=w.run(dict(params,max_dt_s=.1));audit(fine,s,params)
        delta=max(float(np.max(np.abs(np.array(r['temperature_k'][k])-fine['temperature_k'][k]))) for k in v1.CHANNELS)
        exported=bake(s,params);export_path=out/(name+'-fitted-scenario.json');save(export_path,exported);ew=Worker(a.probe,export_path)
        try:baked=ew.run({})
        finally:ew.close()
        if baked['temperature_k']!=r['temperature_k']:raise ValueError('Export mismatch')
        report=dict(experiment=name,parameters=params,protocol=PROTOCOL,meta=meta,validation_refitted=False,all_capacities_fixed=True,
            frozen_profile_sha256=v1.digest(profile_path),v1_replay=comparison,metrics=metrics(r,observed,meta['event']),
            time_check_max_change_k=delta,export_identical=True,
            observed_k={k:[float(v) if np.isfinite(v) else None for v in y] for k,y in observed.items()},result=r)
        save(out/(a.mode+'.json'),report);print(json.dumps({k:report[k] for k in ('experiment','parameters','metrics','time_check_max_change_k')},indent=2),flush=True)
    finally:w.close()

if __name__=='__main__':main()

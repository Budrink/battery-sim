#!/usr/bin/env python3
"""First post-event model trial: prescribed primary event, solved C2 source.

Fit E, pulse duration and two destroyed source contacts on no-cooling ONLY.
All thermal integration is CUDA. Ambient/cooling temperatures remain measured
boundary conditions; this is NOT a gas, overcharge or secondary-ignition fit.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from calibrate_upv import ROOT, Worker, model_hash, save

DATA=ROOT/'data/experiments/upv'
PRIOR=ROOT/'data/calibration/upv-links-contrast-3x3x3'
OUTPUT=ROOT/'data/calibration/upv-failure-v1'
NAMES=('energy_j','duration_s','destroyed_iss_h','destroyed_tim_h')
BOUNDS=np.array([(1000.,1000000.),(.5,120.),(1.,10000.),(1.,10000.)]).T
CHANNELS={'101':('101',),'102':('102',),'107':('107',),'108':('108',),
          'C2-ISS':('105','106'),'C2-TIM':('103','104')}
PROTOCOL=dict(revision='post-event-constant-pulse-raw-scans-v1',fit='no-cooling',validation='active-cooling',
    duration_s=300.,output_interval_s=1.,max_dt_s=.2,
    primary_event='First enclosure-pressure step >=0.5 bar in <=2 s, corroborated by ambient or source thermal jump within the following 2 s; not a learned ignition threshold',
    state_start='Last sample before the pressure event; intact CUDA warmup for all existing volumes; C2 initialized by bilinear interpolation of its four surface readings',
    source='Uniform finite-energy constant-power pulse in C2, all 27 volumes ignited at measured event; E is heat deposited in modeled solid, NOT total chemical heat',
    fitted=list(NAMES),fixed='All Cp, k, intact directed connectivity, geometry, probes, initial fields, measured ambient/cooling boundaries; no secondary ignition',
    contact_switch='Existing irreversible D=1 branch only on two C2 contacts; destroyed h is constant, independently fitted for ISS and TIM',
    score='Unweighted Kelvin residuals at original instrument scan times of four witness channels and two C2 face means; gaps remain masked; no fitted channel offsets or time shifts',
    limitations=['Post-event gas jets, ejected mass and radiation not modeled',
        'Single ambient sensor is not established as a uniform gas boundary',
        'Low-temperature connectivity extrapolation and constant Cp after destruction are unvalidated',
        'C1/C3 acquisition numbering remains an unconfirmed hypothesis',
        'No secondary-ignition threshold can be inferred from isolated surface spikes',
        'Thermocouple attachment after venting is not established'],
    promotion='Exploratory model-adequacy trial only; never replaces accepted heat-transfer scenes')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fingerprint():
    return hashlib.sha256((model_hash()+digest(Path(__file__))+digest(ROOT/'native/tools/failure_probe.cpp')).encode()).hexdigest()

def load_record(name,directory=DATA):
    manifest=json.loads((directory/'manifest.json').read_text());meta=manifest['experiments'][name]
    path=directory/meta['canonical_csv']
    if digest(path)!=meta['canonical_sha256']:raise ValueError('Canonical data checksum mismatch')
    with path.open() as f:rows=list(csv.DictReader(f))
    d={k:np.array([float(r[k]) if r[k] else np.nan for r in rows])
       for k in rows[0] if k=='time_s' or k.startswith(('T','U'))}
    return d,meta

def event_marker(d):
    t=d['time_s'];pressure=2.5*d['U114_v']
    candidates=np.flatnonzero((np.diff(pressure)>=.5)&(np.diff(t)<=2)) + 1
    for i in candidates:
        # Multiplexed channels need not see pressure and heating in one scan.
        # Keep the FIRST pressure step, rather than delaying the trigger until
        # the temperature sensors catch up with venting.
        end=np.searchsorted(t,t[i]+2,side='right')
        thermal=any(np.isfinite(d[f'T{k}_k'][i-1]) and np.any(d[f'T{k}_k'][i:end]-d[f'T{k}_k'][i-1]>50)
                    for k in ('103','104','105','106','109'))
        if thermal and all(np.isfinite(d[f'T{k}_k'][i-1]) for k in ('103','104','105','106')):
            return dict(index=int(i),last_pre_event_s=float(t[i-1]),first_event_s=float(t[i]),
                        bracket_width_s=float(t[i]-t[i-1]),pressure_before_bar=float(pressure[i-1]),
                        pressure_first_bar=float(pressure[i]),source_pre_event_c={k:float(d[f'T{k}_k'][i-1]-273.15) for k in ('103','104','105','106')})
    raise ValueError('No corroborated pressure event with valid pre-event source temperatures')

def resample(t,y,at):
    """No extrapolation, clipping or interpolation across invalid/missing scans."""
    at=np.asarray(at);t=np.asarray(t);y=np.asarray(y)
    if np.any(at<t[0]) or np.any(at>t[-1]):raise ValueError('Observation interval not covered')
    i=np.clip(np.searchsorted(t,at,side='right')-1,0,len(t)-2)
    dt=t[i+1]-t[i];fraction=(at-t[i])/dt
    a=y[i];b=y[i+1];v=a+(b-a)*fraction
    v=np.where(fraction==0,a,np.where(fraction==1,b,v))
    return np.where((dt<=2*np.median(np.diff(t)))|((fraction==0)|(fraction==1)),v,np.nan)

def curve(d,channels,start,duration):
    # Preserve sharp post-event boundary signals at their original timestamps.
    times=np.unique(np.r_[0.,d['time_s'][(d['time_s']>start)&(d['time_s']<start+duration)]-start,duration])
    y=np.mean([d[f'T{k}_k'] for k in channels],axis=0)
    values=resample(d['time_s'],y,times+start)
    if not np.isfinite(values).all():raise ValueError('Invalid prescribed boundary; no gap filling')
    return np.column_stack([times,values]).tolist()

def observations(d,start,duration):
    t=d['time_s'];select=(t>=start)&(t<=start+duration);at=t[select]-start
    if len(at)<2 or at[0]!=0:raise ValueError('Initial state must coincide with a source scan')
    result={k:np.mean([d['T'+c+'_k'][select] for c in cs],axis=0) for k,cs in CHANNELS.items()}
    if at[-1]<duration:
        at=np.r_[at,duration]
        for k,cs in CHANNELS.items():
            y=np.mean([d['T'+c+'_k'] for c in cs],axis=0)
            result[k]=np.r_[result[k],resample(t,y,[start+duration])]
    return at,result

def refresh_boundaries(s,d,start,duration):
    for b in s['boundaries']:
        if b['id']=='coolant-reservoir':continue  # fixed external reservoir, never ch110
        if b['id'].startswith('source-'):keys=('105','106') if b['id'].endswith('ISS') else ('103','104')
        elif b['id'].startswith('cooling-'):keys=('110',)
        elif b['id'].startswith('ambient-'):keys=('109',)
        else:raise ValueError('Unknown measured boundary '+b['id'])
        b['temperature_profile']=curve(d,keys,start,duration);b['temperature_k']=b['temperature_profile'][0][1]

def check_parameters(p):
    if set(p)!=set(NAMES):raise ValueError('Only pulse and destroyed C2 contacts can be fitted')
    for k,lo,hi in zip(NAMES,*BOUNDS):
        if not np.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError('Failure parameter bounds: '+k)

def bake(s,p):
    check_parameters(p);s=copy.deepcopy(s);c=next(c for c in s['cells'] if c['id']=='C2')
    c['runaway']={'energy_j':p['energy_j'],'duration_s':p['duration_s']}
    ports={v['id']:v for v in s['ports']}
    for e in s['interfaces']:
        if e['id'].startswith('failure-C2-'):
            key='destroyed_iss_h' if e['id'].endswith('ISS') else 'destroyed_tim_h'
            e['law']['destroyed_background_conductance_w_per_k']=p[key]*ports[e['port_a']]['area_m2']
    return s

def post_scene(base,warm,d,event,duration):
    s=copy.deepcopy(base);start=event['last_pre_event_s']
    for c in s['cells']:
        if c['thermal'].get('heat_capacity_law'):raise ValueError('All capacities must stay fixed')
        c['thermal']['initial_temperature_field_k']=warm['final_fields_k'][c['id']]
        c['thermal']['initial_temperature_k']=float(np.mean(c['thermal']['initial_temperature_field_k']))
    c=copy.deepcopy(next(c for c in base['cells'] if c['id']=='C1'));c['id']='C2'
    c['degradation']={'initial_damage':0,'rate_at_onset_per_s':0}
    c['runaway']={'energy_j':150000.,'duration_s':10.}
    sources=[b for b in s['boundaries'] if b['id'].startswith('source-')]
    s['boundaries']=[b for b in s['boundaries'] if not b['id'].startswith('source-')]
    ports={p['id']:p for p in s['ports']}
    faces={b['id'].split('-')[-1]:ports[b['port_id']] for b in sources}
    x0=faces['ISS']['centroid_m'][0];x1=faces['TIM']['centroid_m'][0]
    if abs(x1-x0-.011)>1e-10:raise ValueError('Wrong source gap')
    c['pose']['position_m']=[(x0+x1)/2,0,0]
    if c['thermal']['divisions']!=[3,3,3]:raise ValueError('Frozen 3^3 grid required')
    i=event['index']-1;temps={k:d[f'T{k}_k'][i] for k in ('103','104','105','106')}
    field=[]
    for z in range(3):
        fz=(z+.5)/3;iss=temps['105']*(1-fz)+temps['106']*fz;tim=temps['103']*(1-fz)+temps['104']*fz
        for y in range(3):
            for x in range(3):fx=(x+.5)/3;field.append(float(iss*(1-fx)+tim*fx))
    c['thermal']['initial_temperature_field_k']=field;c['thermal']['initial_temperature_k']=float(np.mean(field));s['cells'].append(c)
    def port(face):
        axis='xyz'.index(face[1]);sign=1 if face[0]=='+' else -1;dims=c['geometry']['dimensions_m']
        pos=c['pose']['position_m'].copy();pos[axis]+=sign*dims[axis]/2;normal=[0,0,0];normal[axis]=sign
        p=dict(id='C2:'+face,cell_id='C2',kind='contact',surface=face,area_m2=dims[(axis+1)%3]*dims[(axis+2)%3],centroid_m=pos,normal=normal)
        s['ports'].append(p);return p
    for b in sources:
        side=b['id'].split('-')[-1];p=port('-x' if side=='ISS' else '+x');law=copy.deepcopy(b['connectivity'])
        law.update(endpoint_damage_weight=1.,destruction_damage=1.,destroyed_active_conductance_w_per_k=0.,destroyed_background_conductance_w_per_k=100*p['area_m2'])
        e=dict(id='failure-C2-'+side,port_a=b['port_id'] if side=='ISS' else p['id'],port_b=p['id'] if side=='ISS' else b['port_id'],law=law)
        s['interfaces'].append(e);s['measurement_probes'].append(dict(id='C2-'+side,cell_id='C2',interface_id=e['id']))
    # New C2 exposed edges use exactly the already frozen environmental h.
    sample=next(b for b in s['boundaries'] if b['id']=='ambient-C1-y')
    h=sample['conductance_w_per_k']/ports[sample['port_id']]['area_m2']
    for face in ('-y','+y','-z','+z'):
        p=port(face);s['boundaries'].append(dict(id='ambient-C2'+face,port_id=p['id'],temperature_k=298.15,conductance_w_per_k=h*p['area_m2']))
    refresh_boundaries(s,d,start,duration)
    s['primary_failures']=[dict(cell_id='C2',time_s=event['bracket_width_s'])]
    s['simulation'].update(duration_s=duration,max_dt_s=PROTOCOL['max_dt_s'],output_interval_s=1.)
    s['failure_sample_times_s']=observations(d,start,duration)[0].tolist()
    s['metadata'].update(failure_calibration=True,calibration_status='Exploratory post-event source/contact trial, NOT accepted destruction calibration',
        validation_scope='Conditional post-event solid response to prescribed primary failure and measured environment',
        failure_protocol=PROTOCOL,time_origin_s=start,primary_event=event,
        drivers=['Measured primary event time','Measured environment and cooling contact'],source_cell_not_solved=False,
        target_channels=list(CHANNELS),window_rule='Fixed 300 seconds after last pre-event sample')
    # These keys describe an obsolete intact-fitting interface, not this probe.
    for k in ('staged_calibration','internal_contrast_calibration'):s['metadata'].pop(k,None)
    s['name']='UPV '+base['metadata']['experiment_id']+': exploratory post-event CUDA source'
    return s

def prepare(name,probe,out,directory=DATA,duration=300.):
    d,record=load_record(name,directory);event=event_marker(d)
    base_path=PRIOR/(name+'-fitted-scenario.json');base=json.loads(base_path.read_text());warm_scene=copy.deepcopy(base)
    warm_scene['simulation']['duration_s']=event['last_pre_event_s'];refresh_boundaries(warm_scene,d,0.,event['last_pre_event_s'])
    path=out/(name+'-warmup.json');save(path,warm_scene);w=Worker(probe,path)
    try:warm=w.run({})
    finally:w.close()
    audit(warm,warm_scene,{})
    s=post_scene(base,warm,d,event,duration);path=out/(name+'-trial.json');save(path,s)
    at,observed=observations(d,event['last_pre_event_s'],duration)
    return s,path,observed,dict(event=event,source_csv_sha256=record['canonical_sha256'],intact_scene_sha256=digest(base_path),warmup_energy_residual_j=warm['energy_residual_j'])

def audit(run,s,p):
    objs={c['id']:c for c in run['applied_objects']}
    for c in s['cells']:
        a=objs[c['id']];t=c['thermal']
        if a['capacity_j_k']!=t['heat_capacity_j_per_k'] or a['conductivity_w_m_k']!=t['conductivity_w_per_m_k'] or a['divisions']!=t['divisions']:raise ValueError('Frozen material/grid changed')
        for k,jsonkey in [('energy_j','energy_j'),('duration_s','duration_s')]:
            expected=p.get(k,c['runaway'][jsonkey]) if c['id']=='C2' else c['runaway'][jsonkey]
            if a[k]!=expected:raise ValueError('Wrong source parameter')
    ports={q['id']:q for q in s['ports']}
    for e in s['interfaces']:
        if not e['id'].startswith('failure-C2-'):continue
        a=next(a for a in run['applied_source_contacts'] if a['id']==e['id']);key='destroyed_iss_h' if e['id'].endswith('ISS') else 'destroyed_tim_h'
        expected=p[key]*ports[e['port_a']]['area_m2'] if key in p else e['law']['destroyed_background_conductance_w_per_k']
        if abs(a['destroyed_g_w_k']-expected)>1e-10 or a['intact_reference_logit']!=e['law']['geometry_reference_logit']:raise ValueError('Wrong source contact')
    if abs(run['energy_residual_j'])>1e-5:raise ValueError('Energy conservation failed')

def errors(run,observed):
    return {k:np.array(run['temperature_k'][k])[np.isfinite(y)]-y[np.isfinite(y)] for k,y in observed.items()}

def metrics(run,observed):
    e=errors(run,observed)
    def score(v):return dict(rmse_k=float(np.sqrt(np.mean(v*v))),bias_k=float(v.mean()),max_abs_error_k=float(np.max(np.abs(v))),count=len(v))
    return dict(**score(np.concatenate(list(e.values()))),per_channel={k:score(v) for k,v in e.items()},
        witness_rmse_k=score(np.concatenate([e[k] for k in ('101','102','107','108')]))['rmse_k'],
        source_rmse_k=score(np.concatenate([e[k] for k in ('C2-ISS','C2-TIM')]))['rmse_k'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('fit','validate'))
    p.add_argument('--probe',required=True,type=Path);p.add_argument('--output',default=OUTPUT,type=Path)
    p.add_argument('--max-evaluations',default=70,type=int);a=p.parse_args();out=a.output;name='no-cooling' if a.mode=='fit' else 'active-cooling'
    profile_path=out/'profile.json'
    if a.mode=='validate':
        profile=json.loads(profile_path.read_text())
        if profile['model_sha256']!=fingerprint() or profile['probe_sha256']!=digest(a.probe):raise ValueError('Model/probe differs from frozen fit')
        params=profile['parameters'];check_parameters(params)
    s,path,observed,meta=prepare(name,a.probe,out);w=Worker(a.probe,path)
    try:
        baseline=w.run({});audit(baseline,s,{})
        if a.mode=='fit':
            from scipy.optimize import least_squares
            def unpack(x):return dict(zip(NAMES,map(float,np.exp(x))))
            evaluations=0
            def residual(x):
                nonlocal evaluations
                q=unpack(x);run=w.run(q);audit(run,s,q);evaluations+=1
                e=np.concatenate(list(errors(run,observed).values()))
                if evaluations%15==0:print(f'candidate {evaluations}: RMSE={np.sqrt(np.mean(e*e)):.3f} K',flush=True)
                return e
            trials=[]
            for initial in ((150000.,10.,100.,100.),(300000.,20.,1000.,1000.)):
                f=least_squares(residual,np.log(initial),bounds=np.log(BOUNDS),diff_step=.002,max_nfev=a.max_evaluations,
                                ftol=1e-5,xtol=1e-5,gtol=1e-5,x_scale='jac')
                trials.append(dict(parameters=unpack(f.x),cost=float(f.cost),success=bool(f.success),message=f.message,nfev=f.nfev))
            params=min(trials,key=lambda r:r['cost'])['parameters']
            profile=dict(status='exploratory_not_promoted',parameters=params,protocol=PROTOCOL,model_sha256=fingerprint(),probe_sha256=digest(a.probe),
                calibration_data=meta,starts=trials,evaluations=evaluations,validation_refitted=False,
                bounds=dict(zip(NAMES,BOUNDS.T.tolist())),near_bounds=[k for k,lo,hi in zip(NAMES,*BOUNDS) if params[k]<lo*1.01 or params[k]>hi/1.01])
            save(profile_path,profile)
        fitted=w.run(params);audit(fitted,s,params)
        refined=w.run(dict(params,max_dt_s=.1));audit(refined,s,params)
        delta=max(float(np.max(np.abs(np.array(fitted['temperature_k'][k])-refined['temperature_k'][k]))) for k in CHANNELS)
        scene_path=out/(name+'-fitted-scenario.json');save(scene_path,bake(s,params));export=Worker(a.probe,scene_path)
        try:exported=export.run({})
        finally:export.close()
        if exported['temperature_k']!=fitted['temperature_k']:raise ValueError('Baked scenario differs from fitted run')
        report=dict(experiment=name,status='exploratory_model_adequacy_trial',protocol=PROTOCOL,parameters=params,
                    validation_refitted=False,all_capacities_fixed=True,meta=meta,frozen_profile_sha256=digest(profile_path),
                    baseline_metrics=metrics(baseline,observed),metrics=metrics(fitted,observed),time_check_max_change_k=delta,
                    export_identical=True,observed_k={k:[float(v) if np.isfinite(v) else None for v in y] for k,y in observed.items()},result=fitted)
        save(out/(a.mode+'.json'),report);print(json.dumps({k:report[k] for k in ('experiment','parameters','metrics','time_check_max_change_k')},indent=2))
    finally:w.close()

if __name__=='__main__':main()

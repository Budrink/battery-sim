#!/usr/bin/env python3
"""Conditional post-event kinetic/anisotropic fit; no-cooling only, frozen validation.

All evolution is native CUDA. Python supplies optimization and artifact/audit code.
No fitting of measured Cp/k, intact transport, initial state, time shifts or vents.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import calibrate_upv_failure as base
import calibrate_upv_failure_transition as prior
from calibrate_upv import Worker, save

OUTPUT=base.ROOT/'data/calibration/upv-kinetic-v4'
NAMES=('energy_j','rate_400_per_s','destroyed_iss_h','destroyed_tim_h','residue_through','residue_inplane')
BOUNDS=np.array([(1000.,1000000.),(.0001,10.),(.01,10000.),(.01,10000.),(.005,10.),(.005,10.)]).T
KINETICS=dict(reference_temperature_k=400.,rate_at_reference_per_s=.05,activation_energy_j_per_mol=60000.,
    activation_damage=.2,damage_sensitivity=2.,damage_per_conversion=1.,quench_rate_per_s=1e-5,
    max_fraction_per_step=.005,max_temperature_rise_per_step_k=.5)
STRUCTURE=dict(active_factor=[3.,8.,8.],residue_factor=[.02,.1,.1],activity_rate_per_s=.1)
STARTS=((140000.,.05,5.,5.,.02,.1),(70000.,.5,50.,50.,.01,.01),(200000.,.005,100.,10.,.1,.3))
PROTOCOL=dict(prior.PROTOCOL,revision='kinetic-process-late-witness-v4',fitted=list(NAMES),
    source='Local Arrhenius reaction with finite accessible solid-heat reserve and temperature/damage feedback; not a timed pulse',
    structure='Source-owned anisotropic process multiplier; separate through-plane x and tied in-plane y/z residue; same factor on boundary half-volumes',
    fixed='All Cp, reference k, intact contact and internal laws, geometry, measured boundary histories, initial fields, primary timing and sensor map; no witness chemistry or secondary-ignition fit',
    kinetic_prior=KINETICS,structure_prior=STRUCTURE,
    pulse_prior=None,contact_switch='Existing C2 interface branch at local D=1, reached progressively by the reaction; destroyed effective areal h fitted separately on ISS/TIM',
    score=prior.PROTOCOL['score'],
    limits='No vent geometry or outlet enabled: fitted E is an effective accessible solid-heat budget, NOT total cell chemistry or calorimetric/ejecta heat. Late observations may not identify early kinetics.',
    comparison='Same unchanged late-witness window as v3; no active-cooling objective used for fitting or selection',
    validation_note='Held out from this fit, but previously inspected in model development; not a pristine blind dataset',
    numerical_controls=dict(max_dt_s=.2,max_fraction_per_step=.005,max_temperature_rise_per_step_k=.5))

def fingerprint():
    return base.hashlib.sha256((base.fingerprint()+base.digest(Path(__file__))+base.digest(Path(prior.__file__))).encode()).hexdigest()

def check(p):
    if set(p)!=set(NAMES):raise ValueError('Only the six kinetic failure parameters may be fitted')
    for k,lo,hi in zip(NAMES,*BOUNDS):
        if not np.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError('Kinetic parameter bounds: '+k)

def activate(scene):
    s=copy.deepcopy(scene)
    s['metadata'].update(failure_model='kinetic_process_v4',failure_protocol=PROTOCOL,
        calibration_status='Exploratory kinetic post-event calibration; not promoted',
        target_channels=list(prior.WITNESSES),diagnostic_channels=['C2-ISS','C2-TIM'],
        window_rule=PROTOCOL['score'])
    c=next(c for c in s['cells'] if c['id']=='C2')
    c['runaway']=dict(profile='kinetic',energy_j=STARTS[0][0],kinetics=copy.deepcopy(KINETICS))
    t=c['thermal'];t.pop('bulk_destruction_on_boundary',None);t.pop('bulk_failure_transient',None)
    t['bulk_connectivity']['endpoint_damage_weight']=0.
    t['bulk_connectivity']['damage_rate_at_onset_per_s']=0.
    t['reaction_structure']=copy.deepcopy(STRUCTURE)
    s['simulation']['max_dt_s']=PROTOCOL['numerical_controls']['max_dt_s']
    return s

def bake(scene,p):
    check(p);s=copy.deepcopy(scene);c=next(c for c in s['cells'] if c['id']=='C2')
    c['runaway']['energy_j']=p['energy_j'];c['runaway']['kinetics']['rate_at_reference_per_s']=p['rate_400_per_s']
    c['thermal']['reaction_structure']['residue_factor']=[p['residue_through'],p['residue_inplane'],p['residue_inplane']]
    ports={x['id']:x for x in s['ports']}
    for e in s['interfaces']:
        if e['id'] in ('failure-C2-ISS','failure-C2-TIM'):
            key='destroyed_iss_h' if e['id'].endswith('ISS') else 'destroyed_tim_h'
            e['law']['destroyed_background_conductance_w_per_k']=p[key]*ports[e['port_a']]['area_m2']
    return s

def audit(run,scene,p):
    check(p);actual={c['id']:c for c in run['applied_objects']}
    for c in scene['cells']:
        a=actual[c['id']];t=c['thermal'];source=c['id']=='C2'
        if a['capacity_j_k']!=t['heat_capacity_j_per_k'] or a['conductivity_w_m_k']!=t['conductivity_w_per_m_k'] or a['divisions']!=t['divisions']:
            raise ValueError('Frozen material capacity, reference conductivity or grid changed')
        if a['energy_j']!=(p['energy_j'] if source else c['runaway']['energy_j']):raise ValueError('Unexpected chemical source')
        if source:
            k=a['kinetic_state'];expected=c['runaway']['kinetics'];f=t['reaction_structure']
            for key in ('reference_temperature_k','activation_energy_j_per_mol','activation_damage','damage_sensitivity','damage_per_conversion'):
                if k[key]!=expected[key]:raise ValueError('Kinetic prior changed: '+key)
            if k['rate_400_per_s']!=p['rate_400_per_s'] or k['residue_factor']!=[p['residue_through'],p['residue_inplane'],p['residue_inplane']]:
                raise ValueError('Wrong fitted kinetics/anisotropy')
            if k['active_factor']!=f['active_factor'] or k['activity_rate_per_s']!=f['activity_rate_per_s'] or k['outlet_enabled']:
                raise ValueError('Unidentified transient/outlet was changed')
    ports={q['id']:q for q in scene['ports']}
    for e in scene['interfaces']:
        if not e['id'].startswith('failure-C2-'):continue
        a=next(a for a in run['applied_source_contacts'] if a['id']==e['id'])
        key='destroyed_iss_h' if e['id'].endswith('ISS') else 'destroyed_tim_h'
        if abs(a['destroyed_g_w_k']-p[key]*ports[e['port_a']]['area_m2'])>1e-10 or a['intact_reference_logit']!=e['law']['geometry_reference_logit']:
            raise ValueError('Contact binding or intact contact altered')
    if run['time_s']!=scene['failure_sample_times_s']:raise ValueError('Observation timestamps changed')
    released=np.asarray(run['released_energy_j_history']);remaining=np.asarray(run['remaining_energy_j'])
    if np.max(np.abs(released+remaining-p['energy_j']))>1e-5 or np.any(np.diff(released)<-1e-8) or np.min(remaining)<0:
        raise ValueError('Chemical reserve conservation failed')
    if run['max_thermal_residual_j']>1e-5 or run['max_chemical_residual_j']>1e-5 or run['vent_heat_out_j']!=0 or run['ejected_chemical_energy_j']!=0:
        raise ValueError('Energy conservation or nonexistent outlet')
    for k,y in run['temperature_k'].items():
        if not np.isfinite(y).all() or np.min(y)<=0:raise ValueError('Invalid predicted temperature: '+k)

def vector(run,observed,event):return np.concatenate(list(prior.errors(run,observed,event).values()))
def metrics(run,observed,event):return prior.metrics(run,observed,event)
def unpack(x):return dict(zip(NAMES,map(float,np.exp(x))))
def pack(p):return np.log([p[k] for k in NAMES])

def process_summary(r,event):
    t=np.asarray(r['time_s'])-event['bracket_width_s'];heat=np.asarray(r['released_energy_j_history'])
    total=heat[-1];summary=dict(released_kj=float(total/1000),remaining_kj=r['remaining_energy_j'][-1]/1000,
        sampled_peak_power_kw=max(r['chemical_power_w'])/1000,sampled_peak_local_temperature_k=max(r['source_max_temperature_k']),
        final_source_damage=r['source_damage'][-1])
    for f in (.1,.5,.9):
        target=f*(heat[0]+r['remaining_energy_j'][0]);where=np.flatnonzero(heat>=target)
        summary[f't{int(f*100)}_s']=float(t[where[0]]) if len(where) else None
    return summary

def prepare(name,probe,out):
    scene,_,observed,meta=base.prepare(name,probe,out,duration=PROTOCOL['duration_s'])
    scene=activate(scene);path=out/(name+'-kinetic-trial.json');save(path,scene)
    return scene,path,observed,meta

def fit(w,scene,observed,event,out,max_nfev,starts):
    from scipy.optimize import least_squares
    trials=[];count=0;best=float('inf');started=time.monotonic();cache={}
    def residual(x):
        nonlocal count,best
        key=tuple(x)
        if key in cache:return cache[key]
        p=unpack(x);r=w.run(p);audit(r,scene,p);e=vector(r,observed,event);count+=1
        rmse=float(np.sqrt(np.mean(e*e)));cache[key]=e
        if rmse<best:
            best=rmse;save(out/'checkpoint.json',dict(parameters=p,training_rmse_k=rmse,evaluations=count,elapsed_s=time.monotonic()-started))
        if count%10==0:print(f'kinetic candidate {count}: train RMSE {rmse:.5f} K, best {best:.5f} K, {time.monotonic()-started:.0f}s',flush=True)
        return e
    for i,start in enumerate(STARTS[:starts]):
        print('starting kinetic search',i+1,flush=True)
        f=least_squares(residual,np.log(start),bounds=np.log(BOUNDS),diff_step=.01,x_scale='jac',max_nfev=max_nfev,
                        ftol=1e-5,xtol=1e-5,gtol=1e-5)
        singular=np.linalg.svd(f.jac,compute_uv=False)
        trials.append(dict(parameters=unpack(f.x),rmse_k=float(np.sqrt(np.mean(f.fun*f.fun))),nfev=f.nfev,success=bool(f.success),
            message=f.message,jacobian_singular_values=singular.tolist()))
        save(out/'searches.json',trials)
    # Keep the best evaluated candidate, not an inferior final trust-region iterate.
    checkpoint=json.loads((out/'checkpoint.json').read_text())
    return checkpoint['parameters'],trials,count

def sensitivity(w,scene,p,observed,event):
    result={}
    for key in ('rate_400_per_s','residue_inplane'):
        result[key]=[];index=NAMES.index(key)
        for scale in (.1,1.,10.):
            candidate=dict(p);candidate[key]=float(np.clip(p[key]*scale,BOUNDS[0,index],BOUNDS[1,index]))
            r=w.run(candidate);audit(r,scene,candidate)
            result[key].append(dict(value=candidate[key],metrics=metrics(r,observed,event),process=process_summary(r,event)))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('fit','validate','baseline'))
    parser.add_argument('--probe',required=True,type=Path);parser.add_argument('--output',default=OUTPUT,type=Path)
    parser.add_argument('--max-evaluations',type=int,default=40);parser.add_argument('--starts',type=int,default=3)
    a=parser.parse_args();out=a.output;profile_path=out/'profile.json';name='active-cooling' if a.mode=='validate' else 'no-cooling'
    if a.mode=='validate':
        profile=json.loads(profile_path.read_text())
        if profile['model_sha256']!=fingerprint() or profile['probe_sha256']!=base.digest(a.probe):raise ValueError('Changed model/probe since training')
        p=profile['parameters'];check(p)
    s,path,observed,meta=prepare(name,a.probe,out);w=Worker(a.probe,path)
    try:
        if a.mode=='baseline':
            p=dict(zip(NAMES,STARTS[0]));r=w.run(p);audit(r,s,p)
            print(json.dumps(dict(parameters=p,metrics=metrics(r,observed,meta['event']),process=process_summary(r,meta['event']),wall_s=r['wall_s']),indent=2));return
        if a.mode=='fit':
            p,trials,count=fit(w,s,observed,meta['event'],out,a.max_evaluations,a.starts)
            probes=sensitivity(w,s,p,observed,meta['event'])
            profile=dict(status='conditional_fit_not_full_failure_validation',parameters=p,protocol=PROTOCOL,
                model_sha256=fingerprint(),probe_sha256=base.digest(a.probe),calibration_data=meta,starts=trials,evaluations=count,
                sensitivity=probes,validation_refitted=False,validation_used_for_selection=False,
                bounds=dict(zip(NAMES,BOUNDS.T.tolist())),near_bounds=[k for k,lo,hi in zip(NAMES,*BOUNDS)
                    if min(np.log(p[k]/lo),np.log(hi/p[k]))<.001*np.log(hi/lo)])
            save(profile_path,profile)
        r=w.run(p);audit(r,s,p);controls=PROTOCOL['numerical_controls']
        fine=w.run(dict(p,**{k:v/2 for k,v in controls.items()}));audit(fine,s,p)
        check_time={}
        mask=np.asarray(r['time_s'])>=meta['event']['bracket_width_s']+PROTOCOL['fit_start_s']
        for label,keys,selection in [('scored_witnesses',prior.WITNESSES,mask),('all_witnesses',prior.WITNESSES,slice(None)),('all_channels',base.CHANNELS,slice(None))]:
            check_time[label]=max(float(np.max(np.abs(np.asarray(r['temperature_k'][k])[selection]-np.asarray(fine['temperature_k'][k])[selection]))) for k in keys)
        export_path=out/(name+'-fitted-scenario.json');save(export_path,bake(s,p));ew=Worker(a.probe,export_path)
        try:exported=ew.run({})
        finally:ew.close()
        audit(exported,bake(s,p),p)
        if r['temperature_k']!=exported['temperature_k'] or r['released_energy_j_history']!=exported['released_energy_j_history']:raise ValueError('Export differs from calibrated model')
        report=dict(experiment=name,parameters=p,protocol=PROTOCOL,meta=meta,validation_refitted=False,
            frozen_profile_sha256=base.digest(profile_path),metrics=metrics(r,observed,meta['event']),
            process=process_summary(r,meta['event']),time_check_max_change_k=check_time,
            refined_metrics=metrics(fine,observed,meta['event']),export_identical=True,
            observed_k={k:[float(v) if np.isfinite(v) else None for v in y] for k,y in observed.items()},result=r)
        save(out/(a.mode+'.json'),report)
        print(json.dumps({k:report[k] for k in ('experiment','parameters','metrics','process','time_check_max_change_k')},indent=2),flush=True)
    finally:w.close()

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Sequential conductivity then node-capacity fit. All integration is CUDA.

Fit reads no-cooling ONLY. Validate freezes BOTH stages before loading the
previously inspected active-cooling experiment. No default scene is replaced.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from calibrate_upv import Worker, metrics, model_hash, residuals, save, verify_fixed_properties
from calibrate_upv_internal import internal_scene
from check_upv_core_shell import compare
from export_upv_state_geometry import contact_group
from generate_upv import ROOT, load_case

G_NAMES=('internal_log_35','internal_log_50','internal_log_70','battery_iss_scale','battery_tim_scale')
C_NAMES=('capacity_45','capacity_70')
TEMPERATURES=[298.15,308.15,323.15,343.15]
PROTOCOL=dict(revision='staged-table-energy-v1',grid=[3,3,3],calibration='no-cooling',validation='active-cooling',
    order=['conductivity at ALL constant capacities','capacity at frozen ALL transport laws'],
    internal_law='0.2+2.8*sigmoid(log(0.8/2)+piecewise_linear_log_response(Tdonor))',
    internal_knots_k=TEMPERATURES,internal_anchor=0.,internal_log_bounds=[-6.,6.],
    contact_scope='Only battery/ISS and battery/TIM reference h; within factor two of previous contact fit and 1<h<10000',
    fixed_transport='Other contacts, existing external temperature/contrast responses, bulk reference tensor, surface half-cell resistances, ambient/cooling boundaries',
    capacity_knots_k=[298.15,318.15,343.15],capacity_factor_bounds=[.8,1.2],capacity_anchor=1.,
    capacity_scope='Shared battery-node C(T)/C_ref; all passive capacities fixed, defect factor=1',
    bounds_status='Exploratory regularization bounds, NOT measured material uncertainty',
    extrapolation='Constant endpoint extension; not validated outside observed temperatures',
    regularization='0.03 K RMS per internal log-response slope difference; zero for capacity',
    caveat='Grid-specific pre-event graph fit; unconfirmed witness numbering; validation previously inspected; no defect/runaway calibration')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def table_scene(base):
    scene=internal_scene(base)
    scene['metadata'].pop('internal_calibration');scene['metadata'].pop('internal_protocol')
    scene['metadata'].update(staged_calibration='conductivity',staged_protocol=PROTOCOL,
        calibration_status='Exploratory sequential transport/capacity fit; not runaway validation')
    for c in scene['cells']:
        if c['role']=='battery':
            c['thermal']['bulk_response']['temperature_log_response']=[[t,0.] for t in TEMPERATURES]
    return scene

def bake_transport(scene,params):
    if set(params)!=set(G_NAMES) or any(not math.isfinite(x) for x in params.values()):raise ValueError('Incomplete transport parameters')
    if any(abs(params[k])>6 for k in G_NAMES[:3]):raise ValueError('Internal table out of bounds')
    out=copy.deepcopy(scene);objects={c['id']:c for c in out['cells']};ports={p['id']:p for p in out['ports']}
    for c in out['cells']:
        if c['role']=='battery':
            c['thermal']['bulk_response']['temperature_log_response']=[[t,v] for t,v in zip(TEMPERATURES,[0.]+[params[k] for k in G_NAMES[:3]])]
    def reference(law,group):
        if group not in params:return
        h=100*params[group]
        if not 1<h<10000:raise ValueError('Contact h outside model domain')
        law['geometry_reference_logit']=math.log((h-1)/(10000-h))
    for e in out['interfaces']:
        group=contact_group(objects[ports[e['port_a']]['cell_id']]['material_id'],objects[ports[e['port_b']]['cell_id']]['material_id'])
        reference(e['law'],group)
    for b in out['boundaries']:
        if b['id'].startswith('source-'):
            group=contact_group('battery',objects[ports[b['port_id']]['cell_id']]['material_id'])
            reference(b['connectivity'],group)
            if group in params:b['conductance_w_per_k']*=params[group] # mirror probe; dynamic boundary uses its law
    out['metadata']['transport_parameters']=params
    return out

def capacity_scene(transport):
    out=copy.deepcopy(transport);out['metadata']['staged_calibration']='capacity'
    for c in out['cells']:
        if c['role']=='battery':
            c['thermal']['heat_capacity_law']=dict(reference_temperature_k=298.15,damage_factor_at_one=1.,
                factor=[[298.15,1.],[318.15,1.],[343.15,1.]])
    return out

def bake_capacity(scene,params):
    if set(params)!=set(C_NAMES) or any(not math.isfinite(v) or not .8<=v<=1.2 for v in params.values()):
        raise ValueError('Capacity factors outside exploratory bounds')
    out=copy.deepcopy(scene)
    for c in out['cells']:
        if c['role']=='battery':c['thermal']['heat_capacity_law']['factor']=[[298.15,1.],[318.15,params['capacity_45']],[343.15,params['capacity_70']]]
    out['metadata']['capacity_parameters']=params
    return out

def audit_stage(run,scene,params,baseline_audit):
    verify_fixed_properties(run,scene)
    if abs(run['energy_residual_j'])>1e-6:raise ValueError('Energy balance failed')
    audit=copy.deepcopy(run['applied_model']);stage=scene['metadata']['staged_calibration']
    for actual,prior,c in zip(audit['objects'],baseline_audit['objects'],scene['cells']):
        if c['role']=='battery':
            key='internal_table' if stage=='conductivity' else 'capacity_table'
            expected=(bake_transport if stage=='conductivity' else bake_capacity)(scene,params)
            target=next(x for x in expected['cells'] if x['id']==c['id'])['thermal']
            values=target['bulk_response']['temperature_log_response'] if stage=='conductivity' else target['heat_capacity_law']['factor']
            if actual[key]!=values:raise ValueError('Native constitutive table not applied')
            actual[key]=prior[key]
    if stage=='conductivity':
        for actual,prior in zip(audit['contacts'],baseline_audit['contacts']):
            if actual['group'] in params:
                if abs(actual['reference_h_w_m2_k']-100.*params[actual['group']])>1e-8:raise ValueError('Wrong contact reference')
                for key in ('scale','reference_h_w_m2_k','conductance_w_per_k'):actual[key]=prior[key]
        for actual,prior in zip(audit['boundaries'],baseline_audit['boundaries']):
            if actual['id'].startswith('source-') and actual['group'] in params:
                actual['conductance_w_per_k']=prior['conductance_w_per_k']
    if audit!=baseline_audit:raise ValueError('Frozen model block changed')

def search(worker,scene,observed,names,starts,lower,upper,max_nfev,stage):
    from scipy.optimize import least_squares
    baseline=worker.run({});audit=baseline['applied_model'];history=[];cache={}
    def objective(x):
        key=tuple(x)
        if key not in cache:
            params=dict(zip(names,map(float,x)));run=worker.run(params);audit_stage(run,scene,params,audit)
            r=np.concatenate(list(residuals(run,observed).values()))
            history.append(dict(parameters=params,rmse_k=float(np.sqrt(np.mean(r*r)))))
            if stage=='conductivity':
                slopes=np.diff(np.r_[0.,x[:3]])/np.diff(np.asarray(TEMPERATURES)/10.)
                r=np.r_[r,.03*math.sqrt(len(r))*np.diff(slopes)]
            cache[key]=r
            if len(history)%20==1:print(stage,len(history),history[-1],flush=True)
        return cache[key]
    def jacobian(x):
        columns=[]
        for i in range(len(x)):
            h=.002 if i<3 and stage=='conductivity' else max(.0002,abs(x[i])*.0002)
            lo=x.copy();hi=x.copy();lo[i]=max(lower[i],x[i]-h);hi[i]=min(upper[i],x[i]+h)
            columns.append((objective(hi)-objective(lo))/(hi[i]-lo[i]))
        return np.stack(columns,axis=1)
    fits=[least_squares(objective,np.clip(start,lower,upper),jac=jacobian,bounds=(lower,upper),max_nfev=max_nfev,
                        ftol=1e-7,xtol=1e-6,gtol=1e-5) for start in starts]
    best=min([f for f in fits if f.success] or fits,key=lambda f:np.dot(f.fun,f.fun))
    params=dict(zip(names,map(float,best.x)));run=worker.run(params);audit_stage(run,scene,params,audit)
    sv=np.linalg.svd(best.jac,compute_uv=False)
    return dict(parameters=params,optimizer_success=bool(best.success),message=best.message,
        near_bounds=[n for i,n in enumerate(names) if min(best.x[i]-lower[i],upper[i]-best.x[i])<.001*(upper[i]-lower[i])],
        jacobian_condition_number=float(sv[0]/sv[-1]) if sv[-1]>0 else None,
        metrics=metrics(run,observed),bounds=dict(zip(names,zip(map(float,lower),map(float,upper)))),
        starts=[dict(success=bool(f.success),parameters=dict(zip(names,map(float,f.x))),objective=float(np.dot(f.fun,f.fun))) for f in fits],
        evaluation_history=history),run,baseline

def run_saved(binary,path,scene):
    save(path,scene);w=Worker(binary,path)
    try:return w.run({})
    finally:w.close()

def numerical_check(binary,path,run):
    w=Worker(binary,path)
    try:
        cap=min(json.loads(path.read_text())['simulation']['max_dt_s']/2,run['time_s'][-1]/run['steps']/2)
        fine=w.run({'max_dt_s':cap})
        if fine['steps']<=run['steps'] or abs(fine['energy_residual_j'])>1e-6:raise ValueError('Numerical check failed')
        check=dict(**compare(run,fine),max_dt_s=cap,baseline_steps=run['steps'],refined_steps=fine['steps'])
        if check['max_change_k']>.05:raise ValueError('Time refinement changes temperatures >0.05K')
        return check
    finally:w.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('fit','validate'))
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--max-evaluations',type=int,default=80)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/calibration/upv-staged-3x3x3')
    a=p.parse_args();out=a.output_dir;profile_path=out/'profile.json'
    provenance=dict(model_sha256=model_hash(),probe_sha256=sha(a.probe),script_sha256=sha(__file__),protocol=PROTOCOL)
    calibration_base=ROOT/'data/calibration/upv-core-shell-check/no-cooling-3x3x3-scenario.json'
    if a.mode=='validate':
        profile=json.loads(profile_path.read_text())
        for k,v in provenance.items():
            if profile[k]!=v:raise ValueError('Changed after fit: '+k)
        if not all(profile[k]['optimizer_success'] for k in ('conductivity','capacity')):raise ValueError('Unconverged stage')
        if profile['baseline_sha256']!=sha(calibration_base):raise ValueError('Baseline changed')
    experiment='no-cooling' if a.mode=='fit' else 'active-cooling'
    base_path=ROOT/f'data/calibration/upv-core-shell-check/{experiment}-3x3x3-scenario.json'
    base=json.loads(base_path.read_text());observed,record,manifest=load_case(ROOT/'data/experiments/upv',experiment)
    if record['split']!=('calibration' if a.mode=='fit' else 'validation'):raise ValueError('Split violation')
    if record['canonical_sha256']!=base['metadata']['canonical_sha256']:raise ValueError('Dataset changed')
    if a.mode=='validate' and (manifest['archive_sha256']!=profile['archive_sha256'] or
        base['metadata']['applied_parameters']!=profile['prior_external_parameters']):raise ValueError('Prior/data mismatch')
    baseline=run_saved(a.probe,out/f'{experiment}-baseline-scenario.json',base)
    legacy=json.loads((base_path.parent/f'{experiment}-3x3x3.json').read_text())['result']
    reproduction=compare(baseline,legacy)
    if reproduction['max_change_k']>1e-7:raise ValueError('Constant baseline regression')
    scene=table_scene(base);path=out/f'{experiment}-transport-trial-scenario.json';save(path,scene)
    if a.mode=='fit':
        old=base['metadata']['applied_parameters'];contacts=[old[k] for k in G_NAMES[3:]]
        lower=np.array([-6.]*3+[max(.0101,.5*v) for v in contacts]);upper=np.array([6.]*3+[min(99.99,2*v) for v in contacts])
        theta=(np.asarray(TEMPERATURES[1:])-298.15)/100.
        prior=-9.374269586*theta+12*theta*theta
        w=Worker(a.probe,path)
        try:g,g_run,neutral=search(w,scene,observed,G_NAMES,[np.r_[np.zeros(3),contacts],np.r_[prior,contacts]],lower,upper,a.max_evaluations,'conductivity')
        finally:w.close()
        if not g['optimizer_success']:raise ValueError('Conductivity search did not converge')
        save(out/'conductivity-profile.json',dict(**provenance,**g,validation_used=False))
    else:g=profile['conductivity']
    transport=bake_transport(scene,g['parameters']);g_path=out/f'{experiment}-conductivity-scenario.json'
    exported_g=run_saved(a.probe,g_path,transport)
    if a.mode=='fit' and compare(exported_g,g_run)['max_change_k']>1e-9:raise ValueError('Transport export mismatch')
    g_run=exported_g;cp_scene=capacity_scene(transport);cp_path=out/f'{experiment}-capacity-trial-scenario.json';save(cp_path,cp_scene)
    if a.mode=='fit':
        w=Worker(a.probe,cp_path)
        try:c,c_run,neutral_c=search(w,cp_scene,observed,C_NAMES,[np.ones(2),np.array([1.1,.9])],np.full(2,.8),np.full(2,1.2),a.max_evaluations,'capacity')
        finally:w.close()
        neutral_check=compare(neutral_c,g_run)
        if neutral_check['max_change_k']>1e-7:raise ValueError('Neutral energy law changes temperatures')
        if not c['optimizer_success']:raise ValueError('Capacity search did not converge')
        profile=dict(**provenance,conductivity=g,capacity=c,validation_used=False,baseline_sha256=sha(calibration_base),
            calibration_csv_sha256=record['canonical_sha256'],archive_sha256=manifest['archive_sha256'],
            prior_external_parameters=base['metadata']['applied_parameters'],neutral_capacity_check=neutral_check,
            baseline_reproduction=reproduction)
        save(profile_path,profile)
    else:c=profile['capacity']
    final=bake_capacity(cp_scene,c['parameters']);final_path=out/f'{experiment}-capacity-scenario.json'
    exported_c=run_saved(a.probe,final_path,final)
    if a.mode=='fit' and compare(exported_c,c_run)['max_change_k']>1e-9:raise ValueError('Capacity export mismatch')
    report=dict(experiment=experiment,model_sha256=model_hash(),frozen_profile_sha256=sha(profile_path),
        baseline=dict(metrics=metrics(baseline,observed),result=baseline),
        conductivity=dict(metrics=metrics(g_run,observed),result=g_run,time_check=numerical_check(a.probe,g_path,g_run)),
        capacity=dict(metrics=metrics(exported_c,observed),result=exported_c,time_check=numerical_check(a.probe,final_path,exported_c)),
        baseline_reproduction=reproduction,source_csv_sha256=record['canonical_sha256'],validation_refitted=False)
    save(out/f'{a.mode}.json',report)
    print(json.dumps(dict(experiment=experiment,conductivity=g['parameters'],capacity=c['parameters'],
        metrics={k:report[k]['metrics'] for k in ('baseline','conductivity','capacity')}),indent=2),flush=True)

if __name__=='__main__':main()

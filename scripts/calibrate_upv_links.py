#!/usr/bin/env python3
"""Continue 3^3 connectivity calibration with ALL capacities constant.

Calibration-only sensitivity selected battery/ISS and ISS/aluminium references.
Insensitive saturated contact references, boundary conditions and external
temperature responses are frozen. All time integration is native CUDA.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
from calibrate_upv import Worker, metrics, model_hash, residuals, save, verify_fixed_properties
from calibrate_upv_staged import numerical_check, run_saved, sha
from check_upv_core_shell import compare
from export_upv_state_geometry import contact_group
from generate_upv import ROOT, load_case

INTERNAL=('internal_log_35','internal_log_50','internal_log_70')
CONTACTS=('battery_iss_scale','iss_aluminium_scale')
NAMES=INTERNAL+CONTACTS
INTERNAL_CONTRAST='internal_temperature_contrast'
TEMPERATURES=[298.15,308.15,323.15,343.15]
PRIOR_DIR=ROOT/'data/calibration/upv-staged-3x3x3'
PROTOCOL=dict(revision='links-only-v1',grid=[3,3,3],calibration='no-cooling',validation='active-cooling',fit_start_s=60.,
    fixed='ALL capacities (battery C=412.87 J/K), reference k tensors, external temperature/contrast response, geometry, cooling/ambient boundaries, source forcing, sensors, defects',
    fitted=list(NAMES),internal_knots_k=TEMPERATURES,internal_log_bounds=[-6.,6.],internal_factor_bounds=[.2,3.],
    contact_h_bounds_w_m2_k=[2.,9999.],
    selection='No-cooling finite perturbations: battery/ISS and ISS/aluminium sensitive; saturated TIM contacts weakly observable. No validation-guided selection.',
    bounds_note='Remove prior factor-two trust box for battery/ISS; keep the existing physical-model h ceiling. Exploratory bounds, not measured material intervals.',
    regularization='0.03 K RMS per internal log-response slope difference, unchanged from previous transport fit',
    caveat='Pre-event conditional graph fit, not runaway/defect calibration or measured bulk k(T). Numbering unconfirmed; validation previously inspected. Mesh remains 3^3.')

def bindings(scene):
    objects={c['id']:c for c in scene['cells']};ports={p['id']:p for p in scene['ports']}
    edges=[(e,contact_group(objects[ports[e['port_a']]['cell_id']]['material_id'],objects[ports[e['port_b']]['cell_id']]['material_id'])) for e in scene['interfaces']]
    boundaries=[(b,contact_group('battery',objects[ports[b['port_id']]['cell_id']]['material_id'])) for b in scene['boundaries'] if b['id'].startswith('source-')]
    return edges,boundaries

def protocol(with_contrast=False):
    result=copy.deepcopy(PROTOCOL)
    if with_contrast:
        result.update(revision='links-only-contrast-v1',fitted=list(NAMES)+( [INTERNAL_CONTRAST]),
            internal_contrast_bounds=[-12.,12.],
            internal_response='tabulated(T_donor) + beta*(T_donor-T_receiver)/100 K; never average directions',
            development_note='Additional constitutive trial after inspecting reference-contact trial validation; not blind model selection')
    return result

def links_scene(prior,with_contrast=False):
    s=copy.deepcopy(prior)
    for c in s['cells']:
        if 'heat_capacity_law' in c['thermal']:raise ValueError('Links-only fit refuses a variable-capacity scenario')
        if c['role']=='battery':
            if c['thermal']['divisions']!=[3]*3:raise ValueError('Battery mesh must stay 3^3')
            r=c['thermal']['bulk_response']
            if [p[0] for p in r['temperature_log_response']]!=TEMPERATURES or r['temperature_log_response'][0][1]!=0.:
                raise ValueError('Unexpected internal table anchor/knots')
    s['metadata'].pop('staged_protocol',None)
    s['metadata'].update(staged_calibration='links',links_protocol=protocol(with_contrast),internal_contrast_calibration=with_contrast,
        calibration_status='Links-only calibration, ALL capacities constant; exploratory pre-event profile')
    return s

def current_parameters(scene):
    responses=[c['thermal']['bulk_response']['temperature_log_response'] for c in scene['cells'] if c['role']=='battery']
    if any(r!=responses[0] for r in responses):raise ValueError('Shared battery internal table required')
    result=dict(zip(INTERNAL,[v for t,v in responses[0][1:]]))
    for e,group in bindings(scene)[0]:
        if group not in CONTACTS:continue
        h=1.+9999./(1.+math.exp(-e['law']['geometry_reference_logit']))
        if group in result and abs(result[group]-h/100.)>1e-10:raise ValueError('Inconsistent material-pair reference')
        result[group]=h/100.
    if set(result)!=set(NAMES):raise ValueError('Missing calibrated contact class')
    if scene['metadata'].get('internal_contrast_calibration'):
        values=[c['thermal']['bulk_response']['temperature_contrast'] for c in scene['cells'] if c['role']=='battery']
        if any(v!=[values[0][0]]*3 for v in values):raise ValueError('Shared isotropic internal contrast required')
        result[INTERNAL_CONTRAST]=values[0][0]
    return result

def check_parameters(params):
    if set(params) not in (set(NAMES),set(NAMES+(INTERNAL_CONTRAST,))):raise ValueError('Only selected link parameters are accepted')
    for k,v in params.items():
        if not math.isfinite(v) or (k in INTERNAL and not -6<=v<=6) or (k in CONTACTS and not .02<=v<=99.99) or (k==INTERNAL_CONTRAST and not -12<=v<=12):
            raise ValueError('Link parameter out of protocol bounds: '+k)

def bake(scene,params):
    check_parameters(params);out=copy.deepcopy(scene)
    if bool(scene['metadata'].get('internal_contrast_calibration'))!=(INTERNAL_CONTRAST in params):raise ValueError('Contrast protocol mismatch')
    for c in out['cells']:
        if c['role']=='battery':
            c['thermal']['bulk_response']['temperature_log_response']=[[t,v] for t,v in zip(TEMPERATURES,[0.]+[params[k] for k in INTERNAL])]
            if INTERNAL_CONTRAST in params:c['thermal']['bulk_response']['temperature_contrast']=[params[INTERNAL_CONTRAST]]*3
    def apply(law,group):
        if group in CONTACTS:
            h=100.*params[group];law['geometry_reference_logit']=math.log((h-1.)/(10000.-h))
    edges,boundaries=bindings(out)
    for e,group in edges:apply(e['law'],group)
    for b,group in boundaries:apply(b['connectivity'],group)
    # The legacy constant-G fallback is not the dynamic contact and stays fixed.
    out['metadata']['links_parameters']=params
    return out

def audit_run(run,scene,params,baseline):
    check_parameters(params);verify_fixed_properties(run,scene)
    audit=copy.deepcopy(run['applied_model'])
    for actual,prior,c in zip(audit['objects'],baseline['objects'],scene['cells']):
        if 'capacity_table' in actual:raise ValueError('A variable capacity leaked into links-only calibration')
        if actual['heat_capacity_j_per_k']!=c['thermal']['heat_capacity_j_per_k']:raise ValueError('Capacity changed')
        if c['role']=='battery':
            expected=[[t,v] for t,v in zip(TEMPERATURES,[0.]+[params[k] for k in INTERNAL])]
            if actual['internal_table']!=expected:raise ValueError('Wrong native internal response')
            actual['internal_table']=prior['internal_table']
            if INTERNAL_CONTRAST in params:
                if actual[INTERNAL_CONTRAST]!=params[INTERNAL_CONTRAST]:raise ValueError('Wrong internal contrast')
                actual[INTERNAL_CONTRAST]=prior[INTERNAL_CONTRAST]
    for actual,prior in zip(audit['contacts'],baseline['contacts']):
        if actual['group'] in CONTACTS:
            if abs(actual['reference_h_w_m2_k']-100.*params[actual['group']])>1e-8:raise ValueError('Wrong native contact reference')
            for key in ('scale','reference_h_w_m2_k','conductance_w_per_k'):actual[key]=prior[key]
    for actual,prior in zip(audit['boundaries'],baseline['boundaries']):
        if actual['id'].startswith('source-') and actual['group'] in CONTACTS:
            if abs(actual['reference_h_w_m2_k']-100.*params[actual['group']])>1e-8:raise ValueError('Source/witness contact-class mismatch')
            actual['reference_h_w_m2_k']=prior['reference_h_w_m2_k']
    if audit!=baseline:raise ValueError('A frozen part of the actual native model changed')
    if abs(run['energy_residual_j'])>1e-6:raise ValueError('Energy conservation failed')

def parameter_names(params):return NAMES+((INTERNAL_CONTRAST,) if INTERNAL_CONTRAST in params else ())
def pack(params):return np.array([math.log(params[k]) if k in CONTACTS else params[k] for k in parameter_names(params)])
def unpack(x,names=NAMES):return {k:math.exp(float(v)) if k in CONTACTS else float(v) for k,v in zip(names,x)}

def search(worker,scene,observed,max_nfev):
    from scipy.optimize import least_squares
    prior_params=current_parameters(scene);baseline=worker.run({});audit_run(baseline,scene,prior_params,baseline['applied_model'])
    names=parameter_names(prior_params);extra=INTERNAL_CONTRAST in prior_params
    lower=np.r_[[-6.]*3,[math.log(.02)]*2,[-12.] if extra else []];upper=np.r_[[6.]*3,[math.log(99.99)]*2,[12.] if extra else []]
    first=pack(prior_params);second=first.copy();second[3:5]-=math.log(2.)
    cache={};history=[]
    def objective(x):
        key=tuple(x)
        if key not in cache:
            params=unpack(x,names);run=worker.run(params);audit_run(run,scene,params,baseline['applied_model'])
            r=np.concatenate(list(residuals(run,observed).values()))
            slopes=np.diff(np.r_[0.,x[:3]])/np.diff(np.asarray(TEMPERATURES)/10.)
            cache[key]=np.r_[r,.03*math.sqrt(len(r))*np.diff(slopes)]
            history.append(dict(parameters=params,rmse_k=float(np.sqrt(np.mean(r*r)))))
            if len(history)%20==1:print('links',len(history),history[-1],flush=True)
        return cache[key]
    def jacobian(x):
        cols=[]
        for i in range(len(x)):
            lo=x.copy();hi=x.copy();lo[i]=max(lower[i],x[i]-.002);hi[i]=min(upper[i],x[i]+.002)
            cols.append((objective(hi)-objective(lo))/(hi[i]-lo[i]))
        return np.stack(cols,axis=1)
    fits=[least_squares(objective,np.clip(start,lower,upper),jac=jacobian,bounds=(lower,upper),max_nfev=max_nfev,
                        ftol=1e-7,xtol=1e-6,gtol=1e-5) for start in (first,second)]
    best=min([f for f in fits if f.success] or fits,key=lambda f:np.dot(f.fun,f.fun))
    params=unpack(best.x,names);run=worker.run(params);audit_run(run,scene,params,baseline['applied_model'])
    # Conditioning of DATA ONLY, not of the smoothness regularizer.
    n=sum(len(e) for e in residuals(run,observed).values());u,sv,vt=np.linalg.svd(best.jac[:n],full_matrices=False)
    return dict(parameters=params,optimizer_success=bool(best.success),message=best.message,
        near_bounds=[k for i,k in enumerate(names) if min(best.x[i]-lower[i],upper[i]-best.x[i])<.001*(upper[i]-lower[i])],
        data_jacobian_condition_number=float(sv[0]/sv[-1]) if sv[-1]>0 else None,
        data_jacobian_singular_values=sv.tolist(),parameter_coordinates='internal log response; natural log contact h/100',
        metrics=metrics(run,observed),prior_metrics=metrics(baseline,observed),
        starts=[dict(success=bool(f.success),parameters=unpack(f.x,names),objective=float(np.dot(f.fun,f.fun))) for f in fits],
        search=dict(max_nfev_per_start=max_nfev,absolute_jacobian_step=.002,ftol=1e-7,xtol=1e-6,gtol=1e-5),
        evaluation_history=history),run,baseline

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('fit','validate'))
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--max-evaluations',type=int,default=100)
    p.add_argument('--internal-contrast',action='store_true')
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args();out=a.output_dir or ROOT/('data/calibration/upv-links-contrast-3x3x3' if a.internal_contrast else 'data/calibration/upv-links-3x3x3');profile_path=out/'profile.json'
    dependencies=['calibrate_upv_staged.py','calibrate_upv_internal.py','check_upv_core_shell.py','export_upv_state_geometry.py']
    provenance=dict(model_sha256=model_hash(),probe_sha256=sha(a.probe),script_sha256=sha(__file__),protocol=protocol(a.internal_contrast),
        helper_sha256={k:sha(ROOT/'scripts'/k) for k in dependencies},
        calibration_prior_sha256=sha(PRIOR_DIR/'no-cooling-conductivity-scenario.json'))
    if a.mode=='validate':
        profile=json.loads(profile_path.read_text())
        for k,v in provenance.items():
            if profile[k]!=v:raise ValueError('Changed after fit: '+k)
        if not profile['optimizer_success']:raise ValueError('Cannot validate unconverged fit')
    experiment='no-cooling' if a.mode=='fit' else 'active-cooling'
    prior_path=PRIOR_DIR/f'{experiment}-conductivity-scenario.json'
    prior=json.loads(prior_path.read_text());scene=links_scene(prior,a.internal_contrast)
    observed,record,manifest=load_case(ROOT/'data/experiments/upv',experiment)
    if record['split']!=('calibration' if a.mode=='fit' else 'validation'):raise ValueError('Split violation')
    if record['canonical_sha256']!=prior['metadata']['canonical_sha256']:raise ValueError('Prior data changed')
    if a.mode=='validate' and (manifest['archive_sha256']!=profile['archive_sha256'] or
        current_parameters(scene)!=profile['prior_parameters']):raise ValueError('Validation baseline does not share frozen link parameters')
    path=out/f'{experiment}-trial-scenario.json';save(path,scene);w=Worker(a.probe,path)
    try:
        if a.mode=='fit':
            fit,run,baseline=search(w,scene,observed,a.max_evaluations)
            profile=dict(**provenance,**fit,validation_used=False,all_capacities_fixed=True,
                prior_parameters=current_parameters(scene),calibration_csv_sha256=record['canonical_sha256'],archive_sha256=manifest['archive_sha256'])
            save(profile_path,profile)
            if not profile['optimizer_success']:raise ValueError('Search unconverged; profile retained as diagnostic only')
        else:
            baseline=w.run({});run=w.run(profile['parameters']);audit_run(run,scene,profile['parameters'],baseline['applied_model'])
        legacy=json.loads((PRIOR_DIR/('fit.json' if a.mode=='fit' else 'validate.json')).read_text())['conductivity']['result']
        reproduction=compare(baseline,legacy)
        if reproduction['max_change_k']>1e-7:raise ValueError('Prior transport baseline changed')
        baked=bake(scene,profile['parameters']);export_path=out/f'{experiment}-fitted-scenario.json'
        exported=run_saved(a.probe,export_path,baked);audit_run(exported,scene,profile['parameters'],baseline['applied_model'])
        equivalence=compare(exported,run)
        if equivalence['max_change_k']>1e-9:raise ValueError('Export disagrees with fit')
        report=dict(experiment=experiment,parameters=profile['parameters'],metrics=metrics(run,observed),result=run,
            baseline_metrics=metrics(baseline,observed),baseline_result=baseline,baseline_reproduction=reproduction,
            export_equivalence=equivalence,time_check=numerical_check(a.probe,export_path,exported),
            frozen_profile_sha256=sha(profile_path),model_sha256=model_hash(),source_csv_sha256=record['canonical_sha256'],
            all_capacities_fixed=True,validation_refitted=False,scenario_path=str(export_path))
        save(out/f'{a.mode}.json',report)
        print(json.dumps({k:report[k] for k in ('experiment','parameters','baseline_metrics','metrics','time_check')},indent=2),flush=True)
    finally:w.close()

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Training-only shell-opening fit atop frozen v7 materials, chemistry and forcing.

All thermal/gas evolution is CUDA. No re-fit of heat capacity or conductivity.
The chamber pressure cannot identify weld geometry and strength separately.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
import calibrate_upv_gas_energy as gas
from calibrate_upv import save

ROOT=gas.t.ROOT
PRIOR=ROOT/'data/calibration/upv-gas-energy-v7'
HEAT=ROOT/'data/calibration/upv-physical-casing-v7/heat'
OUT=ROOT/'data/calibration/upv-progressive-shell-v9'
NAMES=('strength_pa','damage_rate_per_s','total_area_m2')
INITIAL=dict(strength_pa=1e5,damage_rate_per_s=1.,total_area_m2=1e-5)
LIMITS=np.array([[1e4,.05,1e-7],[5e6,100.,1e-3]])
PROTOCOL=dict(revision='progressive-shell-v9-terminal-closure',calibration='no-cooling',validation='active-cooling',
    frozen='All v7 solid/gas energy, gas yield, chemistry, free volume, wall/exposure, thermal properties, sensors and initial fields',
    fitted=list(NAMES),seam_width_m=.0005,seam_strength_factor=.3,damage_onset=.05,exponent=2.,
    area='Total effective area distributed by panel face area over six C2 panels; not recovered fracture geometry',
    routing='Single chamber receiving compartment is an effective peripheral escape path, including discharge from covered faces; no resolved interfacial gas gap or jet. Thermal solid contacts are retained, not bypassed.',
    scope='Conditional primary C2 event only, not prediction of ignition or cascade; C1/C3 remain the previous thermal witnesses',
    identifiability='Only chamber pressure observed; seam geometry/ratio fixed priors, not measured or identified weld data',
    objective=gas.PROTOCOL['objective'],validation_note='Previously inspected experiment, excluded from all parameter selection; not blind validation')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(probe):
    return hashlib.sha256((gas.t.fingerprint(probe)+digest(__file__)+digest(PRIOR/'profile.json')+digest(HEAT/'profile.json')).encode()).hexdigest()

def inputs(name):
    mode='fit' if name=='no-cooling' else 'validate'
    scene=json.loads((PRIOR/(name+'-'+mode+'-scenario.json')).read_text())
    report=json.loads((PRIOR/(mode+'.json')).read_text())
    if scene['failure_sample_times_s']!=report['result']['time_s']:
        if not np.allclose(scene['failure_sample_times_s'],report['result']['time_s'],rtol=0,atol=1e-8):raise ValueError('Observation alignment changed')
    return scene,report

def bake(scene,parameters):
    if set(parameters)!=set(NAMES):raise ValueError('Only shell opening parameters may be fitted')
    for k,lo,hi in zip(NAMES,*LIMITS):
        if not math.isfinite(parameters[k]) or not lo<=parameters[k]<=hi:raise ValueError('Invalid shell parameter '+k)
    s=copy.deepcopy(scene)
    shell=next(x for x in s['shells'] if x['contents_id']=='C2')
    core=next(x for x in s['cells'] if x['id']=='C2');dims=core['geometry']['dimensions_m'];thickness=shell['thickness_m']
    faces=[]
    for axis in range(3):
        panel=[thickness if k==axis else dims[k]+(2*thickness if k>axis else 0) for k in range(3)]
        area=panel[(axis+1)%3]*panel[(axis+2)%3]
        for sign in ('-','+'):faces.append((sign+'xyz'[axis],area))
    total=sum(area for _,area in faces);prototype=s['gas_network']['channels'][0]
    if len(s['gas_network']['channels'])!=1 or prototype['from']!='inside-C2':raise ValueError('Unexpected prior gas topology')
    channels=[]
    for face,area in faces:
        ch=copy.deepcopy(prototype)
        ch.update(id='C2-progressive-'+face,port_id='C2:case:'+face+':outer',
            area_m2=parameters['total_area_m2']*area/total,opening_pressure_pa=parameters['strength_pa'],
            damage_opening=dict(damage_onset=PROTOCOL['damage_onset'],exponent=PROTOCOL['exponent'],
                pressure_damage_rate_per_s=parameters['damage_rate_per_s'],max_damage_step=.02,
                seam_width_m=PROTOCOL['seam_width_m'],seam_strength_factor=PROTOCOL['seam_strength_factor']))
        if ch['area_m2']>area:raise ValueError('Effective aperture exceeds face')
        channels.append(ch)
    s['gas_network']['channels']=channels
    s['metadata']['progressive_shell_protocol']=PROTOCOL
    s['metadata']['calibration_status']='Exploratory frozen-v7 shell-only calibration; not validated destruction physics'
    return s

def score(run,report):
    obs={k:np.asarray(v,dtype=float) for k,v in report['observed_temperature_k'].items()}
    return gas.errors(run,obs,np.asarray(report['observed_pressure_bar']),report['meta']['event'])

def metrics(run,report):
    obs={k:np.asarray(v,dtype=float) for k,v in report['observed_temperature_k'].items()}
    return gas.metrics(run,obs,np.asarray(report['observed_pressure_bar']),report['meta']['event'])

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('mode',choices=['verify-heat','fit','validate','check'])
    ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--max-evaluations',type=int,default=6)
    ap.add_argument('--probe',type=Path,default=Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);fp=fingerprint(args.probe)
    runner=gas.t.Runner(args.probe,args.output)
    try:
        if args.mode=='verify-heat':
            results={}
            for mode,name in [('fit','no-cooling'),('validate','active-cooling')]:
                scene=json.loads((HEAT/(name+'-fitted-scenario.json')).read_text())
                old=json.loads((HEAT/(mode+'.json')).read_text())['result'];run=runner.run(scene)
                delta=max(float(np.max(np.abs(np.asarray(run['temperature_k'][k])-v))) for k,v in old['temperature_k'].items())
                if delta>1e-10 or run['applied_objects']!=old['applied_objects']:raise ValueError('Frozen intact heating changed')
                results[name]=dict(max_change_k=delta,steps=run['steps']);print(name,results[name],flush=True)
            save(args.output/'heat-replay.json',dict(fingerprint=fp,results=results));return
        proof=json.loads((args.output/'heat-replay.json').read_text())
        if proof['fingerprint']!=fp:raise ValueError('Re-run intact heating after code changes')
        if args.mode=='fit':
            if (args.output/'profile.json').exists():raise ValueError('Profile is frozen; use a new directory')
            scene,prior=inputs('no-cooling');cache={};best=[math.inf,None,None];history=[];started=time.monotonic()
            def residual(x):
                key=tuple(x)
                if key in cache:return cache[key]
                p=dict(zip(NAMES,map(float,np.exp(x))));p={k:float(np.clip(p[k],LIMITS[0,i],LIMITS[1,i])) for i,k in enumerate(NAMES)}
                run=runner.run(bake(scene,p));v=score(run,prior);cost=float(v@v)
                history.append(dict(candidate=len(history)+1,parameters=p,cost=cost,metrics=metrics(run,prior),steps=run['steps'],wall_s=run['wall_s']))
                if cost<best[0]:
                    best[:]=[cost,p,run];save(args.output/'checkpoint.json',history[-1]);save(args.output/'best-run.json',run)
                if len(history)==1:save(args.output/'initial-run.json',run)
                save(args.output/'history.json',history)
                print(f'candidate {len(history)} cost={cost:.6f} best={best[0]:.6f} pressure={history[-1]["metrics"]["predicted_peak_bar"]:.4f} bar T_RMSE={history[-1]["metrics"]["witness"]["rmse_k"]:.4f} K native={run["wall_s"]:.1f}s elapsed={time.monotonic()-started:.0f}s {p}',flush=True)
                cache[key]=v;return v
            def jacobian(x):
                base=residual(x);columns=[]
                for i in range(len(x)):
                    # Fixed 5% logarithmic perturbation, including log(rate)=0.
                    dx=.05 if x[i]+.05<=math.log(LIMITS[1,i]) else -.05
                    shifted=x.copy();shifted[i]+=dx
                    columns.append((residual(shifted)-base)/dx)
                return np.column_stack(columns)
            fit=least_squares(residual,np.log([INITIAL[k] for k in NAMES]),bounds=np.log(LIMITS),
                jac=jacobian,x_scale='jac',max_nfev=args.max_evaluations,ftol=1e-4,xtol=1e-4,gtol=1e-4)
            p=best[1];run=best[2]
            profile=dict(parameters=p,protocol=PROTOCOL,fingerprint=fp,validation_used_for_selection=False,
                provenance={str(path.relative_to(ROOT)):digest(path) for path in [PRIOR/'profile.json',HEAT/'profile.json',PRIOR/'no-cooling-fit-scenario.json',PRIOR/'fit.json']},
                search=dict(success=bool(fit.success),message=fit.message,nfev=fit.nfev,candidates=len(history),
                    jacobian_singular_values=np.linalg.svd(fit.jac,compute_uv=False).tolist(),initial_cost=history[0]['cost'],best_cost=best[0]))
            save(args.output/'profile.json',profile);name='no-cooling';mode='fit'
        else:
            profile=json.loads((args.output/'profile.json').read_text())
            if profile['fingerprint']!=fp:raise ValueError('Frozen model changed')
            p=profile['parameters']
            if args.mode=='check':
                result={}
                for mode,name in [('fit','no-cooling'),('validate','active-cooling')]:
                    saved=json.loads((args.output/(mode+'.json')).read_text());s=json.loads((args.output/(name+'-'+mode+'-scenario.json')).read_text())
                    s['simulation']['stability_factor']*=.5;fine=runner.run(s,.5)
                    coarse=saved['result'];delta=dict(max_pressure_change_bar=float(np.max(np.abs(np.asarray(coarse['gas']['chamber']['pressure_pa'])-fine['gas']['chamber']['pressure_pa']))/1e5),
                        max_witness_change_k=max(float(np.max(np.abs(np.asarray(coarse['temperature_k'][k])-fine['temperature_k'][k]))) for k in gas.t.late.WITNESSES),coarse_steps=coarse['steps'],fine_steps=fine['steps'])
                    _,prior=inputs(name);result[name]=dict(refinement=delta,metrics=metrics(fine,prior));save(args.output/(name+'-refined.json'),fine);print(name,result[name],flush=True)
                save(args.output/'refinement.json',result);return
            name='active-cooling';mode='validate';scene,prior=inputs(name);run=runner.run(bake(scene,p))
        report=dict(experiment=name,parameters=p,protocol=PROTOCOL,metrics=metrics(run,prior),previous_metrics=prior['metrics'],
            observed_temperature_k=prior['observed_temperature_k'],observed_pressure_bar=prior['observed_pressure_bar'],meta=prior['meta'],result=run)
        save(args.output/(name+'-'+mode+'-scenario.json'),bake(scene,p));save(args.output/(mode+'.json'),report)
        print(json.dumps(report['metrics'],indent=2),flush=True)
    finally:runner.close()

if __name__=='__main__':main()

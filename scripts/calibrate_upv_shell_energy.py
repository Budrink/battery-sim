#!/usr/bin/env python3
"""Refit the existing six gas-energy parameters after freezing shell opening.

No new physics or battery heat-capacity/conductivity degrees of freedom.
Calibration stays no-cooling; active cooling is replayed only after freezing.
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
import calibrate_upv_shell_opening as shell
from calibrate_upv import save

gas=shell.gas
PRIOR=shell.PRIOR
SHELL=shell.OUT
OUT=shell.ROOT/'data/calibration/upv-progressive-energy-v9'
NAMES=gas.NAMES
PROTOCOL=dict(shell.PROTOCOL,revision='progressive-shell-frozen-energy-refit-v9',
    frozen='Measured solid heat capacity, reference conductivity, intact contacts, shell profile, gas volume and gas species constants',
    fitted=list(NAMES),energy='Same previously fitted six gas-energy parameters as v7; energy reserve is finite, no independent extra heat source',
    thermal='Wall h and common gas exposure can change; intact solid conduction/capacity cannot')
digest=shell.digest

def fingerprint(probe):
    return hashlib.sha256((shell.fingerprint(probe)+digest(__file__)+digest(SHELL/'profile.json')).encode()).hexdigest()

def bake(scene,p):
    if set(p)!=set(NAMES):raise ValueError('Unexpected energy parameters')
    for key,lo,hi in zip(NAMES,*np.exp(gas.BOUNDS)):
        if not math.isfinite(p[key]) or not lo<=p[key]<=hi:raise ValueError('Parameter outside exploratory bounds')
    parameters=json.loads((SHELL/'profile.json').read_text())['parameters']
    s=shell.bake(scene,parameters)
    # Input is already the v7 baked scene: scale exposure relative to that fit,
    # not a second multiplication by the absolute exposure multiplier.
    prior=json.loads((PRIOR/'profile.json').read_text())['parameters']
    relative=dict(p,exposure_scale=p['exposure_scale']/prior['exposure_scale'])
    s=gas.bake(s,relative);s['metadata']['progressive_energy_protocol']=PROTOCOL
    return s

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('mode',choices=['fit','validate','check'])
    ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--max-evaluations',type=int,default=5)
    ap.add_argument('--probe',type=Path,default=Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'));a=ap.parse_args()
    sh=json.loads((SHELL/'profile.json').read_text())
    if sh['fingerprint']!=shell.fingerprint(a.probe) or sh['validation_used_for_selection']:raise ValueError('Shell fit changed or used validation')
    a.output.mkdir(parents=True,exist_ok=True);fp=fingerprint(a.probe);runner=gas.t.Runner(a.probe,a.output)
    try:
        if a.mode=='fit':
            if (a.output/'profile.json').exists():raise ValueError('Output profile frozen')
            scene,prior=shell.inputs('no-cooling');initial=json.loads((PRIOR/'profile.json').read_text())['parameters']
            cache={};best=[math.inf,None,None];history=[];started=time.monotonic()
            def residual(x):
                key=tuple(x)
                if key in cache:return cache[key]
                p={k:float(np.clip(np.exp(x[i]),np.exp(gas.BOUNDS[0,i]),np.exp(gas.BOUNDS[1,i]))) for i,k in enumerate(NAMES)}
                run=runner.run(bake(scene,p));v=shell.score(run,prior);cost=float(v@v)
                history.append(dict(candidate=len(history)+1,cost=cost,parameters=p,metrics=shell.metrics(run,prior),wall_s=run['wall_s']))
                if cost<best[0]:best[:]=[cost,p,run];save(a.output/'checkpoint.json',history[-1]);save(a.output/'best-run.json',run)
                save(a.output/'history.json',history);cache[key]=v
                print(f'energy candidate {len(history)} cost={cost:.6f} best={best[0]:.6f} P={history[-1]["metrics"]["predicted_peak_bar"]:.4f} bar T={history[-1]["metrics"]["witness"]["rmse_k"]:.4f} K elapsed={time.monotonic()-started:.0f}s',flush=True)
                return v
            def jacobian(x):
                base=residual(x);columns=[]
                for i in range(len(x)):
                    dx=.03 if x[i]+.03<=gas.BOUNDS[1,i] else -.03
                    xx=x.copy();xx[i]+=dx;columns.append((residual(xx)-base)/dx)
                return np.column_stack(columns)
            result=least_squares(residual,np.log([initial[k] for k in NAMES]),bounds=gas.BOUNDS,jac=jacobian,
                x_scale='jac',max_nfev=a.max_evaluations,ftol=1e-4,xtol=1e-4,gtol=1e-4)
            p=best[1];run=best[2];mode='fit';name='no-cooling'
            save(a.output/'profile.json',dict(parameters=p,protocol=PROTOCOL,fingerprint=fp,validation_used_for_selection=False,
                shell_profile_sha256=digest(SHELL/'profile.json'),v7_profile_sha256=digest(PRIOR/'profile.json'),
                search=dict(success=bool(result.success),message=result.message,nfev=result.nfev,candidates=len(history),
                    initial_cost=history[0]['cost'],best_cost=best[0],jacobian_singular_values=np.linalg.svd(result.jac,compute_uv=False).tolist())))
        else:
            profile=json.loads((a.output/'profile.json').read_text())
            if profile['fingerprint']!=fp:raise ValueError('Frozen energy model changed')
            p=profile['parameters']
            if a.mode=='check':
                summary={}
                for mode,name in [('fit','no-cooling'),('validate','active-cooling')]:
                    saved=json.loads((a.output/(mode+'.json')).read_text());s=json.loads((a.output/(name+'-'+mode+'-scenario.json')).read_text())
                    s['simulation']['stability_factor']*=.5;fine=runner.run(s,.5);coarse=saved['result']
                    _,prior=shell.inputs(name)
                    summary[name]=dict(metrics=shell.metrics(fine,prior),refinement=dict(
                        max_pressure_change_bar=float(np.max(np.abs(np.asarray(coarse['gas']['chamber']['pressure_pa'])-fine['gas']['chamber']['pressure_pa']))/1e5),
                        max_witness_change_k=max(float(np.max(np.abs(np.asarray(coarse['temperature_k'][k])-fine['temperature_k'][k]))) for k in gas.t.late.WITNESSES),
                        coarse_steps=coarse['steps'],fine_steps=fine['steps']))
                    save(a.output/(name+'-refined.json'),fine);print(name,summary[name],flush=True)
                save(a.output/'refinement.json',summary);return
            mode='validate';name='active-cooling';scene,prior=shell.inputs(name);run=runner.run(bake(scene,p))
        report=dict(experiment=name,parameters=p,protocol=PROTOCOL,metrics=shell.metrics(run,prior),previous_metrics=prior['metrics'],
            observed_temperature_k=prior['observed_temperature_k'],observed_pressure_bar=prior['observed_pressure_bar'],meta=prior['meta'],result=run)
        save(a.output/(mode+'.json'),report);save(a.output/(name+'-'+mode+'-scenario.json'),bake(scene,p));print(json.dumps(report['metrics'],indent=2),flush=True)
    finally:runner.close()

if __name__=='__main__':main()

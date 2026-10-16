#!/usr/bin/env python3
"""Training-only ablation: refit static contacts with EXACTLY the same fixed materials.

Keep the dynamic CUDA implementation and set its state response to zero. This
separates the benefit of temperature response from the updated pad-cp priors.
The validation case is never opened by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from calibrate_upv import (CONTACTS,CONTRASTS,Worker,metrics,model_hash,parameter_bounds,
                           residuals,save,validate_profile,verify_fixed_properties)
from generate_upv import ROOT,load_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();profile=json.loads(a.profile.read_text());report=json.loads(a.report.read_text())
    validate_profile(profile)
    if report['kind']!='fit' or report['experiment']!='no-cooling':raise ValueError('Training report required')
    if profile['model_sha256']!=model_hash() or report['model_sha256']!=model_hash():raise ValueError('Model changed')
    if profile['probe_sha256']!=hashlib.sha256(a.probe.read_bytes()).hexdigest():raise ValueError('Probe changed')
    if report['frozen_profile_sha256']!=hashlib.sha256(a.profile.read_bytes()).hexdigest():raise ValueError('Profile changed')
    observed,record,_=load_case(ROOT/'data/experiments/upv','no-cooling')
    if record['canonical_sha256']!=profile['calibration_csv_sha256']:raise ValueError('Data changed')
    path=Path(report['scenario_path']);path=path if path.is_absolute() else ROOT/path
    scene=json.loads(path.read_text());worker=Worker(a.probe,path)
    names=CONTACTS+('ambient_scale',);bounds=np.log(parameter_bounds(names,True));history=[]
    def unpack(x):
        result=dict(zip(names,map(float,np.exp(x))),surface_linear=0.,surface_quadratic=0.)
        if profile['parameter_set']=='directed_geometry':result.update({k:0. for k in CONTRASTS})
        return result
    def objective(x):
        run=worker.run(unpack(x));verify_fixed_properties(run,scene)
        err=np.concatenate(list(residuals(run,observed).values()))
        history.append(float(np.sqrt(np.mean(err**2))))
        return err
    def jac(x):
        columns=[]
        for i in range(len(x)):
            lo=x.copy();hi=x.copy();lo[i]=max(bounds[0,i],x[i]-.002);hi[i]=min(bounds[1,i],x[i]+.002)
            columns.append((objective(hi)-objective(lo))/(hi[i]-lo[i]))
        return np.stack(columns,axis=1)
    try:
        fit=least_squares(objective,np.log([50.]*5+[3.]),bounds=bounds,jac=jac,max_nfev=100,ftol=1e-6,xtol=1e-5,gtol=1e-4)
        run=worker.run(unpack(fit.x));verify_fixed_properties(run,scene);score=metrics(run,observed)
        save(a.output,dict(kind='training_static_ablation',model_sha256=model_hash(),
                          frozen_dynamic_profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
                          parameters=unpack(fit.x),metrics=score,result=run,
                          optimizer_success=bool(fit.success),optimizer_message=fit.message,
                          evaluations=len(history),rmse_history=history,validation_used=False,
                          dynamic_rmse_k=profile['fit_metrics']['rmse_k'],
                          material_properties_identical=True,scenario_path=str(path)))
        print(json.dumps(score,indent=2))
    finally:worker.close()


if __name__=='__main__':main()

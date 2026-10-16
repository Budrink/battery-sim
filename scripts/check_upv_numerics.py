#!/usr/bin/env python3
"""Check a frozen UPV result without repeating or extending the parameter fit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from calibrate_upv import Worker,metrics,model_hash,save,validate_profile,verify_fixed_properties
from generate_upv import ROOT,TARGETS,load_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.report.read_text());profile=json.loads(a.profile.read_text())
    validate_profile(profile)
    if report.get('status')=='rejected':raise ValueError('Rejected report')
    if report['model_sha256']!=model_hash() or profile['model_sha256']!=model_hash():raise ValueError('Model changed')
    digest=hashlib.sha256(a.profile.read_bytes()).hexdigest()
    if report['frozen_profile_sha256']!=digest:raise ValueError('Profile changed')
    if profile['probe_sha256']!=hashlib.sha256(a.probe.read_bytes()).hexdigest():raise ValueError('Probe changed')
    scene_path=Path(report['scenario_path'])
    if not scene_path.is_absolute():scene_path=ROOT/scene_path
    scene=json.loads(scene_path.read_text())
    observed,record,_=load_case(ROOT/'data/experiments/upv',report['experiment'])
    if record['canonical_sha256']!=report['source_csv_sha256']:raise ValueError('Data changed')
    baseline=report['result'];checks={}
    worker=Worker(a.probe,scene_path)
    try:
        for name,settings in [('half_dt',{'max_dt_s':.25}),('refined_passive',{'mesh_scale':2})]:
            print('Checking',name,flush=True)
            result=worker.run(dict(profile['parameters'],**settings));verify_fixed_properties(result,scene)
            if result['time_s']!=baseline['time_s']:raise ValueError('Output times changed')
            diff=np.concatenate([np.asarray(result['temperature_k'][c])-baseline['temperature_k'][c] for c in TARGETS])
            checks[name]=dict(max_change_k=float(np.max(np.abs(diff))),rms_change_k=float(np.sqrt(np.mean(diff**2))),
                              metrics=metrics(result,observed))
    finally:worker.close()
    save(a.output,dict(model_sha256=model_hash(),frozen_profile_sha256=digest,experiment=report['experiment'],
                       baseline_report=str(a.report),parameters_refitted=False,numerical_checks=checks))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

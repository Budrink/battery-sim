#!/usr/bin/env python3
"""Failure-stage driver with a previous training fit as initialization.

Does not change/refit the frozen heat profile. Gas numerical controls are coarser
for the search and explicitly checked against smaller CFL/source/gas steps.
"""
import argparse
import copy
import json
import numpy as np
import calibrate_upv_casing as t
from calibrate_upv import save


def main():
    cli=argparse.ArgumentParser(__doc__);cli.add_argument('mode',choices=['baseline','fit','validate'])
    cli.add_argument('--output',type=t.Path,default=t.OUT)
    cli.add_argument('--probe',type=t.Path,default=t.Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'))
    cli.add_argument('--max-evaluations',type=int,default=15)
    args=cli.parse_args();out=args.output/'failure-fast';out.mkdir(exist_ok=True);name='active-cooling' if args.mode=='validate' else 'no-cooling'
    heat_path=args.output/'heat/profile.json';heat=json.loads(heat_path.read_text())
    if heat['fingerprint']!=t.fingerprint(args.probe):raise ValueError('Model changed since frozen heat fit')
    profile_path=out/'profile.json';fingerprint=t.old.hashlib.sha256((t.fingerprint(args.probe)+t.old.digest(t.Path(__file__))).encode()).hexdigest()
    if args.mode=='fit' and profile_path.exists():raise ValueError('Frozen profile exists: choose a fresh directory')
    if args.mode=='validate':
        profile=json.loads(profile_path.read_text())
        if profile['fingerprint']!=fingerprint or profile['heat_profile_sha256']!=t.old.digest(heat_path):raise ValueError('Frozen model/heat mismatch')
    runner=t.Runner(args.probe,out)
    try:
        scene,obs,meta=t.failure_prepare(name,heat['parameters'],runner)
        scene['gas_network'].update(max_fraction_per_step=.1,max_temperature_step_k=5.)
        scene['metadata']['casing_protocol']['gas_numerics']='Search max fraction .1, max gas temperature step 5 K; final independent CFL/source/gas refinement required'
        save(out/(name+'-unfitted-scenario.json'),scene)
        build=lambda p:t.fail_bake(scene,p)
        error=lambda r:np.concatenate(list(t.late.errors(r,obs,meta['event']).values()))
        # Use the earlier no-cooling-only conditional fit as a START, not truth.
        old=json.loads((t.ROOT/'data/calibration/upv-kinetic-v4/profile.json').read_text())['parameters']
        start={k:old['rate_400_per_s'] if k=='rate_per_s' else old[k] for k in t.FAIL_NAMES}
        selected=('energy_j','residue_through','residue_inplane')
        checkpoint=args.output/'failure-six-parameter-trial/checkpoint.json'
        if checkpoint.exists():
            explored=json.loads(checkpoint.read_text())['parameters']
            for k in selected:start[k]=explored[k]
        if args.mode=='validate':p=profile['parameters']
        elif args.mode=='baseline':p=start
        else:
            indices=[t.FAIL_NAMES.index(k) for k in selected]
            p,search=t.optimize(runner,build,error,np.log([start[k] for k in selected]),t.FAIL_LO[indices],t.FAIL_HI[indices],
                lambda x:dict(start,**dict(zip(selected,map(float,np.exp(x))))),out,args.max_evaluations)
            profile=dict(parameters=p,search=search,protocol=scene['metadata']['casing_protocol'],meta=meta,fingerprint=fingerprint,
                heat_profile_sha256=t.old.digest(heat_path),validation_used_for_selection=False,
                fitted=list(selected),fixed_unidentified={k:start[k] for k in t.FAIL_NAMES if k not in selected},
                selection='Training-only six-parameter perturbation trial: destroyed contacts had zero observable sensitivity; rate sensitivity was below thermal numerical error. Freeze those priors, fit energy and residue factors only.',
                initialization_source='Frozen v4 training-only profile; rate is now a tied two-stage parameter, not identical v4 chemistry',
                status='Conditional late-temperature fit; gas/rupture constants NOT identified')
            save(profile_path,profile)
        s=build(p);r=runner.run(s)
        report=dict(stage='failure',experiment=name,parameters=p,meta=meta,protocol=scene['metadata']['casing_protocol'],metrics=t.late.metrics(r,obs,meta['event']),result=r)
        if args.mode!='baseline':
            fine_scene=copy.deepcopy(s);fine_scene['simulation']['stability_factor']*=.5
            fine=runner.run(fine_scene,.5)
            mask=np.asarray(r['time_s'])>=meta['event']['bracket_width_s']+300
            report['max_refinement_difference_k']=max(float(np.max(np.abs(np.asarray(r['temperature_k'][k])-np.asarray(fine['temperature_k'][k])))) for k in r['temperature_k'])
            report['max_scored_refinement_difference_k']=max(float(np.max(np.abs(np.asarray(r['temperature_k'][k])[mask]-np.asarray(fine['temperature_k'][k])[mask]))) for k in t.late.WITNESSES)
            report['refined_metrics']=t.late.metrics(fine,obs,meta['event']);report['refined_steps']=fine['steps']
            save(out/(name+'-fitted-scenario.json'),s)
        save(out/(args.mode+'.json'),report)
        print(json.dumps({k:report[k] for k in ('experiment','parameters','metrics')},indent=2),flush=True)
    finally:runner.close()


if __name__=='__main__':main()

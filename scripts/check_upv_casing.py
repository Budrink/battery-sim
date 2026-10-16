#!/usr/bin/env python3
"""Refine the CFL bound as well as dt/source/gas caps; packaging sensitivity is training-only."""
import argparse
import copy
import json
import numpy as np
import calibrate_upv_casing as trial
from calibrate_upv import save, metrics


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=trial.Path,default=trial.OUT)
    p.add_argument('--probe',type=trial.Path,default=trial.Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'))
    p.add_argument('--reverify-heat-fingerprint',action='store_true',help='Update provenance only after exact replay of BOTH frozen heat curves; preserve the original profile')
    args=p.parse_args();dest=args.output/'numerics';dest.mkdir(exist_ok=True);runner=trial.Runner(args.probe,dest)
    reports={}
    try:
        if args.reverify_heat_fingerprint:
            profile_path=args.output/'heat/profile.json';profile=json.loads(profile_path.read_text())
            backup=args.output/'heat/profile-before-coupled-conversion.json'
            if backup.exists():raise ValueError('Provenance backup already exists; do not overwrite')
            for mode,name in [('fit','no-cooling'),('validate','active-cooling')]:
                scene=json.loads((args.output/'heat'/(name+'-fitted-scenario.json')).read_text())
                old=json.loads((args.output/'heat'/(mode+'.json')).read_text())['result'];r=runner.run(scene)
                if r['temperature_k']!=old['temperature_k'] or r['steps']!=old['steps'] or r['applied_objects']!=old['applied_objects']:
                    raise ValueError('Frozen heating changed; cannot renew fingerprint')
            save(backup,profile);profile['previous_fingerprint']=profile['fingerprint'];profile['fingerprint']=trial.fingerprint(args.probe)
            profile['provenance_renewal']='Exact replay of both frozen heat curves, steps and applied materials after conversion-only integrator change; no refit'
            save(profile_path,profile)
            print('Both heating replays identical; original profile retained',flush=True)
        for stage in ('heat','failure'):
            for mode,name in [('fit','no-cooling'),('validate','active-cooling')]:
                path=args.output/stage/(mode+'.json')
                if not path.exists():continue
                old=json.loads(path.read_text());scene=json.loads((args.output/stage/(name+'-fitted-scenario.json')).read_text())
                scene['simulation']['stability_factor']*=.5
                fine=runner.run(scene,.5);base=old['result'];changes={}
                for key,values in base['temperature_k'].items():changes[key]=float(np.max(np.abs(np.asarray(values)-np.asarray(fine['temperature_k'][key]))))
                scored=np.asarray(base['time_s'])>= (60 if stage=='heat' else old['meta']['event']['bracket_width_s']+300)
                selected=max(float(np.max(np.abs(np.asarray(base['temperature_k'][k])[scored]-np.asarray(fine['temperature_k'][k])[scored]))) for k in ('101','102','107','108'))
                report=dict(max_channel_difference_k=changes,max_scored_witness_difference_k=selected,
                    original_steps=base['steps'],refined_steps=fine['steps'],max_thermal_residual_j=fine['max_thermal_residual_j'])
                reports[stage+'/'+name]=report;save(dest/(stage+'-'+name+'.json'),report)
                print(stage,name,report,flush=True)
        heat_p=json.loads((args.output/'heat/profile.json').read_text())['parameters'];base=trial.heat_bake(trial.read_scene('no-cooling'),heat_p)
        data,_,_=trial.load_case(trial.old.DATA,'no-cooling');sens=[]
        for t in (.0001,.0002):
            scene=trial.casing(base,heat_p,thickness=t);r=runner.run(scene)
            sens.append(dict(thickness_m=t,rmse_k=metrics(r,data)['rmse_k'],note='All fitted parameters frozen; measured total capacity and outer dimensions fixed'))
        reports['training_only_packaging_sensitivity']=sens
        save(dest/'assessment.json',reports)
    finally:runner.close()


if __name__=='__main__':main()

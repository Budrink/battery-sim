#!/usr/bin/env python3
"""Choose a small 3-D battery grid by numerical error, never by fitting data.

Frozen material/contact parameters; passive meshes and sensor bindings unchanged.
3^3 and 4^3 are compared with 8^3, checked against 12^3. The old 2x2x2 run
must still reproduce its recorded temperatures before any comparison is accepted.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from calibrate_upv import Worker, metrics, model_hash, save, validate_profile, verify_fixed_properties
from export_upv_state_geometry import calibrated_scene
from generate_upv import ROOT, TARGETS, load_case


def grid_scene(scene, n):
    if not isinstance(n,int) or n<3:raise ValueError('Core/shell requires at least three volumes per axis')
    out=copy.deepcopy(scene)
    for c in out['cells']:
        if c['role']=='battery':
            if 'initial_temperature_field_k' in c['thermal']:raise ValueError('Initial field remapping is not specified')
            c['thermal']['divisions']=[n]*3
            c['thermal']['internal_heat_transfer']='finite_volume'
    out['metadata']['core_shell_grid']=dict(divisions=[n]*3,core_volumes=(n-2)**3,shell_volumes=n**3-(n-2)**3,
                                           definition='One numerical layer on ALL six exterior faces; not a measured casing thickness',
                                           parameters_refitted=False)
    out['metadata']['battery_grid']=[n]*3
    out['metadata']['source_profile_model_sha256']=out['metadata'].pop('model_sha256',None)
    out['metadata']['calibration_status']='Prior contact fit carried unchanged onto a new grid; not recalibrated'
    return out


def compare(run,reference):
    t=np.asarray(run['time_s']);r=np.asarray(reference['time_s'])
    if t.shape!=r.shape or not np.allclose(t,r,rtol=0,atol=1e-8):raise ValueError('Different sample times')
    e={c:np.asarray(run['temperature_k'][c])-reference['temperature_k'][c] for c in TARGETS}
    mask=t>=60;flat=np.concatenate(list(e.values()));scored=np.concatenate([v[mask] for v in e.values()])
    return dict(max_change_k=float(np.max(np.abs(flat))),scored_max_change_k=float(np.max(np.abs(scored))),
                rms_change_k=float(np.sqrt(np.mean(scored**2))),
                per_channel_max_change_k={c:float(np.max(np.abs(v[mask]))) for c,v in e.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--profile',type=Path,default=ROOT/'data/calibration/upv-directed-geometry-fit.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--tolerance-k',type=float,default=.5)
    p.add_argument('--export',type=Path,help='Export the selected no-cooling case for the native viewer')
    a=p.parse_args();profile=json.loads(a.profile.read_text());validate_profile(profile)
    if not 0<a.tolerance_k<=1:raise ValueError('Expected a positive numerical tolerance <=1 K')
    digest=hashlib.sha256(a.profile.read_bytes()).hexdigest()
    result=dict(kind='core-shell-grid-check',parameters_refitted=False,current_model_sha256=model_hash(),
                source_profile_model_sha256=profile['model_sha256'],frozen_profile_sha256=digest,
                probe_sha256=hashlib.sha256(a.probe.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                tolerance_k=a.tolerance_k,reference_tolerance_k=.1,scoring_start_s=60,
                criterion='Maximum surface-temperature difference over the FULL interval versus refined calculation, not error against experiment.',
                caveats=['Fixed-property pre-event grid check only; repeat after internal-law calibration.',
                         'Core/shell tags are resolution-dependent numerical regions, not physical casing material.',
                         'Fine reference changes batteries only; this is not full-domain convergence.',
                         'C1/C3 acquisition-numbering hypothesis remains unconfirmed.'],experiments={})
    for experiment,stage in (('no-cooling','calibration'),('active-cooling','validation')):
        old=json.loads((ROOT/f'data/calibration/upv-directed-geometry-{stage}.json').read_text())
        if old['frozen_profile_sha256']!=digest or old['parameters']!=profile['parameters']:raise ValueError('Frozen fit mismatch')
        path=Path(old['scenario_path']);path=path if path.is_absolute() else ROOT/path
        scene=calibrated_scene(json.loads(path.read_text()),profile)
        observed,record,_=load_case(ROOT/'data/experiments/upv',experiment)
        if record['canonical_sha256']!=old['source_csv_sha256']:raise ValueError('Data changed')
        baseline=a.output_dir/f'{experiment}-legacy-scenario.json';save(baseline,scene)
        worker=Worker(a.probe,baseline)
        try:baseline_run=worker.run({});verify_fixed_properties(baseline_run,scene)
        finally:worker.close()
        if compare(baseline_run,old['result'])['max_change_k']>1e-7:raise ValueError('Legacy run changed; cannot isolate grid effect')
        cases={};runs={}
        for n in (3,4,8,12):
            print(f'{experiment}: {n}x{n}x{n}',flush=True)
            s=grid_scene(scene,n);path=a.output_dir/f'{experiment}-{n}x{n}x{n}-scenario.json';save(path,s)
            w=Worker(a.probe,path)
            try:
                run=w.run({});verify_fixed_properties(run,s)
                cap=min(s['simulation']['max_dt_s']/2,run['time_s'][-1]/run['steps']/2)
                finer=w.run({'max_dt_s':cap});verify_fixed_properties(finer,s)
                if finer['steps']<=run['steps']:raise ValueError('Timestep check did not refine actual steps')
            finally:w.close()
            case=dict(divisions=[n]*3,core_count=(n-2)**3,shell_count=n**3-(n-2)**3,metrics=metrics(run,observed),
                      time_step_check=dict(**compare(finer,run),max_dt_s=cap,baseline_steps=run['steps'],refined_steps=finer['steps']),
                      scenario_path=str(path),scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),result=run)
            save(a.output_dir/f'{experiment}-{n}x{n}x{n}.json',case)
            runs[n]=run;cases[str(n)]={k:v for k,v in case.items() if k!='result'}
            print(json.dumps(dict(grid=n,rmse_k=case['metrics']['rmse_k'],time_step_check=case['time_step_check'])),flush=True)
        for n in (3,4,8):cases[str(n)]['difference_from_12']=compare(runs[n],runs[12])
        for n in (3,4):cases[str(n)]['difference_from_8']=compare(runs[n],runs[8])
        result['experiments'][experiment]=dict(cases=cases,baseline_reproduction=compare(baseline_run,old['result']))
        save(a.output_dir/'summary.json',result)
    qualified=[]
    for n in (3,4):
        if all(ex['cases'][str(n)]['difference_from_12']['max_change_k']<=a.tolerance_k and
               ex['cases']['8']['difference_from_12']['max_change_k']<=result['reference_tolerance_k'] and
               max(case['time_step_check']['max_change_k'] for case in ex['cases'].values())<=.05
               for ex in result['experiments'].values()):qualified.append(n)
    result['selected_divisions']=[qualified[0]]*3 if qualified else None
    save(a.output_dir/'summary.json',result)
    if a.export:
        if not qualified:raise ValueError('Neither grid passed; refusing to export a selected case')
        case=result['experiments']['no-cooling']['cases'][str(qualified[0])]
        chosen=json.loads(Path(case['scenario_path']).read_text())
        chosen['metadata'].update(current_model_sha256=result['current_model_sha256'],
                                  frozen_profile_sha256=digest,
                                  grid_check_summary_sha256=hashlib.sha256((a.output_dir/'summary.json').read_bytes()).hexdigest())
        chosen['name']+='; core-shell grid '+str(qualified[0])+'^3'
        save(a.export,chosen)
    print('Selected:',result['selected_divisions'],'; summary:',a.output_dir/'summary.json')


if __name__=='__main__':main()

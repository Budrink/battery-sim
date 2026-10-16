#!/usr/bin/env python3
"""Frozen UPV battery-depth sweep, not parameter fitting or model selection.

Only battery x divisions / the separate isothermal closure change. All contacts,
bulk k, capacities, passive grids, forcing and observer bindings are preserved.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from calibrate_upv import Worker, metrics, model_hash, save, validate_profile, verify_fixed_properties
from export_upv_state_geometry import calibrated_scene
from generate_upv import ROOT, TARGETS, load_case


def variant(scene, nx, isothermal=False):
    if nx < 2 or nx % 2:
        raise ValueError('An even number of depth slabs >=2 is required')
    out=copy.deepcopy(scene)
    for cell in out['cells']:
        if cell['role']=='battery':
            cell['thermal']['divisions'][0]=nx
            cell['thermal']['internal_heat_transfer']='isothermal' if isothermal else 'finite_volume'
    out['metadata']['mesh_diagnostic']=dict(depth_slabs=nx,isothermal=isothermal,parameters_refitted=False)
    return out


def difference(run, baseline):
    if run['time_s']!=baseline['time_s']:raise ValueError('Inconsistent sample times')
    errors={c:np.asarray(run['temperature_k'][c])-baseline['temperature_k'][c] for c in TARGETS}
    flat=np.concatenate(list(errors.values()))
    scored=np.concatenate([e[np.asarray(run['time_s'])>=60] for e in errors.values()])
    return dict(max_change_k=float(np.max(np.abs(flat))),rms_change_k=float(np.sqrt(np.mean(flat**2))),
                scored_max_change_k=float(np.max(np.abs(scored))),
                per_channel_max_change_k={c:float(np.max(np.abs(e))) for c,e in errors.items()})


def surface_summary(run, observed):
    times=run['time_s'];out={}
    for cell,iss,tim in [('C1','101','102'),('C3','108','107')]:
        simulated=np.asarray(run['temperature_k'][iss])-run['temperature_k'][tim]
        measured=np.interp(times,observed['time_s'],observed['T'+iss+'_k'])-np.interp(times,observed['time_s'],observed['T'+tim+'_k'])
        out[cell]=dict(convention='ISS face minus TIM face',model_final_delta_k=float(simulated[-1]),
                       observed_final_delta_k=float(measured[-1]),
                       final_face_errors_k={c:float(run['temperature_k'][c][-1]-np.interp(times[-1],observed['time_s'],observed['T'+c+'_k'])) for c in (iss,tim)})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--profile-probe',type=Path,required=True)
    p.add_argument('--profile',type=Path,default=ROOT/'data/calibration/upv-directed-geometry-fit.json')
    p.add_argument('--calibration-report',type=Path,default=ROOT/'data/calibration/upv-directed-geometry-calibration.json')
    p.add_argument('--validation-report',type=Path,default=ROOT/'data/calibration/upv-directed-geometry-validation.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--depths',type=int,nargs='+',default=[2,4,8])
    a=p.parse_args();profile=json.loads(a.profile.read_text());validate_profile(profile)
    digest=hashlib.sha256(a.profile.read_bytes()).hexdigest()
    if profile['model_sha256']!=model_hash():raise ValueError('Core model changed since fitting')
    if profile['probe_sha256']!=hashlib.sha256(a.probe.read_bytes()).hexdigest():raise ValueError('Observer binary changed since fitting')
    if not a.depths or a.depths[0]!=2 or a.depths!=sorted(set(a.depths)):raise ValueError('Depth sequence must increase, starting at 2')
    for n in a.depths:
        if n<2 or n%2:raise ValueError('Even depth required')
    output=dict(kind='frozen_battery_mesh_sensitivity',parameters_refitted=False,model_sha256=model_hash(),
                frozen_profile_sha256=digest,diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                profile_probe_sha256=hashlib.sha256(a.profile_probe.read_bytes()).hexdigest(),
                topology_status=profile['protocol']['topology_status'],
                caveats=['Isothermal is a different physical closure, not a mesh refinement.',
                         'The calibrated coefficients were originally fitted on two depth slabs; no mesh is selected by holdout error.',
                         'Directed contact response uses adjacent volume states: changing depth resolution also changes its state samples, not its coefficients.',
                         'Temperatures are reconstructed face means; exact thermocouple coordinates remain unknown.',
                         'No runaway or degradation validation.'],experiments={})
    for path in (a.calibration_report,a.validation_report):
        report=json.loads(path.read_text());experiment=report['experiment']
        if report['model_sha256']!=model_hash() or report['frozen_profile_sha256']!=digest:raise ValueError('Report provenance mismatch')
        if report['parameters']!=profile['parameters']:raise ValueError('Parameters changed')
        scene_path=Path(report['scenario_path']);scene_path=scene_path if scene_path.is_absolute() else ROOT/scene_path
        baked=calibrated_scene(json.loads(scene_path.read_text()),profile)
        observed,record,_=load_case(ROOT/'data/experiments/upv',experiment)
        if record['canonical_sha256']!=report['source_csv_sha256']:raise ValueError('Data changed')
        cases={};runs={};numerics={}
        settings=[(f'nx{n}',n,False) for n in a.depths]+[('isothermal',2,True)]
        for label,nx,isothermal in settings:
            print(f'{experiment}: {label}',flush=True)
            scene=variant(baked,nx,isothermal)
            path=a.output_dir/f'{experiment}-{label}-scenario.json';save(path,scene)
            worker=Worker(a.probe,path)
            try:
                run=worker.run({});verify_fixed_properties(run,scene)
                if label in ('nx8',f'nx{a.depths[-1]}','isothermal'):
                    # A CFL-limited run can ignore a halved configured ceiling.
                    # Force a smaller actual step and verify the step count grows.
                    cap=min(scene['simulation']['max_dt_s']/2,run['time_s'][-1]/run['steps']/2)
                    finer=worker.run({'max_dt_s':cap})
                    verify_fixed_properties(finer,scene)
                    if finer['steps']<=run['steps']:raise ValueError('Time refinement did not reduce actual steps')
                    numerics[label]=dict(**difference(finer,run),metrics=metrics(finer,observed),
                                         baseline_steps=run['steps'],refined_steps=finer['steps'],max_dt_s=cap)
            finally:worker.close()
            if label=='nx2' and difference(run,report['result'])['max_change_k']>1e-7:
                raise ValueError('Baked baseline no longer reproduces frozen fit')
            native=subprocess.run([str(a.profile_probe.resolve()),str(path.resolve())],check=True,capture_output=True,text=True)
            depth=json.loads(native.stdout)
            if depth['time_s']!=run['time_s'] or depth['volumes']!=run['volumes']:raise ValueError('Profile and observer integration differ')
            if abs(depth['energy_residual_j']-run['energy_residual_j'])>1e-7:raise ValueError('Integration energy mismatch')
            case=dict(metrics=metrics(run,observed),difference_from_nx2=difference(run,runs.get('nx2',run)),
                      face_summary=surface_summary(run,observed),scenario_path=str(path),
                      scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),result=run,depth_profile=depth)
            save(a.output_dir/f'{experiment}-{label}.json',case)
            runs[label]=run;cases[label]={k:v for k,v in case.items() if k not in ('result','depth_profile')}
            print(json.dumps(dict(case=label,metrics=case['metrics'],face_summary=case['face_summary'])),flush=True)
        successive={f'nx{x}_to_nx{y}':difference(runs[f'nx{y}'],runs[f'nx{x}']) for x,y in zip(a.depths,a.depths[1:])}
        output['experiments'][experiment]=dict(cases=cases,successive_mesh_changes=successive,time_step_checks=numerics,
                                               source_csv_sha256=record['canonical_sha256'])
        save(a.output_dir/'summary.json',output)
    print('Summary:',a.output_dir/'summary.json')


if __name__=='__main__':main()

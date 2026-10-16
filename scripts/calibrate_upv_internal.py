#!/usr/bin/env python3
"""Two-parameter internal-edge trial on 3^3/4^3/8^3, with frozen external contacts.

Fit opens no-cooling only. Validate reads the frozen profile before opening
active-cooling. Python optimizes/scores; integration is exclusively native CUDA.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from calibrate_upv import Worker, metrics, model_hash, residuals, save, verify_fixed_properties
from check_upv_core_shell import compare
from generate_upv import ROOT, load_case

NAMES=('internal_linear','internal_quadratic')
LOW,HIGH=.2,3.
ZREF=math.log((1.-LOW)/(HIGH-1.))
PROTOCOL=dict(revision='internal-edges-3x3x3-v1',calibration='no-cooling',validation='active-cooling',
              grid=[3,3,3],fit_start_s=60.,coefficient_bounds=[-12.,12.],factor_bounds=[LOW,HIGH],
              reference_temperature_k=298.15,temperature_scale_k=100.,
              law='G=Gref*(0.2+2.8*sigmoid(log(0.8/2)+a*theta_donor+b*theta_donor^2)); theta=(Tdonor-298.15)/100',
              scope='Internal numerical edges only, not a measured bulk k(T); surface half-volume resistances remain fixed.',
              fixed='External contact laws, k_ref anisotropy, ALL heat capacities, geometry, boundaries, sensor mapping',
              caveat='Pre-event only. No damage/destruction identification. C1/C3 hypothesis unconfirmed; validation previously inspected.')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid_protocol(grid):
    if type(grid) is not int or grid not in (3,4,8):raise ValueError('Internal fit grid must be 3, 4 or 8')
    protocol=copy.deepcopy(PROTOCOL)
    protocol.update(revision=f'internal-edges-{grid}x{grid}x{grid}-v1',grid=[grid]*3)
    return protocol


def internal_scene(base,grid=3):
    protocol=grid_protocol(grid)
    scene=copy.deepcopy(base)
    if scene['metadata']['connectivity_model']!='directed_geometry':raise ValueError('Frozen directed scene required')
    for c in scene['cells']:
        if c['role']!='battery':continue
        if c['thermal']['divisions']!=[grid]*3:raise ValueError(f'Fit requires {grid}x{grid}x{grid} batteries')
        if 'bulk_connectivity' in c['thermal']:raise ValueError('Baseline must have constant internal edges')
        c['thermal']['bulk_connectivity']=dict(kind='directed_geometry',l0_m=1e-4,activation_width_m=1e-5,
            geometry_reference_logit=ZREF,background_factor=LOW,active_factor=HIGH-LOW,
            destroyed_l0_m=2e-4,destroyed_l_m=2e-4,destroyed_background_factor=1.,destroyed_active_factor=0.)
        c['thermal']['bulk_response']=dict(reference_temperature_k=298.15,temperature_scale_k=100.,
            linear=[0.]*3,quadratic=[0.]*3,damage=[0.]*3,temperature_contrast=[0.]*3,damage_contrast=[0.]*3)
    scene['metadata'].update(internal_calibration=True,internal_protocol=protocol,
        calibration_status='Internal-edge calibration trial; prior external contacts frozen')
    # Historical hashes are retained as provenance, not claims about this model.
    scene['metadata'].pop('current_model_sha256',None)
    return scene


def export_scene(scene,params):
    if set(params)!=set(NAMES) or any(not math.isfinite(v) or abs(v)>12 for v in params.values()):
        raise ValueError('Two finite internal coefficients in [-12,12] required')
    out=copy.deepcopy(scene)
    for c in out['cells']:
        if c['role']=='battery':
            c['thermal']['bulk_response']['linear']=[params['internal_linear']]*3
            c['thermal']['bulk_response']['quadratic']=[params['internal_quadratic']]*3
    out['metadata']['internal_parameters']=params
    return out


def verify_internal(run,scene,params,baseline_audit):
    verify_fixed_properties(run,scene)
    audit=copy.deepcopy(run['applied_model'])
    for c in audit['objects']:
        if c['material']=='battery':
            if not c['has_bulk_response']:raise ValueError('Missing independent internal response')
            for k in NAMES:
                if c[k]!=params[k]:raise ValueError('Native coefficient not applied: '+k)
                c[k]=0.
    if audit!=baseline_audit:raise ValueError('External contact/material audit changed during internal fit')
    if abs(run['energy_residual_j'])>1e-6:raise ValueError('Energy balance failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('fit','validate'))
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--grid',type=int,choices=(3,4,8),default=3)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--max-evaluations',type=int,default=80)
    p.add_argument('--numerical-check',action='store_true')
    a=p.parse_args();grid=a.grid;grid_name=f'{grid}x{grid}x{grid}';protocol=grid_protocol(grid)
    out=a.output_dir or ROOT/f'data/calibration/upv-internal-{grid_name}';profile_path=out/'profile.json'
    provenance=dict(model_sha256=model_hash(),probe_sha256=sha(a.probe),script_sha256=sha(__file__),protocol=protocol)
    if a.mode=='validate':
        profile=json.loads(profile_path.read_text())
        for k,v in provenance.items():
            if profile[k]!=v:raise ValueError('Changed since fitting: '+k)
        if not profile['optimizer_success']:raise ValueError('Fit not converged; validation refused')
    experiment='no-cooling' if a.mode=='fit' else 'active-cooling'
    # Previously checked baked scenes differ in battery resolution only.
    base_path=ROOT/f'data/calibration/upv-core-shell-check/{experiment}-{grid_name}-scenario.json'
    base=json.loads(base_path.read_text());observed,record,manifest=load_case(ROOT/'data/experiments/upv',experiment)
    if record['split']!=('calibration' if a.mode=='fit' else 'validation'):raise ValueError('Split violation')
    if base['metadata']['canonical_sha256']!=record['canonical_sha256']:raise ValueError('Baseline data changed')
    if a.mode=='validate':
        if base['metadata']['applied_parameters']!=profile['frozen_external_parameters']:raise ValueError('External fit changed')
        if manifest['archive_sha256']!=profile['archive_sha256']:raise ValueError('Dataset version changed')
        if sha(ROOT/f'data/calibration/upv-core-shell-check/no-cooling-{grid_name}-scenario.json')!=profile['baseline_scenario_sha256']:
            raise ValueError('Calibration baseline changed')
    scene=internal_scene(base,grid);path=out/f'{experiment}-scenario.json';save(path,scene)
    w=Worker(a.probe,path)
    try:
        neutral=w.run({});verify_fixed_properties(neutral,scene)
        baseline_audit=copy.deepcopy(neutral['applied_model'])
        legacy=json.loads((ROOT/f'data/calibration/upv-core-shell-check/{experiment}-{grid_name}.json').read_text())['result']
        neutral_difference=compare(neutral,legacy)
        # The bounded dynamic law can tighten CFL even at a=b=0. Check the
        # physics at a common actual timestep, not against a different integrator
        # schedule; separately require the old constant model to reproduce.
        common_cap=min(scene['simulation']['max_dt_s']/2,neutral['time_s'][-1]/neutral['steps']/2,
                       legacy['time_s'][-1]/legacy['steps']/2)
        fixed_worker=Worker(a.probe,base_path)
        try:
            fixed_run=fixed_worker.run({});verify_fixed_properties(fixed_run,base)
            baseline_reproduction=compare(fixed_run,legacy)
            if baseline_reproduction['max_change_k']>1e-7:raise ValueError('Constant baseline changed')
            fixed_common=fixed_worker.run({'max_dt_s':common_cap})
            neutral_common=w.run({'max_dt_s':common_cap})
            verify_fixed_properties(fixed_common,base);verify_fixed_properties(neutral_common,scene)
            neutral_common_check=compare(neutral_common,fixed_common)
            if neutral_common['steps']!=fixed_common['steps'] or neutral_common_check['max_change_k']>1e-7:
                raise ValueError('Neutral internal law changed physics at common dt')
        finally:fixed_worker.close()
        prior=metrics(neutral,observed)
        if a.mode=='fit':
            from scipy.optimize import least_squares
            history=[];cache={}
            def objective(x):
                key=tuple(x)
                if key not in cache:
                    params=dict(zip(NAMES,map(float,x)));run=w.run(params);verify_internal(run,scene,params,baseline_audit)
                    r=np.concatenate(list(residuals(run,observed).values()));cache[key]=r
                    history.append(dict(parameters=params,rmse_k=float(np.sqrt(np.mean(r*r)))))
                    if len(history)%10==1:print('candidate',len(history),history[-1],flush=True)
                return cache[key]
            def jacobian(x):
                columns=[]
                for i in range(2):
                    lo=x.copy();hi=x.copy();lo[i]=max(-12.,x[i]-.002);hi[i]=min(12.,x[i]+.002)
                    columns.append((objective(hi)-objective(lo))/(hi[i]-lo[i]))
                return np.stack(columns,axis=1)
            fits=[least_squares(objective,start,jac=jacobian,bounds=(-12.,12.),max_nfev=a.max_evaluations,
                                ftol=1e-7,xtol=1e-6,gtol=1e-5) for start in (np.zeros(2),np.array([-6.,6.]))]
            selected=min([f for f in fits if f.success] or fits,key=lambda f:np.dot(f.fun,f.fun))
            params=dict(zip(NAMES,map(float,selected.x)));run=w.run(params);verify_internal(run,scene,params,baseline_audit)
            sv=np.linalg.svd(selected.jac,compute_uv=False)
            profile=dict(**provenance,parameters=params,optimizer_success=bool(selected.success),
                optimizer_message=selected.message,optimizer_optimality=float(selected.optimality),
                local_jacobian_condition_number=float(sv[0]/sv[-1]) if sv[-1]>0 else None,
                near_bounds=[k for k,v in params.items() if abs(v)>11.99],
                starts=[dict(parameters=dict(zip(NAMES,map(float,f.x))),success=bool(f.success),
                             rmse_k=float(np.sqrt(np.mean(f.fun*f.fun))),message=f.message) for f in fits],
                fit_metrics=metrics(run,observed),prior_metrics=prior,validation_used=False,
                calibration_csv_sha256=record['canonical_sha256'],archive_sha256=manifest['archive_sha256'],
                baseline_scenario_sha256=sha(base_path),frozen_external_parameters=base['metadata']['applied_parameters'],
                neutral_difference=neutral_difference,baseline_reproduction=baseline_reproduction,
                neutral_common_dt=dict(**neutral_common_check,max_dt_s=common_cap,steps=neutral_common['steps']),
                search=dict(starts=[[0.,0.],[-6.,6.]],max_nfev_per_start=a.max_evaluations,
                            absolute_jacobian_step=.002,ftol=1e-7,xtol=1e-6,gtol=1e-5),evaluation_history=history)
            save(profile_path,profile)
        else:
            params=profile['parameters'];run=w.run(params);verify_internal(run,scene,params,baseline_audit)
        report=dict(kind=a.mode,experiment=experiment,protocol=protocol,parameters=params,metrics=metrics(run,observed),
                    baseline_metrics=prior,neutral_difference=neutral_difference,result=run,
                    baseline_reproduction=baseline_reproduction,
                    neutral_common_dt=dict(**neutral_common_check,max_dt_s=common_cap,steps=neutral_common['steps']),
                    topology=base['metadata']['topology'],source_csv_sha256=record['canonical_sha256'],
                    model_sha256=model_hash(),frozen_profile_sha256=sha(profile_path),scenario_path=str(path))
        baked=export_scene(scene,params);export_path=out/f'{experiment}-fitted-scenario.json';save(export_path,baked)
        exported=Worker(a.probe,export_path)
        try:
            export_run=exported.run({});verify_internal(export_run,baked,params,baseline_audit)
            report['export_equivalence']=compare(export_run,run)
            if report['export_equivalence']['max_change_k']>1e-9:raise ValueError('Native exported scene disagrees with fitted run')
        finally:exported.close()
        if a.numerical_check:
            cap=min(scene['simulation']['max_dt_s']/2,run['time_s'][-1]/run['steps']/2)
            fine=w.run(dict(params,max_dt_s=cap));verify_internal(fine,scene,params,baseline_audit)
            if fine['steps']<=run['steps']:raise ValueError('No actual time-step refinement')
            report['numerical_checks']={'time_step':dict(**compare(fine,run),max_dt_s=cap,
                baseline_steps=run['steps'],refined_steps=fine['steps']), 'battery_grid':{}}
            refined_runs={grid:run}
            for n in sorted({4,8,12}-{grid}):
                refined=copy.deepcopy(baked)
                for c in refined['cells']:
                    if c['role']=='battery':c['thermal']['divisions']=[n]*3
                refined['metadata']['battery_grid']=[n]*3
                refined['metadata']['core_shell_grid'].update(divisions=[n]*3,core_volumes=(n-2)**3,shell_volumes=n**3-(n-2)**3)
                refined['metadata']['evaluation_grid']=[n]*3
                refined['metadata']['internal_fit_grid']=[grid]*3
                refined_path=out/f'{experiment}-{n}x{n}x{n}-scenario.json';save(refined_path,refined)
                worker=Worker(a.probe,refined_path)
                try:
                    other=worker.run({});verify_internal(other,refined,params,baseline_audit)
                    refined_runs[n]=other
                    report['numerical_checks']['battery_grid'][str(n)]=compare(run,other)
                    save(out/f'{experiment}-{n}x{n}x{n}.json',dict(result=other,metrics=metrics(other,observed)))
                finally:worker.close()
            report['numerical_checks']['reference_8_vs_12']=compare(refined_runs[8],refined_runs[12])
            report['numerical_checks']['grid_passed']=report['numerical_checks']['battery_grid']['12']['max_change_k']<=.5
            if grid==3:report['numerical_checks']['grid_3_passed']=report['numerical_checks']['grid_passed']
        save(out/f'{a.mode}.json',report)
        print(json.dumps(dict(experiment=experiment,parameters=params,baseline=prior,metrics=report['metrics'],
                              numerical_checks=report.get('numerical_checks')),indent=2),flush=True)
    finally:w.close()


if __name__=='__main__':main()

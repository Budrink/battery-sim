#!/usr/bin/env python3
"""Fit CUDA contact parameters on UPV no-cooling; evaluate a frozen holdout.

Python reads observations, chooses parameters, and scores results. It does not
integrate heat flow. 'fit' cannot open active-cooling observations or scenarios.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from generate_upv import ROOT, TARGETS, generate, load_case

MODEL_FILES=['scripts/generate_upv.py','scripts/calibrate_upv.py','native/src/cuda_solver.cu',
             'native/src/thermal_mesh.cpp','native/src/scenario.cpp','native/src/simulation.cpp',
             'native/tools/experiment_probe.cpp','native/include/battery_native/model.hpp',
             'native/include/battery_native/state_geometry.hpp','native/include/battery_native/constitutive.hpp','native/include/battery_native/thermal_mesh.hpp',
             'native/include/battery_native/cuda_solver.hpp','native/include/battery_native/simulation.hpp',
             'native/include/battery_native/failure.hpp','native/include/battery_native/reaction.hpp']
PROTOCOL=dict(scope='conditional pre-event heat transfer; NOT runaway/aging validation',
              revision='directed-geometry-v6',topology='swapped-witnesses',
              topology_status='UNCONFIRMED acquisition-numbering hypothesis; not a verified correction to the source',
              calibration='no-cooling',validation='active-cooling',
              fit_start_s=60,output_interval_s=2,acceptance_rmse_k=5,
              acceptance_note='Provisional engineering gate, NOT instrument uncertainty or a certification criterion',
              fitted_parameters=['material-pair contact h at 298.15 K','effective environment h',
                                 'directed_geometry: shared sender polynomial plus five material-specific signed temperature contrasts'],
              fixed_parameters=['ALL measured battery properties: conductivity tensor and heat capacity',
                                'ALL bulk conductivities, including ISS=0.145 and TIM=2 W/m/K',
                                'directed/state geometry: ISS cp=770, TIM cp=1175 J/kg/K working assumptions',
                                'Al/steel cp priors','layer thicknesses and densities','cooling-tip boundary conductance prior'],
              validation_note='Excluded from parameter fitting; previously inspected during debugging, not a pristine blind dataset',
              not_identified=['absolute l0','out-of-window hot l(T,D)','damage response and destruction jump',
                              'degradation kinetics','runaway source','gas heat transfer'])
CONTACTS=('battery_iss_scale','battery_tim_scale','iss_aluminium_scale','iss_steel_scale','tim_aluminium_scale')
CONTRASTS=tuple(m+'_temperature_contrast' for m in ('battery','iss','tim','aluminium','steel'))
PARAMETER_SETS={'shared':('contact_scale','ambient_scale'),
                'contacts':CONTACTS+('ambient_scale',),
                'pads':CONTACTS+('ambient_scale','iss_cp_scale','tim_cp_scale'),
                'state_geometry':CONTACTS+('ambient_scale','surface_linear','surface_quadratic'),
                'directed_geometry':CONTACTS+('ambient_scale','surface_linear','surface_quadratic')+CONTRASTS}
SIGNED=('surface_linear','surface_quadratic')+CONTRASTS
PARAMETERS=tuple(dict.fromkeys(k for names in PARAMETER_SETS.values() for k in names))
# Exploratory engineering bounds, NOT confidence intervals of measured properties.
BOUNDS={**{k:(.01,100.) for k in ('contact_scale',)+CONTACTS},'ambient_scale':(.02,6.),
        'iss_cp_scale':(.5,2.),'tim_cp_scale':(.5,2.),**{k:(-12.,12.) for k in SIGNED}}


def parameter_bounds(names, dynamic=False):
    return np.array([(.02,99.) if dynamic and k in CONTACTS else BOUNDS[k] for k in names]).T


def transform(values,names,inverse=False):
    # Temperature response may open OR close contacts, and be nonmonotonic.
    return np.array([v if k in SIGNED else (math.exp(v) if inverse else math.log(v)) for k,v in zip(names,values)])


def validate_parameters(params):
    if set(params)-set(PARAMETERS):
        raise ValueError('Unsupported calibration parameters; measured battery properties are fixed')
    if any(not math.isfinite(v) or (k not in SIGNED and v<=0) or (k in SIGNED and abs(v)>12) for k,v in params.items()):
        raise ValueError('Positive finite calibration scales required')
    if any(k in params for k in SIGNED) and any(k in params for k in ('iss_cp_scale','tim_cp_scale')):
        raise ValueError('Pad heat capacities are fixed during state geometry calibration')
    if 'contact_scale' in params and any(k in params for k in CONTACTS):
        raise ValueError('Cannot combine shared and material-pair contact scales')


def validate_profile(profile):
    if profile.get('status')=='rejected':raise ValueError('Rejected calibration profile: '+profile['rejection_reason'])
    validate_parameters(profile['parameters'])


def verify_fixed_properties(run,scene):
    """Fail the fit if the actual native model changed any known material value."""
    applied={c['id']:c for c in run['applied_model']['objects']}
    for c in scene['cells']:
        actual=applied[c['id']];original=c['thermal']
        if actual['conductivity_w_per_m_k']!=original['conductivity_w_per_m_k']:
            raise ValueError('Bulk conductivity changed during contact calibration: '+c['id'])
        if scene.get('metadata',{}).get('fixed_pad_cp') or c['material_id'] not in ('ISS','TIM') or c['role']=='battery':
            if actual['heat_capacity_j_per_k']!=original['heat_capacity_j_per_k']:
                raise ValueError('Fixed heat capacity changed during contact calibration: '+c['id'])


def model_hash():
    digest=hashlib.sha256()
    for name in MODEL_FILES:
        digest.update(name.encode());digest.update((ROOT/name).read_bytes())
    return digest.hexdigest()


class Worker:
    def __init__(self, binary, scenario):
        self.process=subprocess.Popen([str(binary.resolve()),str(scenario.resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def run(self, params):
        p=self.process;p.stdin.write(json.dumps(params)+'\n');p.stdin.flush()
        line=p.stdout.readline()
        if not line:raise RuntimeError('CUDA probe exited without a result')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def close(self):
        self.process.stdin.close();self.process.stdout.close()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait()


def residuals(run, observed):
    times=np.asarray(run['time_s']);mask=times>=PROTOCOL['fit_start_s']
    errors={}
    for channel in TARGETS:
        expected=np.interp(times,observed['time_s'],observed[f'T{channel}_k'])
        e=np.asarray(run['temperature_k'][channel])-expected
        if not np.isfinite(e[mask]).all():raise ValueError('Invalid target in scoring window; review data, do not silently drop it')
        errors[channel]=e[mask]
    return errors


def metrics(run,observed):
    errors=residuals(run,observed);all_errors=np.concatenate(list(errors.values()))
    result=dict(rmse_k=float(np.sqrt(np.mean(all_errors**2))),max_abs_error_k=float(np.max(np.abs(all_errors))),
                per_channel={k:dict(rmse_k=float(np.sqrt(np.mean(e**2))),bias_k=float(e.mean()),max_abs_error_k=float(np.max(np.abs(e)))) for k,e in errors.items()},
                energy_residual_j=run['energy_residual_j'],wall_s=run['wall_s'],volumes=run['volumes'],
                duration_s=run['time_s'][-1],evaluated_values=len(all_errors))
    result['engineering_gate_passed']=result['rmse_k']<=PROTOCOL['acceptance_rmse_k']
    return result


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['fit','validate'])
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/experiments/upv')
    p.add_argument('--profile',type=Path,default=ROOT/'data/calibration/upv-directed-geometry-fit.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-evaluations',type=int,default=100)
    p.add_argument('--parameter-set',choices=tuple(PARAMETER_SETS),default='directed_geometry',help='Battery properties and all bulk conductivities are fixed')
    p.add_argument('--starts',type=int,choices=(1,2),default=2)
    p.add_argument('--numerical-check',action='store_true')
    a=p.parse_args()
    experiment='no-cooling' if a.mode=='fit' else 'active-cooling'
    if a.mode=='validate':
        profile=json.loads(a.profile.read_text())
        validate_profile(profile)
        if profile['model_sha256']!=model_hash():raise ValueError('Model changed after fitting; freeze/recalibrate before validation')
        if profile['protocol']!=PROTOCOL:raise ValueError('Protocol changed after fitting')
        if profile['probe_sha256']!=hashlib.sha256(a.probe.read_bytes()).hexdigest():raise ValueError('CUDA executable changed after fitting')
        if not profile['optimizer_success']:raise ValueError('Fit did not converge; cannot label this a frozen calibrated holdout')
    observed,record,manifest=load_case(a.data,experiment)
    if record['split']!=('calibration' if a.mode=='fit' else 'validation'):raise ValueError('Split violation')
    family=a.parameter_set if a.mode=='fit' else profile['parameter_set']
    dynamic=family in ('state_geometry','directed_geometry')
    scene=generate(a.data,experiment,PROTOCOL['topology'],family if dynamic else 'constant')
    # Keep v1 reports/scenarios intact; the new snapshot lives beside its report.
    scene_path=a.output.with_name(a.output.stem+'-scenario.json');save(scene_path,scene)
    worker=Worker(a.probe,scene_path)
    try:
        if a.mode=='fit':
            from scipy.optimize import least_squares
            evaluations=[]
            names=PARAMETER_SETS[a.parameter_set]
            bounds=parameter_bounds(names,dynamic)
            optimizer_bounds=np.array([transform(row,names) for row in bounds])
            prior_run=worker.run({});prior_score=metrics(prior_run,observed)
            def unpack(x):return dict(zip(names,map(float,transform(x,names,inverse=True))))
            def objective(x):
                params=unpack(x)
                run=worker.run(params);verify_fixed_properties(run,scene);score=metrics(run,observed)
                evaluations.append(dict(parameters=params,rmse_k=score['rmse_k']))
                if len(evaluations)%10==1:print(f"candidate {len(evaluations)}: RMSE={score['rmse_k']:.6f} K; {params}",flush=True)
                return np.concatenate(list(residuals(run,observed).values()))
            def jacobian(x):
                # Absolute log-parameter step avoids a near-zero relative step at scale=1.
                columns=[];lo,hi=optimizer_bounds;step=.002
                for i in range(len(x)):
                    low=x.copy();high=x.copy()
                    low[i]=max(lo[i],x[i]-step);high[i]=min(hi[i],x[i]+step)
                    columns.append((objective(high)-objective(low))/(high[i]-low[i]))
                return np.stack(columns,axis=1)
            fits=[]
            starts=[{k:(0. if k in SIGNED else .5 if k=='ambient_scale' else .9 if '_cp_' in k else 5.) for k in names},
                    {k:(-2. if k=='surface_linear' else 2. if k=='surface_quadratic' else (-2. if k.startswith('battery') else 2.) if k in CONTRASTS else 3. if k=='ambient_scale' else 1.1 if '_cp_' in k else 80.) for k in names}]
            for start in starts[:a.starts]:
                fits.append(least_squares(objective,transform([start[k] for k in names],names),jac=jacobian,
                                          bounds=optimizer_bounds,max_nfev=a.max_evaluations,
                                          ftol=1e-6,xtol=1e-5,gtol=1e-4))
            converged=[f for f in fits if f.success]
            fit=min(converged or fits,key=lambda f:float(np.dot(f.fun,f.fun)))
            params=unpack(fit.x);run=worker.run(params);verify_fixed_properties(run,scene);score=metrics(run,observed)
            singular=np.linalg.svd(fit.jac,compute_uv=False)
            condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
            profile=dict(protocol=PROTOCOL,model_sha256=model_hash(),archive_sha256=manifest['archive_sha256'],
                         probe_sha256=hashlib.sha256(a.probe.read_bytes()).hexdigest(),parameter_set=a.parameter_set,
                         calibration_csv_sha256=record['canonical_sha256'],parameters=params,
                         contact_h_w_m2_k={k.removesuffix('_scale'):params.get(k,params.get('contact_scale',1.))*100 for k in CONTACTS},
                         environment_h_w_m2_k=params['ambient_scale']*5,
                         pad_specific_heat_j_kg_k=(scene['metadata']['pad_specific_heat_j_kg_k'] if dynamic else
                                                  {m:1000*params.get(m.lower()+'_cp_scale',1.) for m in ('ISS','TIM')}),
                         connectivity_model=family if dynamic else 'constant',
                         geometry_note='Directed v6: z=z0+a*theta_sender+b*theta_sender^2+c_material_sender*(Tsender-Treceiver)/100. No mean of directions; hotter-donor branch selected. Legacy state_geometry remains symmetric. No bulk k/C changed.',
                         fixed_battery_properties=dict(conductivity_w_per_m_k=[.9,28.3,28.3],
                                                       specific_heat_j_kg_k=1007.,mass_kg=.410),
                         bounds={k:list(bounds[:,i]) for i,k in enumerate(names)},
                         bound_note='Exploratory bounds, not measured uncertainty; bound hits are not identified material values',
                         near_bounds=[k for i,k in enumerate(names) if min(fit.x[i]-optimizer_bounds[0,i],optimizer_bounds[1,i]-fit.x[i])<.01],
                         fit_metrics=score,prior_metrics=prior_score,evaluations=len(evaluations),
                         optimizer_success=bool(fit.success),optimizer_message=fit.message,
                         optimizer_optimality=float(fit.optimality),
                         starts=[dict(parameters=unpack(f.x),success=bool(f.success),rmse_k=float(np.sqrt(np.mean(f.fun**2))),message=f.message) for f in fits],
                         evaluation_history=evaluations,
                         local_jacobian_condition_number=condition,validation_used=False,
                         caveat='State-geometry fit: only reference contacts, environment h and nonlinear temperature response; ALL cp and bulk k fixed. Legacy static sets may explore pad cp. Topology/fixture conditional; correlated parameters and bound hits are not material identification. Destruction is implemented but not calibrated by pre-event data.')
            save(a.profile,profile)
        else:
            if manifest['archive_sha256']!=profile['archive_sha256']:raise ValueError('Dataset version changed after fitting')
            params=profile['parameters'];run=worker.run(params);verify_fixed_properties(run,scene);score=metrics(run,observed)
        report=dict(kind=a.mode,protocol=PROTOCOL,experiment=experiment,parameters=params,metrics=score,
                    topology=scene['metadata']['topology'],topology_status=scene['metadata']['topology_status'],
                    measurement_probes=scene['measurement_probes'],scenario_path=str(scene_path),
                    source_csv_sha256=record['canonical_sha256'],model_sha256=model_hash(),
                    frozen_profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
                    assumptions=scene['metadata']['assumptions'],result=run)
        if a.numerical_check:
            checks={}
            for label,extra in [('half_dt',{'max_dt_s':.25}),('refined_passive',{'mesh_scale':2})]:
                refined=worker.run(dict(params,**extra))
                diffs=np.concatenate([np.asarray(refined['temperature_k'][c])-np.asarray(run['temperature_k'][c]) for c in TARGETS])
                checks[label]=dict(max_change_k=float(np.max(np.abs(diffs))),rms_change_k=float(np.sqrt(np.mean(diffs**2))),metrics=metrics(refined,observed))
            report['numerical_checks']=checks
        save(a.output,report)
        print(json.dumps(score,indent=2))
        print('Report:',a.output)
    finally:worker.close()


if __name__=='__main__':main()

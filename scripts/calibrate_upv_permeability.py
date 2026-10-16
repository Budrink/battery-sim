#!/usr/bin/env python3
"""Bounded pressure/permeability identification; thermal model and chemistry frozen.

Python only selects scenarios and audits results. All evolution is on CUDA.
The pressure record identifies an effective closure, not unique material data.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

import calibrate_upv_mass_paths as fit
import calibrate_upv_shell_strength as strength
import upv_mass_paths as geometry
from calibrate_upv import save

OUT = geometry.ROOT/'data/calibration/upv-permeability-v22'
BASE = dict(strength_factor=1., rate_factor=1., exponent=2., discharge_coefficient=.7)
# v17 already screened five broad regimes, but its selected aperture did not
# converge. First recheck baseline and that candidate with feedback-resolved
# stepping. This is NOT another broad search or evidence of a unique fit.
CASES = [('baseline', {}),
         ('delayed-rapid', dict(strength_factor=8., rate_factor=.03, exponent=1.))]
BOUNDS = dict(strength_factor=(1.,8.), rate_factor=(.03,30.), exponent=(1.,4.),
              discharge_coefficient=(.07,1.))


def build(scene, parameters, duration=30.):
    if set(parameters)!=set(BASE):
        raise ValueError('Only permeability parameters may be fitted')
    for key,(lo,hi) in BOUNDS.items():
        if not np.isfinite(parameters[key]) or not lo<=parameters[key]<=hi:
            raise ValueError('Invalid permeability parameter: '+key)
    result=geometry.attach(scene,geometry.initial())
    for channel in result['gas_network']['channels']:
        if channel['kind']!='rupture':
            continue
        channel['opening_pressure_pa']*=parameters['strength_factor']
        channel['discharge_coefficient']=parameters['discharge_coefficient']
        law=channel['damage_opening']
        law['pressure_damage_rate_per_s']*=parameters['rate_factor']
        law['exponent']=parameters['exponent']
    times=[t for t in result['failure_sample_times_s'] if t<=duration]
    times += [float(t) for t in np.arange(0,min(5.,duration),.02)]+[duration]
    result['failure_sample_times_s']=sorted(set(times))
    result['simulation']['duration_s']=duration
    result['metadata']['permeability_trial']=parameters
    if geometry.thermal_signature(result)!=geometry.thermal_signature(scene):
        raise ValueError('Thermal model changed')
    return result


def pressure_score(run, prior):
    # Original measurement times only; dense output adds no fictitious evidence.
    sample=strength.observation_samples(run,prior);t=np.asarray(sample['time_s'])
    measured=np.interp(t,prior['result']['time_s'],prior['observed_pressure_bar'])
    error=np.asarray(sample['gas']['chamber']['pressure_pa'])/1e5-measured
    windows={}
    for lo,hi,scale in [(0.,3.,.25),(3.,30.,.1)]:
        mask=(t>lo)&(t<=hi)&np.isfinite(error)
        if not mask.any():
            raise ValueError('Missing pressure observations')
        windows[f'{lo:g}-{hi:g}']=float(np.sqrt(np.mean(error[mask]**2)))
    return dict(objective=(windows['0-3']/.25)**2+(windows['3-30']/.1)**2,
                rmse_bar_by_window=windows)


def summarize(run, scene, prior):
    sampled=strength.observation_samples(run,prior)
    features=strength.features(run,scene)
    ruptures=[run['channels'][c['id']] for c in scene['gas_network']['channels'] if c['kind']=='rupture']
    features['fracture_work_j']=sum(c['fracture_work_j'][-1] for c in ruptures)
    features['advected_energy_j']=sum(c['delivered_gas_energy_j'][-1] for c in ruptures)
    # Compare proposals and accepted rates FROM THE SAME substep. End-state
    # sampled pressures/apertures cannot reconstruct that previous proposal.
    accepted=np.abs(np.asarray([c['mass_flow_kg_s'] for c in ruptures]))
    proposed=np.abs(np.asarray([c['proposed_mass_flow_kg_s'] for c in ruptures]))
    active=proposed>1e-6
    features['sampled_significant_flow_capped_fraction']=float(np.mean(accepted[active]<proposed[active]*(1-1e-8))) if active.any() else 0.
    return dict(score=pressure_score(run,prior),features=features,metrics=fit.metrics(sampled,prior))


def replay(runner,scene,prior,parameters,duration=30.,refinement=1.):
    candidate=build(scene,parameters,duration)
    candidate['simulation']['stability_factor']*=refinement
    run=runner.run(candidate,refinement)
    if run.get('max_oxidizer_residual_kg',0)>1e-10:
        raise ValueError('Oxidizer conservation failed')
    return dict(parameters=parameters,**summarize(run,candidate,prior),result=run,
                meta=prior['meta'],observed_temperature_k=prior['observed_temperature_k'],
                observed_pressure_bar=prior['observed_pressure_bar']),candidate


def heating_regression(runner):
    results=[]
    for name,folder in [('no-cooling',geometry.PRIOR),('active-cooling',geometry.PRIOR/'cooling-check/nominal')]:
        scene=geometry.read(folder/'warmup-scenario.json');old=geometry.read(folder/'warmup.json')
        new=runner.run(scene)
        if new['time_s']!=old['time_s']:
            raise ValueError('Heating sample times changed')
        delta=max(float(np.max(np.abs(np.asarray(v)-old['temperature_k'][k]))) for k,v in new['temperature_k'].items())
        same=new['final_fields_k']==old['final_fields_k']
        results.append(dict(experiment=name,max_temperature_change_k=delta,identical_final_fields=same,
                            steps=new['steps'],max_thermal_residual_j=new['max_thermal_residual_j']))
        if delta!=0 or not same or new['steps']!=old['steps']:
            raise ValueError('Frozen heating regression failed')
    return results


def refinement_metrics(coarse,fine):
    a,b=coarse['result'],fine['result']
    ids=[k for k in a['channels'] if k.startswith('C2-progressive')]
    opening_change=max(float(np.max(np.abs(np.asarray(a['channels'][k]['opening_fraction'])-b['channels'][k]['opening_fraction']))) for k in ids)
    relative_final=max(abs(a['channels'][k]['opening_fraction'][-1]-b['channels'][k]['opening_fraction'][-1])/
                       max(1e-8,b['channels'][k]['opening_fraction'][-1]) for k in ids)
    wa=sum(a['channels'][k]['fracture_work_j'][-1] for k in ids)
    wb=sum(b['channels'][k]['fracture_work_j'][-1] for k in ids)
    ma=np.sum([a['channels'][k]['transferred_mass_kg'] for k in ids],axis=0)
    mb=np.sum([b['channels'][k]['transferred_mass_kg'] for k in ids],axis=0)
    return dict(max_witness_change_k=max(float(np.max(np.abs(np.asarray(a['temperature_k'][k])-b['temperature_k'][k]))) for k in fit.WITNESSES),
        max_pressure_change_bar=float(np.max(np.abs(np.asarray(a['gas']['chamber']['pressure_pa'])-b['gas']['chamber']['pressure_pa']))/1e5),
        max_open_fraction_change=opening_change,max_final_aperture_relative_change=relative_final,
        fracture_work_change_j=abs(wa-wb),fine_fracture_work_j=wb,
        max_net_export_change_g=float(np.max(np.abs(ma-mb))*1000))


def refinement_passes(delta):
    return (delta['max_witness_change_k']<=.5 and delta['max_pressure_change_bar']<=.05
            and delta['max_open_fraction_change']<=.01 and delta['max_final_aperture_relative_change']<=.05
            and delta['fracture_work_change_j']<=max(1e-6,.05*delta['fine_fracture_work_j'])
            and delta['max_net_export_change_g']<=.1)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=['check','scan','finalize'])
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path(strength.__file__).read_bytes()+fit.fingerprint(fit.PROBE).encode()).hexdigest()
    scene,prior=geometry.inputs('no-cooling');runner=fit.native.Runner(fit.PROBE,args.output)
    manifest=dict(fingerprint=fingerprint,cases=CASES,bounds=BOUNDS,
        design='Baseline plus prior v17 pressure-selected candidate, after numerical feedback correction; no further optimization',
        score='Equal-weight normalized pressure MSE at original observation times: 0..3s /0.25bar and 3..30s /0.1bar',
        fixed='All thermal properties, heat contacts, initial fields, reaction energy/rate/yield, gas geometry, fracture work, seam ratio and thermal softening',
        thermal_sha256=geometry.thermal_signature(scene),validation_used_for_selection=False,
        scope='Effective permeability identification; unmeasured geometry and mechanical priors are not uniquely identifiable')
    try:
        path=args.output/'manifest.json'
        if path.exists() and geometry.digest(geometry.read(path))!=geometry.digest(manifest):
            raise ValueError('Model or protocol changed; use a fresh output folder')
        save(path,manifest)
        if args.mode=='check':
            # Audit the known stiff failure, without fitting or promoting it.
            path=args.output/'heating-regression.json'
            if not path.exists():
                save(path,heating_regression(runner))
            p=dict(BASE,**CASES[1][1]);reports=[]
            for name,refinement in [('stiff-coarse',1.),('stiff-fine',.5)]:
                path=args.output/(name+'.json')
                if path.exists():
                    report=geometry.read(path)
                else:
                    print('checking',name,flush=True)
                    report,candidate=replay(runner,scene,prior,p,refinement=refinement)
                    save(path,report);save(args.output/(name+'-scenario.json'),candidate)
                reports.append(report)
            delta=refinement_metrics(*reports)
            save(args.output/'stiff-numerics.json',dict(delta,passed=refinement_passes(delta),calibration_accepted=False))
            print('stiff refinement',delta,'passed',refinement_passes(delta),flush=True)
            return
        if args.mode=='scan':
            path=args.output/'heating-regression.json'
            if not path.exists():
                print('frozen heating regression (no fitting)',flush=True);save(path,heating_regression(runner))
            history=[]
            for label,changes in CASES:
                p=dict(BASE,**changes);path=args.output/(label+'.json')
                if path.exists():
                    report=geometry.read(path)
                    if report['parameters']!=p:
                        raise ValueError('Cached parameters differ')
                else:
                    print('starting',label,p,flush=True);report,candidate=replay(runner,scene,prior,p)
                    save(path,report);save(args.output/(label+'-scenario.json'),candidate)
                summary={k:report[k] for k in ('parameters','score','features','metrics')}
                history.append(dict(id=label,**summary));save(args.output/'history.json',history)
                print('completed',label,report['score'],report['features'],flush=True)
            best=min(history,key=lambda h:h['score']['objective'])
            save(args.output/'selection.json',dict(fingerprint=fingerprint,id=best['id'],parameters=best['parameters'],
                relative_score_reduction=1-best['score']['objective']/history[0]['score']['objective'],validation_used_for_selection=False))
            return
        selected=geometry.read(args.output/'selection.json')
        if selected['fingerprint']!=fingerprint:
            raise ValueError('Selection belongs to another model')
        p=selected['parameters'];coarse=geometry.read(args.output/(selected['id']+'.json'))
        path=args.output/'refined.json'
        if path.exists():
            fine=geometry.read(path)
        else:
            print('refining',selected['id'],flush=True);fine,_=replay(runner,scene,prior,p,refinement=.5);save(path,fine)
        delta=refinement_metrics(coarse,fine)
        save(args.output/'numerics.json',delta);print('refinement',delta,flush=True)
        if not refinement_passes(delta):
            raise ValueError('Refinement failed; do not freeze numerical artifacts')
        save(args.output/'profile.json',dict(selected,calibration_accepted=False,
            status='Frozen candidate selected on training pressure only; not a validated material law',
            thermal_sha256=manifest['thermal_sha256']))
        for name,stem in [('no-cooling','fit'),('active-cooling','validation')]:
            path=args.output/(stem+'.json');scene,prior=geometry.inputs(name)
            if path.exists():
                report=geometry.read(path)
            else:
                print('starting full',stem,flush=True);report,candidate=replay(runner,scene,prior,p,1800.)
                save(path,report);save(args.output/(stem+'-scenario.json'),candidate)
            print('completed full',stem,report['score'],'late RMSE K',report['metrics']['phases']['300-1800']['rmse_k'],flush=True)
    finally:
        runner.close()


if __name__=='__main__':
    main()

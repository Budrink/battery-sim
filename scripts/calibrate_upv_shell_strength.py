#!/usr/bin/env python3
"""User-directed casing-strength test; every thermal/reaction input stays frozen."""
import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np

import calibrate_upv_mass_paths as fit
import upv_mass_paths as geometry
from calibrate_upv import save

OUT = geometry.ROOT/'data/calibration/upv-shell-strength-v16'
FACTORS = (1., 2., 4., 8.)


def build(scene, factor, duration=None):
    if factor not in FACTORS:
        raise ValueError('Undeclared strength factor')
    result = geometry.attach(scene, geometry.initial())
    for channel in result['gas_network']['channels']:
        if channel['kind']=='rupture':
            channel['opening_pressure_pa'] *= factor
    end = duration or result['simulation']['duration_s']
    times = [t for t in result['failure_sample_times_s'] if t<=end]
    # Additional OUTPUT samples resolve opening/blowdown. They are NOT treated
    # as extra experimental measurements when computing calibration scores.
    times += [float(t) for t in np.arange(0, min(5., end), .02)] + [end]
    result['failure_sample_times_s'] = sorted(set(times))
    result['simulation']['duration_s'] = end
    result['metadata']['strength_trial'] = dict(factor=factor, only_changed_parameter='rupture opening_pressure_pa')
    if geometry.thermal_signature(result)!=geometry.thermal_signature(scene):
        raise ValueError('Thermal model changed')
    return result


def observation_samples(run, prior):
    t=np.asarray(run['time_s']);original=np.asarray(prior['result']['time_s'])
    original=original[original<=t[-1]+1e-9];indices=np.searchsorted(t, original)
    if not np.allclose(t[indices], original, rtol=0, atol=1e-8):
        raise ValueError('Original measurement times are missing')
    def select(value):
        if isinstance(value, dict):
            return {key:select(v) for key,v in value.items()}
        if isinstance(value, list) and len(value)==len(t):
            return [value[int(i)] for i in indices]
        return value
    return select(run)


def features(run, scene):
    times=np.asarray(run['time_s']);pressure=np.asarray(run['gas']['chamber']['pressure_pa'])/1e5
    rupture_ids={c['id'] for c in scene['gas_network']['channels'] if c['kind']=='rupture'}
    openings=[e['time_s'] for e in run['events'] if e['id'] in rupture_ids and e['kind']=='shell_rupture']
    mass=np.sum([run['channels'][k]['transferred_mass_kg'] for k in rupture_ids], axis=0)
    first=next((float(times[i]) for i in range(len(times)) if mass[i]>=.001), None)
    maximum=int(np.argmax(pressure));peak=float(pressure[maximum]);baseline=float(pressure[0])
    after=np.flatnonzero((np.arange(len(times))>maximum)&(pressure<=baseline+.5*(peak-baseline)))
    return dict(first_open_s=min(openings) if openings else None, first_net_gram_s=first,
                sampled_chamber_peak_bar=peak, sampled_peak_s=float(times[maximum]),
                half_pressure_fall_s=float(times[after[0]]-times[maximum]) if len(after) else None,
                sampled_max_source_pressure_bar=max(max(v['pressure_pa']) for k,v in run['gas'].items() if k.startswith('inside-C2'))/1e5,
                net_export_g=float(mass[-1]*1000))


def replay(runner, scene, prior, factor, duration=None, refinement=1.):
    candidate=build(scene, factor, duration);candidate['simulation']['stability_factor']*=refinement
    run=runner.run(candidate, refinement)
    if run.get('max_oxidizer_residual_kg',0)>1e-10:
        raise ValueError('Oxidizer conservation failed')
    sampled=observation_samples(run,prior)
    return dict(parameters=dict(geometry.initial(), strength_factor=factor), metrics=fit.metrics(sampled, prior),
                features=features(run,candidate), result=run, meta=prior['meta'],
                observed_temperature_k=prior['observed_temperature_k'], observed_pressure_bar=prior['observed_pressure_bar']), candidate


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode', choices=['scan','finalize'])
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+fit.fingerprint(fit.PROBE).encode()).hexdigest()
    scene,prior=geometry.inputs('no-cooling');runner=fit.native.Runner(fit.PROBE,args.output)
    try:
        if args.mode=='scan':
            manifest=dict(fingerprint=fingerprint, factors=FACTORS, duration_s=30.,
                selection='Lowest chamber-pressure RMSE on original 0..30 s measurement times among increased-strength cases; 1x is reference only',
                thermal_sha256=geometry.thermal_signature(scene), fixed_reaction=geometry.initial(),
                changed='Only rupture opening_pressure_pa; seam ratio, softening, damage kinetics and fracture work unchanged',
                validation_used_for_selection=False)
            path=args.output/'manifest.json'
            if path.exists() and geometry.digest(geometry.read(path))!=geometry.digest(manifest):
                raise ValueError('Different strength protocol')
            save(path,manifest);history=[]
            for factor in FACTORS:
                stem='strength-'+str(int(factor));path=args.output/(stem+'.json')
                if path.exists():
                    report=geometry.read(path)
                else:
                    print('starting',stem,flush=True);report,candidate=replay(runner,scene,prior,factor,30.)
                    save(path,report);save(args.output/(stem+'-scenario.json'),candidate)
                history.append(dict(factor=factor, features=report['features'], metrics=report['metrics']))
                save(args.output/'history.json',history)
                print(stem, report['features'], 'pressure_RMSE',report['metrics']['pressure_0_30_rmse_bar'],flush=True)
            best=min(history[1:],key=lambda x:x['metrics']['pressure_0_30_rmse_bar'])
            save(args.output/'selection.json',dict(fingerprint=fingerprint, factor=best['factor'], validation_used_for_selection=False))
            return
        selection=geometry.read(args.output/'selection.json')
        if selection['fingerprint']!=fingerprint:
            raise ValueError('Selected model changed')
        factor=selection['factor']
        for name,stem in [('no-cooling','fit'),('active-cooling','validation')]:
            scene,prior=geometry.inputs(name);path=args.output/(stem+'.json')
            if path.exists():
                report=geometry.read(path)
            else:
                print('starting',stem,'strength',factor,flush=True);report,candidate=replay(runner,scene,prior,factor)
                report['old_metrics']=fit.metrics(prior['result'],prior)
                save(path,report);save(args.output/(stem+'-scenario.json'),candidate)
            print(stem, report['features'], 'late_RMSE',report['metrics']['phases']['300-1800']['rmse_k'],flush=True)
            if name=='no-cooling':
                fine_path=args.output/'refined.json'
                if fine_path.exists():
                    fine=geometry.read(fine_path)
                else:
                    print('refining selected strength',flush=True);fine,_=replay(runner,scene,prior,factor,refinement=.5);save(fine_path,fine)
                delta=dict(max_witness_change_k=max(float(np.max(np.abs(np.asarray(report['result']['temperature_k'][k])-fine['result']['temperature_k'][k]))) for k in fit.WITNESSES),
                           max_pressure_change_bar=float(np.max(np.abs(np.asarray(report['result']['gas']['chamber']['pressure_pa'])-fine['result']['gas']['chamber']['pressure_pa']))/1e5))
                save(args.output/'numerics.json',delta);print('refinement',delta,flush=True)
                if delta['max_witness_change_k']>.5 or delta['max_pressure_change_bar']>.05:
                    raise ValueError('Numerical refinement failed before validation')
                save(args.output/'profile.json',dict(selection, thermal_sha256=geometry.thermal_signature(scene),
                    parameters=report['parameters'], calibration_accepted=False,
                    status='User-directed stronger-casing candidate; thermal and reaction parameters frozen'))
    finally:
        runner.close()


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Fixed area increments on the local void graph. Destruction-only screening.

No thermal, chemical or gas-geometry fitting. Checks precede fitting; failed
numerics can be screened diagnostically, but never promoted or validated.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import calibrate_upv_area as area
import upv_local_mass_paths as local
import gas_void_geometry as voids
from calibrate_upv import save

OUT=area.old.geometry.ROOT/'data/calibration/upv-fixed-fraction-v26'
BASE=dict(pressure_pa=1e6,rate_per_s=20.,fraction=.02)
CASES=[('fraction02',BASE),('fraction05',dict(BASE,fraction=.05)),
       ('strong-slow',dict(BASE,pressure_pa=5e6,rate_per_s=.2)),
       ('strong-fast',dict(BASE,pressure_pa=5e6,rate_per_s=200.)),
       ('weak-fast',dict(BASE,pressure_pa=2e5,rate_per_s=200.))]
BOUNDS=dict(pressure_pa=(2e5,5e6),rate_per_s=(.02,2000.),fraction=(.02,.05))


def build(scene,p,seed=17,duration=30.,refinement=.5):
    if set(p)!=set(BOUNDS) or any(not np.isfinite(p[k]) or not lo<=p[k]<=hi for k,(lo,hi) in BOUNDS.items()):
        raise ValueError('Only declared rupture parameters may change')
    if not 0<duration<=30 or not 0<refinement<=1:raise ValueError('Invalid replay controls')
    s=local.attach(area.build(scene,dict(pressure_pa=p['pressure_pa'],rate_per_s=p['rate_per_s'],fragment_area_m2=1e-5),seed))
    for c in s['gas_network']['channels']:
        if c['kind']=='rupture':
            c['damage_opening'].pop('fragment_mean_area_m2')
            c['damage_opening'].update(mode='stochastic_fraction',fixed_area_fraction=p['fraction'])
    s['simulation']['duration_s']=duration;s['simulation']['stability_factor']*=refinement
    s['failure_sample_times_s']=sorted({0.,duration,*[t for t in s['failure_sample_times_s'] if t<duration]})
    s['metadata']['fixed_fraction_trial']=dict(parameters=p,seed=seed,refinement=refinement,
        event_rate='rate * max(dp/Pcrit-1,0) * (1-a) / fraction',
        increment='min(fraction,1-a), fraction of original local casing patch area',calibration_accepted=False)
    return s


def summarize(run,s,prior):
    sampled=area.old.protocol.strength.observation_samples(run,prior)
    # A short numerical audit may precede the first acquired sample. It has
    # no experimental residual; do not invent observations by interpolation.
    metrics=area.old.fit.metrics(sampled,prior) if len(sampled['time_s'])>1 else dict(phases={},objective=None)
    features=area.old.protocol.strength.features(run,s)
    ids=[p['id'] for p in s['gas_network']['channels'] if p['kind']=='rupture']
    features['fracture_work_j']=sum(run['channels'][k]['fracture_work_j'][-1] for k in ids)
    if run['time_s'][-1]>=30:
        score=area.old.protocol.pressure_score(run,prior)
        objective=score['objective']+(metrics['phases']['3-30']['rmse_k']/20.)**2
    else:score=None;objective=None
    return dict(metrics=metrics,features=features,score=score,joint_objective=objective)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=['check','scan','finalize'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--case',choices=[k for k,_ in CASES])
    parser.add_argument('--duration',type=float,default=30.)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(b''.join(Path(m.__file__).read_bytes() for m in (area,area.old,local,voids))+
                              Path(__file__).read_bytes()+area.old.fit.fingerprint(area.old.fit.PROBE).encode()).hexdigest()
    manifest=dict(fingerprint=fingerprint,cases=CASES,bounds=BOUNDS,grid_mm=32.,seed=17,
        score='Original observations: pressure 0..3/3..30 s at 0.25/0.1 bar scale + pooled four-witness 3..30 s RMSE/20K squared',
        fixed='Thermal model, chemistry, local gas graph, Cd, fracture work, seam strength, pressure exponent=1',
        validation_used_for_selection=False,calibration_accepted=False)
    path=out/'manifest.json'
    if path.exists() and area.old.geometry.digest(area.old.geometry.read(path))!=area.old.geometry.digest(manifest):raise ValueError('Changed model: use a fresh output folder')
    save(path,manifest)
    scene,prior=area.old.geometry.inputs('no-cooling');runner=area.old.fit.native.Runner(area.old.fit.PROBE,out)
    def get(name,p,duration=30.,refinement=.5,seed=17):
        path=out/(name+'.json')
        if path.exists():return area.old.geometry.read(path)
        s=build(scene,p,seed,duration,refinement)
        save(out/(name+'-scenario.json'),s)
        print('starting',name,p,'duration',duration,'refinement',refinement,flush=True)
        run=runner.run(s,refinement)
        save(out/(name+'-raw.json'),run)
        if run.get('max_oxidizer_residual_kg',0)>1e-10:raise ValueError('Oxidizer balance failed')
        report=dict(parameters=p,duration=duration,refinement=refinement,seed=seed,**summarize(run,s,prior),result=run,
                    meta=prior['meta'],observed_temperature_k=prior['observed_temperature_k'],observed_pressure_bar=prior['observed_pressure_bar'])
        save(path,report)
        print('completed',name,'objective',report['joint_objective'],'features',report['features'],'metrics',report['metrics']['phases'],
              'steps',run['steps'],'wall_s',run['wall_s'],flush=True)
        return report
    try:
        if args.mode=='check':
            checks={}
            for name,p in CASES[:2]:
                a=get(name+'-check',p,.2);b=get(name+'-check-fine',p,.2,.25)
                d=area.old.protocol.refinement_metrics(a,b);checks[name]=dict(d,passed=area.old.protocol.refinement_passes(d))
                save(out/'short-numerics.json',checks);print('numerics',name,checks[name],flush=True)
            return
        if args.mode=='scan':
            for name,p in CASES:
                if args.case and name!=args.case:continue
                get(name+(f'-duration{args.duration:g}' if args.duration!=30 else ''),p,args.duration)
            history=[]
            for name,p in CASES:
                path=out/(name+'.json')
                if path.exists():
                    r=area.old.geometry.read(path);history.append(dict(id=name,**{k:r[k] for k in ('parameters','joint_objective','features','metrics')}))
            save(out/'history.json',history)
            if history:save(out/'selection.json',dict(min(history,key=lambda x:x['joint_objective']),calibration_accepted=False,
                                                   completed_cases=len(history),declared_cases=len(CASES)))
            return
        chosen=area.old.geometry.read(out/'selection.json');name,p=chosen['id'],chosen['parameters']
        a=get(name,p);b=get(name+'-fine',p,refinement=.25)
        d=area.old.protocol.refinement_metrics(a,b);passed=area.old.protocol.refinement_passes(d)
        save(out/'selected-numerics.json',dict(d,passed=passed,calibration_accepted=False))
        if not passed:print('Failed refinement; no promotion or validation',flush=True);return
        for seed in (29,43):get(name+f'-seed{seed}',p,seed=seed)
        scene,prior=area.old.geometry.inputs('active-cooling')
        for seed in (17,29,43):get(f'validation-seed{seed}',p,seed=seed)
    finally:runner.close()


if __name__=='__main__':main()

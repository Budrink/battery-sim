#!/usr/bin/env python3
"""Partial-area stochastic rupture: bounded destruction-only calibration.

Thermal inputs, chemistry, mass paths, discharge coefficient and fracture work
are frozen. Python orchestrates experiments; all state evolution is CUDA.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import calibrate_upv_stochastic as old
import compare_upv_shell_mesh as mesh
from calibrate_upv import save

OUT=old.geometry.ROOT/'data/calibration/upv-stochastic-area-v24'
BASE=dict(pressure_pa=1e6,rate_per_s=20.,fragment_area_m2=1e-5)
CASES=[('baseline',BASE),
       ('strong-slow',dict(pressure_pa=5e6,rate_per_s=.02,fragment_area_m2=1e-5)),
       ('weak-fast',dict(pressure_pa=2e5,rate_per_s=2000.,fragment_area_m2=1e-5)),
       ('small-fragments',dict(BASE,fragment_area_m2=1e-7)),
       ('large-fragments',dict(BASE,fragment_area_m2=1e-4))]
BOUNDS=dict(pressure_pa=(2e5,5e6),rate_per_s=(.02,2000.),fragment_area_m2=(1e-7,1e-4))
REFINEMENT=.5
SEEDS=(17,29,43)


def build(scene,p,seed,size_mm=32.):
    if set(p)!=set(BOUNDS) or any(not np.isfinite(p[k]) or not lo<=p[k]<=hi for k,(lo,hi) in BOUNDS.items()):
        raise ValueError('Only bounded destruction parameters may change')
    s=old.build(scene,dict(pressure_pa=p['pressure_pa'],rate_per_s=p['rate_per_s'],pressure_exponent=1.),seed,
                grid=mesh.grid_for_size(scene,size_mm))
    for c in s['gas_network']['channels']:
        if c['kind']=='rupture':c['damage_opening'].update(mode='stochastic_area',fragment_mean_area_m2=p['fragment_area_m2'])
    s['metadata']['area_rupture_trial']=dict(parameters=p,seed=seed,size_mm=size_mm)
    return s


def replay(runner,scene,prior,p,seed,size_mm=32.,refinement=REFINEMENT):
    s=build(scene,p,seed,size_mm);s['simulation']['stability_factor']*=refinement
    run=runner.run(s,refinement)
    if run.get('max_oxidizer_residual_kg',0)>1e-10:raise ValueError('Oxidizer balance failed')
    summary=old.protocol.summarize(run,s,prior)
    # Do not let a good chamber pressure hide the C3/ISS discrepancy again.
    # 0..3 s thermocouple spikes remain diagnostic, not a fitted target.
    summary['joint_objective']=summary['score']['objective']+(summary['metrics']['phases']['3-30']['rmse_k']/20.)**2
    return dict(parameters=p,seed=seed,size_mm=size_mm,refinement=refinement,**summary,result=run,
                meta=prior['meta'],observed_temperature_k=prior['observed_temperature_k'],observed_pressure_bar=prior['observed_pressure_bar']),s


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=['check','scan','finalize'])
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    fp=hashlib.sha256(Path(__file__).read_bytes()+Path(old.__file__).read_bytes()+Path(old.protocol.__file__).read_bytes()+Path(mesh.__file__).read_bytes()+old.fit.fingerprint(old.fit.PROBE).encode()).hexdigest()
    manifest=dict(fingerprint=fp,cases=CASES,bounds=BOUNDS,seeds=SEEDS,refinement=REFINEMENT,
                  objective='Pressure score (0..3/3..30s) + mean squared four-witness temperature error (3..30s)/20K^2; original observation times',
                  fixed='All thermal inputs, chemistry, mass paths, Cd, fracture work, seam strength; pressure exponent=1',
                  changed='Reference strength, event rate, mean physical fragment area only',calibration_accepted=False)
    path=folder/'manifest.json'
    if path.exists() and old.geometry.digest(old.geometry.read(path))!=old.geometry.digest(manifest):raise ValueError('Changed model/protocol: use fresh output folder')
    save(path,manifest);scene,prior=old.geometry.inputs('no-cooling');runner=old.fit.native.Runner(old.fit.PROBE,folder)
    def get(stem,p,seed,**kw):
        path=folder/(stem+'.json')
        if path.exists():return old.geometry.read(path)
        print('starting',stem,p,kw,flush=True)
        report,candidate=replay(runner,scene,prior,p,seed,**kw)
        save(path,report);save(folder/(stem+'-scenario.json'),candidate)
        print('completed',stem,'joint',report['joint_objective'],'pressure',report['metrics']['pressure_0_30_rmse_bar'],
              'temperature',report['metrics']['phases']['3-30']['rmse_k'],'features',report['features'],
              'steps',report['result']['steps'],'wall_s',report['result']['wall_s'],flush=True)
        return report
    try:
        if args.mode=='check':
            a=get('baseline-seed17',BASE,17)
            b=get('baseline-seed17-fine',BASE,17,refinement=REFINEMENT/2)
            delta=old.protocol.refinement_metrics(a,b)
            save(folder/'baseline-numerics.json',dict(delta,passed=old.protocol.refinement_passes(delta),calibration_accepted=False))
            print('baseline numerics',delta,'passed',old.protocol.refinement_passes(delta),flush=True)
            get('baseline-64mm-seed17',BASE,17,size_mm=64.)
            return
        if args.mode=='scan':
            # A bounded diagnostic sweep remains useful when baseline numerical
            # convergence fails, but nothing is accepted by the ranking alone.
            history=[]
            for name,p in CASES:
                r=get(name+'-seed17',p,17)
                history.append(dict(id=name,parameters=p,joint_objective=r['joint_objective'],metrics=r['metrics'],features=r['features']))
                save(folder/'history.json',history)
            best=min(history,key=lambda r:r['joint_objective'])
            save(folder/'selection.json',dict(best,fingerprint=fp,calibration_accepted=False,
                 improvement=1-best['joint_objective']/history[0]['joint_objective'],scope='Single-seed broad screening, not a calibrated profile'))
            return
        selected=old.geometry.read(folder/'selection.json');p=selected['parameters'];name=selected['id']
        if selected['fingerprint']!=fp:raise ValueError('Stale selection')
        a=get(name+'-seed17',p,17);b=get(name+'-seed17-fine',p,17,refinement=REFINEMENT/2)
        delta=old.protocol.refinement_metrics(a,b)
        save(folder/'selected-numerics.json',dict(delta,passed=old.protocol.refinement_passes(delta),calibration_accepted=False))
        if not old.protocol.refinement_passes(delta):
            print('Selected candidate fails refinement; stop without promoting or validating',flush=True);return
        reports=[a]+[get(name+f'-seed{seed}',p,seed) for seed in SEEDS[1:]]
        save(folder/'ensemble.json',dict(mean_joint_objective=float(np.mean([r['joint_objective'] for r in reports])),
             std_joint_objective=float(np.std([r['joint_objective'] for r in reports])),calibration_accepted=False))
        scene,prior=old.geometry.inputs('active-cooling')
        for seed in SEEDS:get(f'validation-seed{seed}',p,seed)
    finally:runner.close()


if __name__=='__main__':main()

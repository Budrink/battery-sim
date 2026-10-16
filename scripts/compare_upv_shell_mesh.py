#!/usr/bin/env python3
"""Fixed-parameter stochastic casing mesh comparison; NOT parameter fitting."""
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import calibrate_upv_stochastic as model
from calibrate_upv import save

OUT=model.geometry.ROOT/'data/calibration/upv-shell-mesh-64-32-v1'
SIZES_MM=(64.,32.)


def grid_for_size(scene,size_mm):
    # Ceil gives cells no larger than the requested in-plane size. Casing
    # thickness remains its physical 150 um; each face has one normal layer.
    grids=[]
    cells={c['id']:c for c in scene['cells']}
    for shell in scene['shells']:
        outer=np.asarray(cells[shell['contents_id']]['geometry']['dimensions_m'])+2*shell['thickness_m']
        grids.append(tuple(max(1,math.ceil(d*1000/size_mm-1e-12)) for d in outer))
    if len(set(grids))!=1:raise ValueError('This comparison expects equal battery dimensions')
    return grids[0]


def stats(values):
    return dict(mean=float(np.mean(values)),std=float(np.std(values)),min=float(np.min(values)),max=float(np.max(values)))


def summary(reports):
    metrics=dict(pressure_rmse_bar=lambda r:r['metrics']['pressure_0_30_rmse_bar'],
                 pressure_peak_bar=lambda r:r['features']['sampled_chamber_peak_bar'],
                 temperature_0_3_rmse_k=lambda r:r['metrics']['phases']['0-3']['rmse_k'],
                 temperature_3_30_rmse_k=lambda r:r['metrics']['phases']['3-30']['rmse_k'])
    result={key:stats([fn(r) for r in reports]) for key,fn in metrics.items()}
    result['channels_3_30']={k:stats([r['metrics']['phases']['3-30']['per_channel'][k]['rmse_k'] for r in reports]) for k in model.fit.WITNESSES}
    result['seeds']=[r['seed'] for r in reports]
    result['wall_s']=sum(r['result']['wall_s'] for r in reports)
    result['steps']=[r['result']['steps'] for r in reports]
    return result


def plot(folder,reports,prior):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    obs_t=np.asarray(prior['result']['time_s']);mask=obs_t<=30
    colors=['#2563eb','#ea580c']
    keys=['pressure','pressure','101','102','107','108']
    names={'pressure':'Давление в камере, бар (абс.)','101':'C1 / ISS, °C','102':'C1 / TIM, °C','107':'C3 / TIM, °C','108':'C3 / ISS, °C'}
    for index,(ax,key) in enumerate(zip(axes.flat,keys)):
        for color,(size,runs) in zip(colors,reports.items()):
            curves=[]
            for report in runs:
                r=report['result'];t=np.asarray(r['time_s'])
                y=np.asarray(r['gas']['chamber']['pressure_pa'])/1e5 if key=='pressure' else np.asarray(r['temperature_k'][key])-273.15
                ax.plot(t,y,color=color,alpha=.25,lw=.7);curves.append(y)
            ax.plot(t,np.mean(curves,axis=0),color=color,lw=1.8,label=f'{size:g} мм: среднее, n={len(runs)}')
        y=np.asarray(prior['observed_pressure_bar']) if key=='pressure' else np.asarray(prior['observed_temperature_k'][key],dtype=float)-273.15
        ax.plot(obs_t[mask],y[mask],color='#111827',lw=1.2,label='Эксперимент')
        ax.set(title=names[key],xlim=(0,3 if index==0 else 30),xlabel='Время после начала аварии, с')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Оболочка: 64 → 32 мм, одинаковые параметры; тонкие линии — отдельные реализации\nОпыт без охлаждения, первые 30 с; это проверка сетки, не завершённая калибровка')
    fig.savefig(folder/'mesh-comparison.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    scene,prior=model.geometry.inputs('no-cooling')
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path(model.__file__).read_bytes()+Path(model.protocol.__file__).read_bytes()+model.fit.fingerprint(model.fit.PROBE).encode()).hexdigest()
    grids={size:grid_for_size(scene,size) for size in SIZES_MM}
    manifest=dict(fingerprint=fingerprint,grids={str(k):v for k,v in grids.items()},parameters=model.BASE,seeds=model.SEEDS,
                  duration_s=30,experiment='no-cooling',thermal_parameters_fitted=False,calibration_accepted=False,
                  initial_field='Conservative volume-overlap projection of frozen pre-event temperatures',
                  stochastic_note='Fixed per-pixel hazard law. Same seed labels across meshes are NOT the same physical random field.',
                  scope='Combined casing thermal discretization and whole-pixel fracture discretization; no parameter optimization')
    path=folder/'manifest.json'
    if path.exists() and model.geometry.digest(model.geometry.read(path))!=model.geometry.digest(manifest):
        raise ValueError('Changed protocol: choose a fresh output directory')
    save(path,manifest);runner=model.fit.native.Runner(model.fit.PROBE,folder);reports={}
    try:
        # Interleave grids so even a partial run compares identical seed sets.
        for seed in model.SEEDS:
            for size,grid in grids.items():
                stem=f'mesh-{size:g}mm-seed{seed}';path=folder/(stem+'.json')
                if path.exists():report=model.geometry.read(path)
                else:
                    print('starting',stem,'grid',grid,flush=True)
                    report,candidate=model.replay(runner,scene,prior,model.BASE,seed,grid=grid)
                    save(path,report);save(folder/(stem+'-scenario.json'),candidate)
                reports.setdefault(size,[]).append(report)
                rows={str(s):summary(r) for s,r in reports.items()}
                save(folder/'summary.json',dict(meshes=rows,complete=all(len(r)==len(model.SEEDS) for r in reports.values()) and len(reports)==len(SIZES_MM),
                    calibration_accepted=False,scope=manifest['scope']))
                print('completed',stem,'summary',summary([report]),flush=True)
            plot(folder,reports,prior)
    finally:runner.close()


if __name__=='__main__':main()

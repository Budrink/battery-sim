#!/usr/bin/env python3
"""Show every completed area-rupture regime, including failed fits."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import calibrate_upv_area as model

def assessment(folder):
    rows=[]
    for name,_ in model.CASES:
        path=folder/(name+'-seed17.json')
        if not path.exists():continue
        report=model.old.geometry.read(path);r=report['result']
        scene=model.old.geometry.read(folder/(name+'-seed17-scenario.json'))
        iss3=lambda identifier:identifier.startswith('gap-ISS3-')
        net=sum((int(iss3(c['to']))-int(iss3(c['from'])))*r['channels'][c['id']]['delivered_gas_energy_j'][-1]
                for c in scene['gas_network']['channels'])
        gas_total=sum(g['released_energy_j'][-1] for g in r['gas'].values())
        rows.append(dict(id=name,parameters=report['parameters'],joint_objective=report['joint_objective'],
            pressure_rmse_bar=report['metrics']['pressure_0_30_rmse_bar'],
            c3_iss_rmse_k=report['metrics']['phases']['3-30']['per_channel']['108']['rmse_k'],
            gas_reaction_total_j=gas_total,chamber_gas_reaction_j=r['gas']['chamber']['released_energy_j'][-1],
            iss3_gas_reaction_j=sum(g['released_energy_j'][-1] for k,g in r['gas'].items() if iss3(k)),
            iss3_net_advected_energy_j=net,
            max_thermal_residual_j=r['max_thermal_residual_j'],max_mass_residual_kg=r['max_mass_residual_kg']))
    if not rows:return
    baseline=rows[0];best=min(rows,key=lambda r:r['joint_objective'])
    result=dict(rows=rows,complete=len(rows)==len(model.CASES),best_screened=best['id'],
                improvement_fraction=1-best['joint_objective']/baseline['joint_objective'],calibration_accepted=False,
                caveat='Single-seed bounded screening. Best score does not prove temporal/spatial convergence or validation; thermal properties and chemistry were not fitted.')
    path=folder/'baseline-seed17-fine.json'
    if path.exists():
        fine=model.old.geometry.read(path)
        result['baseline_temporal_objective_change']=abs(fine['joint_objective']-baseline['joint_objective'])
        result['screened_gain_smaller_than_baseline_temporal_change']=baseline['joint_objective']-best['joint_objective']<result['baseline_temporal_objective_change']
    path=folder/'baseline-numerics.json'
    if path.exists():result['baseline_numerics']=model.old.geometry.read(path)
    result['mesh64_completed']=(folder/'baseline-64mm-seed17.json').exists()
    model.save(folder/'assessment.json',result)

def main():
    cli=argparse.ArgumentParser(__doc__);cli.add_argument('--output',type=Path,default=model.OUT)
    folder=cli.parse_args().output;assessment(folder);_,prior=model.old.geometry.inputs('no-cooling')
    obs_t=np.asarray(prior['result']['time_s']);mask=obs_t<=30
    fig,axes=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    keys=['pressure','pressure','101','102','107','108']
    labels={'pressure':'Давление в камере, бар (абс.)','101':'C1 / ISS, °C','102':'C1 / TIM, °C','107':'C3 / TIM, °C','108':'C3 / ISS, °C'}
    for index,(ax,key) in enumerate(zip(axes.flat,keys)):
        for name,_ in model.CASES:
            path=folder/(name+'-seed17.json')
            if not path.exists():continue
            r=model.old.geometry.read(path)['result']
            y=np.asarray(r['gas']['chamber']['pressure_pa'])/1e5 if key=='pressure' else np.asarray(r['temperature_k'][key])-273.15
            ax.plot(r['time_s'],y,lw=1.1,label=name)
        y=np.asarray(prior['observed_pressure_bar']) if key=='pressure' else np.asarray(prior['observed_temperature_k'][key],dtype=float)-273.15
        ax.plot(obs_t[mask],y[mask],color='black',lw=1.8,label='Эксперимент')
        ax.set(title=labels[key],xlim=(0,3 if index==0 else 30),xlabel='Время после начала аварии, с')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Случайный прорыв части площади: проверка разных режимов\nТепловая модель и химия зафиксированы; это не принятая калибровка')
    fig.savefig(folder/'area-regimes.png',dpi=150);plt.close(fig)
    coarse=folder/'baseline-seed17.json';fine=folder/'baseline-seed17-fine.json'
    if not coarse.exists() or not fine.exists():return
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for path,label in [(coarse,'Ограничения шага ×0,5'),(fine,'Ограничения шага ×0,25')]:
        r=model.old.geometry.read(path)['result'];t=np.asarray(r['time_s'])
        channels=[c for k,c in r['channels'].items() if k.startswith('C2-progressive')]
        series=[np.max([g['pressure_pa'] for k,g in r['gas'].items() if k.startswith('inside-C2')],axis=0)/1e5,
                np.sum([c['effective_area_m2'] for c in channels],axis=0)*1e6,
                np.sum([c['fracture_work_j'] for c in channels],axis=0),
                np.sum([c['transferred_mass_kg'] for c in channels],axis=0)*1000]
        for ax,y in zip(axes.flat,series):ax.plot(t,y,label=label)
    titles=['Максимум давления внутренних узлов, бар (по отсчётам)','Суммарная открытая площадь, мм²','Работа разрушения, Дж','Суммарный нетто-вынос газа, г']
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title,xlim=(0,3),xlabel='Время, с');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Одинаковый seed и параметры, разные численные ограничения\nПроверять нужно не только ошибку внешних датчиков')
    fig.savefig(folder/'area-time-refinement.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()

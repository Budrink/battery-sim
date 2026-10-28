#!/usr/bin/env python3
"""Run one declared, unfitted FSRI development case; never calibrate silently."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

import calibrate_upv_r123 as replay_settings
from check_upv_restart import native
from fsri30_data import load, save
from generate_fsri30 import build, BOX, DIAMETER, HEATED


def summarize(scene, run, data):
    ids=sorted(run['cell_mean_temperature_k']);times=np.array(run['time_s'])
    mean=np.array([run['cell_mean_temperature_k'][k]for k in ids]).T
    maximum=np.array(run['source_max_temperature_k']).reshape(len(times),len(ids))
    r2=np.array(run['released_energy_j_history']).reshape(len(times),len(ids))
    initial=np.array([next(c for c in scene['cells']if c['id']==k)['conversion']['mass_kg']*
        next(c for c in scene['cells']if c['id']==k)['conversion']['net_heat_j_per_kg']*
        (1.+next(c for c in scene['cells']if c['id']==k)['conversion'].get('initial_charged_fraction',0.)) for k in ids])
    # Reporting conventions only; these do not initiate/stop any reaction.
    ten_percent={k:float(times[np.flatnonzero(r2[:,j]>=.1*initial[j])[0]])
        for j,k in enumerate(ids)if np.any(r2[:,j]>=.1*initial[j])}
    event_first={key:float(data['time_s'][np.flatnonzero(data[key])[0]])for key in ('runaway','flaming')if np.any(data[key])}
    return dict(status='partial'if run.get('partial')else'complete',diagnostic_only=True,
        calibrated=False,validation_passed=False,sensor_component_mapping_confirmed=False,
        advanced_to_s=run['advanced_to_s'],wall_s=run['wall_s'],
        realtime_ratio=(run['advanced_to_s']-run['started_at_s'])/run['wall_s'],
        battery_count=len(ids),thermal_volumes=run['volumes'],
        gas_compartments=len(scene['gas_network']['compartments']),
        minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
        main_condensed_heat_pulse_sample_end_s=float(times[1:][int(np.argmax(np.diff(q)/np.diff(times)))])
            if (q:=r2.sum(axis=1)).max()>0 else None,
        maximum_other_component_temperature_c=float(np.max(maximum[:,[j for j,k in enumerate(ids)if k!=HEATED]])-273.15),
        external_gas_oxidation_energy_j=float(sum(v['released_energy_j'][-1]for v in run['gas'].values())),
        final_hood_unburned_chemical_j=float(run['gas']['hood']['chemical_j'][-1]),
        final_hood_temperature_k=float(run['gas']['hood']['temperature_k'][-1]),
        final_hood_oxygen_kg=float(run['gas']['hood']['oxidizer_kg'][-1]),
        model_first_reported_reaction_s=min([e['time_s']for e in run['events']if e['kind'].startswith('runaway_')and e['kind']!='runaway_complete'],default=None),
        components_releasing_10pct_of_initial_reserve=len(ten_percent),
        ten_percent_release_times_s=ten_percent,
        maximum_battery_temperature_c=float(np.max(maximum)-273.15),
        measured_event_markers_s=event_first,
        observed_hrr_peak_kw=float(np.nanmax(data['hrr_kw'])),
        observed_hrr_peak_s=float(data['time_s'][int(np.nanargmax(data['hrr_kw']))]),
        observed_hrr_integral_j=float(np.trapz(data['hrr_kw']*1000.,data['time_s'])),
        balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k},
        comparison_limits=['No numeric validation claim; uncalibrated chemistry and unknown sensor positions',
            'Experimental oxygen-consumption HRR includes ABS fire; plotted model R3 is gas oxidation only',
            *(['Whole-experiment tail beyond selected horizon is not simulated']
                if run['advanced_to_s']<data['time_s'][-1] else [])])


def plot(out, scene, run, data):
    ids=sorted(run['cell_mean_temperature_k']);t=np.array(run['time_s'])
    mean=np.array([run['cell_mean_temperature_k'][k]for k in ids]).T-273.15
    maximum=np.array(run['source_max_temperature_k']).reshape(len(t),len(ids))-273.15
    observed=np.array([data[f'T{i}_k']for i in range(1,10)]).T-273.15
    event=float(data['time_s'][np.flatnonzero(data['runaway'])[0]])
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    ax=axs[0,0]
    for j in range(9):ax.plot(data['time_s'],observed[:,j],lw=.8,label=f'Канал {j+1}')
    ax.set_title('Эксперимент: 9 опубликованных термопар');ax.set_ylabel('°C')
    ax.legend(ncol=3,fontsize=7);ax.axvline(event,c='k',ls=':',lw=1)
    ax=axs[0,1]
    for j,k in enumerate(ids):ax.plot(t,mean[:,j],lw=.7,alpha=.6)
    ax.plot(t,mean[:,ids.index(HEATED)],c='k',lw=1.5,label='Нагреваемый '+HEATED)
    ax.set_title('Расчёт: средняя температура каждой из 30 батарей');ax.set_ylabel('°C');ax.legend(fontsize=8)
    ax=axs[1,0]
    ax.fill_between(t,mean.min(axis=1),mean.max(axis=1),color='tab:blue',alpha=.2,label='Расчёт: диапазон средних T')
    ax.plot(t,maximum.max(axis=1),'--',c='tab:blue',lw=1,label='Расчёт: самая горячая внутренняя зона')
    ax.plot(data['time_s'],np.nanmax(observed,axis=1),c='k',lw=1,label='Эксперимент: максимум 9 термопар')
    ax.set_title('Общая форма — без выдуманного соответствия датчиков');ax.set_ylabel('°C');ax.legend(fontsize=8)
    ax=axs[1,1]
    release=np.sum([np.array(v['released_energy_j'])for v in run.get('gas',{}).values()],axis=0)
    power=np.diff(release)/np.diff(t)/1000.
    ax.plot(data['time_s'],data['hrr_kw'],c='k',lw=1,label='Эксперимент: HRR, включая горение ABS')
    ax.plot((t[1:]+t[:-1])/2,power,c='tab:orange',lw=1,label='Расчёт: окисление газа R3')
    ax.set_title('Разные измеряемые величины: не критерий точной подгонки');ax.set_ylabel('кВт');ax.legend(fontsize=8)
    for ax in axs.flat:ax.set_xlim(0,t[-1]);ax.set_xlabel('Время от начала нагрева, с');ax.grid(alpha=.2)
    fig.suptitle('UL FSRI: первый НЕОТКАЛИБРОВАННЫЙ прогон; геометрия и химия содержат допущения',fontsize=12)
    fig.savefig(out/'comparison.png',dpi=145);plt.close(fig)
    poses=scene['metadata']['approximate_positions_m']
    peak=float(t[np.argmax(mean[:,ids.index(HEATED)])])
    targets=[0.,peak,min(t[-1],peak+60.),t[-1]]
    fig,axes=plt.subplots(1,4,figsize=(12,7),constrained_layout=True)
    norm=matplotlib.colors.Normalize(25,max(300,float(np.max(mean))))
    for ax,at in zip(axes,targets):
        idx=int(np.argmin(np.abs(t-at)))
        ax.add_patch(Rectangle((-BOX[0]*500,-BOX[1]*500),BOX[0]*1000,BOX[1]*1000,fill=False,color='gray'))
        for j,k in enumerate(ids):
            x,y,_=poses[k];ax.add_patch(Circle((x*1000,y*1000),DIAMETER*500,
                facecolor=plt.cm.inferno(norm(mean[idx,j])),edgecolor='cyan'if k==HEATED else 'gray',lw=1.2 if k==HEATED else .5))
            ax.text(x*1000,y*1000,k[1:],ha='center',va='center',fontsize=6,color='white')
        ax.set_aspect('equal');ax.set_xlim(-BOX[0]*550,BOX[0]*550);ax.set_ylim(-BOX[1]*550,BOX[1]*550)
        ax.invert_yaxis()  # drawing coordinates: positive y points downward
        ax.set_title(f'{t[idx]:.0f} с');ax.set_xlabel('мм')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='inferno'),ax=list(axes),label='Средняя T, °C',shrink=.6)
    fig.suptitle('Схема 7+8+8+7; цилиндры показаны физической формы, расчётные объёмы эквивалентные')
    fig.savefig(out/'temperature-map.png',dpi=145);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=3600.)
    ap.add_argument('--scenario',type=Path,help='Explicit existing scene, no rebuild from defaults')
    args=ap.parse_args();out=args.output.resolve();out.mkdir()
    data,manifest=load();scene=json.loads(args.scenario.read_text())if args.scenario else build(args.duration)
    scene['metadata']['data_manifest']=manifest;save(out/'scenario.json',scene)
    replay_settings.PROBE=args.probe.resolve();fingerprint=hashlib.sha256(replay_settings.PROBE.read_bytes()).hexdigest()
    state=dict(status='running',pid=os.getpid(),native_sha256=fingerprint,calibrated=False,
        scenario_sha256=hashlib.sha256((out/'scenario.json').read_bytes()).hexdigest(),duration_s=scene['simulation']['duration_s'])
    save(out/'status.json',state)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    run=None
    try:
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=fingerprint,stop_after_first_rupture=False),
            5.2*scene['simulation']['duration_s']+30.,minimum_speed=.2,rolling_minimum=.2)
        state.update(summarize(scene,run,data));save(out/'summary.json',state)
    except (Exception,KeyboardInterrupt)as exc:
        state.update(status='stopped'if isinstance(exc,KeyboardInterrupt)else'failed',error=str(exc))
        partial=out/'replay/incomplete-run.json.gz'
        if partial.exists():
            with gzip.open(partial,'rt')as f:run=json.load(f)
            state.update(advanced_to_s=run['advanced_to_s'],known_until_s=run['time_s'][-1],wall_s=run['wall_s'])
    finally:
        if run is not None:plot(out,scene,run,data)
        save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

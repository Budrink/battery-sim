#!/usr/bin/env python3
"""Plot current training/held-out trajectories with strict shared provenance."""
import argparse
import gzip
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import upv_r123_data as dataio
from upv_cooling_review import phase_report
from upv_frozen_validation import digest
from upv_observation_policy import CALIBRATION_CHANNELS,metadata as observation_policy
from realtime_guard import RealtimeFloor

LABELS={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS',
        'C2_TIM':'C2 / TIM','pressure_bar':'Давление камеры'}


def boundary_provenance(scene):
    metadata=scene.get('metadata',{})
    conditioned='cooling_boundary_identification'in metadata
    return dict(validation_refitted=bool(conditioned or metadata.get('frozen_validation',{}).get('parameters_refitted',False)),
        cooling_boundary_conditioned=conditioned,
        validation_scope='Cooling identified from 110--112; other channels checking only, NOT fully blind validation'if conditioned else'Frozen training laws; unmeasured cooling priors')


def main_pressure_peak_time(run):
    if not run.get('gas_observers'):return None
    pressure=np.asarray(dataio.chamber_pressure_pa(run))/1e5
    if not len(pressure)or np.max(pressure)-pressure[0]<.5:return None
    return float(run['time_s'][int(np.argmax(pressure))])


def replay_timing(folder,run,event):
    text=(folder/'cuda.log').read_text()
    start=float(run['started_at_s']);end=float(run['advanced_to_s'])
    native_wall_s=float(run['wall_s']);points=[(0.,start)]
    # A rejected replay preserves the last committed snapshot, while the log
    # can contain later progress. Never append that older snapshot after it.
    # Native progress and snapshot timers can also differ by startup overhead.
    for m in re.finditer(r'Replay: t=([\d.eE+-]+).*?wall_s=([\d.eE+-]+)',text):
        wall_time,sim_time=float(m[2]),float(m[1])
        if (points[-1][0]<wall_time<native_wall_s and
                points[-1][1]<=sim_time<=end):
            points.append((wall_time,sim_time))
    points.append((native_wall_s,end))
    wall,times=np.asarray(points).T
    stages={};windows=[('pre_burst',0.,event-20.),('observed_event_and_relaxation',event-20.,event+60.),
                     ('cooling_tail',event+60.,run['time_s'][-1])]
    peak=main_pressure_peak_time(run)
    if peak is not None:windows.append(('predicted_event_and_relaxation',peak-5.,peak+25.))
    for name,a,b in windows:
        if a>=times[0]and b<=times[-1]and b>a:
            elapsed=float(np.interp(b,times,wall)-np.interp(a,times,wall))
            stages[name]=dict(simulated_s=b-a,estimated_wall_s=elapsed,realtime_ratio=(b-a)/elapsed)
    guard=RealtimeFloor();rolling=[]
    for w,t in points:
        status=guard.observe(w,t)
        if status:rolling.append(status)
    return dict(whole_run_realtime_ratio=(end-start)/native_wall_s,
        timing_scope='preserved committed prefix'if run.get('partial')else'completed replay',
        worst_20s_window=min(rolling,key=lambda s:s['realtime_ratio'])if rolling else None,
        phase_estimates=stages,phase_timing_resolution='Interpolated native progress samples, nominally every 10 wall seconds')


def electrical_boundary_summary(scene,run):
    result={}
    for c in scene['cells']:
        law=c.get('electrical')
        if not law:continue
        e=run['electrical'][c['id']]
        row=dict(current_input=law.get('current_input','measured'))
        if law.get('current_input')=='voltage_limited_setpoint':
            knots=np.asarray(law['current_profile_a']);t0,t1=run['time_s'][0],run['time_s'][-1]
            t=np.unique(np.r_[t0,knots[(knots[:,0]>t0)&(knots[:,0]<t1),0],t1])
            command=np.interp(t,knots[:,0],knots[:,1])
            requested=float(np.sum(np.diff(t)*(command[:-1]+command[1:])/2.))
            actual=float(e['actual_positive_charge_c'][-1]);work=float(e['actual_terminal_work_j'][-1])
            joule=float(run['joule_energy_j'][c['id']][-1]);chemical=float(run['chemistry'][c['id']]['electrical_chemical_input_j'][-1])
            row.update(commanded_positive_charge_c=requested,actual_positive_charge_c=actual,
                charge_deficit_fraction=(requested-actual)/requested if requested else 0.,
                voltage_limited_time_s=e['voltage_limited_time_s'][-1],actual_terminal_work_j=work,
                joule_energy_j=joule,electrical_chemical_input_j=chemical,
                accounted_energy_minus_delivered_work_j=joule+chemical-work,
                measured_actual_current_available=False)
        result[c['id']]=row
    return result


def load_run(folder):
    for name in ('run.json.gz','incomplete-run.json.gz'):
        if (folder/name).exists():
            with gzip.open(folder/name,'rt')as stream:return json.load(stream)
    raise ValueError('No completed or explicitly partial replay: '+str(folder))


def plot(run,data,end,title,path,mode='event'):
    event=dataio.event(data)['time_s'];t=np.asarray(run['time_s'])
    raw=data['time_s'];obs=dataio.observed(data,raw)
    ruptures=[float(e['time_s'])for e in run.get('events',[])if e['kind']=='shell_rupture']
    first_rupture=min(ruptures)if ruptures else None
    left=0. if mode=='overview'else event+20. if mode=='cooling'else event-20.
    main_peak=main_pressure_peak_time(run)
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for ax,key in zip(axes.flat,[*CALIBRATION_CHANNELS,'pressure_bar']):
        pressure=key=='pressure_bar';offset=0. if pressure else 273.15
        mask=(raw>=left)&(raw<=end)
        ax.plot(raw[mask],obs[key][mask]-offset,color='#20242b',lw=1.35,label='Эксперимент')
        values=np.asarray(dataio.chamber_pressure_pa(run))/1e5 if pressure else np.asarray(run['temperature_k'][key])
        known=(t>=left)&(t<=end)
        if known.any():ax.plot(t[known],values[known]-offset,color='#1479bd',lw=1.6,label='Расчёт')
        if t[-1]<end-1e-7:
            if t[-1]>=left:ax.axvline(t[-1],color='#b74a3b',ls='--',lw=1.)
            ax.text(.97,.93,f'Расчёт до {t[-1]:.1f} с',ha='right',transform=ax.transAxes,
                    fontsize=9,color='#a0392b',bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        burst_left,burst_right=max(left,event-5.),min(end,event+20.)
        if burst_right>burst_left:ax.axvspan(burst_left,burst_right,color='#dba85b',alpha=.14)
        if first_rupture is not None and left<=first_rupture<=end:
            ax.axvline(first_rupture,color='#888888',ls=':',lw=1.,label='Первое вскрытие, не взрыв')
        if mode!='cooling'and main_peak is not None and left<=main_peak<=end:
            ax.axvline(main_peak,color='#1479bd',ls=':',lw=.8,label='Пик давления расчёта')
        ax.set(title=LABELS[key],xlabel='Время от начала опыта, с',ylabel='бар'if pressure else '°C',xlim=(left,end))
        ax.grid(alpha=.18);ax.legend(fontsize=8,loc='best')
    fig.suptitle(title+'\nC2/ISS исключена; без подгонки времени и масштаба',fontsize=13)
    fig.savefig(path,dpi=145);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--validation',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False)
    training=json.loads((args.calibration/'trial-000/scenario.json').read_text())
    heldout=json.loads((args.validation/'scenario.json').read_text())
    cal_status=json.loads((args.calibration/'status.json').read_text())
    val_status=json.loads((args.validation/'status.json').read_text())
    if cal_status.get('accepted_calibration')and cal_status.get('best',{}).get('trial','trial-000')!='trial-000':
        raise ValueError('Select the actual accepted trial explicitly; trial-000 is not this folder\'s best scene')
    if heldout['metadata']['frozen_validation']['frozen_scene_sha256']!=digest(training):
        raise ValueError('Validation does not use these actual training laws')
    if cal_status['native_sha256']!=val_status['native_sha256']:
        raise ValueError('Training and validation use different calculation cores')
    report=dict(observation_policy=observation_policy(),native_sha256=cal_status['native_sha256'],
                frozen_scene_sha256=digest(training),**boundary_provenance(heldout),
                calibration_accepted=bool(cal_status.get('accepted_calibration',False)),experiments={})
    report['first_shell_opening_is_not_a_runaway_marker']=True
    modified_current=any(c.get('electrical',{}).get('current_input')=='voltage_limited_setpoint'for c in heldout['cells'])
    report['validation_protocol_faithful']=not modified_current
    report['validation_accepted']=bool(val_status.get('accepted_validation',False)and not modified_current)
    if modified_current:
        report['validation_limitation']='DIAGNOSTIC ONLY: measured terminal U used as a source envelope; published protocol holds 35 A with 80 V source compliance, not compliance at measured U'
    validation_label=('Диагностика ядра с изменённой границей тока — НЕ физическая валидация'if modified_current else
        'Проверка с охлаждением, определённым по 110–112; химия заморожена'if report['cooling_boundary_conditioned']else
        'Валидация с охлаждением — параметры заморожены')
    for name,scene,folder,status,label in [
        ('no-cooling',training,args.calibration/'trial-000/replay',cal_status,'Калибровочный опыт — лучший текущий набор'),
        ('active-cooling',heldout,args.validation/'replay',val_status,validation_label)]:
        run=load_run(folder);data,_=dataio.load(name);end=scene['simulation']['duration_s']
        pressure=np.asarray(dataio.chamber_pressure_pa(run))/1e5
        imax=int(np.argmax(pressure));ruptures=[e for e in run.get('events',[])if e['kind']=='shell_rupture']
        observed=dataio.observed(data,data['time_s']);mask=data['time_s']<=end
        op=observed['pressure_bar'][mask];ot=data['time_s'][mask];oi=int(np.argmax(op))
        suffix=' (НЕПОЛНЫЙ ПРОГОН)'if run.get('partial')or run['time_s'][-1]<end-1e-7 else ''
        for mode in ('overview','event','cooling'):
            plot(run,data,end,label+suffix,out/f'{name}-{mode}.png',mode)
        if suffix:
            plot(run,data,float(run['time_s'][-1]),label+' — ТОЛЬКО СОХРАНЁННЫЙ ПРЕФИКС',
                 out/f'{name}-known-prefix.png','overview')
        report['experiments'][name]=dict(status=status.get('status'),error=status.get('error'),
            requested_end_s=end,known_until_s=run['time_s'][-1],wall_s=run['wall_s'],
            minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
            phases=phase_report(run,data,end),
            timing=replay_timing(folder,run,dataio.event(data)['time_s']),
            electrical_boundary=electrical_boundary_summary(scene,run),
            predicted_shell_ruptures=ruptures,
            first_rupture_error_s=min(e['time_s']for e in ruptures)-dataio.event(data)['time_s']if ruptures else None,
            pressure_peak_bar=float(pressure[imax]),pressure_peak_time_s=float(run['time_s'][imax]),
            observed_pressure_peak_bar=float(op[oi]),observed_pressure_peak_time_s=float(ot[oi]),
            pressure_peak_time_error_s=float(run['time_s'][imax]-ot[oi]),
            pressure_peak_relative_error=float(abs(pressure[imax]-op[oi])/op[oi]),
            predicted_peak_sample_spacing_s=float(max(np.diff(run['time_s'][max(0,imax-1):imax+2]),default=0.)),
            source_assumptions=scene.get('metadata',{}).get('frozen_validation',{}),
            balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':main()

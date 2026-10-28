#!/usr/bin/env python3
"""Same-core coarse-floor check against a completed fresh geometry reference.

The 5% budget is time-weighted RMS and endpoint error divided by the reference
excursion, not by absolute Kelvin or resident chamber air. Pointwise errors are
also exposed, not claimed to be bounded by 5%. Mass/energy ledgers stay strict.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import numpy as np
import calibrate_upv_r123 as cal
from check_upv_restart import native
from probe_upv_gas_access import plot
from upv_gas_access import CHANNELS
import upv_r123_data as dataio


def series_error(times,reference,coarse,floor):
    t=np.asarray(times);a=np.asarray(reference);b=np.asarray(coarse)
    if len(t)<2 or np.any(np.diff(t)<=0)or a.shape!=t.shape or b.shape!=t.shape:
        raise ValueError('Invalid numerical comparison series')
    if not np.isfinite(a).all()or not np.isfinite(b).all():raise ValueError('Nonfinite comparison')
    scale=max(float(np.ptp(a)),floor);d=b-a
    rms=float(np.sqrt(np.trapz(d*d,t)/(t[-1]-t[0])))
    return dict(reference_excursion=scale,time_weighted_nrmse=rms/scale,
        normalized_endpoint_error=abs(float(d[-1]))/scale,
        maximum_normalized_error=float(np.max(np.abs(d)))/scale,
        maximum_absolute_error=float(np.max(np.abs(d))))


def compare(reference,run):
    if reference['time_s']!=run['time_s']:raise ValueError('Observation times changed')
    times=np.asarray(reference['time_s']);keep=times>=881.;t=times[keep];metrics={}
    def add(name,a,b,floor,peak=False):
        a=np.asarray(a)[keep];b=np.asarray(b)[keep];row=series_error(t,a,b,floor)
        if peak:
            row['normalized_peak_height_error']=abs(float(np.max(a)-np.max(b)))/row['reference_excursion']
            row['peak_time_difference_s']=float(t[np.argmax(b)]-t[np.argmax(a)])
        row['accepted']=(row['time_weighted_nrmse']<=.05 and row['normalized_endpoint_error']<=.05
            and row.get('normalized_peak_height_error',0.)<=.05)
        metrics[name]=row
    for key in CHANNELS:add('temperature/'+key,reference['temperature_k'][key],run['temperature_k'][key],1.,True)
    add('pressure/chamber',dataio.chamber_pressure_pa(reference),dataio.chamber_pressure_pa(run),100.,True)
    add('mass/chamber',reference['gas_observers']['chamber']['mass_kg'],run['gas_observers']['chamber']['mass_kg'],1e-8)
    for group in ['inside-C2-']+['gap-'+p+'-'for p in ['ISS1','TIM1','ISS2','TIM2','ISS3','TIM3']]:
        names=[n for n in reference['gas']if n.startswith(group)]
        if not names:raise ValueError('Missing local gas group: '+group)
        for field,floor in [('mass_kg',1e-8),('chemical_j',1.)]:
            a=np.sum([reference['gas'][n][field]for n in names],axis=0)
            b=np.sum([run['gas'][n][field]for n in names],axis=0)
            add('local-gas/'+group+'/'+field,a,b,floor)
    def heats(r):
        result={}
        for row in r['gas_film_trace'].values():
            name=row['solid'];result.setdefault(name,np.zeros(len(times)))
            result[name]+=row['heat_to_solid_j']
        return result
    a,b=heats(reference),heats(run)
    if a.keys()!=b.keys():raise ValueError('Gas surface topology changed')
    for name in a:add('gas-heat/'+name,a[name],b[name],1.)
    failed={k:v for k,v in metrics.items()if not v['accepted']}
    return dict(accepted=not failed,tolerance=.05,interval_s=[float(t[0]),float(t[-1])],metrics=metrics,
        failures=list(failed),criterion='Time-weighted NRMSE, endpoint, and temperature/pressure peak-height excursion errors <=5%; pointwise error separately reported',
        conservation_relaxed=False,no_time_shift=True,
        normalization_floors=dict(temperature_k=1.,pressure_pa=100.,mass_kg=1e-8,energy_j=1.))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=Path,required=True);ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--floor',type=float,default=.04)
    a=ap.parse_args();seed=a.seed.resolve();out=a.output.resolve();out.mkdir(exist_ok=False)
    cal.PROBE=a.probe.resolve();contract=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    prior=json.loads((seed/'status.json').read_text())
    if prior['native_sha256']!=contract or prior['status']!='complete':raise ValueError('Require completed exact-core reference')
    with gzip.open(seed/'trial-000/replay/run.json.gz','rt')as stream:reference=json.load(stream)
    if reference['partial']or reference['advanced_to_s']!=950.:raise ValueError('Incomplete reference')
    original=json.loads((seed/'trial-000/scenario.json').read_text());scene=copy.deepcopy(original)
    if not original['simulation']['minimum_dt_s']<a.floor<=.0446975:
        raise ValueError('Coarse floor must exceed reference floor and stay below ordinary thermal stability step')
    scene['simulation']['minimum_dt_s']=a.floor;cal.save(out/'scenario.json',scene)
    state=dict(status='running',pid=os.getpid(),reference=str(seed),native_sha256=contract,
        tolerance=.05,conservation_relaxed=False,only_minimum_dt_changed=True,calibration=False)
    cal.save(out/'status.json',state)
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=contract,
            restart_from=str(seed/'sealed-state.bin'),restart_gas_response_scenario=str(seed/'prefix-scenario.json'),
            stop_after_first_rupture=False),700.)
        report=compare(reference,run)
        report.update(reference_wall_s=reference['wall_s'],coarse_wall_s=run['wall_s'],
            speedup=reference['wall_s']/run['wall_s'],realtime_ratio=115./run['wall_s'],
            minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
            balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k})
        cal.save(out/'comparison.json',report)
        data,_=dataio.load('no-cooling')
        plot(out,{'Эталон dt ≥ 0,02 с':reference,'Огрубление dt ≥ 0,04 с':run},data,
            title='Проверка временного огрубления: физика и геометрия фиксированы\nБез подгонки параметров, сдвига времени и ослабления балансов')
        state.update(status='complete',accepted=report['accepted'],speedup=report['speedup'],failures=report['failures'])
    except KeyboardInterrupt:state['status']='stopped'
    except BaseException as exc:state.update(status='failed',error=str(exc));raise
    finally:cal.save(out/'status.json',state)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Calibrate only heater control on the fresh aligned six-zone FSRI scene."""
import argparse
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import signal
import time

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_contact_heater import metrics, next_gain
from calibrate_fsri30_sequence import assess_run, physical_error
from check_upv_restart import native
from fsri30_data import load,save,sha
from fsri30_early_stop import rejection
from fsri30_wave_fit import reference

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/development/fsri30-aligned-325-seed40-20261027-r1/scenario.json'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-packed-precise-progress-20261027-r1'
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'
HORIZON=1500.


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--max-trials',type=int,default=12)
    p.add_argument('--source',type=Path,default=SOURCE,help='Fresh initial scenario, with inherited chemistry already included')
    p.add_argument('--r2-scale',type=float,default=1.,help='Shared R2 multiplier applied to the source scenario')
    p.add_argument('--initial-gain',type=float,default=.21405515446686751)
    p.add_argument('--allow-early-primary',action='store_true',help='Look for secondary conversion even if the first peak is early')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if not a.resume:out.mkdir(parents=True,exist_ok=False)
    if sha(PROBE)!=CORE:raise ValueError('Pinned CUDA core changed')
    if not .001<=a.r2_scale<=2. or not .125<=a.initial_gain<=16.:raise ValueError('Joint search parameter outside bounds')
    source=a.source.resolve();base=json.loads(source.read_text());data,manifest=load();ref=reference(data)
    experiment={'time_s':data['time_s'].tolist()}
    for i in range(1,10):
        experiment[f'T{i}']=[float(v-273.15) if math.isfinite(v) else None for v in data[f'T{i}_k']]
    atomic_save(out/'experiment.json',experiment)
    warmup_plot=None
    def publish(run,phase):
        nonlocal warmup_plot
        times=list(run.get('time_s',[])); cells=run.get('cell_mean_temperature_k',{})
        if phase=='cascade' and warmup_plot:
            old_t=warmup_plot['time_s']; take=sum(t<times[0]-1e-7 for t in old_t) if times else len(old_t)
            times=old_t[:take]+times
            cells={name:values[:take]+list(cells.get(name,[])) for name,values in warmup_plot['cells'].items()}
        else:cells={name:list(values) for name,values in cells.items()}
        atomic_save(out/'live.json',dict(phase=phase,time_s=times,
            cells={name:[float(v-273.15) for v in values] for name,values in cells.items()},
            simulated_s=run.get('advanced_to_s',times[-1] if times else 0.),wall_s=run.get('wall_s')))
    for cell in base['cells']:
        if cell.get('role')=='battery':cell['conversion']['reaction_rate_per_s']*=a.r2_scale
    heater=base['heating_protocols'][0];heater.update(sensor_id='B10:case:+x',feedforward_w=0.)
    base['metadata']['heater_control_hypothesis']='Opposite +x can panel as proxy, not measured thermocouple location'
    state=dict(status='running',pid=os.getpid(),stage='heater-calibration',trials=[],
        source_sha256=sha(source),core_sha256=CORE,experiment_sha256=manifest['csv_sha256'],
        sensor_face='+x',sensor_mapping_confirmed=False,feedforward_w=0.,max_power_w=heater['maximum_power_w'],
        r2_multiplier_on_source=a.r2_scale,initial_gain=a.initial_gain,early_primary_allowed=a.allow_early_primary,
        fixed='geometry, materials, contacts, other chemistry, heater ramp and maximum power',calibrated=False,validation_passed=False)
    rows=[]
    if a.resume:
        state=json.loads((out/'status.json').read_text())
        if (state.get('source_sha256')!=sha(source) or state.get('core_sha256')!=CORE or
            state.get('stage')!='heater-calibration' or state.get('r2_multiplier_on_source',1.)!=a.r2_scale):
            raise ValueError('Resume requires the same warmup scenario and core')
        rows=list(state.get('trials',[]))
        if state.get('active_trial') and state.get('gain') not in [r['gain'] for r in rows]:
            partial=Path(state['active_trial'])/'replay/incomplete-run.json.gz'
            if partial.exists():
                with gzip.open(partial,'rt') as f:previous=json.load(f)
                m=metrics(previous,data)
                if m and m['max_abs_k']>10.:
                    rows.append(dict(gain=state['gain'],status='pruned',reason='Saved prefix already exceeds 10 K acceptance limit',
                        metrics=m,last_observed_s=previous['time_s'][-1],run_path=str(partial),run_sha256=sha(partial)))
                else:raise ValueError('Interrupted prefix does not justify rejection')
        state.update(status='running',pid=os.getpid(),trials=rows,progress={})
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);atomic_save(out/'status.json',state)
    try:
        for index in range(len(rows),a.max_trials):
            gain=next_gain(rows,a.initial_gain)
            if gain is None:break
            scene=deepcopy(base);scene['heating_protocols'][0]['proportional_w_per_k']=gain
            scene['simulation']['duration_s']=HORIZON
            scene['failure_sample_times_s']=[t for t in scene['failure_sample_times_s'] if t<=HORIZON]
            folder=out/f'heater-{index:02d}';folder.mkdir();save(folder/'scenario.json',scene)
            state.update(stage='heater-calibration',active_trial=str(folder),gain=gain,progress={});atomic_save(out/'status.json',state)
            def inspect(run):
                publish(run,'warmup')
                error=physical_error(run);m=metrics(run,data)
                state['progress']=dict(time_s=run['time_s'][-1],wall_s=run.get('wall_s'),metrics=m,physical_error=error)
                atomic_save(out/'status.json',state)
                if error:return error
                if m and m['final_time_s']>=300.:
                    count=sum(100.<=t<=HORIZON for t in run['time_s'])
                    lower=m['rmse_k']*math.sqrt(count/281.)
                    if m['max_abs_k']>10. or lower>5.:return 'Warmup cannot meet the required error bound'
                return None
            run=None;reason=None;status='complete'
            try:
                run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=CORE,stop_after_first_rupture=False,
                    save_committed_restart=str(folder/'committed-1500.bin'),committed_restart_at_s=HORIZON),
                    timeout=HORIZON/.2+120.,prune_fn=inspect,minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
            except (RuntimeError,TimeoutError) as exc:
                reason=str(exc);status='pruned' if reason.startswith('Pruned:') else 'failed'
                f=folder/'replay/incomplete-run.json.gz'
                if f.exists():
                    with gzip.open(f,'rt') as stream:run=json.load(stream)
            row=dict(gain=gain,status=status,reason=reason,metrics=metrics(run,data) if run else None,
                     last_observed_s=run['time_s'][-1] if run else None)
            checkpoint=folder/'committed-1500.bin'
            if status=='complete' and checkpoint.exists():
                row.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                           warmup_history=str(folder/'replay/run.json.gz'))
            rows.append(row);save(folder/'result.json',row);state['trials']=rows;atomic_save(out/'status.json',state)
            if status=='failed':state.update(status='numerical-review-needed',reason=reason);return
        valid=[r for r in rows if r['status']=='complete' and r['metrics']]
        if not valid:state.update(status='heater-fit-insufficient',reason='No complete warmup');return
        best=min(valid,key=lambda r:r['metrics']['rmse_k'])
        if best['metrics']['rmse_k']>5. or best['metrics']['max_abs_k']>10.:
            state.update(status='heater-fit-insufficient',best=best);return
        fitted=deepcopy(base);fitted['heating_protocols'][0]['proportional_w_per_k']=best['gain']
        fitted['metadata']['heater_calibration_candidate']=dict(gain=best['gain'],metrics=best['metrics'],
            target='B10 mean vs T5',mapping_confirmed=False,qualification='warmup fit only')
        save(out/'selected-scenario.json',fitted);state.update(stage='cascade-check',best=best,progress={});atomic_save(out/'status.json',state)
        folder=out/'cascade';folder.mkdir();save(folder/'scenario.json',fitted)
        with gzip.open(best['warmup_history'],'rt') as stream:
            previous=json.load(stream)
        warmup_plot=dict(time_s=previous['time_s'],cells=previous['cell_mean_temperature_k'])
        def check(run):
            publish(run,'cascade')
            error=physical_error(run);assessment=assess_run(run,ref,20,None)
            temp=rejection(fitted,run,data) if not a.allow_early_primary and len(run.get('time_s',[]))>1 else None
            secondary=[]
            for cell in fitted['cells']:
                if cell.get('role')!='battery' or cell['id']=='B10':continue
                threshold=.1*cell['conversion']['mass_kg']*cell['conversion']['gas_yield']
                for t,v in zip(run['time_s'],run.get('chemistry',{}).get(cell['id'],{}).get('r2_generated_gas_kg',[])):
                    if v>=threshold:secondary.append(dict(cell=cell['id'],time_s=t));break
            secondary.sort(key=lambda e:e['time_s'])
            state['progress']=dict(time_s=run['time_s'][-1],wall_s=run.get('wall_s'),assessment=assessment,
                                   physical_error=error,temperature_rejection=temp,secondary_events=secondary)
            atomic_save(out/'status.json',state)
            if error:return error
            if any(values and max(values)>3500. for values in run.get('cell_mean_temperature_k',{}).values()):
                return 'Critical component mean temperature above 3500 K'
            if a.allow_early_primary:
                pulses=assessment.get('model',{}).get('pulse_groups',[])
                if pulses and not secondary and run['time_s'][-1]>=pulses[0]['time_s']+471.:
                    return 'No secondary R2 event within experimental first-to-second interval plus 100 s'
                if run['time_s'][-1]>=3000.:return 'No two secondary conversions through 3000 s'
            elif assessment.get('accepted') is False:return assessment.get('reason')
            if temp:return temp['reason']
            return None
        try:
            request=dict(restart_contract=CORE,
                save_committed_restart=str(folder/'committed-1800.bin'),committed_restart_at_s=1800.,
                stop_after_first_rupture=False)
            if best.get('checkpoint'):
                if sha(Path(best['checkpoint']))!=best['checkpoint_sha256']:raise ValueError('Warmup checkpoint changed')
                request['restart_from']=best['checkpoint']
                state['restart_from_warmup']=best['checkpoint'];state['warmup_history']=best['warmup_history']
                atomic_save(out/'status.json',state)
            run=native(folder/'scenario.json',folder/'replay',request,timeout=18120.,prune_fn=check,
                minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
            check(run);state.update(status='cascade-check-complete',note='Observed fit, not independent validation')
        except (RuntimeError,TimeoutError) as exc:
            status=('secondary-observed' if 'Observed two secondary R2 conversion events' in str(exc)
                    else 'cascade-pruned' if str(exc).startswith('Pruned:') else 'numerical-review-needed')
            state.update(status=status,reason=str(exc))
    except KeyboardInterrupt:state.update(status='stopped')
    except Exception as exc:state.update(status='failed',reason=str(exc));raise
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

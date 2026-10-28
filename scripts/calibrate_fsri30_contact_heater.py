#!/usr/bin/env python3
"""Fit only the heater controller for 0.5 mm exploratory can strips.

T5 is used as a proxy for heated B10's mean, not an identified point sensor.
Each trial is fresh and keeps material/contact laws, the chosen shared R2 seed
and the reported heater ramp fixed. No temperatures or failures are imposed.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import signal

import numpy as np

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import BASE, BASE_HASH, MINIMUM_SPEED, physical_error
from check_fsri30_vent_retention import SEED
from check_upv_restart import native
from fsri30_can_contacts import add_can_contacts
from fsri30_data import load, save, sha
from fsri30_external_screen import exterior_screen
from fsri30_packed_geometry import close_packed
from fsri30_packed_pockets import split_tangent_pockets
from fsri30_packed_cascade_search import search
from fsri30_sequence_model import candidate

PROBE=Path('build/calibration-probes/battery_sim_casing_probe-joint-enthalpy-20261026-r1').resolve()
CORE='6aa586496f59297caa81356c0fff28a17ffe733ba4a9693fb6f4a7e951f7813e'
HORIZON=1500.


def metrics(run, data):
    t=np.asarray(run['time_s']); y=np.asarray(run['cell_mean_temperature_k']['B10'])
    mask=(t>=100.) & (t<=HORIZON) & (t<=data['time_s'][-1])
    if mask.sum()<10: return None
    truth=np.interp(t[mask],data['time_s'],data['T5_k'])
    err=y[mask]-truth
    return dict(rmse_k=float(np.sqrt(np.mean(err**2))), max_abs_k=float(np.max(np.abs(err))),
        final_error_k=float(err[-1]), final_time_s=float(t[mask][-1]),
        final_mean_c=float(y[mask][-1]-273.15), final_t5_c=float(truth[-1]-273.15))


def next_gain(rows,initial_gain=1.):
    if not rows: return initial_gain
    if len(rows)==1:
        m=rows[0].get('metrics');g=rows[0]['gain']
        if m and m['final_time_s']>=HORIZON-1. and m['rmse_k']<=5. and m['max_abs_k']<=10.:return None
        if m and m['max_abs_k']>20.:
            value=max(.125,g/4.) if m['final_error_k']>0. else min(16.,g*4.)
            return None if value==g else value
        if g<3.:return 4.
        return min(16.,g*2.) if m and m['final_error_k']<0 else 1.
    # Failed early prefixes constrain the sign of an overheating/underheating
    # bracket, but are never eligible as completed scores or fine validation.
    valid=[r for r in rows if r.get('metrics') and r['metrics']['final_time_s']>=150.]
    if len(valid)<2:return None
    if any(r['status']=='complete' and r['metrics']['rmse_k']<=5. and r['metrics']['max_abs_k']<=10. for r in valid):return None
    low=[r for r in valid if r['metrics']['final_error_k']<0]
    high=[r for r in valid if r['metrics']['final_error_k']>=0]
    if low and high:
        pairs=[(a,b) for a in low for b in high if a['gain']<b['gain']]
        if not pairs:return None
        a,b=min(pairs,key=lambda p:p[1]['gain']-p[0]['gain'])
        f=max(.2,min(.8,-a['metrics']['final_error_k']/(b['metrics']['final_error_k']-a['metrics']['final_error_k'])))
        value=math.exp((1-f)*math.log(a['gain'])+f*math.log(b['gain']))
    elif low:value=min(16.,max(r['gain'] for r in valid)*2.)
    else:value=max(.125,min(r['gain'] for r in valid)/2.)
    return None if any(abs(value-r['gain'])<1e-6 for r in rows) else value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-trials',type=int,default=12)
    parser.add_argument('--prior-prefix',type=Path,help='Preserved same-scene 80 mm gain=1 censored prefix')
    parser.add_argument('--sensor-face',choices=('-x','+x'),default='-x',help='Unmeasured heater control sensor face')
    parser.add_argument('--feedforward-w',type=float,default=4.5,help='Unmeasured base controller power')
    parser.add_argument('--initial-gain',type=float,default=1.)
    parser.add_argument('--r2-seed-factor',type=float,default=1.,help='Shared multiplier of inherited R2 rate')
    args=parser.parse_args()
    if not 0<=args.feedforward_w<=30 or not .125<=args.initial_gain<=16:raise ValueError('Controller prior out of bounds')
    if not .001<=args.r2_seed_factor<=20:raise ValueError('Shared R2 factor out of bounds')
    if sha(BASE)!=BASE_HASH or sha(PROBE)!=CORE:raise ValueError('Pinned base/core changed')
    seed=json.loads((SEED/'status.json').read_text())['parameters']
    raw=json.loads(BASE.read_text())
    def scene_at(level):
        base=candidate(exterior_screen(raw,level),seed)
        return add_can_contacts(split_tangent_pockets(close_packed(base)),.0005,1000.)
    base=scene_at(20);screen=scene_at(80)
    for scene in (base,screen):
        scene['heating_protocols'][0]['sensor_id']='B10:case:'+args.sensor_face
        scene['heating_protocols'][0]['feedforward_w']=args.feedforward_w
        scene['metadata']['heater_protocol']['sensor_location_prior']='reference-capacity mean of heated can '+args.sensor_face+' panel; exact TC position unknown'
        for cell in scene['cells']:
            if cell.get('role')=='battery':cell['conversion']['reaction_rate_per_s']*=args.r2_seed_factor
        scene['metadata']['r2_transfer']['used_as_unmodified_law']=False
    data,manifest=load();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'contract.json',dict(base_sha256=BASE_HASH,seed=seed,core=CORE,csv_sha256=manifest['csv_sha256'],
        strip_width_m=.0005,contact_h_w_m2_k=1000.,contact_h_measured=False,
        target='B10 volume mean versus raw T5; mapping unconfirmed',
        fit_parameters=['heater proportional gain'],gain_bounds=[.125,16.],heater_ramp_k_s=.1,
        assumed_control_sensor_face=args.sensor_face,control_sensor_position_measured=False,
        feedforward_w=args.feedforward_w,feedforward_measured=False,
        shared_r2_seed_factor=args.r2_seed_factor,rate_calibrated=False,
        packed_positions_and_internal_passages_rebuilt=True,
        heater_screen_external_mm=80,heater_confirmation_external_mm=20,
        gas_partition_limit='40 tangent pockets and outer space; each pocket well-mixed',
        fresh_from_zero=True,calibrated=False,validation_passed=False))
    state=dict(status='running',pid=os.getpid(),stage='heater-calibration',trials=[],calibrated=False,validation_passed=False)
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);atomic_save(out/'status.json',state)
    try:
        rows=[]
        if args.prior_prefix:
            prior_scene_path=args.prior_prefix.parent.parent/'scenario.json'
            prior_scene=json.loads(prior_scene_path.read_text())
            for key in ('cells','shells','interfaces','ports','boundaries','gas_network','heating_protocols'):
                if prior_scene[key]!=screen[key]:raise ValueError('Prior prefix physics mismatch: '+key)
            prior=json.load(gzip.open(args.prior_prefix,'rt'));m=metrics(prior,data)
            if not m or m['final_time_s']<150. or m['max_abs_k']<=20.:
                raise ValueError('Prior prefix does not demonstrate a gross warmup error')
            rows.append(dict(gain=1.,status='censored',reason='Prior bad warmup prefix preserved',metrics=m,
                last_observed_s=prior['time_s'][-1],run_path=str(args.prior_prefix.resolve()),run_sha256=sha(args.prior_prefix)))
            state['trials']=rows;atomic_save(out/'status.json',state)
        def warmup(reference,gain,index,stage):
            scene=json.loads(json.dumps(reference));scene['heating_protocols'][0]['proportional_w_per_k']=gain
            scene['metadata']['heater_calibration_candidate']=dict(gain=gain,target='B10 mean vs T5',mapping_confirmed=False)
            scene['simulation']['duration_s']=HORIZON
            scene['failure_sample_times_s']=[t for t in scene['failure_sample_times_s'] if t<=HORIZON]
            if scene['failure_sample_times_s'][-1]!=HORIZON:scene['failure_sample_times_s'].append(HORIZON)
            folder=out/f'{stage}-{index:02d}';folder.mkdir();save(folder/'scenario.json',scene)
            state.update(stage=stage,active_trial=index,gain=gain,progress={});atomic_save(out/'status.json',state)
            def inspect(run):
                err=physical_error(run)
                if err:return err
                m=metrics(run,data);state['progress']=dict(time_s=run['time_s'][-1],wall_s=run['wall_s'],metrics=m)
                atomic_save(out/'status.json',state)
                if m and m['final_time_s']>=150:
                    count=sum(100.<=t<=HORIZON for t in run['time_s'])
                    lower_bound=m['rmse_k']*math.sqrt(count/281.)
                    if m['max_abs_k']>20 or lower_bound>8.:
                        return 'Warmup candidate cannot satisfy final error bound'
                return None
            try:
                run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=CORE,stop_after_first_rupture=False),
                    timeout=HORIZON/MINIMUM_SPEED+120,prune_fn=inspect,minimum_speed=MINIMUM_SPEED,
                    rolling_minimum=MINIMUM_SPEED,probe=PROBE)
                reason=physical_error(run);status='complete' if not reason else 'invalid'
            except (RuntimeError,TimeoutError) as exc:
                path=folder/'replay/incomplete-run.json.gz'
                run=json.load(gzip.open(path,'rt')) if path.exists() else None
                reason=str(exc);status='pruned' if reason.startswith('Pruned:') else 'failed'
            row=dict(gain=gain,status=status,reason=reason,metrics=metrics(run,data) if run else None,
                     last_observed_s=run['time_s'][-1] if run else None)
            save(folder/'result.json',row)
            return row
        for index in range(args.max_trials):
            gain=next_gain(rows,args.initial_gain)
            if gain is None:break
            row=warmup(screen,gain,index,'heater-coarse')
            rows.append(row);state['trials']=rows;atomic_save(out/'status.json',state)
            if row['status']=='failed':state['status']='numerical-review-needed';return
        valid=[r for r in rows if r['status']=='complete' and r['metrics']]
        if not valid:state['status']='no-complete-heater-trial';return
        best=min(valid,key=lambda r:r['metrics']['rmse_k'])
        # Independently recompute the selected candidate on the physical 20 mm
        # exterior grid; never inherit its coarse score or state.
        coarse_best=best;fine=[]
        for index in range(4):
            gain=coarse_best['gain'] if not fine else next_gain(fine)
            if gain is None:break
            row=warmup(base,gain,index,'heater-fine')
            fine.append(row);state['fine_trials']=fine;atomic_save(out/'status.json',state)
            if row['status']=='failed':state['status']='numerical-review-needed';return
            if row['status']=='complete' and row['metrics']['rmse_k']<=3. and row['metrics']['max_abs_k']<=8.:break
        valid=[r for r in fine if r['status']=='complete' and r['metrics']]
        if not valid:state.update(status='no-complete-fine-heater-trial',coarse_best=coarse_best);return
        best=min(valid,key=lambda r:r['metrics']['rmse_k'])
        if best['metrics']['rmse_k']>8. or best['metrics']['max_abs_k']>20.:
            state.update(status='heater-fit-insufficient',best=best,coarse_best=coarse_best);return
        fitted=json.loads(json.dumps(base));fitted['heating_protocols'][0]['proportional_w_per_k']=best['gain']
        fitted['metadata']['heater_calibration_candidate']=dict(gain=best['gain'],target='B10 mean vs T5',
            mapping_confirmed=False,metrics=best['metrics'],qualification='fitted proxy, not validation')
        save(out/'selected-scenario.json',fitted);state.update(stage='cascade-search',best=best,progress={});atomic_save(out/'status.json',state)
        search(fitted,out,state,data,PROBE,metrics)
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',reason=str(exc));raise
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

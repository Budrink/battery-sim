#!/usr/bin/env python3
"""Fresh coarse packed-module screening; never a fine-grid validation."""
import argparse
from dataclasses import replace
import gzip
import json
import math
import os
from pathlib import Path

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_contact_heater import metrics
from calibrate_fsri30_sequence import BASE, BASE_HASH, physical_error
from check_fsri30_vent_retention import SEED
from check_upv_restart import native
from fsri30_can_contacts import add_can_contacts
from fsri30_data import load, save, sha
from fsri30_external_screen import exterior_screen
from fsri30_objective import model_fields
from fsri30_packed_geometry import close_packed
from fsri30_packed_pockets import split_tangent_pockets
from fsri30_sequence_model import candidate
from fsri30_wave_fit import POLICY, assess_fields, reference


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--probe-sha256',required=True);ap.add_argument('--r2-seed-factor',type=float,required=True)
    ap.add_argument('--contact-h',type=float,default=1000.,help='Shared, unmeasured metal contact conductance W/(m² K)')
    ap.add_argument('--gain',type=float,default=.21405515446686751);args=ap.parse_args()
    out=args.output.resolve();probe=args.probe.resolve()
    if sha(BASE)!=BASE_HASH or sha(probe)!=args.probe_sha256:raise ValueError('Pinned source/core mismatch')
    if not .001<=args.r2_seed_factor<=20.:raise ValueError('Invalid R2 seed factor')
    seed=json.loads((SEED/'status.json').read_text())['parameters']
    base=candidate(exterior_screen(json.loads(BASE.read_text()),80),seed)
    scene=add_can_contacts(split_tangent_pockets(close_packed(base)),.0005,args.contact_h)
    scene['gas_network']['averaged_pressure_iterations']=8192
    scene['heating_protocols'][0].update(sensor_id='B10:case:+x',feedforward_w=0.,proportional_w_per_k=args.gain)
    for c in scene['cells']:
        if c.get('role')=='battery':c['conversion']['reaction_rate_per_s']*=args.r2_seed_factor
    scene.setdefault('process_history_selection',{}).setdefault('gas_prefixes',[]).append('inside-B10-')
    out.mkdir(parents=True,exist_ok=False);save(out/'scenario.json',scene)
    data,manifest=load();ref=reference(data)
    coarse_policy=replace(POLICY,timing_limit_s=250.)
    state=dict(status='running',pid=os.getpid(),scenario_sha256=sha(out/'scenario.json'),core_sha256=args.probe_sha256,
        csv_sha256=manifest['csv_sha256'],r2_seed_factor=args.r2_seed_factor,gain=args.gain,
        coarse_spacing_mm=80,primary_search_allowance_s=250.,contact_h_w_m2_k=args.contact_h,
        contact_h_measured=False,calibrated=False,validation_passed=False)
    atomic_save(out/'status.json',state)
    def inspect(run):
        err=physical_error(run)
        if err:return err
        t=run['time_s'][-1];m=metrics(run,data);events=[];wave=None
        if len(run['time_s'])>=2:
            ts,ids,values=model_fields(run);wave=assess_fields(ts,values,ids,ref,coarse_policy)
            events=wave['model']['pulse_groups']
        primary=events[0]['time_s'] if events else None
        state['progress']=dict(time_s=t,wall_s=run['wall_s'],warmup_metrics=m,primary_pulse_s=primary,
                               pulse_groups=[dict(time_s=e['time_s'],channels=e['channels']) for e in events],
                               wave_reason=wave['reason'] if wave else None)
        atomic_save(out/'status.json',state)
        if m and m['final_time_s']>=150.:
            count=sum(100.<=x<=1500. for x in run['time_s'])
            if m['max_abs_k']>20. or m['rmse_k']*math.sqrt(count/281.)>8.:
                return 'Coarse warmup cannot satisfy the accepted temperature bound'
        if wave and wave['accepted'] is False:
            return 'Coarse all-wave screening: '+wave['reason']
        return None
    try:
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=args.probe_sha256,
            stop_after_first_rupture=False,save_committed_restart=str(out/'committed-1800.bin'),
            committed_restart_at_s=1800.),timeout=18120.,prune_fn=inspect,
            minimum_speed=.2,rolling_minimum=.2,probe=probe)
        state.update(status='screen-complete',last_time_s=run['time_s'][-1])
    except (RuntimeError,TimeoutError) as exc:
        path=out/'replay/incomplete-run.json.gz'
        if path.exists():
            with gzip.open(path,'rt') as f:run=json.load(f)
            state['last_time_s']=run['time_s'][-1]
        state.update(status='screen-pruned' if str(exc).startswith('Pruned:') else 'numerical-review-needed',reason=str(exc))
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

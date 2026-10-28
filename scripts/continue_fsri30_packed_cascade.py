#!/usr/bin/env python3
"""Verify 20 mm warmup and then continue the same trajectories to cascade."""
import argparse
import json
import os
from pathlib import Path
import signal

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_contact_heater import metrics
from calibrate_fsri30_sequence import BASE, BASE_HASH
from check_fsri30_vent_retention import SEED
from fsri30_can_contacts import add_can_contacts
from fsri30_data import load, save, sha
from fsri30_packed_geometry import close_packed
from fsri30_packed_pockets import split_tangent_pockets
from fsri30_packed_cascade_search import search
from fsri30_sequence_model import candidate

PROBE=Path('build/calibration-probes/battery_sim_casing_probe-joint-enthalpy-20261026-r1').resolve()
CORE='6aa586496f59297caa81356c0fff28a17ffe733ba4a9693fb6f4a7e951f7813e'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--coarse-result',type=Path,required=True)
    ap.add_argument('--probe',type=Path,default=PROBE)
    ap.add_argument('--probe-sha256',default=CORE)
    ap.add_argument('--pressure-iterations',type=int,default=2048)
    ap.add_argument('--r2-seed-factor',type=float,default=.1)
    ap.add_argument('--contact-h',type=float,default=1000.)
    args=ap.parse_args()
    probe=args.probe.resolve()
    if sha(BASE)!=BASE_HASH or sha(probe)!=args.probe_sha256:raise ValueError('Pinned core or source changed')
    if not 4<=args.pressure_iterations<=8192:raise ValueError('Invalid pressure iteration budget')
    if not .001<=args.r2_seed_factor<=20.:raise ValueError('Invalid shared R2 seed factor')
    result=json.loads(args.coarse_result.read_text());m=result['metrics']
    if result['status']!='complete' or m['final_time_s']<1499. or m['rmse_k']>5. or m['max_abs_k']>10.:
        raise ValueError('Expected a completed acceptable coarse warmup')
    seed=json.loads((SEED/'status.json').read_text())['parameters'];base=candidate(json.loads(BASE.read_text()),seed)
    scene=add_can_contacts(split_tangent_pockets(close_packed(base)),.0005,args.contact_h)
    scene['gas_network']['averaged_pressure_iterations']=args.pressure_iterations
    # Record the primary cell's local gas states in subsequent fresh runs so
    # an early valve opening can be checked against its actual pressure drop.
    # Observation selection is excluded from the physical restart identity.
    selection=scene.setdefault('process_history_selection',{}).setdefault('gas_prefixes',[])
    if 'inside-B10-' not in selection:selection.append('inside-B10-')
    p=scene['heating_protocols'][0];p.update(sensor_id='B10:case:+x',feedforward_w=0.,proportional_w_per_k=result['gain'])
    for c in scene['cells']:
        if c.get('role')=='battery':c['conversion']['reaction_rate_per_s']*=args.r2_seed_factor
    scene['metadata']['heater_protocol']['sensor_location_prior']='reference-capacity mean of heated can +x panel; exact TC position unknown'
    scene['metadata']['r2_transfer']['used_as_unmodified_law']=False
    scene['metadata']['heater_calibration_candidate']=dict(gain=result['gain'],target='B10 mean vs T5',
        mapping_confirmed=False,coarse_metrics=m,coarse_metrics_apply_to_current_scene=False,
        shared_r2_seed_factor=args.r2_seed_factor,qualification='transferred initial gain; fine warmup must be rechecked')
    data,manifest=load();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'selected-scenario.json',scene)
    save(out/'contract.json',dict(base_sha256=BASE_HASH,native_sha256=args.probe_sha256,csv_sha256=manifest['csv_sha256'],
        coarse_result=str(args.coarse_result.resolve()),coarse_result_sha256=sha(args.coarse_result),
        pressure_iterations=args.pressure_iterations,
        fine_warmup_is_prefix_of_cascade=True,shared_r2_seed_factor=args.r2_seed_factor,
        contact_h_w_m2_k=args.contact_h,contact_h_measured=False,calibrated=False,validation_passed=False))
    state=dict(status='running',pid=os.getpid(),stage='cascade-search',coarse_best=result,
        calibrated=False,validation_passed=False)
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);atomic_save(out/'status.json',state)
    try:search(scene,out,state,data,probe,metrics)
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',reason=str(exc));raise
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

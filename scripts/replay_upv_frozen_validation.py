#!/usr/bin/env python3
"""Fresh held-out replay of one explicit frozen scene; no parameter search."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as cal
from check_upv_restart import native
from refine_upv_gas_access import temperature_pruner
from upv_cooling_fit import rejection_reason
from upv_cooling_review import phase_report,plot_cooling
from upv_frozen_validation import build,digest
from upv_observation_policy import metadata as observation_policy
from upv_observation_policy import CALIBRATION_CHANNELS
from upv_gas_route_refinement import assessment
import upv_r123_data as dataio


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--generate-only',action='store_true')
    ap.add_argument('--voltage-limited-setpoint',action='store_true',help='DIAGNOSTIC ONLY: use measured U as source envelope, violating the actual constant-35 A/80 V-compliance protocol')
    ap.add_argument('--complete-window',action='store_true',help='No observation-error pruning for this one held-out evaluation; numerical, conservation and speed guards remain')
    ap.add_argument('--max-slowdown',type=float,default=2.,help='Explicit wall/sim limit; at most the user-approved 5x')
    args=ap.parse_args();source=args.source.resolve();out=args.output.resolve();out.mkdir()
    if not 1. <= args.max_slowdown <= 5.:
        raise ValueError('Maximum slowdown must be in [1, 5]')
    frozen=json.loads(source.read_text());source_hash=digest(frozen)
    scene,data=build(frozen,voltage_limited_setpoint=args.voltage_limited_setpoint);end=scene['simulation']['duration_s'];event=dataio.event(data)
    cal.PROBE=args.probe.resolve();core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    cal.save(out/'scenario.json',scene)
    state=dict(status='generated',pid=os.getpid(),source=str(source),source_scene_sha256=source_hash,
        native_sha256=core,refitted=False,observation_policy=observation_policy(),
        requested_end_s=end,observed_event_s=event['time_s'],
        initialization='Fresh from measured initial state; never reuse training checkpoint',
        provenance=scene['metadata']['frozen_validation'])
    state['observation_error_pruning']=not args.complete_window
    state['protocol_faithful']=not args.voltage_limited_setpoint
    state['accepted_validation']=False
    state['max_slowdown']=args.max_slowdown
    if args.voltage_limited_setpoint:
        state['validation_claim']=False
        state['diagnostic_only_reason']='Measured terminal U is not source compliance: published source is 35 A constant-current with 80 V compliance. This mode tests the core under a different electrical boundary.'
    cal.save(out/'status.json',state)
    if args.generate_only:return
    state['status']='running';cal.save(out/'status.json',state)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    strict=temperature_pruner(data)
    def prune(run):return strict(run) or rejection_reason(run,data,event)
    run=None
    try:
        print('START held-out active cooling; all training parameters frozen',flush=True)
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=core,
                   stop_after_first_rupture=False),1.1*args.max_slowdown*end,
                   prune_fn=None if args.complete_window else prune,
                   minimum_speed=1./args.max_slowdown,
                   rolling_minimum=min(.25,1./args.max_slowdown))
        state.update(status='complete',wall_s=run['wall_s'],realtime_ratio=end/run['wall_s'])
    except KeyboardInterrupt:state['status']='stopped'
    except (RuntimeError,TimeoutError)as exc:
        state.update(status='rejected',error=str(exc));print('REJECTED',str(exc),flush=True)
    finally:
        if digest(json.loads(source.read_text()))!=source_hash:raise RuntimeError('Frozen source changed during validation')
        if run is None and (out/'replay/incomplete-run.json.gz').exists():
            with gzip.open(out/'replay/incomplete-run.json.gz','rt')as stream:run=json.load(stream)
        if run is not None:
            state.update(known_until_s=run['time_s'][-1],
                balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k})
            phases=phase_report(run,data,end)
            cal.save(out/'phases.json',phases)
            if state['status']=='complete':
                result=assessment(run,data,end)
                cal.save(out/'assessment.json',result)
                warm={k:phases['phases'].get('pre_burst',{}).get('channels',{}).get(k,{}).get('time_weighted_rmse_k')for k in CALIBRATION_CHANNELS}
                state['warmup_rmse_k']=warm
                state['accepted_validation']=bool(state['protocol_faithful']and result['accepted']and all(v is not None and v <= 5. for v in warm.values()))
            plot_cooling(out,{'held-out':run},data,{'held-out':'Расчёт без подгонки'},pressure_panel=True)
        cal.save(out/'status.json',state)
        print('RESULT',state['status'],state.get('known_until_s'),flush=True)


if __name__=='__main__':main()

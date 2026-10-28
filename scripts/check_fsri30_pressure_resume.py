#!/usr/bin/env python3
"""Test a numerical core on an existing scenario and committed state only."""
import argparse
import json
import os
from pathlib import Path

from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import native
from fsri30_data import sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True);ap.add_argument('--probe-sha256',required=True)
    ap.add_argument('--stop-at',type=float,default=2000.);args=ap.parse_args()
    source=args.source.resolve();out=args.output.resolve();probe=args.probe.resolve()
    if sha(probe)!=args.probe_sha256:raise ValueError('Probe hash mismatch')
    old=json.loads((source/'status.json').read_text());old_core=old['core_sha256']
    checkpoint=source/'committed-1800.bin';scene=source/'scenario.json'
    if sha(scene)!=old['scenario_sha256']:raise ValueError('Source scenario mismatch')
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',pid=os.getpid(),old_core=old_core,new_core=args.probe_sha256,
        checkpoint_sha256=sha(checkpoint),scenario_sha256=sha(scene),target_s=args.stop_at,
        diagnostic_only=True,calibrated=False,validation_passed=False)
    atomic_save(out/'status.json',state)
    try:
        run=native(scene,out/'replay',dict(restart_contract=old_core,restart_from=str(checkpoint),
            save_committed_restart=str(out/'committed.bin'),committed_restart_at_s=args.stop_at,
            stop_after_committed_restart=True,stop_after_first_rupture=False),
            timeout=1800.,minimum_speed=.2,rolling_minimum=.2,probe=probe)
        state.update(status='complete',advanced_to_s=run['advanced_to_s'],
            balances={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k},
            minimum_regular_step_s=run.get('solver_profile',{}).get('minimum_regular_step_s'))
    except Exception as exc:
        state.update(status='failed',reason=str(exc));raise
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

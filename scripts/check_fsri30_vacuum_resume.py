#!/usr/bin/env python3
"""Resume the saved 1800 s state to test the pressure-solver correction.

This is a numerical diagnostic, not a fitted candidate or validation. The
checkpoint retains its original core identity; the actual new binary hash is
recorded separately. No physical input, reaction, or time is modified.
"""
import gzip
import argparse
import json
import os
from pathlib import Path

from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import compare, native
from fsri30_data import sha


ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'data/development/fsri30-packed-cascade-20261027-r2/cascade-0000'
OUT=ROOT/'data/development/fsri30-vacuum-resume-20261027-r1'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-packed-vacuum-residual-20261027-r1'
OLD_CORE='8c95042a139a125033fb178a9c8c5528847ab71f4ecb713691278a69cdca8ab5'
NEW_CORE='f78e44a0d2a1740837e706b6f94c4cadf39fb69db7e145ba379ca5b558316655'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe',type=Path,default=PROBE);ap.add_argument('--probe-sha256',default=NEW_CORE)
    ap.add_argument('--output',type=Path,default=OUT);args=ap.parse_args()
    probe=args.probe.resolve();out=args.output.resolve()
    if sha(probe)!=args.probe_sha256:raise ValueError('New core fingerprint changed')
    old_status=json.loads((OLD/'status.json').read_text())
    if old_status['native_sha256']!=OLD_CORE:raise ValueError('Checkpoint core identity mismatch')
    checkpoint=OLD/'committed-state.bin';scene=OLD/'scenario.json'
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',pid=os.getpid(),old_core=OLD_CORE,new_core=args.probe_sha256,
        checkpoint_sha256=sha(checkpoint),scenario_sha256=sha(scene),start_s=1800.,target_s=1950.,
        diagnostic_only=True,calibrated=False,validation_passed=False)
    atomic_save(out/'status.json',state)
    try:
        run=native(scene,out/'replay',dict(restart_contract=OLD_CORE,restart_from=str(checkpoint),
            save_committed_restart=str(out/'committed-1950.bin'),committed_restart_at_s=1950.,
            stop_after_committed_restart=True,stop_after_first_rupture=False),
            timeout=1000.,minimum_speed=.2,rolling_minimum=.2,probe=probe)
        with gzip.open(OLD/'replay/incomplete-run.json.gz','rt') as stream:previous=json.load(stream)
        state['comparison_through_1885']=compare(previous,run,until_s=1885.)
        state.update(status='complete',advanced_to_s=run['advanced_to_s'],
            balances={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k},
            minimum_regular_step_s=run.get('solver_profile',{}).get('minimum_regular_step_s'))
    except Exception as exc:
        state.update(status='failed',reason=str(exc));raise
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

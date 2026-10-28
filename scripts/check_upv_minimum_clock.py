#!/usr/bin/env python3
"""One numerical-only fork: change thermal minimum dt, keep aperture physics.

The same committed pre-event checkpoint, native binary, graph and all material
and reaction laws are retained. This short accuracy reference is not a fitted
candidate or a production speed claim. UI/conservation checks remain strict.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

import calibrate_upv_r123 as cal
from check_upv_restart import native
from check_upv_gas_access_accuracy import compare
from upv_gas_response_refinement import front
import upv_r123_data as dataio


def select_history(value,indices,count):
    if isinstance(value,dict):return {k:select_history(v,indices,count)for k,v in value.items()}
    if isinstance(value,list)and len(value)==count:return [value[i]for i in indices]
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--floor',type=float,default=.005)
    ap.add_argument('--end',type=float,default=905.)
    args=ap.parse_args();source=args.reference.resolve();out=args.output.resolve()
    out.mkdir()
    status=json.loads((source/'status.json').read_text());cal.PROBE=args.probe.resolve()
    core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    if core!=status['native_sha256']:raise ValueError('Native revision mismatch')
    original=json.loads((source/'trial-000/scenario.json').read_text())
    if not .005<=args.floor<original['simulation']['minimum_dt_s']:
        raise ValueError('Reference floor must be smaller, but never below 5 ms')
    replay=source/'trial-000/replay'
    path=replay/'run.json.gz'
    if not path.exists():path=replay/'incomplete-run.json.gz'
    with gzip.open(path,'rt')as stream:reference=json.load(stream)
    scene=copy.deepcopy(original)
    # End on an already recorded observation: no extra time alignment is added.
    end=max(t for t in reference['time_s']if t<=args.end)
    scene['simulation'].update(minimum_dt_s=args.floor,duration_s=end)
    scene['failure_sample_times_s']=[t for t in original['failure_sample_times_s']if t<=end]
    cal.save(out/'scenario.json',scene)
    state=dict(status='running',reference=str(source),native_sha256=core,
        only_thermal_minimum_dt_changed=True,aperture_law_unchanged=True,
        material_graph_chemistry_unchanged=True,calibration=False,
        reference_minimum_dt_s=original['simulation']['minimum_dt_s'],trial_minimum_dt_s=args.floor,
        diagnostic_speed_not_a_production_claim=True,regular_steps_below_5ms_forbidden=True)
    cal.save(out/'status.json',state)
    try:
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=core,
            restart_from=str(source/'trial-000/sealed-state.bin'),
            restart_gas_response_scenario=str(source/'trial-000/scenario.json'),
            stop_after_first_rupture=False),600.,enforce_speed=False)
        indices=np.flatnonzero((np.asarray(reference['time_s'])>=run['started_at_s']-1e-8)&
                              (np.asarray(reference['time_s'])<=end+1e-8))
        ref=select_history(reference,indices,len(reference['time_s']))
        report=compare(ref,run)
        data,_=dataio.load('no-cooling')
        report.update(reference_front=front(ref,data),refined_front=front(run,data),
            started_from_same_checkpoint_s=run['started_at_s'],
            refined_realtime_ratio=(end-run['started_at_s'])/run['wall_s'],
            minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'])
        names=[k for k in run['channels']if k.startswith('rupture-C2-')]
        report['c2_total_open_area_m2']=dict(time_s=run['time_s'],
            reference=np.sum([ref['channels'][n]['effective_area_m2']for n in names],axis=0).tolist(),
            refined=np.sum([run['channels'][n]['effective_area_m2']for n in names],axis=0).tolist())
        cal.save(out/'comparison.json',report)
        state.update(status='complete',accepted_numerical_comparison=report['accepted'],
            minimum_regular_step_s=report['minimum_regular_step_s'],
            refined_realtime_ratio=report['refined_realtime_ratio'])
        print(json.dumps({k:report[k]for k in ['accepted','refined_realtime_ratio','reference_front','refined_front']},indent=2))
    except(RuntimeError,TimeoutError,ValueError)as exc:
        state.update(status='rejected',error=str(exc));print('REJECTED',str(exc),flush=True)
    finally:cal.save(out/'status.json',state)


if __name__=='__main__':main()

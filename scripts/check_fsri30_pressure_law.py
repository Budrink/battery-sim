#!/usr/bin/env python3
"""Fresh matched-parameter nonlinear-pressure audit, not a validation claim.

First measure the numerical correction at the learned primary-aligned seed.
Then repeat the already-observed exit-area sensitivity on the new core. No
historical state or objective is reused, and no chemistry is retuned here.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import tarfile

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import BASE, BASE_HASH, MINIMUM_SPEED, read_observed_trial, trial
from check_fsri30_vent_retention import SEED, budget_report, read_run, restrict_exits
from fsri30_data import ROOT, load, save, sha
from fsri30_multilevel import remesh
from fsri30_wave_fit import POLICY, reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    args=parser.parse_args();probe=args.probe.resolve();core=sha(probe)
    if sha(BASE)!=BASE_HASH:raise ValueError('Retained base changed')
    data,manifest=load();ref=reference(data);mesh=remesh(json.loads(BASE.read_text()),20)
    seed=read_observed_trial(SEED,20,mesh,ref)
    if not seed or seed.get('physical_error'):raise ValueError('Invalid learned seed')
    if core==seed['native_sha256']:raise ValueError('Expected a new pressure-consistent core')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    lock=(out/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['BATTERY_AVERAGED_FAILURE_SNAPSHOT']=str(out/'pressure-failure.json')
    paths=sorted(p for folder in ('scripts','native') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.cu','.hpp','.cuh'))
    paths+=[ROOT/'CMakeLists.txt']
    contract=dict(native_sha256=core,probe=str(probe),base_sha256=BASE_HASH,
        seed_trial=str(SEED),seed_core=seed['native_sha256'],seed_run_sha256=seed['run_sha256'],
        seed_parameters=seed['parameters'],csv_sha256=manifest['csv_sha256'],policy=asdict(POLICY),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        outlet_access_scales=[1.,.1],fresh_from_zero=True,checkpoints_reused=False,
        mesh_mm=20,minimum_regular_step_s=.01,minimum_realtime_ratio=MINIMUM_SPEED,
        chemistry_changed=False,solid_geometry_changed=False,diagnostic_only=True,
        calibrated=False,validation_passed=False)
    save(out/'contract.json',contract)
    with tarfile.open(out/'source.tar.gz','w:gz') as archive:
        for p in paths:archive.add(p,arcname=str(p.relative_to(ROOT)))
    state=dict(status='running',pid=os.getpid(),native_sha256=core,diagnostic_only=True,
               calibrated=False,validation_passed=False,completed_variants=0)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    def update(row):
        state.update(active_trial=row['trial'],progress=row)
        atomic_save(out/'status.json',state)
    rows=[];atomic_save(out/'status.json',state)
    try:
        for scale in (1.,.1):
            if sha(probe)!=core:raise ValueError('Native executable changed during diagnostic')
            row=trial(out/('exits-'+str(scale)),restrict_exits(mesh,scale),seed['parameters'],
                      20,ref,data,update,probe=probe)
            row['outlet_access_scale']=scale
            if row.get('run_path'):
                scene=json.loads((Path(row['trial'])/'scenario.json').read_text())
                save(Path(row['trial'])/'gas-budget.json',budget_report(scene,read_run(row),row))
            rows.append(row);save(out/'comparison.json',dict(variants=rows,contract=contract))
            state.update(completed_variants=len(rows),last_result=row)
            atomic_save(out/'status.json',state)
            if row['status']=='numerical-failed':
                state.update(status='numerical-failed',note='Fix the numerical rejection before more physical fitting');break
        else:state.update(status='complete',note='Numerical/exit sensitivity only, not completed calibration')
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc));raise
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

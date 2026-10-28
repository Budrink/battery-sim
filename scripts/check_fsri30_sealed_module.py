#!/usr/bin/env python3
"""User-requested exactly sealed 20 mm module, same learned gas/thermal core.

No new can radiation, no changed reaction rate, no gas-contact fit. Compare
against the actual open replay on this SAME core. This is a boundary diagnostic,
not the open experimental geometry or an independently validated profile.
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
from calibrate_fsri30_sequence import BASE, BASE_HASH, MINIMUM_SPEED, trial
from check_fsri30_vent_retention import SEED, budget_report, read_run
from fsri30_data import ROOT,load,save,sha
from fsri30_vent_geometry import seal_module,sealed_cut_audit
from fsri30_wave_fit import POLICY,reference

CONTROL=ROOT/'data/development/fsri30-joint-enthalpy-20261026-r1/exits-1.0'
CORE='6aa586496f59297caa81356c0fff28a17ffe733ba4a9693fb6f4a7e951f7813e'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-joint-enthalpy-20261026-r1'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if sha(PROBE)!=CORE or sha(BASE)!=BASE_HASH:raise ValueError('Pinned retained physics changed')
    base=json.loads(BASE.read_text());seed=json.loads((SEED/'status.json').read_text())
    control=json.loads((CONTROL/'status.json').read_text())
    if control['native_sha256']!=CORE or control['parameters']!=seed['parameters']:raise ValueError('Open control mismatch')
    open_run=read_run(control);closed=seal_module(base);data,manifest=load();ref=reference(data)
    out.mkdir(parents=True,exist_ok=False);lock=(out/'diagnostic.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=sorted((ROOT/'scripts').glob('*.py'))
    contract=dict(native_sha256=CORE,probe=str(PROBE),base_sha256=BASE_HASH,seed_parameters=seed['parameters'],
        control_trial=str(CONTROL),control_history_sha256=control['run_sha256'],csv_sha256=manifest['csv_sha256'],
        closed_channels=closed['metadata']['sealed_module_diagnostic']['channels'],
        policy=asdict(POLICY),minimum_regular_step_s=.01,minimum_realtime_ratio=MINIMUM_SPEED,
        fresh_from_zero=True,checkpoints_reused=False,chemistry_changed=False,thermal_links_changed=False,
        new_internal_radiation=False,diagnostic_only=True,calibrated=False,validation_passed=False,
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources})
    save(out/'contract.json',contract)
    with tarfile.open(out/'source.tar.gz','w:gz') as archive:
        for p in sources:archive.add(p,arcname=str(p.relative_to(ROOT)))
    save(out/'open-control-budget.json',budget_report(json.loads((CONTROL/'scenario.json').read_text()),open_run,control))
    state=dict(status='running',pid=os.getpid(),native_sha256=CORE,calibrated=False,validation_passed=False)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    def update(row):state.update(progress=row);atomic_save(out/'status.json',state)
    atomic_save(out/'status.json',state)
    os.environ['BATTERY_AVERAGED_FAILURE_SNAPSHOT']=str(out/'numerical-failure.json')
    try:
        row=trial(out/'sealed',closed,seed['parameters'],20,ref,data,update,probe=PROBE)
        if row.get('run_path'):
            run=read_run(row);actual=json.loads((out/'sealed/scenario.json').read_text())
            cut=sealed_cut_audit(actual,run);save(out/'sealed-cut-audit.json',cut)
            save(out/'sealed-budget.json',budget_report(actual,run,row))
            if not cut['accepted']:raise ValueError('Sealed module leaked mass or counterexchange')
        state.update(status='diagnostic-finished' if row['status']!='numerical-failed' else 'numerical-review-needed',result=row)
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc));raise
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

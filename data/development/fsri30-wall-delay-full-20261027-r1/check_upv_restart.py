#!/usr/bin/env python3
"""Freeze chemistry; save last sealed state once, verify gas/heat suffix restart."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import numpy as np
import calibrate_upv_r123 as cal
import fit_upv_kinetics as fit
from realtime_guard import RealtimeFloor, completed_performance


def native(scene_path,folder,request,timeout,prune_fn=None,enforce_speed=True,minimum_speed=.5,rolling_minimum=.25,probe=None):
    folder.mkdir()
    with tempfile.TemporaryDirectory(prefix='battery-gas-restart-')as temp:
        raw=Path(temp)/'output.json'
        request=dict(request,checkpoint_path=str(Path(temp)/'partial.json'))
        with raw.open('w')as out,(folder/'cuda.log').open('w')as log:
            child=subprocess.Popen([str(probe if probe is not None else cal.PROBE),str(scene_path)],stdin=subprocess.PIPE,
                stdout=out,stderr=log,text=True,env=dict(os.environ,BATTERY_SIM_PROGRESS='1'))
            try:
                child.stdin.write(json.dumps(request)+'\n');child.stdin.close()
                started=time.monotonic();guard=RealtimeFloor(minimum=rolling_minimum);last_partial_stamp=None
                while child.poll()is None:
                    if time.monotonic()-started>timeout:raise TimeoutError('Restart diagnostic timeout')
                    try:child.wait(timeout=2.)
                    except subprocess.TimeoutExpired:pass
                    status=guard.inspect((folder/'cuda.log').read_text())
                    if enforce_speed and status and not status['accepted']:
                        cal.save(folder/'speed-rejected.json',status)
                        raise RuntimeError(f'Restart suffix below rolling {rolling_minimum:g}x speed floor')
                    partial=Path(temp)/'partial.json'
                    if prune_fn is not None and partial.exists():
                        stamp=partial.stat().st_mtime_ns
                        if stamp!=last_partial_stamp:
                            last_partial_stamp=stamp
                            reason=prune_fn(json.loads(partial.read_text()))
                            if reason:raise RuntimeError('Pruned: '+reason)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=5.)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                partial=Path(temp)/'partial.json'
                if partial.exists():
                    with gzip.open(folder/'incomplete-run.json.gz','wb')as dst:dst.write(partial.read_bytes())
                raise
        def preserve_partial():
            partial=Path(temp)/'partial.json'
            if partial.exists():
                with gzip.open(folder/'incomplete-run.json.gz','wb')as dst:dst.write(partial.read_bytes())
        if child.returncode:
            preserve_partial()
            raise RuntimeError((folder/'cuda.log').read_text()[-3000:])
        run=json.loads(raw.read_text())
        if 'error'in run:
            preserve_partial()
            cal.save(folder/'error.json',run)
            raise RuntimeError(run['error'])
    speed=completed_performance(run['advanced_to_s']-run['started_at_s'],run['wall_s'],minimum=minimum_speed)
    if enforce_speed and not speed['accepted']:
        with gzip.open(folder/'speed-rejected-run.json.gz','wt')as dst:json.dump(run,dst,allow_nan=False)
        cal.save(folder/'speed-rejected.json',speed)
        raise RuntimeError(f'Restart suffix below average {minimum_speed:g}x speed floor')
    for key,limit in [('max_thermal_residual_j',1e-4),('max_chemical_residual_j',1e-4),('max_mass_residual_kg',1e-10)]:
        if run[key]>limit:raise RuntimeError('Restart conservation failed: '+key)
    with gzip.open(folder/'run.json.gz','wt')as dst:json.dump(run,dst,allow_nan=False)
    cal.save(folder/'timing.json',dict(**speed,steps=run['steps']-run['steps_at_start']))
    return run


def compare(reference,resumed,until_s=None):
    start=resumed['started_at_s'];times=np.asarray(reference['time_s'])
    index=int(np.searchsorted(times,start-1e-9));stop=len(times);resumed_stop=len(resumed['time_s'])
    if until_s is not None:
        if until_s>min(times[-1],resumed['time_s'][-1])+1e-9:raise ValueError('Comparison exceeds known interval')
        stop=int(np.searchsorted(times,until_s+1e-9,side='right'))
        resumed_stop=int(np.searchsorted(resumed['time_s'],until_s+1e-9,side='right'))
    expected=times[index:stop]
    if not len(expected)or not np.array_equal(expected,np.asarray(resumed['time_s'])[:resumed_stop]):raise ValueError('Restart observations do not align')
    result={}
    def visit(a,b,path):
        if isinstance(a,dict):
            if a.keys()!=b.keys():raise ValueError('Restart keys changed: '+path)
            for key in a:visit(a[key],b[key],path+'/'+key)
        else:
            aa=np.asarray(a)[index:stop];bb=np.asarray(b)[:resumed_stop]
            if aa.shape!=bb.shape:raise ValueError('Restart shape changed: '+path)
            error=float(np.max(np.abs(aa-bb)))if aa.size else 0.
            result[path]=error
    for section in ('temperature_k','gas','gas_observers','channels','chemistry','chemistry_zones','electrical','joule_energy_j'):
        if section in reference:visit(reference[section],resumed[section],section)
    return dict(bitwise_equal=all(v==0.for v in result.values()),max_absolute_difference=max(result.values(),default=0.),
        compared_fields=len(result),started_at_s=start,compared_until_s=float(expected[-1]),
        differences={k:v for k,v in result.items()if v!=0.})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=950.);ap.add_argument('--timeout',type=float,default=1900.)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
    seed=json.loads(a.seed.read_text());controls=seed['numerical_controls']
    scene,_=fit.trial_scene(seed['parameters'],a.duration,seed['gas_extensions'],controls)
    scene['metadata']['zone_diagnostics']=True
    scene_path=out/'scenario.json';cal.save(scene_path,scene)
    cal.save(out/'parameters.json',dict(parameters=seed['parameters'],gas_extensions=seed['gas_extensions'],
        numerical_controls=controls,source=str(a.seed.resolve()),calibrated=False,chemistry_frozen=True))
    contract=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    state=dict(status='running',phase='reference_and_capture',pid=os.getpid(),native_sha256=contract,
        chemistry_frozen=True,diagnostic_only=True)
    cal.save(out/'status.json',state)
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        checkpoint=out/'sealed-state.bin'
        reference=native(scene_path,out/'reference',dict(restart_contract=contract,
            save_unvented_restart=str(checkpoint),stop_after_first_rupture=False),a.timeout)
        if not checkpoint.exists()or 'first_rupture_s'not in reference:raise RuntimeError('No pre-rupture restart was obtained')
        state.update(phase='restart_suffix',restart=reference['unvented_restart'],first_rupture_s=reference['first_rupture_s'])
        cal.save(out/'status.json',state)
        resumed=native(scene_path,out/'resumed',dict(restart_contract=contract,restart_from=str(checkpoint)),a.timeout)
        result=compare(reference,resumed);cal.save(out/'comparison.json',result)
        state.update(status='complete'if result['bitwise_equal']else'failed',phase='verified',comparison=result)
    except KeyboardInterrupt:state.update(status='stopped')
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:cal.save(out/'status.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()

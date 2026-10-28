#!/usr/bin/env python3
"""Sequential fresh mesh/pattern benchmarks: one GPU job, frozen physics.

No experimental fitting or score/checkpoint reuse across cores. The optional
baseline is an explicitly checked current-core run, not an old calibration.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from calibrate_fsri30_bayes import atomic_save
from check_fsri30_resolved_gas import compact,compare_mesh,plot
from check_fsri30_source_reconstruction import DEFAULT_BASE
from fsri30_data import ROOT,load,sha


def pattern_agreement(a,b,ma,mb):
    result=compare_mesh(a,b,ma,mb)
    # This changes an algebraic initial guess, not physical resolution. Require
    # substantially tighter agreement than the spatial 5% engineering budget.
    keys=('maximum_component_rms_relative_kelvin','r3_energy_relative_difference',
          'r3_cumulative_rms_relative_difference','hottest_external_gas_rms_relative_kelvin',
          'maximum_conversion_time_relative_difference')
    result['accepted']=bool(result['accepted'] and all(result[k]<=.001 for k in keys))
    result['tolerance']=.001
    result['scope']='same physical mesh; numerical predictor verification, not spatial convergence'
    return result


def gas_field_agreement(pa,pb,tolerance=.001):
    """Same-mesh accelerator check: maxima curves cannot hide redistributed heat."""
    def read(path):
        with path.open('rb') as f:
            if f.read(8)!=b'BATTGAS1':raise ValueError('Unknown gas field format')
            count=int.from_bytes(f.read(8),'little')
        dtype=np.dtype([('time','<f8'),('fields','<f4',(count,3))])
        size=path.stat().st_size-16
        if not count or size%dtype.itemsize:raise ValueError('Truncated gas field frame')
        return np.memmap(path,mode='r',offset=16,dtype=dtype)
    a,b=read(pa),read(pb)
    if a.shape!=b.shape or a.dtype!=b.dtype or not np.allclose(a['time'],b['time'],rtol=0.,atol=1e-7):
        raise ValueError('Same-mesh gas fields do not align')
    x,y=a['fields'],b['fields'];difference=np.abs(x.astype(float)-y.astype(float))
    temperature_relative=difference[:,:,0]/np.maximum(x[:,:,0],y[:,:,0])
    frame,node=np.unravel_index(temperature_relative.argmax(),temperature_relative.shape)
    fuel=float(np.max(difference[:,:,1].sum(axis=1)/np.maximum(100.,np.maximum(x[:,:,1].sum(axis=1),y[:,:,1].sum(axis=1)))))
    oxygen=float(np.max(difference[:,:,2].sum(axis=1)/np.maximum(1e-12,np.maximum(x[:,:,2].sum(axis=1),y[:,:,2].sum(axis=1)))))
    maximum=float(temperature_relative[frame,node])
    return dict(accepted=maximum<=tolerance and fuel<=tolerance and oxygen<=tolerance,tolerance=tolerance,
        maximum_local_relative_kelvin=maximum,rms_local_relative_kelvin=float(np.sqrt(np.mean(temperature_relative**2))),
        maximum_absolute_temperature_difference_k=float(difference[:,:,0].max()),
        maximum_fuel_field_l1_relative=fuel,maximum_oxygen_field_l1_relative=oxygen,
        worst_local_relative=dict(time_s=float(a['time'][frame]),node_index=int(node),
            baseline_k=float(x[frame,node,0]),predictor_k=float(y[frame,node,0])),
        scope='full same-mesh gas field, not only hottest node or battery mean')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--base',type=Path,default=DEFAULT_BASE)
    ap.add_argument('--baseline',type=Path,help='Already running/completed fresh 40 mm, predictor OFF, wall 10 mm')
    ap.add_argument('--pattern-trial',type=Path,help='Matching current-core predictor trial to preserve instead of recomputing')
    ap.add_argument('--duration',type=float,default=1800.)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False)
    probe=args.probe.resolve();native_sha=sha(probe);child=None
    base_path=args.base.resolve();base_sha=sha(base_path)
    state=dict(status='running',pid=os.getpid(),duration_s=args.duration,native_sha256=native_sha,
        scope='fresh reactive numerical/grid comparison; NOT experimental validation',
        base_scenario=str(base_path),base_sha256=base_sha,
        results={},comparisons={},predictor_qualified=False,selected_case=None,calibrated=False,validation_passed=False)
    source_files=['check_fsri30_resolved_gas.py','fsri30_resolved_gas.py','free_gas_mesh.py']
    sources={name:sha(ROOT/'scripts'/name) for name in source_files}
    atomic_save(out/'contract.json',dict(state,source_sha256=sources,controller_sha256=sha(Path(__file__))))
    runs={};data,_=load()
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)

    def ensure_frozen():
        if sha(probe)!=native_sha or sha(base_path)!=base_sha or any(sha(ROOT/'scripts'/name)!=value for name,value in sources.items()):
            raise ValueError('Pinned native core, physical profile or mesh code changed during the benchmark')

    def ingest(name,folder,h,near,patterns):
        ensure_frozen();s=json.loads((folder/'status.json').read_text())
        if s['native_sha256']!=native_sha or s['duration_s']!=args.duration or s.get('base_sha256')!=base_sha:
            raise ValueError('Baseline/replay core, physical profile or horizon mismatch')
        if s.get('wall_spacing_mm')!=10. or s.get('vent_spacing_mm')!=near or s.get('pressure_patterns')!=patterns:
            raise ValueError('Baseline/replay grid or predictor mismatch')
        if s['status'] not in ('complete','mesh-limit-not-converged'):raise ValueError('Uncompleted replay: '+s['status'])
        label=f'h-{h:g}mm';summary=s['results'][label];folder=folder/label
        if summary['status']!='complete' or not summary['fresh_from_zero']:raise ValueError('Not a fresh completed trajectory')
        with gzip.open(folder/'replay/run.json.gz','rt') as f:run=json.load(f)
        profile=run['solver_profile'];runs[name]=compact(run)
        result=dict(summary,folder=str(folder),gas_spacing_mm=h,vent_spacing_mm=near,pressure_patterns=patterns,
            pattern_attempts=profile.get('gas_pressure_pattern_attempts',0),
            pattern_accepted=profile.get('gas_pressure_pattern_accepted',0),
            core_sha256=native_sha)
        state['results'][name]=result;atomic_save(out/'status.json',state)
        plot(out,runs,data)
        return result

    def run_case(name,h,near=None,patterns=False,existing=None):
        nonlocal child
        state['phase']=name;atomic_save(out/'status.json',state);ensure_frozen()
        folder=existing or out/name
        if existing:
            while True:
                status=json.loads((folder/'status.json').read_text())
                if status['status']!='running':break
                os.kill(status['pid'],0);time.sleep(5.)
        else:
            cmd=[sys.executable,str(ROOT/'scripts/check_fsri30_resolved_gas.py'),'--probe',str(probe),
                '--base',str(base_path),'--output',str(folder),'--duration',str(args.duration),
                '--levels-mm',str(h),'--wall-spacing-mm','10']
            if patterns:cmd.append('--pressure-patterns')
            if near is not None:cmd.extend(['--vent-spacing-mm',str(near)])
            with (out/(name+'.log')).open('w') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
                child.wait()
            if child.returncode:raise ValueError('Mesh controller process failed: '+name)
            child=None
        return ingest(name,folder,h,near,patterns)

    def comparison(a,b,pattern=False):
        fn=pattern_agreement if pattern else compare_mesh
        result=fn(runs[a],runs[b],state['results'][a],state['results'][b])
        state['comparisons'][a+' / '+b]=result
        atomic_save(out/'status.json',state)
        return result

    try:
        baseline=run_case('uniform-40-off',40.,existing=args.baseline.resolve() if args.baseline else None)
        enabled=False
        try:
            trial=run_case('uniform-40-on',40.,patterns=True,
                existing=args.pattern_trial.resolve() if args.pattern_trial else None)
            check=comparison('uniform-40-off','uniform-40-on',pattern=True)
            field=gas_field_agreement(Path(baseline['folder'])/'gas-fields.bin',Path(trial['folder'])/'gas-fields.bin')
            state['pattern_field_agreement']=field
            speedup=baseline['wall_s']/trial['wall_s'];state['pattern_speedup']=speedup
            enabled=bool(check['accepted'] and field['accepted'] and speedup>1.02)
        except Exception as exc:
            # A failed optional predictor does not invalidate the already
            # completed baseline pressure method. Preserve it and continue OFF.
            state['predictor_failure']=str(exc)
        # Never carry a slower/changed predictor into the expensive refinements.
        state['predictor_qualified']=enabled
        forty='uniform-40-on' if enabled else 'uniform-40-off'
        run_case('uniform-20',20.,patterns=enabled);comparison(forty,'uniform-20')
        run_case('uniform-10',10.,patterns=enabled)
        control=comparison('uniform-20','uniform-10')
        for name,h,near in (('focused-40-10',40.,10.),('focused-20-5',20.,5.)):
            run_case(name,h,near,enabled);comparison(name,'uniform-10')
        # A converged reference pair is mandatory. A local jet refinement that
        # agrees with an unconverged reference cannot qualify by itself.
        candidates=[]
        if control['accepted']:
            candidates.append('uniform-20')
            if state['comparisons'][forty+' / uniform-20']['accepted']:candidates.append(forty)
            for name in ('focused-40-10','focused-20-5'):
                if state['comparisons'][name+' / uniform-10']['accepted']:candidates.append(name)
        candidates=[k for k in candidates if state['results'][k]['realtime_ratio']>=.2]
        state['selected_case']=max(candidates,key=lambda k:state['results'][k]['realtime_ratio']) if candidates else None
        state['status']='pilot-converged' if candidates else 'mesh-limit-not-converged'
        state['full_horizon_convergence_passed']=False
        state['next_required_check']='fresh selected/reference pair on full 5945 s horizon, then experimental comparison'
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=15.)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

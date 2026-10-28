#!/usr/bin/env python3
"""Replay the previous aligned candidate without rejecting its early primary."""
import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import time

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import assess_run,physical_error
from check_upv_restart import native
from fsri30_data import load,save,sha
from fsri30_wave_fit import reference

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/development/fsri30-aligned-325-joint-r2x010-20261027-r1/cascade/scenario.json'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-packed-precise-progress-20261027-r1'
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if sha(PROBE)!=CORE:raise ValueError('Pinned CUDA core changed')
    scene=json.loads(SOURCE.read_text());scene['simulation']['duration_s']=3000.
    scene['failure_sample_times_s']=[t for t in scene['failure_sample_times_s'] if t<=3000.]
    if scene['failure_sample_times_s'][-1]!=3000.:raise ValueError('No existing 3000s observation')
    save(out/'scenario.json',scene);data,manifest=load();ref=reference(data)
    state=dict(status='running',pid=os.getpid(),stage='second-wave-diagnostic',source=str(SOURCE),
        source_sha256=sha(SOURCE),scenario_sha256=sha(out/'scenario.json'),core_sha256=CORE,
        experiment_sha256=manifest['csv_sha256'],early_primary_allowed=True,
        secondary_definition='10% of modeled R2 gas yield in a battery other than B10',
        stop_after_two_secondary_conversions=True,calibrated=False,validation_passed=False)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    atomic_save(out/'status.json',state);start=time.monotonic();found=[]
    def inspect(run):
        nonlocal found
        error=physical_error(run);assessment=assess_run(run,ref,20,None)
        events=[]
        for cell in scene['cells']:
            if cell.get('role')!='battery' or cell['id']=='B10':continue
            threshold=.1*cell['conversion']['mass_kg']*cell['conversion']['gas_yield']
            chemistry=run.get('chemistry',{}).get(cell['id'],{})
            for t,v in zip(run['time_s'],chemistry.get('r2_generated_gas_kg',[])):
                if v>=threshold:events.append(dict(cell=cell['id'],time_s=t));break
        events.sort(key=lambda e:e['time_s']);found=events
        state['progress']=dict(time_s=run['time_s'][-1],live_advanced_to_s=run.get('live_advanced_to_s',run.get('advanced_to_s')),
            wall_s=time.monotonic()-start,secondary_events=events,assessment=assessment,physical_error=error)
        atomic_save(out/'status.json',state)
        if error:return error
        if len(events)>=2:return 'Observed two secondary R2 conversion events'
        return None
    try:
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=CORE,
            save_committed_restart=str(out/'committed-1800.bin'),committed_restart_at_s=1800.,
            stop_after_first_rupture=False),timeout=15120.,prune_fn=inspect,
            minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
        inspect(run);state['status']='complete'
    except KeyboardInterrupt:state.update(status='stopped',reason='Explicitly stopped')
    except (RuntimeError,TimeoutError) as exc:
        state.update(status='secondary-observed' if found and len(found)>=2 and 'Observed two secondary' in str(exc)
                     else 'failed',reason=str(exc))
    finally:
        for name in ('run.json.gz','speed-rejected-run.json.gz','incomplete-run.json.gz'):
            path=out/'replay'/name
            if path.exists():
                state['run_path']=str(path);state['run_sha256']=sha(path)
                with gzip.open(path,'rt') as f:run=json.load(f)
                state['balances']={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k}
                break
        state['wall_s']=time.monotonic()-start;atomic_save(out/'status.json',state)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Fresh 32.5 mm aligned-geometry run seeded by the saved 40 mm experiment."""
import argparse
import gzip
import json
import os
from pathlib import Path
import time
import numpy as np

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import assess_run, physical_error
from check_upv_restart import native
from fsri30_aligned_gas import align_inner_gas
from fsri30_data import load, save, sha
from fsri30_six_zones import six_zone_batteries
from fsri30_wave_fit import reference

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'data/development/fsri30-packed-fine-2406-20261027-r1/cascade-0000/scenario.json'
OLD=ROOT/'data/development/fsri30-candidate-refinement-20261026-r1/h-40mm/scenario.json'
OLD_STATUS=OLD.parent/'status.json'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-packed-precise-progress-20261027-r1'
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if sha(PROBE)!=CORE:raise ValueError('Pinned CUDA core changed')
    base=json.loads(NEW.read_text());old=json.loads(OLD.read_text());old_status=json.loads(OLD_STATUS.read_text())
    if sha(OLD)!=old_status['scenario_sha256']:raise ValueError('40 mm scenario fingerprint changed')
    old_cells={c['id']:c for c in old['cells'] if c.get('role')=='battery'}
    for c in base['cells']:
        if c.get('role')=='battery':
            previous=old_cells[c['id']]['conversion'];current=c['conversion']
            changed={k for k in set(previous)|set(current) if previous.get(k)!=current.get(k)}
            if changed!={'reaction_rate_per_s'}:raise ValueError('Unexpected chemistry difference '+str(changed))
            current['reaction_rate_per_s']=previous['reaction_rate_per_s']
    base['heating_protocols']=old['heating_protocols']
    scene=align_inner_gas(six_zone_batteries(base))
    scene['metadata']['seed_40mm']=dict(source=str(OLD),sha256=sha(OLD),
        source_status=old_status['status'],source_primary_s=1786.,
        source_secondary_wave='not reproduced through 2505 s',
        transferred='R2 rate and heater protocol; shared chemistry and gas laws already identical',
        packed_geometry_and_half_mm_contacts='newer reconstruction; not in old 40 mm run',
        scores_or_checkpoints_transferred=False,calibrated=False)
    save(out/'scenario.json',scene);data,manifest=load();ref=reference(data)
    state=dict(status='running',pid=os.getpid(),stage='fresh-aligned-325',
        scenario_sha256=sha(out/'scenario.json'),seed_40mm_sha256=sha(OLD),core_sha256=CORE,
        experiment_sha256=manifest['csv_sha256'],calibrated=False,validation_passed=False)
    atomic_save(out/'status.json',state);start=time.monotonic()
    def inspect(run):
        assessment=assess_run(run,ref,20,None);error=physical_error(run)
        warmup_rejection=None
        times=np.asarray(run.get('time_s',[]),dtype=float)
        if len(times)>1 and 300.<=times[-1]<=1500.:
            choose=(times>=times[-1]-60.)&(times<=times[-1])
            measured=np.interp(times[choose],data['time_s'],data['T5_k'])
            modeled=np.asarray(run['cell_mean_temperature_k']['B10'])[choose]
            finite=np.isfinite(measured)&np.isfinite(modeled)
            if finite.sum()>=10 and np.all(np.abs(modeled[finite]-measured[finite])>25.):
                warmup_rejection='B10 differs from T5 by more than 25 K for 60 s before runaway'
        state.update(progress=dict(last_observed_s=assessment.get('last_observed_s'),
            live_advanced_to_s=run.get('live_advanced_to_s',run.get('advanced_to_s')),
            assessment=assessment,physical_error=error,warmup_rejection=warmup_rejection,
            wall_s=time.monotonic()-start))
        atomic_save(out/'status.json',state)
        if error:return error
        if warmup_rejection:return warmup_rejection
        if assessment.get('accepted') is False:return assessment.get('reason')
        return None
    try:
        result=native(out/'scenario.json',out/'replay',dict(restart_contract=CORE,
            save_committed_restart=str(out/'committed-1800.bin'),committed_restart_at_s=1800.,
            stop_after_first_rupture=False),timeout=19000.,prune_fn=inspect,
            minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
        inspect(result);state['status']='complete';state['note']='Diagnostic run only; no independent validation'
    except KeyboardInterrupt:
        state.update(status='interrupted',reason='Stopped by user or diagnostic supervisor')
    except (RuntimeError,TimeoutError) as exc:
        state.update(status='pruned' if str(exc).startswith('Pruned:') else 'failed',reason=str(exc))
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

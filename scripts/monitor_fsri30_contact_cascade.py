#!/usr/bin/env python3
"""Observe a delayed contact-strip cascade without rejecting its timing."""
import argparse
import json
import os
from pathlib import Path
import signal

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import MINIMUM_SPEED, physical_error
from check_upv_restart import native
from fsri30_data import save, sha

BASE = Path('data/development/fsri30-can-contacts-20261026-r1/trial-0000/scenario.json')
PROBE = Path('build/calibration-probes/battery_sim_casing_probe-joint-enthalpy-20261026-r1').resolve()
CORE = '6aa586496f59297caa81356c0fff28a17ffe733ba4a9693fb6f4a7e951f7813e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out=args.output.resolve()
    if sha(PROBE) != CORE: raise ValueError('Unexpected CUDA core')
    scene=json.loads(BASE.read_text()); out.mkdir(parents=True,exist_ok=False)
    save(out/'scenario.json',scene)
    save(out/'contract.json',dict(source_scenario=str(BASE.resolve()),scenario_sha256=sha(BASE),native_sha256=CORE,
        purpose='Observe onset even when the first front is late; not a calibration',
        fresh_from_zero=True,minimum_regular_step_s=.01,minimum_realtime_ratio=MINIMUM_SPEED,
        no_wave_timing_rejection=True,calibrated=False,validation_passed=False))
    state=dict(status='running',pid=os.getpid(),calibrated=False,validation_passed=False)
    atomic_save(out/'status.json',state)
    source=scene['metadata']['heated_component']
    batteries={c['id']:c for c in scene['cells'] if c.get('role')=='battery'}
    def inspect(run):
        error=physical_error(run)
        if error:return error
        times=run['time_s']; events=[]
        for name,c in batteries.items():
            mass=c['conversion']['mass_kg']*c['conversion']['gas_yield']
            series=run['chemistry'][name]['r2_generated_gas_kg']
            hit=next((t for t,x in zip(times,series) if x>=.1*mass),None)
            if hit is not None:events.append(dict(cell=name,conversion_10pct_s=hit))
        vents=[]
        for name in batteries:
            channels=[v for k,v in run['channels'].items() if k.startswith('rupture-'+name+'-')]
            hit=next((t for i,t in enumerate(times) if any(ch['opening_fraction'][i]>1e-8 for ch in channels)),None)
            if hit is not None:vents.append(dict(cell=name,opening_s=hit))
        end=len(times)-1
        state['progress']=dict(last_observed_s=times[-1],wall_s=run['wall_s'],events=events,vents=vents,
            primary_temperature_c=run['cell_mean_temperature_k'][source][end]-273.15,
            max_other_temperature_c=max(v[end]-273.15 for k,v in run['cell_mean_temperature_k'].items() if k!=source),
            max_thermal_residual_j=run['max_thermal_residual_j'],
            minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'])
        atomic_save(out/'status.json',state)
        if sum(e['cell']!=source for e in events)>=2:return 'Diagnostic detected two secondary 10% R2 conversions'
        return None
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=CORE,stop_after_first_rupture=False),
            timeout=scene['simulation']['duration_s']/MINIMUM_SPEED+120.,prune_fn=inspect,
            minimum_speed=MINIMUM_SPEED,rolling_minimum=MINIMUM_SPEED,probe=PROBE)
        inspect(run);state['status']='horizon-complete'
    except KeyboardInterrupt:state['status']='stopped'
    except (RuntimeError,TimeoutError) as exc:
        state['reason']=str(exc)
        state['status']='cascade-detected' if 'Diagnostic detected two secondary' in str(exc) else 'stopped-or-failed'
    except Exception as exc:
        state.update(status='failed',reason=str(exc));raise
    finally:atomic_save(out/'status.json',state)


if __name__=='__main__':main()

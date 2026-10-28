#!/usr/bin/env python3
"""Observe the actual CUDA child snapshots without changing solver state."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from calibrate_fsri30_bayes import atomic_save


def checkpoint(controller_pid):
    children=Path(f'/proc/{controller_pid}/task/{controller_pid}/children').read_text().split()
    paths=[]
    for child in children:
        try:
            output=Path(os.readlink(f'/proc/{child}/fd/1'))
            if output.name=='output.json' and output.parent.name.startswith('battery-gas-restart-'):
                paths.append(output.parent/'partial.json')
        except OSError:pass
    if len(paths)>1:raise ValueError('Ambiguous active CUDA children')
    return paths[0] if paths else None


def summarize(run,trial):
    times=run['time_s'];i=len(times)-1
    means=run['cell_mean_temperature_k'];chem=run['chemistry'];gases=run['gas']
    others=sorted(((v[i]-273.15,k) for k,v in means.items() if k!='B10'),reverse=True)
    fuel=sorted(((v['chemical_j'][i],k) for k,v in gases.items() if k.startswith('body-pocket-')),reverse=True)
    heated=sorted(((v['temperature_k'][i]-273.15,k) for k,v in gases.items() if k.startswith('body-pocket-')),reverse=True)
    conversions=[]
    for name,v in chem.items():
        hits=[t for t,x in zip(times,v['r2_generated_gas_kg']) if x>=.1*.025*.33]
        if hits:conversions.append(dict(cell=name,first_10pct_s=hits[0]))
    ruptures=[]
    for name,v in run['channels'].items():
        if not name.startswith('rupture-'):continue
        hits=[t for t,x in zip(times,v['opening_fraction']) if x>1e-8]
        if hits:ruptures.append(dict(channel=name,first_open_s=hits[0]))
    return dict(trial=trial,time_s=times[i],primary_mean_c=means['B10'][i]-273.15,
        hottest_other_cells=[dict(cell=k,mean_c=v) for v,k in others[:5]],
        max_cell_r2_power_w=max(v['r2_power_w'][i] for v in chem.values()),
        conversions=conversions,ruptures=ruptures,
        hottest_pockets=[dict(pocket=k,temperature_c=v) for v,k in heated[:5]],
        richest_fuel_pockets=[dict(pocket=k,chemical_j=v) for v,k in fuel[:5]],
        module_gas={k:v[i] for k,v in run['gas_groups']['module'].items()},
        thermal_residual_j=run['max_thermal_residual_j'],mass_residual_kg=run['max_mass_residual_kg'],
        chemical_residual_j=run['max_chemical_residual_j'],oxygen_residual_kg=run['max_oxidizer_residual_kg'])


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--interval',type=float,default=20.);args=ap.parse_args()
    root=args.run.resolve();path=root/'live-gas-audit.json';rows=[];last=None
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        while True:
            state=json.loads((root/'status.json').read_text());trial=state.get('active_trial')
            if state['status']!='running':break
            try:
                p=checkpoint(state['pid'])
                if p and p.exists():
                    stamp=(str(p),p.stat().st_mtime_ns)
                    if stamp!=last:
                        run=json.loads(p.read_text());rows.append(summarize(run,trial));last=stamp
                        atomic_save(path,dict(status='watching',rows=rows))
            except (FileNotFoundError,ProcessLookupError,json.JSONDecodeError):pass
            time.sleep(args.interval)
    except KeyboardInterrupt:pass
    finally:atomic_save(path,dict(status='stopped',rows=rows))


if __name__=='__main__':main()

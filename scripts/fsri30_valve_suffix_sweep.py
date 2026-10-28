"""Conditioned 1600-1800 s safety-valve threshold diagnostics.

Initial state is the *same* saved physical state for every run; this is not a
fresh-from-zero calibration or validation. A replay of the unmodified control
must agree before any changed-physics run is admitted.
"""
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import numpy as np
from check_upv_restart import native
from fsri30_data import save,sha

ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'data/development/fsri30-valve-1600-state-20261027-r1'
CONTROL=ROOT/'data/development/fsri30-valve-control-1600-1800-20261027-r1'
OUT=ROOT/'data/development/fsri30-valve-suffix-sweep-20261027-r1'
PROBE=Path('/tmp/battery-valve-suffix-20261027/build/valve_suffix')
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'
THRESHOLDS=(40000.,80000.,160000.)


def run_trial(scene, index, checkpoint, original, suppress):
    folder=OUT/index;folder.mkdir();path=folder/'scenario.json';save(path,scene)
    request=dict(restart_contract=CORE,restart_from=str(checkpoint),restart_valve_scenario=str(original),
        diagnostic_shell_suppressed=suppress,stop_after_first_rupture=False)
    result=native(path,folder/'replay',request,timeout=1800.,minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
    if abs(result['started_at_s']-1600)>1e-6 or abs(result['advanced_to_s']-1800)>1e-6:
        raise ValueError('Wrong replay time window')
    return result


def verify_control(a,b):
    if a['time_s']!=b['time_s']:raise ValueError('Control sample times differ')
    err=0.
    for cell in ('B10','B17','B18'):
        delta=np.asarray(a['cell_mean_temperature_k'][cell])-np.asarray(b['cell_mean_temperature_k'][cell])
        err=max(err,float(np.max(np.abs(delta))))
    if err>.05:raise ValueError(f'Rebuilt core is not equivalent: max delta {err:g} K')
    return err


def variant(base,threshold):
    s=deepcopy(base)
    for shell in s['shells']:
        if shell['contents_id'].startswith('B'):shell['damage_rate_per_s']=0.
    for edge in s['gas_network']['channels']:
        if edge.get('kind')!='rupture':continue
        edge['opening_pressure_pa']=1e12  # diagnostic suppression, not a physical material strength
        valve=edge.get('damage_opening',{}).get('valve')
        if valve is not None:valve['pressure_factor']=threshold/1e12
    s['metadata']['conditioned_valve_probe']=dict(valve_initial_dp_pa=threshold,
        shell_damage_after_restart='disabled',other_shell_failure_barrier_pa=1e12,
        saved_damage_retained=True,calibrated=False,validation_passed=False)
    return s


def main():
    state=json.loads((SEED/'status.json').read_text());control=json.loads((CONTROL/'status.json').read_text())
    checkpoint=SEED/'committed-1600.bin';original=SEED/'scenario.json'
    if state['status']!='complete' or control['status']!='complete' or sha(checkpoint)!=state['checkpoint_sha']:
        raise ValueError('Missing validated checkpoint or control')
    OUT.mkdir(parents=True,exist_ok=False);status=dict(status='running',stage='control-equivalence',
        checkpoint_sha=sha(checkpoint),original_core_sha=CORE,diagnostic_probe_sha=sha(PROBE),trials=[])
    save(OUT/'status.json',status)
    try:
        base=json.loads(original.read_text());base['simulation']['duration_s']=1800.
        base.setdefault('process_history_selection',{}).setdefault('gas_prefixes',[]).append('inside-B')
        base['failure_sample_times_s']=[t for t in base['failure_sample_times_s'] if t<=1800.]
        replay=run_trial(base,'control-rebuilt',checkpoint,original,False)
        with gzip.open(CONTROL/'replay/run.json.gz','rt') as f:reference=json.load(f)
        status.update(stage='threshold-sweep',control_max_temperature_delta_k=verify_control(reference,replay));save(OUT/'status.json',status)
        for threshold in THRESHOLDS:
            status['current_threshold_pa']=threshold;save(OUT/'status.json',status)
            name=f'valve-{threshold/1000:g}kpa'
            try:
                run=run_trial(variant(base,threshold),name,checkpoint,original,True)
                inside=[g['pressure_pa'] for name,g in run['gas'].items() if name.startswith('inside-B')]
                if not inside or max(max(p) for p in inside)>=1e12:
                    raise ValueError('Cannot establish suppression of non-valve shell rupture')
                t=run['time_s']; rows={}
                for cell in ('B10','B17','B18'):
                    c=run['channels'][f'rupture-{cell}-1'];gas=run['gas'].get(f'inside-{cell}-4',{})
                    opened=next((t[i] for i,x in enumerate(c['opening_fraction']) if x>1e-10),None)
                    rows[cell]=dict(first_observed_open_s=opened,final_open_fraction=c['opening_fraction'][-1],
                        transferred_mass_kg=c['transferred_mass_kg'][-1],final_mean_c=run['cell_mean_temperature_k'][cell][-1]-273.15,
                        final_pressure_pa=gas.get('pressure_pa',[None])[-1])
                status['trials'].append(dict(threshold_pa=threshold,status='complete',cells=rows,wall_s=run['wall_s']))
            except (RuntimeError,TimeoutError,ValueError) as exc:
                status['trials'].append(dict(threshold_pa=threshold,status='failed',reason=str(exc)))
            save(OUT/'status.json',status)
        status.update(status='complete',stage='done');save(OUT/'status.json',status)
    except Exception as exc:
        status.update(status='failed',reason=str(exc));save(OUT/'status.json',status);raise


if __name__=='__main__':main()

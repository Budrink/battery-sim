"""Conditioned hot gas-contact ablation on B17/B18 from actual 1600 s state."""
from copy import deepcopy
import json
import math
from pathlib import Path
from check_upv_restart import native
from calibrate_fsri30_bayes import atomic_save
from fsri30_data import load,save,sha
from fsri30_wave_fit import signatures
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'data/development/fsri30-valve-1600-state-20261027-r1'
OUT=ROOT/'data/development/fsri30-b17-hot-film-half-20261027-r1'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-packed-precise-progress-20261027-r1'
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--strength',type=float,default=.5);p.add_argument('--end',type=float,default=2000.)
    p.add_argument('--save-at',type=float);p.add_argument('--reject-pulse-before',type=float)
    p.add_argument('--from-run',type=Path,help='Condition on this prior run\u2019s saved committed.bin')
    a=p.parse_args();out=a.output.resolve()
    if not 0.<=a.strength<=1. or a.end<1600. or (a.save_at is not None and not 1600.<a.save_at<=a.end):
        raise ValueError('Invalid diagnostic interval or response')
    if a.from_run:
        import gzip
        source=a.from_run.resolve();ck=source/'committed.bin';original=source/'scenario.json'
        partial=source/'replay/incomplete-run.json.gz'
        if not partial.exists():partial=source/'replay/run.json.gz'
        with gzip.open(partial,'rt') as f:previous=json.load(f)
        commit=previous.get('committed_restart',{})
        if Path(commit.get('path','')).resolve()!=ck or not 1600.<commit.get('time_s',0.)<a.end:
            raise ValueError('Source committed checkpoint does not match requested interval')
        if a.save_at is not None and a.save_at<=commit['time_s']:raise ValueError('New checkpoint must be after source')
        seed={'checkpoint_sha':sha(ck),'source_run':str(source),'checkpoint_time_s':commit['time_s']}
    else:
        ck=SEED/'committed-1600.bin';original=SEED/'scenario.json';seed=json.loads((SEED/'status.json').read_text())
    if sha(ck)!=seed['checkpoint_sha'] or sha(PROBE)!=CORE:raise ValueError('Checkpoint/core mismatch')
    out.mkdir(parents=True,exist_ok=False);scene=deepcopy(json.loads(original.read_text()))
    scene['simulation']['duration_s']=a.end;scene['failure_sample_times_s']=[v for v in scene['failure_sample_times_s'] if v<=a.end]
    count=0
    for film in scene['gas_network']['thermal_contacts']:
        if not film['port_id'].startswith(('B17:case:','B18:case:')):continue
        law=film['gas_surface_connectivity'];curve=law['temperature_log_response']
        if not all(len(x)==2 for x in curve):raise ValueError('Unexpected response curve')
        for point in curve:
            if point[0]>law['reference_temperature_k']:point[1]*=a.strength
        count+=1
    if not count:raise ValueError('No selected films')
    scene['metadata']['hot_film_diagnostic']=dict(selected=['B17','B18'],log_response_multiplier=a.strength,films=count,
        qualification='Local causal intervention, not universal material calibration; cold law unchanged')
    scene_path=out/'scenario.json';save(scene_path,scene)
    data,_=load();experiment=dict(time_s=data['time_s'].tolist())
    for i in range(1,10):experiment[f'T{i}']=[float(v-273.15) if math.isfinite(v) else None for v in data[f'T{i}_k']]
    atomic_save(out/'experiment.json',experiment)
    status=dict(status='running',stage='conditioned-hot-film',checkpoint_sha=sha(ck),source_scenario_sha=sha(original),core_sha=CORE,
        source_run=seed.get('source_run'),source_checkpoint_time_s=seed.get('checkpoint_time_s',1600.),
        response_multiplier=a.strength,end_s=a.end,reject_pulse_before_s=a.reject_pulse_before,progress={})
    atomic_save(out/'status.json',status)
    def inspect(run):
        times=run['time_s'];cells=run['cell_mean_temperature_k'];t=times[-1]
        trace=signatures(times,np.stack([cells[n] for n in ('B17','B18')],axis=1),['B17','B18']) if len(times)>30 else {'pulses':[]}
        pulses=trace['pulses']
        status['progress']=dict(time_s=t,wall_s=run.get('wall_s'),pulses=pulses,
            temperatures_c={n:cells[n][-1]-273.15 for n in ('B10','B17','B18')},
            max_thermal_residual_j=run.get('max_thermal_residual_j'),max_mass_residual_kg=run.get('max_mass_residual_kg'))
        atomic_save(out/'live.json',dict(phase='cascade',time_s=times,
            cells={k:[float(x-273.15) for x in v] for k,v in cells.items()}))
        atomic_save(out/'status.json',status)
        if any(max(v)>3500. for v in cells.values() if v):return 'Critical mean component temperature'
        if a.reject_pulse_before is not None and any(x['time_s']<a.reject_pulse_before for x in pulses):
            return 'Secondary thermal impulse is more than 100 s early vs 2134 s experimental group'
        return None
    try:
        request=dict(restart_contract=CORE,restart_from=str(ck),restart_gas_response_scenario=str(original),stop_after_first_rupture=False)
        if a.save_at is not None:request.update(save_committed_restart=str(out/'committed.bin'),committed_restart_at_s=a.save_at)
        run=native(scene_path,out/'replay',request,
            timeout=4000.,prune_fn=inspect,minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
        inspect(run);status.update(status='complete',end_s=run['advanced_to_s'],wall_s=run['wall_s'])
    except Exception as exc:
        status.update(status='pruned' if str(exc).startswith('Pruned:') else 'failed',reason=str(exc));raise
    finally:atomic_save(out/'status.json',status)

if __name__=='__main__':main()

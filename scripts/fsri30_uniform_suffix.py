"""Shared-material check from a 1600 s checkpoint, not fresh validation."""
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import native
from fsri30_data import load,save,sha
from fsri30_wave_fit import signatures

ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'data/development/fsri30-hot075-checkpoint1600-20261027-r1'
OUT=ROOT/'data/development/fsri30-uniform-hot075-inner15-from1600-20261027-r1'
PROBE=ROOT/'build/calibration-probes/battery_sim_uniform_suffix_20261027_r1'
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'

def main():
    seed=json.loads((SEED/'status.json').read_text());ck=SEED/'committed.bin';original=SEED/'scenario.json'
    if sha(ck)!=seed['checkpoint_sha'] or sha(original)!=seed['scene_sha']:raise ValueError('Seed provenance mismatch')
    OUT.mkdir(parents=True,exist_ok=False);scene=deepcopy(json.loads(original.read_text()))
    scene['simulation']['duration_s']=2200.;scene['failure_sample_times_s']=[x for x in scene['failure_sample_times_s'] if x<=2200.]
    for c in scene['cells']:
        if c.get('role')=='battery':c['thermal']['conductivity_w_per_m_k']=[x*1.5 for x in c['thermal']['conductivity_w_per_m_k']]
    for f in scene['gas_network']['thermal_contacts']:
        port=f.get('port_id','')
        if not port.startswith('B') or ':case:' not in port or port.startswith(('B17:','B18:')):continue
        law=f['gas_surface_connectivity']
        for v in law['temperature_log_response']:
            if v[0]>law['reference_temperature_k']:v[1]*=.75
    scene['metadata']['uniform_diagnostic']=dict(hot_log_response=.75,internal_conductivity=1.5,
        scope='all battery materials; conditional from 1600 s',calibrated=False,validation_passed=False)
    path=OUT/'scenario.json';save(path,scene);data,_=load();ex=dict(time_s=data['time_s'].tolist())
    for i in range(1,10):ex[f'T{i}']=[float(v-273.15) if math.isfinite(v) else None for v in data[f'T{i}_k']]
    atomic_save(OUT/'experiment.json',ex)
    status=dict(status='running',stage='uniform-from-1600',checkpoint_sha=sha(ck),source_scenario_sha=sha(original),
        diagnostic_probe_sha=sha(PROBE),original_core_sha=CORE,progress={});atomic_save(OUT/'status.json',status)
    def inspect(run):
        ts=run['time_s'];c=run['cell_mean_temperature_k'];t=ts[-1];ids=['B10','B17','B18']
        sig=signatures(ts,np.stack([c[n] for n in ids],axis=1),ids) if len(ts)>30 else {'pulses':[]}
        pulse=sig['pulses'];status['progress']=dict(time_s=t,wall_s=run.get('wall_s'),pulses=pulse,
            temperatures_c={n:c[n][-1]-273.15 for n in ids},max_thermal_residual_j=run.get('max_thermal_residual_j'),
            max_mass_residual_kg=run.get('max_mass_residual_kg'))
        atomic_save(OUT/'live.json',dict(phase='cascade',time_s=ts,cells={k:[float(v-273.15) for v in x] for k,x in c.items()}))
        atomic_save(OUT/'status.json',status)
        if any(v and max(v)>3500 for v in c.values()):return 'Critical mean component temperature'
        if any(x['channel']=='B10' and x['time_s']<1663 for x in pulse):return 'Primary pulse more than 100 s early'
        if t>1880 and not any(x['channel']=='B10' for x in pulse):return 'No primary pulse within allowed timing window'
        if any(x['channel'] in ('B17','B18') and x['time_s']<2034 for x in pulse):return 'Secondary pulse more than 100 s early'
        return None
    try:
        run=native(path,OUT/'replay',dict(restart_contract=CORE,restart_from=str(ck),restart_uniform_scenario=str(original),
            stop_after_first_rupture=False),timeout=5000.,prune_fn=inspect,minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
        inspect(run);status.update(status='complete',end_s=run['advanced_to_s'],wall_s=run['wall_s'])
    except Exception as exc:status.update(status='pruned' if str(exc).startswith('Pruned:') else 'failed',reason=str(exc));raise
    finally:atomic_save(OUT/'status.json',status)

if __name__=='__main__':main()

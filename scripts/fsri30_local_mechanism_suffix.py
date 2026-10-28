"""Conditioned B17/B18 evaporation or internal-conduction ablation."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from check_upv_restart import native
from calibrate_fsri30_bayes import atomic_save
from fsri30_data import load,save,sha

ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'data/development/fsri30-valve-1600-state-20261027-r1'
PROBE=ROOT/'build/calibration-probes/battery_sim_mechanism_suffix_20261027_r1'
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('evaporation','internal'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--from-run',type=Path)
    p.add_argument('--end',type=float,default=2000.);p.add_argument('--multiplier',type=float)
    a=p.parse_args();out=a.output.resolve()
    multiplier=a.multiplier if a.multiplier is not None else (10. if a.kind=='evaporation' else 2.)
    if not math.isfinite(multiplier) or not .1<=multiplier<=20.:raise ValueError('Invalid diagnostic multiplier')
    if a.from_run:
        import gzip
        source=a.from_run.resolve();ck=source/'committed.bin';original=source/'scenario.json'
        record=source/'replay/incomplete-run.json.gz'
        if not record.exists():record=source/'replay/run.json.gz'
        with gzip.open(record,'rt') as f:previous=json.load(f)
        commit=previous.get('committed_restart',{})
        if Path(commit.get('path','')).resolve()!=ck or not 0<commit.get('time_s',0)<a.end:raise ValueError('Bad source checkpoint')
        seed=dict(checkpoint_sha=sha(ck),source_run=str(source),checkpoint_time_s=commit['time_s'])
    else:
        ck=SEED/'committed-1600.bin';original=SEED/'scenario.json';seed=json.loads((SEED/'status.json').read_text())
    if sha(ck)!=seed['checkpoint_sha']:raise ValueError('Checkpoint mismatch')
    out.mkdir(parents=True,exist_ok=False);scene=deepcopy(json.loads(original.read_text()))
    scene['simulation']['duration_s']=a.end;scene['failure_sample_times_s']=[v for v in scene['failure_sample_times_s'] if v<=a.end]
    count=0
    for cell in scene['cells']:
        if cell['id'] not in ('B17','B18'):continue
        if a.kind=='evaporation':cell['conversion']['volatile_phase']['relaxation_per_s']*=multiplier
        else:cell['thermal']['conductivity_w_per_m_k']=[multiplier*x for x in cell['thermal']['conductivity_w_per_m_k']]
        count+=1
    if count!=2:raise ValueError('Missing target cells')
    scene['metadata']['local_mechanism_diagnostic']=dict(kind=a.kind,targets=['B17','B18'],
        multiplier=multiplier,
        scope='local conditioned diagnostic; not universal material calibration',validation_passed=False)
    path=out/'scenario.json';save(path,scene)
    data,_=load();experiment=dict(time_s=data['time_s'].tolist())
    for i in range(1,10):experiment[f'T{i}']=[float(v-273.15) if math.isfinite(v) else None for v in data[f'T{i}_k']]
    atomic_save(out/'experiment.json',experiment)
    status=dict(status='running',stage=a.kind,checkpoint_sha=sha(ck),source_scenario_sha=sha(original),
        source_run=seed.get('source_run'),source_checkpoint_time_s=seed.get('checkpoint_time_s',1600.),original_core_sha=CORE,
        diagnostic_probe_sha=sha(PROBE),progress={});atomic_save(out/'status.json',status)
    def inspect(run):
        t=run['time_s'][-1];means=run['cell_mean_temperature_k']
        status['progress']=dict(time_s=t,wall_s=run.get('wall_s'),
            temperatures_c={n:means[n][-1]-273.15 for n in ('B10','B17','B18')},
            max_thermal_residual_j=run.get('max_thermal_residual_j'),max_mass_residual_kg=run.get('max_mass_residual_kg'))
        atomic_save(out/'live.json',dict(phase='cascade',time_s=run['time_s'],
            cells={k:[float(x-273.15) for x in v] for k,v in means.items()}))
        atomic_save(out/'status.json',status)
        if any(v and max(v)>3500 for v in means.values()):return 'Critical mean component temperature'
        return None
    try:
        run=native(path,out/'replay',dict(restart_contract=CORE,restart_from=str(ck),
            restart_mechanism_scenario=str(original),stop_after_first_rupture=False),
            timeout=4000.,prune_fn=inspect,minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
        inspect(run);status.update(status='complete',end_s=run['advanced_to_s'],wall_s=run['wall_s'])
    except Exception as exc:
        status.update(status='failed',reason=str(exc));raise
    finally:atomic_save(out/'status.json',status)

if __name__=='__main__':main()

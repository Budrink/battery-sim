"""Exact 200-second control replay with unchanged constitutive parameters."""
import json
from pathlib import Path
from check_upv_restart import native
from fsri30_data import sha,save

ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'data/development/fsri30-valve-1600-state-20261027-r1'
OUT=ROOT/'data/development/fsri30-valve-control-1600-1800-20261027-r1'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-packed-precise-progress-20261027-r1'
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'

def main():
    state=json.loads((SEED/'status.json').read_text());checkpoint=SEED/'committed-1600.bin'
    if state.get('status')!='complete' or sha(checkpoint)!=state['checkpoint_sha'] or sha(PROBE)!=CORE:
        raise ValueError('Unverified state or core')
    OUT.mkdir(parents=True,exist_ok=False)
    scene=json.loads((SEED/'scenario.json').read_text());scene['simulation']['duration_s']=1800.
    scene['failure_sample_times_s']=[x for x in scene['failure_sample_times_s'] if x<=1800.]
    path=OUT/'scenario.json';save(path,scene)
    save(OUT/'status.json',dict(status='running',checkpoint_sha=sha(checkpoint),core_sha=CORE))
    try:
        run=native(path,OUT/'replay',dict(restart_contract=CORE,restart_from=str(checkpoint),
            stop_after_first_rupture=False),timeout=1800.,minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
        save(OUT/'status.json',dict(status='complete',started_at_s=run['started_at_s'],
            advanced_to_s=run['advanced_to_s'],wall_s=run['wall_s'],checkpoint_sha=sha(checkpoint),core_sha=CORE))
    except Exception as exc:
        save(OUT/'status.json',dict(status='failed',reason=str(exc),checkpoint_sha=sha(checkpoint),core_sha=CORE));raise


if __name__=='__main__':main()

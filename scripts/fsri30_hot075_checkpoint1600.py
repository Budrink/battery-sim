"""Re-identify the unchanged 1600 s state under the selected hot-contact law."""
import json
from pathlib import Path
from check_upv_restart import native
from fsri30_data import sha,save

ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'data/development/fsri30-valve-1600-state-20261027-r1'
HOT=ROOT/'data/development/fsri30-b17-hot-film-075-2200-20261027-r1'
OUT=ROOT/'data/development/fsri30-hot075-checkpoint1600-20261027-r1'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-packed-precise-progress-20261027-r1'
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'

def main():
    seed=json.loads((SEED/'status.json').read_text());source=SEED/'scenario.json';ck=SEED/'committed-1600.bin'
    if sha(ck)!=seed['checkpoint_sha'] or sha(PROBE)!=CORE:raise ValueError('Checkpoint/core mismatch')
    OUT.mkdir(parents=True,exist_ok=False);scene=HOT/'scenario.json';checkpoint=OUT/'committed.bin'
    status=dict(status='running',source_checkpoint_sha=sha(ck),scene_sha=sha(scene),core_sha=CORE)
    save(OUT/'status.json',status)
    try:
        run=native(scene,OUT/'replay',dict(restart_contract=CORE,restart_from=str(ck),
            restart_gas_response_scenario=str(source),save_committed_restart=str(checkpoint),
            committed_restart_at_s=1600.,stop_after_committed_restart=True,stop_after_first_rupture=False),
            timeout=120.,enforce_speed=False,probe=PROBE)
        if abs(run['advanced_to_s']-1600)>1e-6 or not checkpoint.exists():raise ValueError('Wrong checkpoint time')
        status.update(status='complete',time_s=run['advanced_to_s'],checkpoint_sha=sha(checkpoint))
    except Exception as exc:status.update(status='failed',reason=str(exc));raise
    finally:save(OUT/'status.json',status)

if __name__=='__main__':main()

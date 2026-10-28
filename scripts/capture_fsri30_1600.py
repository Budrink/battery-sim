"""Capture an actual 1600 s state from the saved 80 kPa 1500 s checkpoint."""
import json
from pathlib import Path
from check_upv_restart import native
from fsri30_data import sha, save

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/development/fsri30-valve80kpa-only-20261027-r1/cascade/scenario.json'
PREVIOUS=ROOT/'data/development/fsri30-valve80kpa-only-20261027-r1/heater-00/committed-1500.bin'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-packed-precise-progress-20261027-r1'
CORE='4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'
OUT=ROOT/'data/development/fsri30-valve-1600-state-20261027-r1'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    if sha(PROBE)!=CORE:raise ValueError('Core fingerprint changed')
    seed=json.loads(SOURCE.read_text());sel=seed.setdefault('process_history_selection',{}).setdefault('gas_prefixes',[])
    for name in ('inside-B17-','inside-B18-'):
        if name not in sel:sel.append(name)
    scene=OUT/'scenario.json';save(scene,seed)
    checkpoint=OUT/'committed-1600.bin'
    save(OUT/'status.json',dict(status='running',source=str(SOURCE),source_sha=sha(SOURCE),
        checkpoint_1500_sha=sha(PREVIOUS),core_sha=CORE))
    try:
        run=native(scene,OUT/'replay',dict(restart_contract=CORE,restart_from=str(PREVIOUS),
            save_committed_restart=str(checkpoint),committed_restart_at_s=1600.,
            stop_after_committed_restart=True,stop_after_first_rupture=False),
            timeout=1200.,minimum_speed=.2,rolling_minimum=.2,probe=PROBE)
        if not checkpoint.exists() or abs(run['advanced_to_s']-1600.)>1e-6:raise ValueError('Missing exact checkpoint')
        save(OUT/'status.json',dict(status='complete',time_s=run['advanced_to_s'],checkpoint=str(checkpoint),
            checkpoint_sha=sha(checkpoint),source=str(SOURCE),source_sha=sha(SOURCE),
            checkpoint_1500_sha=sha(PREVIOUS),core_sha=CORE))
    except Exception as exc:
        save(OUT/'status.json',dict(status='failed',reason=str(exc),core_sha=CORE));raise


if __name__=='__main__':main()

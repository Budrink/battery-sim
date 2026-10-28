#!/usr/bin/env python3
"""Modified-drive core diagnostic, then strict training regression; NOT validation acceptance."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import calibrate_upv_r123 as cal
from check_upv_restart import native,compare


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True,help='Existing completed training folder')
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir()
    source=a.source.resolve();scene_path=source/'trial-000/scenario.json'
    scene=json.loads(scene_path.read_text());cal.PROBE=a.probe.resolve()
    core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    state=dict(status='running',phase='validation',native_sha256=core,source=str(source),parameters_refitted=False,
        heldout_evaluation_kind='modified_drive_diagnostic_not_protocol_faithful_validation',validation_claim=False)
    cal.save(out/'status.json',state)
    scripts=Path(__file__).resolve().parent
    try:
        subprocess.run([sys.executable,str(scripts/'replay_upv_frozen_validation.py'),
            '--source',str(scene_path),'--probe',str(cal.PROBE),'--output',str(out/'validation'),
            '--voltage-limited-setpoint','--complete-window'],check=True)
        heldout=json.loads((out/'validation/status.json').read_text())
        if heldout['status']!='complete':raise RuntimeError('Validation did not complete: '+str(heldout.get('error')))
        state.update(phase='training_regression');cal.save(out/'status.json',state)
        trial=out/'trial-000';trial.mkdir();cal.save(trial/'scenario.json',scene)
        end=scene['simulation']['duration_s']
        print('START same-core frozen training regression',flush=True)
        run=native(trial/'scenario.json',trial/'replay',dict(restart_contract=core,
            stop_after_first_rupture=False,save_unvented_restart=str(out/'sealed-state.bin'),
            unvented_restart_until_s=835.),2.2*end)
        with gzip.open(source/'trial-000/replay/run.json.gz','rt')as stream:previous=json.load(stream)
        regression=compare(previous,run);cal.save(out/'training-regression.json',regression)
        if not regression['bitwise_equal']:raise RuntimeError('Measured-current training trajectory changed')
        state.update(status='complete_not_calibrated',phase='report',wall_s=run['wall_s'],
            realtime_ratio=end/run['wall_s'],training_regression=regression)
        cal.save(out/'status.json',state)
        subprocess.run([sys.executable,str(scripts/'report_upv_frozen_comparison.py'),
            '--calibration',str(out),'--validation',str(out/'validation'),'--output',str(out/'review')],check=True)
        state['phase']='complete'
    except Exception as exc:
        state.update(status='failed',error=str(exc));raise
    finally:cal.save(out/'status.json',state)


if __name__=='__main__':main()

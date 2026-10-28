#!/usr/bin/env python3
"""Finish the frozen 10 mm reference, explicitly waiving performance only.

This does not loosen a production speed guard or any numerical tolerance.
The previous JSON observations cannot restore the complete CUDA state: replay
from zero and capture a real committed checkpoint before the slow cascade.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import signal

import calibrate_upv_r123 as settings
from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import native
from fsri30_data import sha
from report_fsri30_full_mesh import verify_provenance


def reference_request(scene, checkpoint, core, at_s=2200.):
    if at_s not in scene['failure_sample_times_s']:
        raise ValueError('Checkpoint must use an existing observation boundary')
    if scene['simulation']['minimum_dt_s'] != .01:
        raise ValueError('Reference physical timestep floor changed')
    if scene['gas_network'].get('pressure_pattern_predictor', False):
        raise ValueError('The rejected pressure predictor must remain disabled')
    return dict(restart_contract=core, stop_after_first_rupture=False,
        save_committed_restart=str(checkpoint), committed_restart_at_s=at_s,
        stop_after_committed_restart=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--checkpoint-at-s', type=float, default=2200.)
    args=ap.parse_args();source=args.source.resolve();out=args.output.resolve()
    launch=json.loads((source/'launch.json').read_text())
    contract=json.loads((source/'checks/contract.json').read_text())
    verify_provenance(source, launch, contract)
    probe=Path(launch['command'][launch['command'].index('--probe')+1])
    fine=source/'checks/h-10mm';coarse=source/'checks/h-20mm'
    scene=json.loads((fine/'scenario.json').read_text())
    duration=scene['simulation']['duration_s']
    coarse_summary=json.loads((coarse/'summary.json').read_text())
    if duration != 5945. or coarse_summary['advanced_to_s'] != duration:
        raise ValueError('Expected the frozen full 5945 s pair')
    if coarse_summary['status'] != 'complete' or not coarse_summary['fresh_from_zero']:
        raise ValueError('20 mm reference must already be complete from zero')
    request=reference_request(scene, out/'committed-2200s.bin', launch['native_sha256'], args.checkpoint_at_s)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(fine/'scenario.json', out/'scenario.json')
    pinned={str(p):sha(p) for p in (probe,fine/'scenario.json',coarse/'scenario.json',
        coarse/'replay/run.json.gz',fine/'replay/incomplete-run.json.gz',
        Path(__file__).resolve(),Path(__file__).with_name('check_upv_restart.py'))}
    state=dict(status='running', phase='fresh-10mm-replay', pid=os.getpid(), source=str(source),
        native_sha256=launch['native_sha256'], duration_s=duration, fresh_from_zero=True,
        speed_guard_enforced=False, wall_timeout_s=None, performance_waiver='Explicit user instruction: finish slow 10 mm reference',
        numerical_controls_unchanged=True, physical_minimum_step_s=.01,
        parameters_unchanged=True, closed_air_boundary_unchanged=True,
        checkpoint_at_s=args.checkpoint_at_s, request=request, pinned_files=pinned,
        calibrated=False, validation_passed=False, spatially_converged=False)
    atomic_save(out/'contract.json', state);atomic_save(out/'status.json', state)
    def ensure_pinned():
        if any(sha(Path(path)) != value for path,value in pinned.items()):
            raise ValueError('Frozen reference input changed')
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    settings.PROBE=probe
    for variable,name in (('BATTERY_AVERAGED_FAILURE_SNAPSHOT','pressure-failure.json'),
            ('BATTERY_MIXING_FAILURE_SNAPSHOT','mixing-failure.json'),
            ('BATTERY_GAS_FAILURE_SNAPSHOT','constituent-failure.json')):
        os.environ[variable]=str(out/name)
    request['gas_frames_path']=str(out/'gas-fields.bin')
    try:
        # +inf is internal only, not serialized as invalid JSON. The helper's
        # existing numerical/balance checks stay active; speed is still logged.
        run=native(out/'scenario.json',out/'replay',request,math.inf,enforce_speed=False)
        ensure_pinned()
        if run.get('partial') or run['started_at_s'] != 0. or abs(run['advanced_to_s']-duration)>1e-7:
            raise ValueError('Full fine-grid horizon did not complete')
        if run['solver_profile']['minimum_regular_step_s'] < .01-1e-9:
            raise ValueError('Physical timestep floor violated')
        if run.get('max_oxidizer_residual_kg',0.) > 1e-10:
            raise ValueError('Oxygen conservation failed')
        state.update(status='computed',phase='full-pair-and-propagation-analysis',
            advanced_to_s=run['advanced_to_s'],wall_s=run['wall_s'],run_sha256=sha(out/'replay/run.json.gz'),
            checkpoint_sha256=sha(Path(request['save_committed_restart'])))
        del run;atomic_save(out/'status.json',state)
        # Analysis is read-only and versioned separately from the pinned core.
        from fsri30_propagation_patterns import write_report
        state['report']=write_report(source,out,out/'comparison')
        state.update(status='complete',phase='reported',spatially_converged=state['report']['spatially_converged'])
    except KeyboardInterrupt:
        state.update(status='stopped',error='Interrupted; preserved observations are not a completed reference')
    except Exception as exc:
        state.update(status='failed',error=str(exc))
    finally:
        atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)
    return 0 if state['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())

"""Fresh, pinned-core test of the user-approved existing-parameter candidate."""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess

import numpy as np

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import physical_error
from check_upv_restart import native
from fsri30_data import save, sha
from fsri30_wave_fit import signatures

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/development/fsri30-valve80kpa-only-20261027-r1/selected-scenario.json'
PROBE = ROOT / 'build/calibration-probes/battery_sim_casing_probe-packed-precise-progress-20261027-r1'
CORE = '4a97222e8dcb55d00a2431efbf0e4707a76f4ea872123b4ad8e9dd778d996fc6'
MONITOR = ROOT / 'build/ui/battery_sim_calibration_monitor'


def candidate(source):
    scene = deepcopy(source)
    batteries = [c for c in scene['cells'] if c.get('role') == 'battery']
    assert len(batteries) == 30
    for c in batteries:
        c['conversion']['reaction_rate_per_s'] *= 1.28
    gas = scene['gas_network']
    gas['mixing_diffusivity_m2_s'] *= 3.
    vents = [c for c in gas['channels'] if c['id'].startswith('aligned-vent-')]
    assert len(vents) == 8
    for c in vents:
        assert c.get('hydraulic_resistance_factor', 1.) == 1.
        c['hydraulic_resistance_factor'] = 25.
    # Undo exactly these edits to prove no hidden material/geometry/R3 changes.
    restored = deepcopy(scene)
    for a, b in zip(restored['cells'], source['cells']):
        if a.get('role') == 'battery':
            a['conversion']['reaction_rate_per_s'] = b['conversion']['reaction_rate_per_s']
    restored['gas_network']['mixing_diffusivity_m2_s'] = source['gas_network']['mixing_diffusivity_m2_s']
    for a, b in zip(restored['gas_network']['channels'], source['gas_network']['channels']):
        if a['id'].startswith('aligned-vent-'):
            a.pop('hydraulic_resistance_factor', None)
            if 'hydraulic_resistance_factor' in b:
                a['hydraulic_resistance_factor'] = b['hydraulic_resistance_factor']
    assert restored == source
    scene['simulation']['duration_s'] = 2400.
    scene['failure_sample_times_s'] = [t for t in scene['failure_sample_times_s'] if t <= 2400.]
    scene['metadata']['r2_gas_shape_probe'] = dict(
        r2_rate_multiplier=1.28, r3_rate_multiplier=1., outlet_hydraulic_resistance=25.,
        mixing_diffusivity_multiplier=3., outlets=len(vents),
        fresh_initial_state=True, calibrated=False, validation_passed=False,
        objective='Primary timing and cooling shape; peak amplitude is not a rejection criterion.')
    return scene


def main(source=SOURCE, probe=PROBE, expected_core=CORE, candidate_fn=candidate):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--full-trajectory', action='store_true',
                        help='Disable experimental/speed pruning, retain physical checks')
    parser.add_argument('--record-visual', action='store_true')
    parser.add_argument('--checkpoint-at', type=float,
                        help='Save a full native restart at an existing observation time')
    parser.add_argument('--mixing-failure-snapshot', action='store_true')
    parser.add_argument('--probe', type=Path, help='Numerical-fix binary; requires its explicit SHA256')
    parser.add_argument('--probe-sha256')
    parser.add_argument('--restart-from', type=Path)
    parser.add_argument('--restart-contract', help='Original checkpoint core identity, not the new binary hash')
    parser.add_argument('--stop-at-checkpoint', action='store_true')
    args = parser.parse_args()
    if bool(args.probe) != bool(args.probe_sha256):
        raise ValueError('A replacement binary requires both --probe and --probe-sha256')
    if args.probe:
        probe, expected_core = args.probe.resolve(), args.probe_sha256
    if bool(args.restart_from) != bool(args.restart_contract):
        raise ValueError('Restart requires both full checkpoint and its original core identity')
    if args.stop_at_checkpoint and args.checkpoint_at is None:
        raise ValueError('--stop-at-checkpoint requires --checkpoint-at')
    if sha(probe) != expected_core:
        raise ValueError('Pinned native core changed')
    original = json.loads(source.read_text())
    scene = candidate_fn(original)
    out = args.output.resolve()
    if args.prepare_only:
        print(json.dumps(scene['metadata']['r2_gas_shape_probe'], indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    if args.checkpoint_at is not None:
        if not any(abs(t-args.checkpoint_at)<1e-8 for t in scene['failure_sample_times_s']):
            raise ValueError('Checkpoint time must be an existing observation')
    if args.mixing_failure_snapshot:
        os.environ['BATTERY_MIXING_FAILURE_SNAPSHOT']=str(out/'mixing-failure.json')
    save(out/'scenario.json', scene)
    experiment = json.loads((source.parent/'experiment.json').read_text())
    atomic_save(out/'experiment.json', experiment)
    state = dict(status='starting', stage='fresh-from-zero', pid=os.getpid(),
                 source=str(source), source_sha256=sha(source), core_sha256=expected_core,
                 scenario_sha256=sha(out/'scenario.json'), controls=scene['metadata']['r2_gas_shape_probe'],
                 calibrated=False, validation_passed=False, progress={})
    state['stopping_policy'] = 'physical checks only' if args.full_trajectory else 'physical, timing and speed checks'
    if args.restart_from:
        state.update(stage='resumed-full-state', restart_from=str(args.restart_from.resolve()),
                     restart_sha256=sha(args.restart_from), restart_contract=args.restart_contract)
    if args.record_visual:
        state['visual_manifest'] = str(out/'visual/manifest.json')
    atomic_save(out/'status.json', state)
    atomic_save(out/'live.json', dict(phase='fresh', time_s=[], cells={}))
    if args.monitor:
        with (out/'monitor.log').open('w') as log:
            ui = subprocess.Popen([str(MONITOR), str(out)], stdout=log, stderr=log,
                                  start_new_session=True)
        state['monitor_pid'] = ui.pid
    state['status'] = 'running'
    atomic_save(out/'status.json', state)

    def inspect(run):
        ts = np.asarray(run['time_s'])
        cells = run['cell_mean_temperature_k']
        names = ['B10', 'B17', 'B18']
        pulses = signatures(ts, np.stack([cells[n] for n in names], axis=1), names)['pulses'] if len(ts) > 30 else []
        secondary = []
        for c in scene['cells']:
            name = c['id']
            if name == 'B10' or name not in run.get('chemistry', {}):
                continue
            law = c['conversion']
            emitted = np.asarray(run['chemistry'][name]['r2_generated_gas_kg'])
            reached = np.flatnonzero(emitted >= .1*law['mass_kg']*law['gas_yield'])
            if len(reached):
                secondary.append(dict(cell=name, time_s=float(ts[reached[0]])))
        state['progress'] = dict(time_s=float(ts[-1]), wall_s=run.get('wall_s', 0.), pulses=pulses,
            secondary_events=secondary,
            temperatures_c={n:float(cells[n][-1]-273.15) for n in names},
            ledgers={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k})
        atomic_save(out/'live.json', dict(phase='fresh', time_s=ts.tolist(),
            cells={k:[float(v-273.15) for v in values] for k,values in cells.items()}))
        atomic_save(out/'status.json', state)
        error = physical_error(run)
        if error:
            return error
        if args.full_trajectory:
            return None
        primary = [p for p in pulses if p['channel'] == 'B10']
        if primary and primary[0]['time_s'] < 1663.:
            return 'Primary pulse more than 100 s early'
        if ts[-1] > 1880. and not primary:
            return 'No primary pulse within the allowed timing window'
        if scene['metadata'].get('secondary_timing_guard', False):
            late = [p for p in pulses if p['channel'] in ('B17', 'B18')]
            if late and min(p['time_s'] for p in late) < 2057.:
                return 'Secondary pulse over 100 s early relative to T4 proxy (2157 s)'
            if ts[-1] > 2257. and not late:
                return 'Missing secondary pulse by T4 proxy +100 s'
        if 300. <= ts[-1] <= 1500.:
            mask = ts >= ts[-1]-60.
            truth = np.interp(ts[mask], experiment['time_s'], experiment['T5'])
            err = np.asarray(cells['B10'])[mask]-273.15-truth
            if len(err) >= 10 and np.all(np.abs(err) > 25.):
                return 'Warmup differs from T5 by over 25 K for 60 s'
        return None

    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        request = dict(stop_after_first_rupture=False)
        contract = args.restart_contract or expected_core
        if args.restart_from:
            request.update(restart_contract=contract, restart_from=str(args.restart_from.resolve()))
        if args.checkpoint_at is not None:
            request.update(restart_contract=contract,
                save_committed_restart=str(out/'committed-state.bin'),
                committed_restart_at_s=args.checkpoint_at,
                stop_after_committed_restart=args.stop_at_checkpoint)
        if args.record_visual:
            request['visual_recording_path'] = str(out/'visual')
        run = native(out/'scenario.json', out/'replay', request,
                     timeout=float('inf') if args.full_trajectory else 14000.,
                     prune_fn=inspect, enforce_speed=not args.full_trajectory,
                     minimum_speed=.2, rolling_minimum=.2, probe=probe)
        error = inspect(run)
        if error:
            raise RuntimeError(error)
        state.update(status='complete', end_s=run['advanced_to_s'], wall_s=run['wall_s'])
    except KeyboardInterrupt:
        state.update(status='stopped', reason='Stopped by user')
    except Exception as exc:
        state.update(status='pruned' if str(exc).startswith('Pruned:') else 'failed', reason=str(exc))
    finally:
        atomic_save(out/'status.json', state)
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()

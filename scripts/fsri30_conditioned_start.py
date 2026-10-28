"""One unchanged physical prefix and an audited, already-vented restart.

First micro-opening is ~731s; the source's main R2 burst is ~1755s. A sealed
restart cannot isolate that burst. Preserve the ACTUAL state at 1755s instead,
including open apertures, fuel, gas, heat, clocks and all previous ledgers.
Only fixed-volume gas contacts/passages may differ in conditioned suffixes.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal

import numpy as np

from calibrate_fsri30_bayes import atomic_save, canonical_hash, read_run
import calibrate_upv_r123 as settings
from check_upv_restart import compare, native
from fsri30_data import save, sha
from search_fsri30_transport import REFERENCE, ROOT

PROBE = ROOT/'build/calibration-probes/battery_sim_casing_probe-fsri30-conditioned-start-20261024-r1'
START_S = 1755.


def same_series(reference, resumed, until):
    result = compare(reference, resumed, until_s=until)
    start = resumed['started_at_s']; t = np.asarray(reference['time_s'])
    keep = (t >= start-1e-8)&(t <= until+1e-8)
    failures = {}
    for name, values in reference['cell_mean_temperature_k'].items():
        a = np.asarray(values)[keep]; b = np.asarray(resumed['cell_mean_temperature_k'][name])
        if a.shape != b.shape:
            raise ValueError('Conditioned reference observation shapes disagree')
        if not np.array_equal(a, b):
            failures[name] = float(np.max(np.abs(a-b)))
    result.update(cell_mean_bitwise_equal=not failures, cell_mean_differences_k=failures)
    result['accepted'] = result['bitwise_equal'] and not failures
    return result


def prepare(out, reference=REFERENCE, probe=PROBE, timeout=7200.):
    out = out.resolve(); out.mkdir(parents=True, exist_ok=True)
    scene = json.loads((reference/'scenario.json').read_text()); scene['gas_film_trace'] = True
    identity = dict(probe_sha256=sha(probe), reference_scenario_sha256=sha(reference/'scenario.json'),
        original_core_sha256=json.loads((reference/'evaluation.json').read_text())['native_sha256'],
        captured_at_s=START_S, scene_hash=canonical_hash(scene),
        physically_forced_ignition=False, same_state_diagnostic_only=True)
    contract = out/'contract.json'
    if contract.exists() and json.loads(contract.read_text()) != identity:
        raise ValueError('Incompatible conditioned start: use a new output directory')
    save(contract, identity)
    state = dict(status='running', phase='capture-unchanged-prefix', pid=os.getpid(), **identity)
    atomic_save(out/'status.json', state)
    settings.PROBE = probe
    prefix = out/'prefix'; prefix.mkdir(exist_ok=True)
    scene_path = prefix/'scenario.json'; save(scene_path, scene)
    checkpoint = out/'main-burst-state.bin'
    if not (prefix/'replay/run.json.gz').exists():
        run = native(scene_path, prefix/'replay', dict(restart_contract=sha(probe),
            save_committed_restart=str(checkpoint), committed_restart_at_s=START_S,
            stop_after_committed_restart=True, stop_after_first_rupture=False),
            timeout, minimum_speed=.2, rolling_minimum=.2)
    else:
        run = read_run(prefix)
    if not checkpoint.exists() or abs(run['advanced_to_s']-START_S)>1e-8:
        raise ValueError('Did not capture the requested complete committed state')
    # The source is already partially vented. Never call this a sealed state or
    # zero the old channel energy/mass/fracture histories to make it load.
    open_channels = {k: v['opening_fraction'][-1] for k, v in run['channels'].items()
                     if v['opening_fraction'][-1] > 0.}
    if not open_channels:
        raise ValueError('Expected pre-existing vent history before the main burst')
    archived = read_run(reference)
    prefix_check = same_series(archived, run, START_S)
    save(out/'prefix-equivalence.json', prefix_check)
    if not prefix_check['accepted']:
        raise ValueError('New observer changes the original physical prefix')
    state.update(phase='verify-through-main-burst', start_open_channels=open_channels,
                 checkpoint_sha256=sha(checkpoint))
    atomic_save(out/'status.json', state)
    control = out/'identity-check'; control.mkdir(exist_ok=True)
    control_scene = deepcopy(scene); control_scene['simulation']['duration_s'] = 1800.
    control_scene['failure_sample_times_s'] = [t for t in scene['failure_sample_times_s'] if t <= 1800.]
    save(control/'scenario.json', control_scene)
    if not (control/'replay/run.json.gz').exists():
        resumed = native(control/'scenario.json', control/'replay', dict(restart_contract=sha(probe),
            restart_from=str(checkpoint), restart_transport_scenario=str(scene_path),
            stop_after_first_rupture=False), timeout, minimum_speed=.2, rolling_minimum=.2)
    else:
        resumed = read_run(control)
    check = same_series(archived, resumed, 1800.)
    check.update(initial_serialized_state_bitwise_equal=resumed.get('suffix_initial_state_bitwise_equal', False),
                 rupture_history_preserved=True, conditioned_transport_diagnostic=True,
                 physical_laws_changed=False)
    check['accepted'] = check['accepted'] and check['initial_serialized_state_bitwise_equal']
    save(out/'identity-equivalence.json', check)
    if not check['accepted']:
        raise ValueError('Restored state/trajectory does not reproduce the archived main burst')
    state.update(status='ready', phase='conditioned-start-verified', checkpoint=str(checkpoint),
                 source_scenario=str(scene_path), calibrated=False, validation_passed=False)
    atomic_save(out/'status.json', state)
    return scene, checkpoint, prefix


def join_observations(prefix, suffix):
    """Composite temperature/chemistry trajectory, explicitly not a fresh replay.

    Accepted film-heat integrals in suffix stay suffix-only: never fabricate a
    prehistory for a contact which did not exist before this diagnostic.
    """
    if abs(prefix['time_s'][-1]-suffix['time_s'][0])>1e-8:
        raise ValueError('Prefix/suffix do not meet at the same physical time')
    for section in ('cell_mean_temperature_k', 'gas', 'chemistry', 'channels', 'joule_energy_j', 'process_totals'):
        def seam(a,b):
            if isinstance(b,dict):
                return a.keys()==b.keys() and all(seam(a[k],v) for k,v in b.items())
            return a[-1]==b[0]
        if not seam(prefix[section],suffix[section]):
            raise ValueError('Conditioned prefix/suffix has a physical state jump: '+section)
    result = dict(suffix)
    result['time_s'] = prefix['time_s'][:-1]+suffix['time_s']
    def series(a, b):
        if isinstance(b, dict):
            return {k: series(a[k], v) for k, v in b.items()}
        return a[:-1]+b
    for key in ('cell_mean_temperature_k', 'chemistry', 'gas', 'channels',
                'electrical', 'joule_energy_j', 'process_totals'):
        result[key] = series(prefix[key], suffix[key])
    for key in ('released_energy_j_history', 'remaining_energy_j', 'source_damage', 'source_max_temperature_k'):
        result[key] = prefix[key][:-30]+suffix[key]
    result.update(conditioned_transport_diagnostic=True, full_warmup_validation=False,
        prefix_started_at_s=prefix['started_at_s'], suffix_started_at_s=suffix['started_at_s'])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--probe',type=Path,default=PROBE)
    ap.add_argument('--reference',type=Path,default=REFERENCE)
    args = ap.parse_args()
    def stop(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        prepare(args.output,reference=args.reference,probe=args.probe)
    except BaseException as exc:
        status = args.output/'status.json'
        state = json.loads(status.read_text()) if status.exists() else {}
        state.update(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        atomic_save(status, state)
        raise


if __name__ == '__main__':
    main()

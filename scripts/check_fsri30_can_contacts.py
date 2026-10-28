#!/usr/bin/env python3
"""Fresh contact-strip hypothesis on the pinned FSRI CUDA core."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal

from calibrate_fsri30_bayes import atomic_save
from calibrate_fsri30_sequence import BASE, BASE_HASH, MINIMUM_SPEED, trial
from check_fsri30_vent_retention import SEED
from fsri30_can_contacts import add_can_contacts
from fsri30_data import load, save, sha
from fsri30_wave_fit import POLICY, reference

PROBE = Path('build/calibration-probes/battery_sim_casing_probe-joint-enthalpy-20261026-r1').resolve()
CORE = '6aa586496f59297caa81356c0fff28a17ffe733ba4a9693fb6f4a7e951f7813e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width-mm', type=float, default=1.)
    parser.add_argument('--contact-h', type=float, default=1000.)
    args = parser.parse_args()
    if sha(PROBE) != CORE or sha(BASE) != BASE_HASH:
        raise ValueError('Pinned core or physical seed changed')
    seed = json.loads((SEED/'status.json').read_text())['parameters']
    base = json.loads(BASE.read_text())
    scene = add_can_contacts(base, args.width_mm/1000., args.contact_h)
    data, manifest = load()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    save(out/'contract.json', dict(native_sha256=CORE, base_sha256=BASE_HASH,
        csv_sha256=manifest['csv_sha256'], seed_parameters=seed, policy=asdict(POLICY),
        contact_hypothesis=scene['metadata']['can_contact_hypothesis'],
        script_sha256=sha(Path(__file__)), adapter_sha256=sha(Path(__file__).with_name('fsri30_can_contacts.py')),
        fresh_from_zero=True, minimum_realtime_ratio=MINIMUM_SPEED, calibrated=False, validation_passed=False))
    state = dict(status='running', pid=os.getpid(), calibrated=False, validation_passed=False)
    def update(row):
        state['progress'] = row
        atomic_save(out/'status.json',state)
    def stop(*unused): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    atomic_save(out/'status.json',state)
    try:
        row = trial(out/'trial-0000',scene,seed,20,reference(data),data,update,probe=PROBE)
        state.update(status='diagnostic-finished',result=row)
    except KeyboardInterrupt:
        state['status']='stopped'
    except Exception as exc:
        state.update(status='failed',error=str(exc))
        raise
    finally:
        atomic_save(out/'status.json',state)


if __name__ == '__main__': main()

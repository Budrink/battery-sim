#!/usr/bin/env python3
"""Independent gas--C3/ISS casing and gas--left plate OFF diagnostics."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal

import numpy as np
import calibrate_upv_r123 as cal
import upv_r123_data as dataio
from check_upv_local_max import summary, local_plot
from check_upv_monotone_gas import plot
from check_upv_restart import native

LABELS = dict(baseline='Исходный', no_casing='Без газ ↔ оболочка C3/ISS',
              no_plate='Без газ ↔ левая плита')
TITLE = 'C3/ISS: независимое отключение контактов; исходный газоперенос'
SCALES = ('background_conductance_w_per_k', 'active_conductance_w_per_k',
          'destroyed_background_conductance_w_per_k', 'destroyed_active_conductance_w_per_k')


def without_contact(original, target):
    if target not in ('C3:case:-x', 'steel-L'):
        raise ValueError('Unexpected ablation target')
    scene = copy.deepcopy(original)
    ports = {p['id']: p for p in scene['ports']}
    changed = []
    for index, contact in enumerate(scene['gas_network']['thermal_contacts']):
        if ports[contact['port_id']]['cell_id'] != target:
            continue
        before = copy.deepcopy(contact)
        if contact.get('connectivity', {}).get('kind') != 'directed_geometry':
            raise ValueError('Expected common directed contact law')
        for key in SCALES:
            contact['connectivity'][key] = 0.
        changed.append(dict(index=index, before=before, after=copy.deepcopy(contact)))
    if not changed:
        raise ValueError('No gas contacts for ablation target')
    expected = copy.deepcopy(original)
    for entry in changed:
        expected['gas_network']['thermal_contacts'][entry['index']] = entry['after']
    if scene != expected:
        raise ValueError('Non-contact physics changed')
    return scene, dict(diagnostic_only=True, calibrated=False, target=target,
        gas_transport='unchanged baseline', contacts=changed,
        changed='Only four conductance scales set to zero; shell and plate still exist')


def verify_disabled_heat(scene, run, target):
    records = [c for c in run['gas_film_trace'].values() if c['solid'] == target]
    if not records:
        raise ValueError('Missing accepted-step records for disabled contact')
    maximum = max(float(np.max(np.abs(c[key]))) for c in records
        for key in ('heat_to_solid_j', 'positive_step_heat_j', 'negative_step_heat_j'))
    if maximum != 0.:
        raise ValueError('Disabled gas contact still transfers heat: '+str(maximum))
    return dict(target=target, trace_records=len(records), max_absolute_heat_j=maximum)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--timeout', type=float, default=1800.)
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve()
    if args.plot_only:
        local_plot(out, labels=LABELS, title=TITLE)
        return 0
    source = dataio.ROOT/'data/calibration/upv-separated-chemistry-v6/thermal-gas-mixing-20261020-r1/temperature'
    original = json.loads((source/'scenario.json').read_text())
    core = hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    prior = json.loads((source.parent/'status.json').read_text())
    if core != prior['native_sha256'] or prior['status'] != 'complete':
        raise ValueError('Archived baseline must use the same native kernel')
    out.mkdir(exist_ok=False)
    state = dict(status='running', pid=os.getpid(), source=str(source), native_sha256=core,
        diagnostic_only=True, calibrated=False, validation_run=False, cases={},
        independent_cases=True, gas_transport='unchanged baseline', labels=LABELS)
    cal.save(out/'status.json', state)
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    runs = {}
    try:
        with gzip.open(source/'replay/run.json.gz', 'rt') as stream:
            runs['baseline'] = json.load(stream)
        data, _ = dataio.load('no-cooling')
        state['cases']['baseline'] = summary(original, runs['baseline'])
        cal.save(out/'status.json', state)
        for label, target in (('no_casing', 'C3:case:-x'), ('no_plate', 'steel-L')):
            # Each case starts from original, never from the preceding ablation.
            scene, manifest = without_contact(original, target)
            folder = out/label; folder.mkdir()
            cal.save(folder/'scenario.json', scene); cal.save(folder/'manifest.json', manifest)
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest() != core:
                raise ValueError('Native binary changed during ablation')
            state['active_case'] = label; cal.save(out/'status.json', state)
            print('START', label, 'disabled_contacts', len(manifest['contacts']), flush=True)
            run = native(folder/'scenario.json', folder/'replay', dict(stop_after_first_rupture=False), args.timeout)
            report = summary(scene, run)
            report['disabled_contact_check'] = verify_disabled_heat(scene, run, target)
            cal.save(folder/'report.json', report)
            state['cases'][label] = report; runs[label] = run; cal.save(out/'status.json', state)
            plot(out, runs, data, labels=LABELS, title=TITLE)
            print('CASE', label, 'speed', report['speed'], 'samples', json.dumps(report['C3_ISS_samples']), flush=True)
        state['status'] = 'complete'
    except KeyboardInterrupt:
        state['status'] = 'stopped'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
    finally:
        state.pop('active_case', None); cal.save(out/'status.json', state)
        print('CONTACT_ABLATION_STATUS', state['status'], state.get('error', ''), flush=True)
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())

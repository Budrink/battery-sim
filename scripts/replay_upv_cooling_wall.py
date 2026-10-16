#!/usr/bin/env python3
"""Replay the explicit wall without fitting on the active-cooling experiment.

Rebuild the whole warmup first, then transfer every existing solid/shell field
into the post-event run. C2 retains the existing measured event-conditioned
initialization. Old calibration artifacts are inputs only and stay untouched.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import calibrate_upv_geometric_rupture as model
from calibrate_upv import save
from upv_cooling_wall import attach, WALL_ID


def transfer_warm_fields(post, warm):
    s = copy.deepcopy(post)
    for c in s['cells']:
        if c['id'] in ('C2', 'chamber-wall'):
            continue
        field = warm['final_fields_k'][c['id']]
        if len(field) != np.prod(c['thermal']['divisions']):
            raise ValueError('Warmup grid mismatch: '+c['id'])
        c['thermal']['initial_temperature_field_k'] = field.copy()
        c['thermal']['initial_temperature_k'] = float(np.mean(field))
    for shell in s['shells']:
        if shell['contents_id'] == 'C2':
            continue
        shell['initial_temperature_fields_k'] = {
            face: warm['final_fields_k'][shell['contents_id']+':case:'+face].copy()
            for face in ('-x', '+x', '-y', '+y', '-z', '+z')}
    return s


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('--output', type=Path, default=model.ROOT/'data/calibration/upv-cooling-wall-v1')
    cli.add_argument('--seed', type=Path, default=model.ROOT/'data/calibration/upv-geometric-joint-v12')
    cli.add_argument('--sink-g', type=float, default=10., help='Constant water-side G, W/K; uncalibrated prior')
    cli.add_argument('--contact-h', type=float, default=10000., help='Reference Al/plate h, W/m2/K; uncalibrated prior')
    cli.add_argument('--probe', type=Path, default=model.previous.PROBE)
    cli.add_argument('--warmup-only', action='store_true')
    args = cli.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory; keep historical replays immutable')
    args.output.mkdir(parents=True)
    data, record = model.native.old.load_record('active-cooling')
    event = model.native.old.event_marker(data)
    opts = dict(coolant_temperature_k=float(data['T111_k'][0]),
                sink_conductance_w_per_k=args.sink_g, contact_h_w_per_m2_k=args.contact_h)
    heat_path = model.shell.HEAT/'active-cooling-fitted-scenario.json'
    heat = json.loads(heat_path.read_text())
    heat['simulation']['duration_s'] = event['last_pre_event_s']
    model.native.old.refresh_boundaries(heat, data, 0., event['last_pre_event_s'])
    heat = attach(heat, **opts)
    # Sample the original acquisition times: channel 110 is a held-out local
    # contact observation now, never an imposed temperature or a tuning target.
    at = np.asarray(data['time_s']); at = at[(at >= 0) & (at <= event['last_pre_event_s'])]
    heat['failure_sample_times_s'] = at.tolist()
    save(args.output/'warmup-scenario.json', heat)
    runner = model.native.Runner(args.probe, args.output)
    try:
        warm = runner.run(heat)
        save(args.output/'warmup.json', warm)
        plate = warm['final_fields_k'][WALL_ID]
        print('warm wall range K', min(plate), max(plate), 'balance J', warm['energy_residual_j'], flush=True)
        post_path = args.seed/'active-cooling-validate-scenario.json'
        post = transfer_warm_fields(attach(json.loads(post_path.read_text()), **opts), warm)
        post['metadata']['cooling_wall']['initialization'] = 'Full re-run of intact warmup; wall and all witness/shell fields transferred'
        save(args.output/'active-cooling-scenario.json', post)
        provenance = dict(status='Implementation replay, not a calibration or accepted validation',
                          parameters_fitted=False, options=opts, event=event,
                          warmup_input_sha256=model.digest(heat_path), failure_input_sha256=model.digest(post_path),
                          model_fingerprint=model.fingerprint(args.probe),
                          adapter_sha256=model.digest(Path(__file__).with_name('upv_cooling_wall.py')),
                          source_csv_sha256=record['canonical_sha256'])
        save(args.output/'provenance.json', provenance)
        if args.warmup_only:
            return
        run = runner.run(post); model.audit(run, post)
        prior = json.loads((args.seed/'validate.json').read_text())
        observed110 = model.native.old.resample(data['time_s'], data['T110_k'],
                                               np.asarray(run['time_s'])+event['last_pre_event_s'])
        report = dict(provenance=provenance, metrics=model.shell.metrics(run, prior),
                      observed_contact_110_k=observed110.tolist(), result=run,
                      observed_temperature_k=prior['observed_temperature_k'],
                      observed_pressure_bar=prior['observed_pressure_bar'])
        save(args.output/'replay.json', report)
        print('unchanged-parameter replay', json.dumps(report['metrics']), flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()

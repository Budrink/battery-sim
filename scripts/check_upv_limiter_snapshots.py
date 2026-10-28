#!/usr/bin/env python3
"""Audit frozen saved states with one actual production CUDA transport window.

No fitting and no trajectory prediction. Reaction/films are deliberately absent
from this operator-isolation experiment. Historical observations are not claimed
to be historical pre-transport substages.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import calibrate_upv_r123 as cal
import upv_r123_data as dataio

PROBE = Path('/tmp/battery-sim-cuda-v1/battery_sim_gas_limiter_snapshot')


def summarize(result, selected):
    incoming = []; outgoing = []; boundary = []
    for edge in result['edges']:
        a = edge['from'] in selected; b = edge['to'] in selected
        if a == b: continue
        boundary.append(edge)
        (outgoing if edge['donor'] in selected else incoming).append(edge)
    dt = result['window_s']
    def flows(edges):
        raw = sum(e['raw_mass_kg'] for e in edges)/dt
        pair = sum(e['pair_request_kg'] for e in edges)/dt
        accepted = sum(e['accepted_mass_kg'] for e in edges)/dt
        return dict(raw_kg_s=raw, after_pair_kg_s=pair, accepted_kg_s=accepted,
            pair_acceptance=pair/raw if raw else None, node_acceptance=accepted/pair if pair else None)
    mix_request = sum(e['mix_request_each_way_kg'] for e in boundary)/dt
    mix_actual = sum(e['mix_accepted_each_way_kg'] for e in boundary)/dt
    bottlenecks = sorted([e for e in boundary if e['raw_mass_kg'] > 0],
        key=lambda e:e['raw_mass_kg']-e['accepted_mass_kg'], reverse=True)[:5]
    return dict(incoming=flows(incoming), outgoing=flows(outgoing),
        mixing_requested_each_way_kg_s=mix_request, mixing_accepted_each_way_kg_s=mix_actual,
        mixing_budget_acceptance=mix_actual/mix_request if mix_request else None,
        min_pressure_node_scale=min(n['pressure_scale'] for n in result['nodes'] if n['id'] in selected),
        min_mixing_node_scale=min(n['mixing_scale'] for n in result['nodes'] if n['id'] in selected),
        top_pressure_bottlenecks=bottlenecks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); out = args.output.resolve(); out.mkdir(exist_ok=False)
    sources = dict(baseline=dataio.ROOT/'data/calibration/upv-separated-chemistry-v6/thermal-gas-mixing-20261020-r1/temperature',
        gas_max=dataio.ROOT/'data/calibration/upv-separated-chemistry-v6/local-max-C3-ISS-20261020-r1/gas_max')
    core = hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    state = dict(status='running', production_sha256=core, audit_sha256=hashlib.sha256(PROBE.read_bytes()).hexdigest(),
        diagnostic_only=True, calibrated=False, frozen_snapshots_not_trajectory_replay=True, cases={})
    cal.save(out/'status.json', state)
    try:
        fixture = json.loads(subprocess.check_output([str(PROBE), '--throughflow'], text=True))
        cal.save(out/'throughflow.json', fixture)
        for case, folder in sources.items():
            scene = json.loads((folder/'scenario.json').read_text())
            with gzip.open(folder/'replay/run.json.gz', 'rt') as stream: run = json.load(stream)
            gas_ids = [g['id'] for g in scene['gas_network']['compartments']]
            gaps = {n for n in gas_ids if n.startswith('gap-ISS1-')}
            region = gaps | {c['gas'] for c in scene['metadata']['pad_openings']['exposed_contacts']
                             if c['port'].startswith('opening-contact:ISS1:')}
            state['cases'][case] = []
            for requested_at in (882.877, 930., 950.):
                i = int(np.argmin(np.abs(np.asarray(run['time_s'])-requested_at)))
                at = run['time_s'][i]
                for dt in (.005, .01):
                    label = f'{case}-{at:.6f}-{dt:.3f}'; snapshot_path = out/(label+'-input.json')
                    snapshot = dict(scenario_path=str(folder/'scenario.json'), window_s=dt,
                        requested_time_s=requested_at, sampled_time_s=at,
                        nodes=[dict(id=n, **{k:v[i] for k,v in run['gas'][n].items()}) for n in gas_ids],
                        channels=[dict(id=e['id'], effective_area_m2=run['channels'][e['id']]['effective_area_m2'][i])
                                  for e in scene['gas_network']['channels']])
                    cal.save(snapshot_path, snapshot)
                    result = json.loads(subprocess.check_output([str(PROBE), str(snapshot_path)], text=True))
                    cal.save(out/(label+'-audit.json'), result)
                    row = dict(sampled_time_s=at, window_s=dt, slit=summarize(result, gaps),
                        slit_and_gallery=summarize(result, region), mass_residual_kg=result['mass_residual_kg'],
                        energy_residual_j=result['energy_residual_j'])
                    state['cases'][case].append(row); cal.save(out/'status.json', state)
                    print('SNAPSHOT', case, at, dt, 'slit', json.dumps(row['slit']['incoming']),
                        'mix_budget', row['slit']['mixing_budget_acceptance'], flush=True)
            del run
        if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest() != core:
            raise ValueError('Production binary changed during read-only audit')
        state['status'] = 'complete'
    except Exception as exc:
        state.update(status='failed', error=str(exc))
    finally:
        cal.save(out/'status.json', state)
        print('LIMITER_SNAPSHOT_STATUS', state['status'], state.get('error', ''), flush=True)
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__': raise SystemExit(main())

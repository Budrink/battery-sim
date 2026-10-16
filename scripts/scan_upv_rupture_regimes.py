#!/usr/bin/env python3
"""Bounded, training-only comparisons at a fixed chemical energy inventory.

The manifest is an explicit experimental design, not an optimizer. Every case
runs the full native model; restart reuses only fingerprinted complete reports.
Changing the solid/gas split never changes the sum of their chemical energies.
"""
import argparse
import json
import math
from pathlib import Path
import re

import calibrate_upv_geometric_rupture as model
from calibrate_upv import save

DEFAULT_CASES = [
    dict(id='gas-rich', gas_fraction=.8),
    dict(id='solid-rich', gas_fraction=.3),
    dict(id='strong-exposure', exposure_factor=3.),
    dict(id='gas-rich-strong-exposure', gas_fraction=.8, exposure_factor=3.),
    dict(id='fast-reaction', rate_factor=5.),
]
FACTORS = dict(exposure_factor='exposure_scale', rate_factor='solid_rate',
               wall_factor='wall_h', yield_factor='gas_yield',
               surface_factor='surface_h', strength_factor='strength_pa',
               damage_factor='damage_rate_per_s')


def parameters(seed, case):
    """Apply an explicit regime relative to a frozen seed, without compounding."""
    if set(case) - {'id', 'gas_fraction', *FACTORS}:
        raise ValueError('Unknown regime control')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', case['id']):
        raise ValueError('Case id must be a simple filename')
    p = dict(seed)
    total = seed['solid_energy_j'] + seed['gas_energy_j']
    if 'gas_fraction' in case:
        fraction = case['gas_fraction']
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('Invalid energy fraction')
        p['gas_energy_j'] = total * fraction
        p['solid_energy_j'] = total - p['gas_energy_j']
    for control, name in FACTORS.items():
        if control in case:
            p[name] *= case[control]
    for name, value in p.items():
        if not math.isfinite(value) or not model.LIMITS[name][0] <= value <= model.LIMITS[name][1]:
            raise ValueError('Out-of-bounds regime: ' + name)
    if not math.isclose(p['solid_energy_j'] + p['gas_energy_j'], total, rel_tol=1e-14):
        raise ValueError('Chemical inventory changed')
    return p


def entry(case_id, p, run, prior):
    return dict(id=case_id, parameters=p, metrics=model.shell.metrics(run, prior),
                energy_ledger=model.previous.ledger(run), steps=run['steps'], wall_s=run['wall_s'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=model.ROOT/'data/calibration/upv-geometric-regimes-v12')
    parser.add_argument('--seed-directory', type=Path, default=model.OUT)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--probe', type=Path, default=model.previous.PROBE)
    args = parser.parse_args()
    fp = model.fingerprint(args.probe)
    for name in ('heat-replay', 'numerics'):
        if json.loads((args.seed_directory/(name+'.json')).read_text())['fingerprint'] != fp:
            raise ValueError('Seed verification is stale')
    seed = json.loads((args.seed_directory/'checkpoint.json').read_text())['parameters']
    cases = json.loads(args.plan.read_text()) if args.plan else DEFAULT_CASES
    ids = [case['id'] for case in cases]
    if len(ids) != len(set(ids)) or 'seed' in ids:
        raise ValueError('Duplicate/reserved case id')
    # Validate the complete plan before spending GPU time or writing any result.
    candidates = [(case['id'], parameters(seed, case)) for case in cases]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(fingerprint=fp, seed=seed, cases=cases, validation_used_for_selection=False,
                    fixed_total_energy_j=seed['solid_energy_j']+seed['gas_energy_j'],
                    script_sha256=model.digest(__file__),
                    seed_run_sha256=model.digest(args.seed_directory/'best-run.json'))
    path = args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Different manifest: use a new output directory')
    save(path, manifest)
    scene, prior = model.inputs('no-cooling')
    seed_run = json.loads((args.seed_directory/'best-run.json').read_text())
    model.audit(seed_run, model.bake(scene, seed))
    history = [entry('seed', seed, seed_run, prior)]
    save(args.output/'seed.json', model.report('no-cooling', seed, seed_run, prior))
    runner = model.native.Runner(args.probe, args.output)
    try:
        for case_id, p in candidates:
            path = args.output/(case_id+'.json')
            if path.exists():
                report = json.loads(path.read_text())
                if report['parameters'] != p or report['fingerprint'] != fp:
                    raise ValueError('Stale candidate report')
                run = report['result']
                model.audit(run, model.bake(scene, p))
            else:
                print('starting', case_id, json.dumps(p), flush=True)
                run = model.run_candidate(runner, scene, p)
                report = model.report('no-cooling', p, run, prior)
                report['fingerprint'] = fp
                save(path, report)
            item = entry(case_id, p, run, prior)
            history.append(item)
            save(args.output/'history.json', history)
            print('completed', json.dumps(item), flush=True)
        best = min(history, key=lambda item: item['metrics']['objective_sum_squares'])
        save(args.output/'screen-summary.json', dict(best=best, history=history, manifest=manifest,
             status='Training-only regime screen; no accepted calibration or held-out selection'))
        print('best regime', best['id'], best['metrics']['witness']['rmse_k'], flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()

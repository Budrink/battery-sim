#!/usr/bin/env python3
"""Training-only refit followed by a separately invoked cooling sensitivity check.

All time integration is native CUDA. The residual interpolant proposes only one
extra trial; acceptance always uses a full native run. No validation selection.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import calibrate_upv_geometric_rupture as model
from calibrate_upv import save, metrics as heat_metrics
from propose_upv_rupture_regime import features
from replay_upv_cooling_wall import transfer_warm_fields
from scan_upv_rupture_regimes import parameters
from upv_cooling_wall import attach

SEED = model.ROOT/'data/calibration/upv-geometric-joint-v12'
# Declared before opening held-out temperatures. These are engineering priors,
# not a measured heat-exchanger UA inferred from the 2 L/min flow alone.
COOLING_CASES = [dict(id=name, sink_conductance_w_per_k=g,
                      contact_h_w_per_m2_k=10000.)
                 for name, g in [('weak', 3.), ('nominal', 10.), ('strong', 30.)]]
CASES = [dict(id='solid-rich', gas_fraction=.4),
         dict(id='gas-rich', gas_fraction=.75),
         dict(id='lower-exposure', exposure_factor=.5),
         dict(id='gas-rich-lower-exposure', gas_fraction=.75, exposure_factor=.5),
         dict(id='slower-reaction', rate_factor=1/3)]


def read(path):
    return json.loads(path.read_text())


def update_post_links(scene, heat):
    """Copy only recalibrated intact links; preserve all damaged-state laws.

The event source has no warmup volume (it is measured forcing). Its intact
bulk law is shared with the two witness cells; its initialization is retained.
"""
    out = copy.deepcopy(scene)
    cells = {c['id']: c for c in heat['cells']}
    for c in out['cells']:
        if c['role'] == 'battery':
            c['thermal']['bulk_response'] = copy.deepcopy(cells['C1']['thermal']['bulk_response'])
    laws = {e['id']: e['law'] for e in heat['interfaces']}
    for b in heat['boundaries']:
        if b['id'].startswith('source-C2-'):
            laws[b['id'].replace('source-', 'failure-')] = b['connectivity']
    for edge in out['interfaces']:
        edge['law']['geometry_reference_logit'] = laws[edge['id']]['geometry_reference_logit']
    h = heat['shells'][0]['contents_contact_h']
    for shell in out['shells']:
        shell['contents_contact_h'] = h
    return out


def prepare(name, heat_profile, runner, cooling=None):
    """Rebuild the warmup, then transfer complete fields to the event scene."""
    data, record = model.native.old.load_record(name)
    event = model.native.old.event_marker(data)
    p = heat_profile['parameters']
    heat = model.native.casing(model.native.heat_bake(model.native.read_scene(name), p), p)
    heat['simulation']['duration_s'] = event['last_pre_event_s']
    model.native.old.refresh_boundaries(heat, data, 0., event['last_pre_event_s'])
    at = np.asarray(data['time_s'])
    heat['failure_sample_times_s'] = at[(at >= 0) & (at <= event['last_pre_event_s'])].tolist()
    post, prior = model.inputs(name)
    post = update_post_links(post, heat)
    if cooling is not None:
        options = {k: v for k, v in cooling.items() if k != 'id'}
        options['coolant_temperature_k'] = float(data['T111_k'][0])
        heat, post = attach(heat, **options), attach(post, **options)
    elif name != 'no-cooling':
        raise ValueError('Held-out replay requires explicit cooling priors')
    if name == 'no-cooling' and any(b['id'].startswith('cooling-') or b['id']=='coolant-reservoir'
                                   for b in heat['boundaries']):
        raise ValueError('Cooling leaked into training')
    warm = runner.run(heat)
    return transfer_warm_fields(post, warm), prior, heat, warm, data, record


def fit(out, probe):
    if (out/'profile.json').exists() or (out/'manifest.json').exists():
        raise ValueError('Use a fresh output directory; do not overwrite a fit')
    heat = read(out/'heat/profile.json')
    seed_profile = read(SEED/'profile.json')
    if heat['validation_used_for_selection'] or seed_profile['validation_used_for_selection']:
        raise ValueError('Training-only initialization required')
    seed = seed_profile['parameters']
    manifest = dict(calibration='no-cooling', validation_used_for_selection=False,
                    cases=CASES, cooling_sensitivity=COOLING_CASES,
                    fixed_total_energy_j=seed['solid_energy_j']+seed['gas_energy_j'],
                    heat_profile_sha256=model.digest(out/'heat/profile.json'),
                    seed_profile_sha256=model.digest(SEED/'profile.json'),
                    fingerprint=model.fingerprint(probe), script_sha256=model.digest(__file__),
                    scope='Bounded conditional refit; no claim of global optimum or identified material constants')
    save(out/'manifest.json', manifest)
    runner = model.native.Runner(probe, out)
    history, runs, residuals = [], {}, {}
    try:
        scene, prior, heat_scene, warm, _, record = prepare('no-cooling', heat, runner)
        save(out/'warmup-scenario.json', heat_scene)
        save(out/'warmup.json', warm)
        def evaluate(case):
            p = parameters(seed, case)
            print('starting training', case['id'], flush=True)
            run = model.run_candidate(runner, scene, p)
            report = model.report('no-cooling', p, run, prior)
            save(out/(case['id']+'.json'), report)
            runs[case['id']] = report
            residuals[case['id']] = model.shell.score(run, prior)
            history.append(dict(id=case['id'], parameters=p, metrics=report['metrics']))
            save(out/'history.json', history)
            print('completed', case['id'], 'RMSE K', report['metrics']['witness']['rmse_k'],
                  'objective', report['metrics']['objective_sum_squares'], flush=True)
        evaluate(dict(id='baseline'))
        for case in CASES:
            evaluate(case)
        total = manifest['fixed_total_energy_j']
        f0 = seed['gas_energy_j']/total
        design, delta = [], []
        for case in CASES:
            p = parameters(seed, case)
            design.append(features([p['gas_energy_j']/total-f0,
                                    np.log(p['exposure_scale']/seed['exposure_scale']),
                                    np.log(p['solid_rate']/seed['solid_rate'])]))
            delta.append(residuals[case['id']]-residuals['baseline'])
        if np.linalg.matrix_rank(design) != 5:
            raise ValueError('Degenerate training design')
        coeff = np.linalg.solve(design, delta)
        def residual(x):
            return residuals['baseline']+features(x)@coeff
        lower = np.array([.4-f0, np.log(.5), np.log(1/3)])
        upper = np.array([.75-f0, 0., 0.])
        proposed = [least_squares(residual, x, bounds=(lower, upper))
                    for x in [np.zeros(3), (lower+upper)/2]]
        x = min(proposed, key=lambda v: v.fun@v.fun).x
        proposal = dict(id='proposal', gas_fraction=float(f0+x[0]),
                        exposure_factor=float(np.exp(x[1])), rate_factor=float(np.exp(x[2])))
        save(out/'proposal-controls.json', proposal)
        evaluate(proposal)
        best = min(history, key=lambda h: h['metrics']['objective_sum_squares'])
        p = best['parameters']; report = runs[best['id']]
        # Numerical refinement is checked on training before any validation read.
        fine = model.run_candidate(runner, scene, p, .5)
        changes = model.previous.changes(report['result'], fine)
        save(out/'numerics.json', dict(changes=changes, result=fine))
        if changes['max_witness_change_k'] > .25 or changes['max_pressure_change_bar'] > .025:
            raise ValueError('Refinement failed; profile not frozen')
        profile = dict(parameters=p, selected_case=best['id'], validation_used_for_selection=False,
                       calibration_accepted=False, manifest=manifest,
                       source_csv_sha256=record['canonical_sha256'],
                       status='Frozen training-only diagnostic fit; cooling held out; not a validated destruction model',
                       fitted=['energy partition at fixed total', 'gas exposure', 'reaction rate'],
                       fixed='Battery/pad capacities and reference k, newly frozen intact links, rupture and gas-yield priors')
        save(out/'profile.json', profile)
        save(out/'fit.json', report)
        save(out/'no-cooling-fit-scenario.json', model.bake(scene, p))
        print('FROZEN', best['id'], json.dumps(best['metrics']), flush=True)
    finally:
        runner.close()


def validate(out, probe):
    profile = read(out/'profile.json'); manifest = profile['manifest']
    if manifest['fingerprint'] != model.fingerprint(probe) or manifest['script_sha256'] != model.digest(__file__):
        raise ValueError('Model changed after freeze')
    if manifest['heat_profile_sha256'] != model.digest(out/'heat/profile.json'):
        raise ValueError('Heat profile changed after freeze')
    target = out/'cooling-check'
    if target.exists():
        raise ValueError('Do not overwrite held-out results')
    target.mkdir()
    profile_hash = model.digest(out/'profile.json')
    runner = model.native.Runner(probe, target); reports = []
    try:
        for case in manifest['cooling_sensitivity']:
            print('starting held-out', case['id'], flush=True)
            scene, prior, heat_scene, warm, data, record = prepare('active-cooling', read(out/'heat/profile.json'), runner, case)
            folder = target/case['id']; folder.mkdir()
            save(folder/'warmup-scenario.json', heat_scene); save(folder/'warmup.json', warm)
            run = model.run_candidate(runner, scene, profile['parameters'])
            report = model.report('active-cooling', profile['parameters'], run, prior)
            report.update(cooling_priors=case, frozen_profile_sha256=profile_hash,
                          status='Held-out sensitivity case; NOT ranked or selected',
                          warmup_metrics=heat_metrics(warm, data), source_csv_sha256=record['canonical_sha256'])
            save(folder/'report.json', report)
            save(folder/'scenario.json', model.bake(scene, profile['parameters']))
            reports.append(dict(id=case['id'], cooling_priors=case, metrics=report['metrics'],
                                warmup_metrics=report['warmup_metrics']))
            save(target/'summary.json', dict(cases=reports, selected_case=None,
                 frozen_profile_sha256=profile_hash, validation_used_for_selection=False,
                 limitation='Water-side G is unmeasured; contact h and plate footprint remain fixed unmeasured priors. Not independent blind validation.'))
            print('held-out', case['id'], json.dumps(report['metrics']), flush=True)
        if profile_hash != model.digest(out/'profile.json'):
            raise ValueError('Profile mutated during validation')
    finally:
        runner.close()


if __name__ == '__main__':
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('mode', choices=['fit', 'validate'])
    cli.add_argument('--output', type=Path, default=model.ROOT/'data/calibration/upv-training-refit-v13')
    cli.add_argument('--probe', type=Path, default=model.previous.PROBE)
    args = cli.parse_args()
    (fit if args.mode == 'fit' else validate)(args.output, args.probe)

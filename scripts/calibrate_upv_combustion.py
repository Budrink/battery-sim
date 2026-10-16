#!/usr/bin/env python3
"""Training-only rupture/combustion search on the frozen v16 pressure geometry."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.optimize import least_squares

import calibrate_upv_mass_paths as score
import calibrate_upv_casing as native
import upv_mass_paths as geometry
from calibrate_upv import save

ROOT = geometry.ROOT
PRIOR = ROOT/'data/calibration/upv-mass-paths-v16-network'
OUT = ROOT/'data/calibration/upv-continuous-combustion-v17'
NAMES = ('shell_strength_pa', 'shell_rate_per_s', 'solid_rate_per_s', 'gas_rate_per_s')
BOUNDS = dict(shell_strength_pa=(1e4, 1e8), shell_rate_per_s=(.002, 20.),
              solid_rate_per_s=(.3, 30.), gas_rate_per_s=(.2, 2000.))
CASES = [('baseline', {}), ('gas-10x', {'gas_rate_per_s': 10.}),
         ('gas-100x', {'gas_rate_per_s': 100.}), ('gas-01x', {'gas_rate_per_s': .1}),
         ('shell-strong', {'shell_strength_pa': 10.}), ('shell-weak', {'shell_strength_pa': .1}),
         ('opening-fast', {'shell_rate_per_s': 10.}), ('opening-slow', {'shell_rate_per_s': .1}),
         ('solid-slow', {'solid_rate_per_s': .25}), ('solid-fast', {'solid_rate_per_s': 3.})]


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def frozen_signature(scene):
    s = copy.deepcopy(scene)
    s.pop('metadata', None)
    core = next(c for c in s['cells'] if c['id'] == 'C2')['conversion']
    core['activation_rate_per_s'] = core['reaction_rate_per_s'] = 0.
    for node in s['gas_network']['compartments']:
        if node.get('reaction_rate_per_s', 0) > 0:
            node['reaction_rate_per_s'] = 0.
    for channel in s['gas_network']['channels']:
        if channel['kind'] == 'rupture':
            channel['opening_pressure_pa'] = 0.
            channel['damage_opening']['pressure_damage_rate_per_s'] = 0.
    return geometry.digest(s)


def seed(scene):
    shell = next(c for c in scene['gas_network']['channels'] if c['kind'] == 'rupture')
    core = next(c for c in scene['cells'] if c['id'] == 'C2')['conversion']
    rates = {n['reaction_rate_per_s'] for n in scene['gas_network']['compartments'] if n.get('reaction_rate_per_s', 0) > 0}
    if len(rates) != 1:
        raise ValueError('Expected one existing external-gas rate')
    return dict(shell_strength_pa=shell['opening_pressure_pa'],
                shell_rate_per_s=shell['damage_opening']['pressure_damage_rate_per_s'],
                solid_rate_per_s=core['reaction_rate_per_s'], gas_rate_per_s=rates.pop())


def bake(scene, parameters):
    if set(parameters) != set(NAMES):
        raise ValueError('Only the four declared rupture/reaction parameters may change')
    for key, (lo, hi) in BOUNDS.items():
        if not math.isfinite(parameters[key]) or not lo <= parameters[key] <= hi:
            raise ValueError('Out of bounds: '+key)
    s = copy.deepcopy(scene)
    core = next(c for c in s['cells'] if c['id'] == 'C2')['conversion']
    core['activation_rate_per_s'] = core['reaction_rate_per_s'] = parameters['solid_rate_per_s']
    for node in s['gas_network']['compartments']:
        if node.get('reaction_rate_per_s', 0) > 0:
            node['reaction_rate_per_s'] = parameters['gas_rate_per_s']
    for channel in s['gas_network']['channels']:
        if channel['kind'] == 'rupture':
            channel['opening_pressure_pa'] = parameters['shell_strength_pa']
            channel['damage_opening']['pressure_damage_rate_per_s'] = parameters['shell_rate_per_s']
    if frozen_signature(s) != frozen_signature(scene) or geometry.thermal_signature(s) != geometry.thermal_signature(scene):
        raise ValueError('A fixed property, energy reserve, geometry or thermal law changed')
    return s


def observations(experiment):
    path = PRIOR/('fit.json' if experiment == 'no-cooling' else 'validation.json')
    prior = read(path)
    return dict(result={'time_s': prior['result']['time_s']},
                observed_temperature_k=prior['observed_temperature_k'],
                observed_pressure_bar=prior['observed_pressure_bar'], meta=prior['meta'])


def fingerprint(binary):
    files = [Path(__file__), Path(score.__file__), binary]
    files += sorted((ROOT/'native').rglob('*.hpp')) + sorted((ROOT/'native').rglob('*.cuh'))
    files += sorted((ROOT/'native').rglob('*.cpp')) + sorted((ROOT/'native').rglob('*.cu'))
    files += [PRIOR/'profile.json', PRIOR/'fit-scenario.json', PRIOR/'validation-scenario.json']
    h = hashlib.sha256()
    for path in files:
        h.update(path.read_bytes())
    return h.hexdigest()


def replay(binary, scene, directory, label, timeout, refinement=1.):
    scene_path, run_path = directory/(label+'-scenario.json'), directory/(label+'-run.json')
    save(scene_path, scene)
    request = dict(scenario_path=str(scene_path.resolve()), refinement=refinement,
                   checkpoint_path=str((directory/(label+'-partial.json')).resolve()))
    env = dict(os.environ, BATTERY_SIM_PROGRESS='1')
    with run_path.open('w') as output, (directory/(label+'-progress.log')).open('w') as progress:
        child = subprocess.Popen([str(binary.resolve()), str(scene_path.resolve())], stdin=subprocess.PIPE,
                                 stdout=output, stderr=progress, text=True, env=env)
        try:
            child.communicate(json.dumps(request)+'\n', timeout=timeout)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
            raise
        if child.returncode:
            raise RuntimeError('Native replay exited: '+str(child.returncode))
    run = read(run_path)
    if 'error' in run:
        raise RuntimeError(run['error'])
    native.audit(run, scene)
    if run.get('max_oxidizer_residual_kg', 0.) > 1e-10:
        raise ValueError('Oxygen ledger failed')
    return run


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('mode', choices=['fit', 'validate'])
    cli.add_argument('--output', type=Path, default=OUT)
    cli.add_argument('--probe', type=Path, default=Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'))
    cli.add_argument('--timeout', type=float, default=600.)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(args.probe)
    experiment = 'no-cooling' if args.mode == 'fit' else 'active-cooling'
    scene = read(PRIOR/('fit-scenario.json' if args.mode == 'fit' else 'validation-scenario.json'))
    # v16's final training artifact omitted its replay backend. Never fall back
    # to the legacy explicit gas integrator when loading that frozen geometry.
    scene['gas_network'].update(mass_backend='cuda_network', network_pressure_fraction=1.)
    prior = observations(experiment)
    if args.mode == 'validate':
        profile = read(args.output/'profile.json')
        if profile['fingerprint'] != fp or profile['validation_used_for_selection']:
            raise ValueError('Frozen model changed or validation used for selection')
        if (args.output/'validation.json').exists():
            raise ValueError('Do not overwrite held-out replay')
        candidate = bake(scene, profile['parameters'])
        run = replay(args.probe, candidate, args.output, 'validation', args.timeout)
        result = dict(parameters=profile['parameters'], metrics=score.metrics(run, prior), result=run,
                      observed_temperature_k=prior['observed_temperature_k'], observed_pressure_bar=prior['observed_pressure_bar'],
                      meta=prior['meta'], frozen_profile_sha256=geometry.digest(profile),
                      validation_note='Previously inspected experiment, excluded from parameter selection; not blind validation')
        save(args.output/'validation.json', result)
        print('VALIDATION', json.dumps(result['metrics']), flush=True)
        return

    if (args.output/'profile.json').exists():
        raise ValueError('Profile is frozen; use a new directory')
    initial = seed(scene)
    manifest = dict(fingerprint=fp, calibration='no-cooling', validation_used_for_selection=False,
                    backend='cuda_network',
                    frozen_signature=frozen_signature(scene), thermal_sha256=geometry.thermal_signature(scene),
                    fixed='Energy reserve, gas yield/fraction, all geometry, materials, capacities, thermal laws, sensor mapping, cooling, initial fields, seam geometry and fracture work',
                    parameters=list(NAMES), bounds=BOUNDS, initial=initial, cases=CASES,
                    temperature_windows=score.WINDOWS, pressure_weights={'0-30': .25, '30-1800': .05},
                    early_0_3='Diagnostic only; direct thermocouple jet hits are unresolved',
                    thermal_regression='Both frozen warmups reproduced bit for bit before this search')
    if (args.output/'manifest.json').exists() and geometry.digest(read(args.output/'manifest.json')) != geometry.digest(manifest):
        raise ValueError('Existing search belongs to different code or inputs')
    save(args.output/'manifest.json', manifest)
    history, residuals = [], {}
    started = time.monotonic()

    def evaluate(label, parameters):
        cache = args.output/(label+'.json')
        save(args.output/'status.json', dict(stage='search', candidate=label, parameters=parameters))
        print('START', label, json.dumps(parameters), flush=True)
        if cache.exists():
            record = read(cache)
            if record['parameters'] != parameters:
                raise ValueError('Cached parameters differ')
            if 'failed' in record:
                history.append(record);return None
            run = read(args.output/(label+'-run.json'))
        else:
            candidate = bake(scene, parameters)
            try:
                run = replay(args.probe, candidate, args.output, label, args.timeout)
            except subprocess.TimeoutExpired:
                record = dict(id=label, parameters=parameters, failed='wall timeout; partial replay excluded from objective')
                save(cache, record);history.append(record);save(args.output/'history.json', history)
                print('TIMEOUT', label, flush=True);return None
            record = dict(id=label, parameters=parameters, metrics=score.metrics(run, prior),
                          steps=run['steps'], wall_s=run['wall_s'])
            save(cache, record)
        r = score.residual(run, prior)
        residuals[label] = r
        history.append(record);save(args.output/'history.json', history)
        m = record['metrics']
        print('DONE', label, 'objective', round(m['objective'], 5),
              'RMSE_30_300', round(m['phases']['30-300']['rmse_k'], 4),
              'RMSE_300_1800', round(m['phases']['300-1800']['rmse_k'], 4),
              'peak_bar', round(m['pressure_peak_bar'], 5), 'wall_s', round(run['wall_s'], 2),
              'elapsed_s', round(time.monotonic()-started), flush=True)
        return r

    base = evaluate('baseline', initial)
    if base is None:
        raise RuntimeError('Baseline exceeded budget; no calibration can be frozen')
    for label, changes in CASES[1:]:
        parameters = dict(initial)
        for key, factor in changes.items():
            parameters[key] *= factor
        evaluate(label, parameters)
    # One bounded multivariate proposal from broad log-space secants. It is
    # accepted only by an actual complete CUDA replay, never by the surrogate.
    columns, active, lower, upper = [], [], [], []
    for key in NAMES:
        scans = [row for row in history if row['id'] != 'baseline' and 'failed' not in row
                 and row['parameters'][key] != initial[key]]
        if not scans:
            continue
        dx = [math.log(row['parameters'][key]/initial[key]) for row in scans]
        columns.append(np.mean([(residuals[row['id']]-base)/x for row, x in zip(scans, dx)], axis=0))
        active.append(key);lower.append(min(0., *dx));upper.append(max(0., *dx))
    if columns:
        jac = np.column_stack(columns)
        fit = least_squares(lambda x: base+jac@x, np.zeros(len(active)), bounds=(lower, upper))
        proposal = dict(initial)
        for key, value in zip(active, fit.x):
            proposal[key] = float(np.clip(initial[key]*math.exp(value), *BOUNDS[key]))
        evaluate('joint-proposal', proposal)
    best = min((row for row in history if 'failed' not in row), key=lambda row: row['metrics']['objective'])
    candidate = bake(scene, best['parameters'])
    save(args.output/'status.json', dict(stage='refinement', selected=best['id'], parameters=best['parameters']))
    print('REFINE', best['id'], flush=True)
    coarse = read(args.output/(best['id']+'-run.json'))
    fine = replay(args.probe, candidate, args.output, 'refined', 2*args.timeout, .5)
    difference = dict(max_witness_change_k=max(float(np.max(np.abs(np.asarray(coarse['temperature_k'][key])-fine['temperature_k'][key]))) for key in score.WITNESSES),
                      max_pressure_change_bar=float(np.max(np.abs(np.asarray(coarse['gas']['chamber']['pressure_pa'])-fine['gas']['chamber']['pressure_pa']))/1e5))
    save(args.output/'numerics.json', difference)
    if difference['max_witness_change_k'] > .5 or difference['max_pressure_change_bar'] > .05:
        raise ValueError('Refinement failed; do not freeze a step-dependent fit: '+str(difference))
    profile = dict(parameters=best['parameters'], selected=best['id'], fingerprint=fp,
                   validation_used_for_selection=False, thermal_sha256=manifest['thermal_sha256'],
                   frozen_signature=manifest['frozen_signature'], calibration_accepted=False,
                   initial_objective=history[0]['metrics']['objective'], best_objective=best['metrics']['objective'],
                   status='Frozen conditional rupture/combustion fit; physical validation pending')
    save(args.output/'profile.json', profile)
    save(args.output/'fit-scenario.json', candidate)
    save(args.output/'fit.json', dict(parameters=best['parameters'], metrics=best['metrics'], result=coarse,
                                    observed_temperature_k=prior['observed_temperature_k'],
                                    observed_pressure_bar=prior['observed_pressure_bar'], meta=prior['meta']))
    save(args.output/'status.json', dict(stage='frozen', selected=best['id']))
    print('FROZEN', json.dumps(profile), 'NUMERICS', json.dumps(difference), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Geometric, pressure-arrested rupture calibration. No fitted aperture ceiling.

Only no-cooling selects coefficients. All evolution is in the native CUDA core;
Python builds immutable candidate scenes, audits them and fits observations.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
import calibrate_upv_aperture as previous
from calibrate_upv import save

shell = previous.shell
gas = previous.gas
native = gas.t
ROOT = previous.ROOT
PRIOR = previous.OUT
OUT = ROOT/'data/calibration/upv-geometric-rupture-v11'
NAMES = gas.NAMES + ('surface_h', 'strength_pa', 'damage_rate_per_s')
LIMITS = dict(zip(gas.NAMES, np.exp(gas.BOUNDS).T.tolist()),
              surface_h=(10., 1e7), strength_pa=(1e4, 5e6), damage_rate_per_s=(.05, 100.))
FIXED_WORK = 1000.
PROTOCOL = dict(
    revision='geometric-pressure-arrested-v11', calibration='no-cooling', validation='active-cooling',
    objective=gas.PROTOCOL['objective'],
    frozen='All measured solid capacities/reference conductivities, geometry, intact and residue connectivity, sensor bindings, pre-event fields, chamber/free volume and gas constants',
    aperture='Physical panel area times local irreversible opening; no independent area ceiling',
    fitted_candidates=list(NAMES),
    priors='Opening work 1000 J/m2, seam geometry/strength ratio and opening exponent remain unmeasured fixed priors',
    bounds='Exploratory optimization limits, not independently measured material-property intervals',
    numerics='Gas temperature budget 50 K (additional native 5% relative bound remains), gas mass budget .1; refinement required, never fitted',
    scope='Conditional primary event; late witness temperatures and chamber pressure. Burst thermocouples and internal pressure are not fit targets',
    validation_note='Previously inspected experiment excluded from parameter selection; not blind validation')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(probe):
    paths = [Path(__file__), PRIOR/'profile.json', previous.SHELL_PRIOR, shell.HEAT/'profile.json']
    return hashlib.sha256((previous.fingerprint(probe)+''.join(digest(p) for p in paths)).encode()).hexdigest()


def inputs(name):
    mode = 'fit' if name == 'no-cooling' else 'validate'
    scene = json.loads((PRIOR/(name+'-'+mode+'-scenario.json')).read_text())
    report = json.loads((PRIOR/(mode+'.json')).read_text())
    if not np.allclose(scene['failure_sample_times_s'], report['result']['time_s'], rtol=0, atol=1e-8):
        raise ValueError('Observation alignment changed')
    return scene, report


def initial_parameters():
    old = json.loads((PRIOR/'profile.json').read_text())['parameters']
    opening = json.loads(previous.SHELL_PRIOR.read_text())['parameters']
    # Old fitted h was normalized by an artificial area; restart this coefficient
    # at the original exploratory prior rather than treating it as a material law.
    return {**{k: old[k] for k in gas.NAMES}, 'surface_h': 1e4,
            **{k: opening[k] for k in ('strength_pa', 'damage_rate_per_s')}}


def bake(scene, parameters):
    if set(parameters) != set(NAMES):
        raise ValueError('Only declared post-event parameters may change; no area/cp/k knob')
    for k, value in parameters.items():
        if not math.isfinite(value) or not LIMITS[k][0] <= value <= LIMITS[k][1]:
            raise ValueError('Out-of-bounds parameter: '+k)
    old = json.loads((PRIOR/'profile.json').read_text())['parameters']
    p = {k: parameters[k] for k in gas.NAMES}
    p['exposure_scale'] /= old['exposure_scale']
    s = gas.bake(scene, p)
    for channel in s['gas_network']['channels']:
        if 'damage_opening' not in channel:
            raise ValueError('Expected whole-shell progressive channels')
        channel.pop('area_m2', None)
        channel['opening_pressure_pa'] = parameters['strength_pa']
        channel['damage_opening']['pressure_damage_rate_per_s'] = parameters['damage_rate_per_s']
        channel['damage_opening']['fracture_energy_j_per_m2'] = FIXED_WORK
        channel['surface_link']['h_w_per_m2_k'] = parameters['surface_h']
    s['gas_network'].update(max_fraction_per_step=.1, max_temperature_step_k=50.)
    for before, after in zip(scene['cells'], s['cells']):
        for key in ('thermal', 'geometry', 'pose', 'degradation'):
            if before.get(key) != after.get(key):
                raise ValueError('Frozen material changed: '+before['id']+'/'+key)
    for key in ('shells', 'interfaces', 'ports', 'boundaries', 'measurement_probes'):
        if scene[key] != s[key]:
            raise ValueError('Frozen geometry/contact changed: '+key)
    s['metadata']['geometric_rupture_protocol'] = PROTOCOL
    s['metadata']['calibration_status'] = 'Exploratory geometric-rupture candidate, not an accepted calibration'
    return s


def verify_heat(runner, output, fp):
    previous.verify_heat(runner, output, fp)


def audit(run, scene):
    native.audit(run, scene)
    if run.get('max_oxidizer_residual_kg', 0.) > 1e-10:
        raise ValueError('Oxidizer balance failed')
    if not np.allclose(run['time_s'], scene['failure_sample_times_s'], rtol=0, atol=1e-8):
        raise ValueError('Incomplete or misaligned replay')


def run_candidate(runner, scene, p, refinement=1.):
    s = bake(scene, p)
    s['simulation']['stability_factor'] *= refinement
    result = runner.run(s, refinement)
    audit(result, s)
    return result


def report(name, p, run, prior):
    return dict(experiment=name, parameters=p, protocol=PROTOCOL,
                metrics=shell.metrics(run, prior), energy_ledger=previous.ledger(run),
                observed_temperature_k=prior['observed_temperature_k'],
                observed_pressure_bar=prior['observed_pressure_bar'], meta=prior['meta'], result=run)


def baseline(runner, output, fp):
    scene, prior = inputs('no-cooling'); p = initial_parameters()
    run = run_candidate(runner, scene, p)
    save(output/'baseline.json', report('no-cooling', p, run, prior))
    save(output/'baseline-scenario.json', bake(scene, p))
    print('baseline', json.dumps(dict(metrics=shell.metrics(run, prior), wall_s=run['wall_s'], steps=run['steps'])), flush=True)
    fine = run_candidate(runner, scene, p, .5)
    change = previous.changes(run, fine)
    save(output/'baseline-refined.json', fine)
    proof = dict(fingerprint=fp, changes=change, coarse_steps=run['steps'], fine_steps=fine['steps'])
    save(output/'numerics.json', proof)
    print('numerical refinement', json.dumps(proof), flush=True)
    if change['max_witness_change_k'] > .25 or change['max_pressure_change_bar'] > .025:
        raise ValueError('Search numerics too coarse: do not calibrate discretization error')


def fit(runner, output, fp, max_evaluations, names):
    if (output/'profile.json').exists():
        raise ValueError('Frozen profile exists; choose a new output directory')
    proof = json.loads((output/'numerics.json').read_text())
    if proof['fingerprint'] != fp:
        raise ValueError('Repeat numerical verification after changes')
    scene, prior = inputs('no-cooling'); initial = initial_parameters()
    names = tuple(names)
    if len(set(names)) != len(names) or not set(names) <= set(NAMES):
        raise ValueError('Invalid fitted parameter selection')
    lo, hi = np.log(np.array([LIMITS[k] for k in names]).T)
    history, cache = [], {}
    best = [math.inf, None, None]
    start = time.monotonic()

    def residual(x):
        key = tuple(x)
        if key in cache:
            return cache[key]
        p = dict(initial)
        for k, value in zip(names, np.exp(x)):
            p[k] = float(np.clip(value, *LIMITS[k]))
        run = run_candidate(runner, scene, p)
        vector = shell.score(run, prior); metrics = shell.metrics(run, prior)
        entry = dict(candidate=len(history)+1, parameters=p, cost=float(vector@vector),
                     metrics=metrics, energy_ledger=previous.ledger(run), wall_s=run['wall_s'], steps=run['steps'])
        history.append(entry); cache[key] = vector
        if entry['cost'] < best[0]:
            best[:] = [entry['cost'], p, run]
            save(output/'checkpoint.json', entry); save(output/'best-run.json', run)
        save(output/'history.json', history)
        print(f"candidate {len(history)} cost={entry['cost']:.6f} best={best[0]:.6f} T={metrics['witness']['rmse_k']:.4f} K peak={metrics['predicted_peak_bar']:.4f} bar native={run['wall_s']:.1f}s elapsed={time.monotonic()-start:.0f}s", flush=True)
        return vector

    scans = []
    def jacobian(x):
        base = residual(x); columns = []
        for i, name in enumerate(names):
            dx = .05 if x[i]+.05 <= hi[i] else -.05
            shifted = x.copy(); shifted[i] += dx
            column = (residual(shifted)-base)/dx
            columns.append(column)
        matrix = np.column_stack(columns)
        scans.append(dict(at=dict(zip(names, np.exp(x).tolist())),
                          column_norms=dict(zip(names, np.linalg.norm(matrix, axis=0).tolist())),
                          singular_values=np.linalg.svd(matrix, compute_uv=False).tolist()))
        save(output/'sensitivity.json', scans)
        return matrix

    result = least_squares(residual, np.log([initial[k] for k in names]), bounds=(lo, hi),
                           jac=jacobian, x_scale='jac', max_nfev=max_evaluations,
                           ftol=5e-4, xtol=1e-4, gtol=1e-4)
    p, run = best[1:]
    profile = dict(parameters=p, fitted=list(names),
                   fixed_priors=dict(fracture_work_j_m2=FIXED_WORK, **{k:v for k,v in initial.items() if k not in names}),
                   fingerprint=fp, protocol=PROTOCOL, validation_used_for_selection=False,
                   search=dict(success=bool(result.success), message=result.message, nfev=result.nfev,
                               candidates=len(history), initial_cost=history[0]['cost'], best_cost=best[0],
                               jacobian_singular_values=np.linalg.svd(result.jac, compute_uv=False).tolist()),
                   input_sha256={str(path.relative_to(ROOT)): digest(path) for path in
                                 (PRIOR/'profile.json', PRIOR/'fit.json', PRIOR/'no-cooling-fit-scenario.json', shell.HEAT/'profile.json')})
    save(output/'profile.json', profile)
    save(output/'fit.json', report('no-cooling', p, run, prior))
    save(output/'no-cooling-fit-scenario.json', bake(scene, p))
    print('frozen training fit', json.dumps(profile), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['verify-heat', 'baseline', 'fit', 'validate', 'check'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--probe', type=Path, default=previous.PROBE)
    parser.add_argument('--max-evaluations', type=int, default=8)
    parser.add_argument('--fit-parameters', nargs='+', default=list(NAMES))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(args.probe); runner = native.Runner(args.probe, args.output)
    try:
        if args.mode == 'verify-heat':
            return verify_heat(runner, args.output, fp)
        if json.loads((args.output/'heat-replay.json').read_text())['fingerprint'] != fp:
            raise ValueError('Repeat intact heat replay after model changes')
        if args.mode == 'baseline':
            return baseline(runner, args.output, fp)
        if args.mode == 'fit':
            return fit(runner, args.output, fp, args.max_evaluations, args.fit_parameters)
        profile = json.loads((args.output/'profile.json').read_text())
        if profile['fingerprint'] != fp:
            raise ValueError('Frozen calibration model changed')
        p = profile['parameters']
        if args.mode == 'validate':
            scene, prior = inputs('active-cooling')
            run = run_candidate(runner, scene, p)
            save(args.output/'validate.json', report('active-cooling', p, run, prior))
            save(args.output/'active-cooling-validate-scenario.json', bake(scene, p))
            print('held-out replay', json.dumps(shell.metrics(run, prior)), flush=True)
            return
        results = {}
        for mode, name in [('fit', 'no-cooling'), ('validate', 'active-cooling')]:
            saved = json.loads((args.output/(mode+'.json')).read_text()); scene, prior = inputs(name)
            fine = run_candidate(runner, scene, p, .5)
            changes = previous.changes(saved['result'], fine)
            results[name] = dict(changes=changes, metrics=shell.metrics(fine, prior), energy_ledger=previous.ledger(fine))
            save(args.output/(name+'-refined.json'), fine)
            save(args.output/'refinement.json', results)
            print('refinement', name, json.dumps(results[name]), flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()

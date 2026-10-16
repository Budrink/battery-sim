#!/usr/bin/env python3
"""Training-only aperture calibration; frozen heating, held-out cooling replay.

The two new coefficients are first screened for observable sensitivity. An
unidentifiable fracture-work coefficient is retained as a declared prior, not
reported as a measured material property. All evolution stays in native CUDA.
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
import calibrate_upv_shell_opening as shell
from calibrate_upv import save

gas = shell.gas
ROOT = shell.ROOT
OUT = ROOT / 'data/calibration/upv-aperture-v10'
ENERGY_PRIOR = ROOT / 'data/calibration/upv-progressive-energy-v9/profile.json'
SHELL_PRIOR = ROOT / 'data/calibration/upv-progressive-shell-v9/profile.json'
PROBE = Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe')
APERTURE_NAMES = ('surface_h', 'fracture_work_j_m2')
APERTURE_INITIAL = dict(surface_h=1e4, fracture_work_j_m2=1e3)
APERTURE_LIMITS = dict(surface_h=(10., 1e7), fracture_work_j_m2=(1., 1e6))
PROTOCOL = dict(
    revision='aperture-directed-energy-paid-v10', calibration='no-cooling', validation='active-cooling',
    objective=gas.PROTOCOL['objective'],
    frozen='All solid geometry, capacity, reference conductivity, intact/residue connections, sensor bindings, initial fields, chamber volume and gas species constants',
    geometry='First visible solid along each outward panel normal; partial occlusion; six openings share the previous strength/rate/area profile',
    energy='Surface heat debited from arriving gas enthalpy; opening work debited from internal gas and stored in its own ledger',
    aperture_priors='h=1e4 W/m2/K, work=1e3 J/m2 are exploratory priors; search bounds are NOT independently established material-property intervals',
    identifiability='Sensitivity on training record before selection; flat coefficients are frozen and explicitly marked unidentifiable',
    validation_note='Previously inspected experiment, excluded from parameter selection; not blind or independent-laboratory validation',
    measurement_note='Same historical unconfirmed witness numbering; burst thermocouples diagnostic only, no channel offsets or fitted time shifts')


def fingerprint(probe):
    paths = [Path(__file__), Path(shell.__file__), Path(gas.__file__), ENERGY_PRIOR, SHELL_PRIOR,
             shell.PRIOR/'profile.json', shell.HEAT/'profile.json']
    return hashlib.sha256((gas.t.fingerprint(probe)+''.join(shell.digest(p) for p in paths)).encode()).hexdigest()


def initial_parameters():
    return dict(json.loads(ENERGY_PRIOR.read_text())['parameters'], **APERTURE_INITIAL)


def bake(scene, parameters, enabled=True):
    """Only explicit post-event energy/contact/channel parameters may change."""
    if set(parameters) != set(gas.NAMES) | set(APERTURE_NAMES):
        raise ValueError('Unexpected calibration parameters')
    for k, (lo, hi) in APERTURE_LIMITS.items():
        if not math.isfinite(parameters[k]) or not lo <= parameters[k] <= hi:
            raise ValueError('Aperture coefficient outside exploratory bounds: '+k)
    for k, lo, hi in zip(gas.NAMES, *np.exp(gas.BOUNDS)):
        if not math.isfinite(parameters[k]) or not lo <= parameters[k] <= hi:
            raise ValueError('Energy coefficient outside bounds: '+k)
    s = shell.bake(scene, json.loads(SHELL_PRIOR.read_text())['parameters'])
    old = json.loads((shell.PRIOR/'profile.json').read_text())['parameters']
    p = {k: parameters[k] for k in gas.NAMES}
    p['exposure_scale'] /= old['exposure_scale']  # input already contains v7 exposure
    s = gas.bake(s, p)
    if enabled:
        for channel in s['gas_network']['channels']:
            channel['surface_link'] = dict(h_w_per_m2_k=parameters['surface_h'])
            channel['damage_opening']['fracture_energy_j_per_m2'] = parameters['fracture_work_j_m2']
    for before, after in zip(scene['cells'], s['cells']):
        for key in ('thermal', 'geometry', 'pose', 'degradation'):
            if before.get(key) != after.get(key):
                raise ValueError('Frozen material/state changed: '+before['id']+'/'+key)
    for key in ('shells', 'interfaces', 'ports', 'boundaries', 'measurement_probes'):
        if scene[key] != s[key]:
            raise ValueError('Frozen geometry/contact/forcing changed: '+key)
    s['metadata']['aperture_calibration_protocol'] = PROTOCOL
    s['metadata']['calibration_status'] = 'Exploratory aperture calibration; no accepted profile is overwritten'
    return s


def ledger(run):
    channels = run['channels'].values()
    return dict(surface_heat_j=sum(c['surface_energy_j'][-1] for c in channels),
                fracture_work_j=sum(c['fracture_work_j'][-1] for c in run['channels'].values()),
                per_channel={k: dict(surface_heat_j=v['surface_energy_j'][-1],
                                    fracture_work_j=v['fracture_work_j'][-1]) for k, v in run['channels'].items()})


def changes(a, b):
    return dict(max_witness_change_k=max(float(np.max(np.abs(np.asarray(a['temperature_k'][k])-b['temperature_k'][k])))
                                        for k in gas.t.late.WITNESSES),
                max_pressure_change_bar=float(np.max(np.abs(np.asarray(a['gas']['chamber']['pressure_pa'])-b['gas']['chamber']['pressure_pa']))/1e5))


def report(name, parameters, run, prior):
    return dict(experiment=name, parameters=parameters, protocol=PROTOCOL,
                metrics=shell.metrics(run, prior), energy_ledger=ledger(run),
                observed_temperature_k=prior['observed_temperature_k'],
                observed_pressure_bar=prior['observed_pressure_bar'], meta=prior['meta'], result=run)


def verify_heat(runner, output, fp):
    results = {}
    for mode, name in [('fit', 'no-cooling'), ('validate', 'active-cooling')]:
        s = json.loads((shell.HEAT/(name+'-fitted-scenario.json')).read_text())
        old = json.loads((shell.HEAT/(mode+'.json')).read_text())['result']
        run = runner.run(s)
        delta = max(float(np.max(np.abs(np.asarray(run['temperature_k'][k])-v))) for k, v in old['temperature_k'].items())
        if delta > 1e-10 or run['applied_objects'] != old['applied_objects']:
            raise ValueError('Pre-event thermal calibration changed')
        results[name] = dict(max_change_k=delta, steps=run['steps'])
        print('intact heat', name, results[name], flush=True)
    save(output/'heat-replay.json', dict(fingerprint=fp, results=results))


def scan(runner, output, fp):
    if (output/'scan.json').exists():
        raise ValueError('Sensitivity scan already exists; use another output directory')
    scene, prior = shell.inputs('no-cooling')
    initial = initial_parameters()
    trials = [('legacy-v9', initial, False), ('initial', initial, True)]
    trials += [('h-'+str(h), dict(initial, surface_h=h), True) for h in (100., 1e6, 1e7)]
    trials += [('work-'+str(w), dict(initial, fracture_work_j_m2=w), True) for w in (1., 1e6)]
    results = []
    reference = None
    for label, p, enabled in trials:
        run = runner.run(bake(scene, p, enabled))
        entry = dict(label=label, parameters=p, enabled=enabled, metrics=shell.metrics(run, prior),
                     ledger=ledger(run), wall_s=run['wall_s'], steps=run['steps'])
        if label == 'initial':
            reference = run
        if reference is not None:
            entry['change_from_initial'] = changes(run, reference)
        results.append(entry)
        save(output/(label+'-run.json'), run)
        save(output/'scan-checkpoint.json', results)
        print('scan', json.dumps(entry), flush=True)
    save(output/'scan.json', dict(fingerprint=fp, protocol=PROTOCOL, trials=results))


def fit(runner, output, fp, max_evaluations, names):
    if (output/'profile.json').exists():
        raise ValueError('Frozen profile already exists')
    screen = json.loads((output/'scan.json').read_text())
    if screen['fingerprint'] != fp:
        raise ValueError('Repeat sensitivity scan after code/model changes')
    scene, prior = shell.inputs('no-cooling')
    initial = initial_parameters()
    names = tuple(names)
    if len(set(names)) != len(names) or not set(names) <= set(initial):
        raise ValueError('Invalid parameter selection')
    trials = [x for x in screen['trials'] if x['enabled']]
    best_scan = min(trials, key=lambda x:x['metrics']['objective_sum_squares'])
    for k in names:
        initial[k] = best_scan['parameters'][k]
    bounds = dict(zip(gas.NAMES, np.exp(gas.BOUNDS).T.tolist()), **APERTURE_LIMITS)
    lo, hi = np.log(np.array([bounds[k] for k in names]).T)
    cache, history = {}, []
    best = [math.inf, None, None]
    started = time.monotonic()

    def residual(x):
        key = tuple(x)
        if key in cache:
            return cache[key]
        p = dict(initial)
        for k, value in zip(names, np.exp(x)):
            p[k] = float(np.clip(value, *bounds[k]))
        run = runner.run(bake(scene, p))
        vector = shell.score(run, prior)
        metrics = shell.metrics(run, prior)
        entry = dict(candidate=len(history)+1, cost=float(vector@vector), parameters=p,
                     metrics=metrics, ledger=ledger(run), wall_s=run['wall_s'])
        history.append(entry)
        if entry['cost'] < best[0]:
            best[:] = [entry['cost'], p, run]
            save(output/'checkpoint.json', entry)
            save(output/'best-run.json', run)
        save(output/'history.json', history)
        cache[key] = vector
        print(f"candidate {len(history)} cost={entry['cost']:.6f} best={best[0]:.6f} T={metrics['witness']['rmse_k']:.4f} K P={metrics['predicted_peak_bar']:.4f} bar elapsed={time.monotonic()-started:.0f}s", flush=True)
        return vector

    def jacobian(x):
        base = residual(x)
        columns = []
        for i in range(len(x)):
            step = .03 if x[i]+.03 <= hi[i] else -.03
            shifted = x.copy(); shifted[i] += step
            columns.append((residual(shifted)-base)/step)
        return np.column_stack(columns)

    result = least_squares(residual, np.log([initial[k] for k in names]), bounds=(lo, hi),
                           jac=jacobian, x_scale='jac', max_nfev=max_evaluations,
                           ftol=5e-4, xtol=1e-4, gtol=1e-4)
    p, run = best[1:]
    save(output/'profile.json', dict(parameters=p, fitted=list(names),
         fixed_priors={k: v for k, v in initial.items() if k not in names},
         fingerprint=fp, protocol=PROTOCOL, validation_used_for_selection=False,
         search=dict(success=bool(result.success), message=result.message, nfev=result.nfev,
                     candidates=len(history), initial_cost=history[0]['cost'], best_cost=best[0],
                     jacobian_singular_values=np.linalg.svd(result.jac, compute_uv=False).tolist())))
    save(output/'fit.json', report('no-cooling', p, run, prior))
    save(output/'no-cooling-fit-scenario.json', bake(scene, p))
    print('frozen fit', json.dumps(shell.metrics(run, prior)), flush=True)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('mode', choices=['verify-heat', 'scan', 'fit', 'validate', 'check'])
    ap.add_argument('--output', type=Path, default=OUT)
    ap.add_argument('--probe', type=Path, default=PROBE)
    ap.add_argument('--max-evaluations', type=int, default=12)
    ap.add_argument('--fit-parameters', nargs='+', default=list(gas.NAMES)+['surface_h'])
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(a.probe); runner = gas.t.Runner(a.probe, a.output)
    try:
        if a.mode == 'verify-heat':
            return verify_heat(runner, a.output, fp)
        if json.loads((a.output/'heat-replay.json').read_text())['fingerprint'] != fp:
            raise ValueError('Replay intact heating after code changes')
        if a.mode == 'scan':
            return scan(runner, a.output, fp)
        if a.mode == 'fit':
            return fit(runner, a.output, fp, a.max_evaluations, a.fit_parameters)
        profile = json.loads((a.output/'profile.json').read_text())
        if profile['fingerprint'] != fp:
            raise ValueError('Frozen model changed')
        p = profile['parameters']
        if a.mode == 'validate':
            scene, prior = shell.inputs('active-cooling')
            run = runner.run(bake(scene, p))
            save(a.output/'validate.json', report('active-cooling', p, run, prior))
            save(a.output/'active-cooling-validate-scenario.json', bake(scene, p))
            print('held-out replay', json.dumps(shell.metrics(run, prior)), flush=True)
            return
        results = {}
        for mode, name in [('fit', 'no-cooling'), ('validate', 'active-cooling')]:
            saved = json.loads((a.output/(mode+'.json')).read_text())
            scene = json.loads((a.output/(name+'-'+mode+'-scenario.json')).read_text())
            scene['simulation']['stability_factor'] *= .5
            fine = runner.run(scene, .5)
            _, prior = shell.inputs(name)
            results[name] = dict(changes=changes(fine, saved['result']), metrics=shell.metrics(fine, prior), ledger=ledger(fine))
            save(a.output/(name+'-refined.json'), fine)
            print('refinement', name, json.dumps(results[name]), flush=True)
        save(a.output/'refinement.json', results)
    finally:
        runner.close()


if __name__ == '__main__':
    main()

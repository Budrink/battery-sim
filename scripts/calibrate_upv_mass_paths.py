#!/usr/bin/env python3
"""Destruction-only bounded search with immutable thermal inputs and held-out replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

import upv_mass_paths as geometry
import calibrate_upv_casing as native
from calibrate_upv import save

OUT = geometry.ROOT / 'data/calibration/upv-mass-paths-v15'
PROBE = Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe')
WITNESSES = ('101', '102', '107', '108')
WINDOWS = [(3., 30., 20.), (30., 300., 10.), (300., 1800., 5.)]
# Declared independently of validation observations. Large moves test distinct
# regimes before attempting one residual-based proposal, not endless local fits.
CASES = [('baseline', {}), ('lower-energy', {'energy_j': 280000.}),
         ('gas-rich', {'gas_fraction': .8}), ('solid-rich', {'gas_fraction': .3}),
         ('slow', {'solid_rate': 1.2}), ('fast', {'solid_rate': 10.}),
         ('narrow-gap', {'gap_m': .00005}), ('wide-gap', {'gap_m': .0008})]


def fingerprint(probe):
    files = [Path(__file__), Path(geometry.__file__), geometry.PRIOR/'profile.json', geometry.PRIOR/'heat/profile.json', probe]
    files += sorted((geometry.ROOT/'native').rglob('*.cuh')) + sorted((geometry.ROOT/'native').rglob('*.cpp'))
    files += sorted((geometry.ROOT/'native').rglob('*.cu')) + sorted((geometry.ROOT/'native').rglob('*.hpp'))
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


def statistics(error):
    return dict(rmse_k=float(np.sqrt(np.mean(error**2))), bias_k=float(np.mean(error)),
                max_abs_k=float(np.max(np.abs(error))), count=len(error))


def residual(run, prior):
    at = np.asarray(run['time_s']);blocks = []
    pressure = np.interp(at, prior['result']['time_s'], prior['observed_pressure_bar'])
    dp = np.asarray(run['gas']['chamber']['pressure_pa'])/1e5-pressure
    for lo, hi, scale in [(0,30,.25), (30,1800,.05)]:
        mask = (at>lo)&(at<=hi)&np.isfinite(dp)
        if mask.any():
            blocks.append(dp[mask]/(scale*np.sqrt(mask.sum())))
    for lo, hi, scale in WINDOWS:
        mask_time = (at>lo)&(at<=hi)
        for channel in WITNESSES:
            observed = np.interp(at, prior['result']['time_s'], np.asarray(prior['observed_temperature_k'][channel], dtype=float))
            error = np.asarray(run['temperature_k'][channel])-observed
            mask = mask_time&np.isfinite(error)
            if mask.any():
                blocks.append(error[mask]/(scale*np.sqrt(mask.sum())))
    return np.concatenate(blocks)


def metrics(run, prior):
    at = np.asarray(run['time_s']);result = dict(phases={}, objective=float(residual(run, prior)@residual(run, prior)))
    for lo, hi in [(0,3), (3,30), (30,300), (300,1800)]:
        rows = {};errors = []
        for channel in WITNESSES:
            y = np.interp(at, prior['result']['time_s'], np.asarray(prior['observed_temperature_k'][channel], dtype=float))
            error = np.asarray(run['temperature_k'][channel])-y
            selected = error[(at>lo)&(at<=hi)&np.isfinite(error)]
            if len(selected):
                rows[channel] = statistics(selected);errors.append(selected)
        if errors:
            result['phases'][f'{lo}-{hi}'] = dict(**statistics(np.concatenate(errors)), per_channel=rows)
    pressure = np.asarray(run['gas']['chamber']['pressure_pa'])/1e5
    observed = np.interp(at, prior['result']['time_s'], prior['observed_pressure_bar'])
    result.update(pressure_peak_bar=float(pressure.max()), observed_peak_bar=float(observed.max()),
                  pressure_peak_time_s=float(at[pressure.argmax()]))
    for lo, hi in [(0,30), (30,1800)]:
        mask = (at>lo)&(at<=hi)&np.isfinite(observed)
        if mask.any():
            result[f'pressure_{lo}_{hi}_rmse_bar'] = float(np.sqrt(np.mean((pressure[mask]-observed[mask])**2)))
    result['gas_released_by_region_j'] = {k: v['released_energy_j'][-1] for k, v in run['gas'].items()}
    result['max_local_pressure_bar'] = max(max(v['pressure_pa']) for v in run['gas'].values())/1e5
    return result


def replay(runner, scene, p, prior, duration=None, refinement=1., backend='cuda_global'):
    candidate = geometry.attach(scene, p)
    candidate['gas_network']['mass_backend'] = backend
    if backend == 'cuda_network':
        candidate['gas_network']['network_pressure_fraction'] = 1.0
    if duration:
        times = [t for t in candidate['failure_sample_times_s'] if t <= duration]
        candidate['failure_sample_times_s'] = times;candidate['simulation']['duration_s'] = times[-1]
    candidate['simulation']['stability_factor'] *= refinement
    run = runner.run(candidate, refinement)
    if run.get('max_oxidizer_residual_kg', 0) > 1e-10:
        raise ValueError('Oxidizer conservation failed')
    return dict(parameters=p, metrics=metrics(run, prior), meta=prior['meta'],
                observed_temperature_k=prior['observed_temperature_k'], observed_pressure_bar=prior['observed_pressure_bar'],
                result=run, topology=candidate['metadata']['mass_paths']), candidate


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument('mode', choices=['trial', 'fit', 'validate'])
    cli.add_argument('--output', type=Path, default=OUT);cli.add_argument('--probe', type=Path, default=PROBE)
    cli.add_argument('--duration', type=float)
    cli.add_argument('--backend', choices=['cuda_global', 'cuda_network'], default='cuda_global',
                     help='gas-mass transport integrator; thermal model remains frozen')
    args = cli.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(args.probe)
    runner = native.Runner(args.probe, args.output)
    try:
        if args.mode == 'validate':
            profile = geometry.read(args.output/'profile.json')
            if profile['fingerprint'] != fp or profile['validation_used_for_selection']:
                raise ValueError('Frozen model changed')
            if (args.output/'validation.json').exists():
                raise ValueError('Do not overwrite validation')
            scene, prior = geometry.inputs('active-cooling')
            report, candidate = replay(runner, scene, profile['parameters'], prior, backend=args.backend)
            report.update(experiment='active-cooling', cooling='Frozen v13 nominal G=10 W/K; not selected on validation',
                          old_metrics=metrics(prior['result'], prior), frozen_profile_sha256=geometry.digest(profile))
            save(args.output/'validation.json', report);save(args.output/'validation-scenario.json', candidate)
            print(json.dumps(report['metrics'], indent=2), flush=True);return
        scene, prior = geometry.inputs('no-cooling');seed = geometry.initial()
        if args.mode == 'trial':
            report, candidate = replay(runner, scene, seed, prior, args.duration, backend=args.backend)
            save(args.output/'trial.json', report);save(args.output/'trial-scenario.json', candidate)
            print(json.dumps(dict(metrics=report['metrics'], steps=report['result']['steps'], wall_s=report['result']['wall_s']), indent=2), flush=True);return
        if args.duration:
            raise ValueError('Calibration must cover the declared 1800 s window')
        if (args.output/'profile.json').exists():
            raise ValueError('Do not overwrite a frozen profile')
        manifest = dict(fingerprint=fp, backend=args.backend, calibration='no-cooling', validation_used_for_selection=False,
            thermal_sha256=geometry.thermal_signature(scene), heat_profile_sha256=geometry.digest(geometry.read(geometry.PRIOR/'heat/profile.json')),
            fixed_thermal='All solid properties, contacts, boundaries, sensors, pre-event fields; no thermal optimization',
            pressure_weights={'0-30_s': .25, '30-1800_s': .05}, temperature_windows=WINDOWS,
            burst_0_3_s='Diagnostic only: direct jet hits and multiplexed thermocouple response are unresolved',
            cases=CASES, parameters=geometry.PARAMETERS, bounds=geometry.BOUNDS,
            old_metrics_same_objective=metrics(prior['result'], prior))
        if (args.output/'manifest.json').exists() and geometry.digest(geometry.read(args.output/'manifest.json')) != geometry.digest(manifest):
            raise ValueError('Existing search belongs to another model')
        save(args.output/'manifest.json', manifest)
        runs = {};history = [];start = time.monotonic()

        def evaluate(label, p):
            path = args.output/(label+'.json')
            if path.exists():
                report = geometry.read(path)
                if report['parameters'] != p:
                    raise ValueError('Cached candidate differs')
            else:
                print('starting', label, p, flush=True)
                report, candidate = replay(runner, scene, p, prior, backend=args.backend)
                save(path, report);save(args.output/(label+'-scenario.json'), candidate)
            # Retain paths, not every full per-channel history (hundreds of MB
            # per candidate). Residual vectors and scalar history are small.
            runs[label] = path
            history.append(dict(id=label, parameters=p, metrics=report['metrics'], wall_s=report['result']['wall_s'], steps=report['result']['steps']))
            save(args.output/'history.json', history)
            print('completed', label, 'objective', report['metrics']['objective'], 'late_K', report['metrics']['phases']['300-1800']['rmse_k'],
                  'peak_bar', report['metrics']['pressure_peak_bar'], 'elapsed_s', round(time.monotonic()-start), flush=True)
            return residual(report['result'], prior)

        r0 = evaluate('baseline', seed);columns = {};left = {};right = {}
        for label, change in CASES[1:]:
            p = dict(seed, **change);value = evaluate(label, p);key = next(iter(change));dx = np.log(p[key]/seed[key])
            # Average separate secants when a parameter has both directions.
            columns.setdefault(key, []).append((value-r0)/dx)
            left[key] = min(left.get(key,0),dx);right[key] = max(right.get(key,0),dx)
        keys = list(geometry.PARAMETERS);jac = np.column_stack([np.mean(columns[k],axis=0) for k in keys])
        lo = np.array([left[k] for k in keys]);hi = np.array([right[k] for k in keys])
        fit = least_squares(lambda x:r0+jac@x, (lo+hi)/2, bounds=(lo,hi))
        proposal = {k:float(seed[k]*np.exp(fit.x[i])) for i,k in enumerate(keys)}
        evaluate('proposal', proposal)
        best = min(history, key=lambda h:h['metrics']['objective']);report = geometry.read(runs[best['id']])
        print('refining', best['id'], flush=True)
        fine, _ = replay(runner, scene, best['parameters'], prior, refinement=.5, backend=args.backend)
        changes = dict(max_witness_change_k=max(float(np.max(np.abs(np.asarray(report['result']['temperature_k'][k])-fine['result']['temperature_k'][k]))) for k in WITNESSES),
                       max_pressure_change_bar=float(np.max(np.abs(np.asarray(report['result']['gas']['chamber']['pressure_pa'])-fine['result']['gas']['chamber']['pressure_pa']))/1e5))
        save(args.output/'refined.json', fine);save(args.output/'numerics.json', changes)
        if changes['max_witness_change_k'] > .5 or changes['max_pressure_change_bar'] > .05:
            raise ValueError('Numerical refinement failed; do not freeze a discretization-dependent fit')
        profile = dict(parameters=best['parameters'], selected=best['id'], fingerprint=fp, validation_used_for_selection=False,
                       thermal_sha256=manifest['thermal_sha256'], heat_profile_sha256=manifest['heat_profile_sha256'],
                       calibration_accepted=False, status='Frozen conditional destruction fit; unmeasured gas geometry remains a hypothesis')
        save(args.output/'profile.json', profile);save(args.output/'fit.json', report)
        selected_scene=geometry.attach(scene, best['parameters'])
        selected_scene['gas_network']['mass_backend']=args.backend
        if args.backend=='cuda_network':selected_scene['gas_network']['network_pressure_fraction']=1.
        save(args.output/'fit-scenario.json', selected_scene)
        print('FROZEN', json.dumps(profile), flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()

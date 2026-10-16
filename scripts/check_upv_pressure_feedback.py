#!/usr/bin/env python3
"""Geometric rupture diagnostic, not a calibration or a held-out refit.

Remove only the obsolete aperture ceiling from the archived v10 training
scenario. Retain its remaining coefficients to expose the effect and its limits.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import calibrate_upv_casing as native
from calibrate_upv import save

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'data/calibration/upv-aperture-v10'


def prepare(scene, times):
    s = copy.deepcopy(scene)
    removed = {}
    for ch in s['gas_network']['channels']:
        if 'damage_opening' in ch:
            removed[ch['id']] = ch.pop('area_m2')
    if not removed:
        raise ValueError('No obsolete rupture ceilings to remove')
    s['failure_sample_times_s'] = times
    s['simulation']['duration_s'] = times[-1]
    s['metadata']['calibration_status'] = 'UNFITTED geometry-only pressure-feedback diagnostic; v10 coefficients are obsolete priors'
    s['metadata']['removed_aperture_ceilings_m2'] = removed
    return s


def metrics(r):
    areas = np.sum([c['effective_area_m2'] for c in r['channels'].values()], axis=0)
    return dict(internal_peak_bar=max(r['gas']['inside-C2']['pressure_pa'])/1e5,
                chamber_peak_bar=max(r['gas']['chamber']['pressure_pa'])/1e5,
                max_open_area_mm2=float(max(areas)*1e6),
                final_open_area_mm2=float(areas[-1]*1e6),
                fracture_work_j=sum(c['fracture_work_j'][-1] for c in r['channels'].values()),
                surface_heat_j=sum(c['surface_energy_j'][-1] for c in r['channels'].values()),
                final_damage={k: dict(bulk=c['max_bulk_damage'][-1], seam=c['max_seam_damage'][-1])
                              for k, c in r['channels'].items()},
                conservation={k: v for k, v in r.items() if 'residual' in k})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'data/diagnostics/upv-geometric-rupture')
    p.add_argument('--probe', type=Path, default=Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'))
    p.add_argument('--duration', type=float, default=30., help='Initial observation window, up to 30 s')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    source = PRIOR/'no-cooling-fit-scenario.json'
    historical = PRIOR/'no-cooling-dense.json'
    old = json.loads(historical.read_text())
    if not np.isfinite(a.duration) or a.duration<=0 or a.duration>old['time_s'][-1]:
        p.error('--duration must be positive and within the historical observation window')
    times = [t for t in old['time_s'] if t<=a.duration]
    if times[-1]!=a.duration:
        p.error('--duration must coincide with a historical sample time')
    for group in ('channels', 'gas'):
        for record in old[group].values():
            for key, values in record.items():
                record[key] = values[:len(times)]
    old['time_s'] = times
    s = prepare(json.loads(source.read_text()), old['time_s'])
    save(a.output/'scenario.json', s)
    runner = native.Runner(a.probe, a.output)
    try:
        coarse = runner.run(s); save(a.output/'coarse.json', coarse)
        print('Coarse geometry-only diagnostic: '+json.dumps(metrics(coarse)), flush=True)
        fine_scene = copy.deepcopy(s); fine_scene['simulation']['stability_factor'] *= .5
        fine = runner.run(fine_scene, .5); save(a.output/'refined.json', fine)
        result = dict(parameters_refitted=False, calibration_accepted=False, observation_duration_s=a.duration,
                      model_fingerprint=native.fingerprint(a.probe),
                      input_sha256={str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest()
                                    for x in (source, historical)},
                      historical=metrics(old), geometric=metrics(coarse), refined=metrics(fine),
                      max_pressure_refinement_bar=float(np.max(np.abs(np.asarray(coarse['gas']['chamber']['pressure_pa'])-
                                                                 fine['gas']['chamber']['pressure_pa']))/1e5),
                      max_surface_probe_refinement_k=max(float(np.max(np.abs(np.asarray(v)-fine['temperature_k'][k])))
                                                   for k, v in coarse['temperature_k'].items()),
                      note='Training-only initial transient. Only area normalization changed; old strength, rate and energy coefficients have NOT been recalibrated. Numerical CFL now uses active aperture conductance.')
        save(a.output/'assessment.json', result)
        print(json.dumps(result), flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()

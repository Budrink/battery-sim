#!/usr/bin/env python3
"""Compare matching leading-aperture replays differing only in the dt cap."""
import argparse
import json
from pathlib import Path
import numpy as np
from calibrate_upv import save


def compare(coarse, fine):
    def read(folder, name):
        return json.loads((folder/name).read_text())
    a_scene, b_scene = read(coarse, 'scenario.json'), read(fine, 'scenario.json')
    caps = [s['simulation'].pop('max_dt_s') for s in (a_scene, b_scene)]
    if a_scene != b_scene or not caps[1] < caps[0]:
        raise ValueError('Expect identical models and smaller second dt cap')
    a, b = read(coarse, 'run.json'), read(fine, 'run.json')
    if len(a['time_s']) != len(b['time_s']) or not np.allclose(a['time_s'], b['time_s'], rtol=0, atol=1e-9):
        raise ValueError('Compare identical output times, do not interpolate peaks')
    ids = [c['id'] for c in a_scene['gas_network']['channels'] if c['kind'] == 'rupture']
    def summed(run, key):
        return sum(np.asarray(run['channels'][k][key]) for k in ids)
    def delta(x, y):
        x, y = np.asarray(x), np.asarray(y)
        scale = float(np.max(np.abs(y)))
        return dict(coarse_final=float(x[-1]), fine_final=float(y[-1]),
            max_abs_difference=float(np.max(np.abs(x-y))),
            max_difference_relative_to_fine_peak=float(np.max(np.abs(x-y))/scale) if scale else None)
    report = dict(calibration_accepted=False, scope='Numerical comparison only; no experimental refit',
        duration_s=a['time_s'][-1], max_dt_s=caps, steps=[a['steps'], b['steps']],
        chamber_bar=delta(np.asarray(a['gas']['chamber']['pressure_pa'])/1e5,
                          np.asarray(b['gas']['chamber']['pressure_pa'])/1e5),
        area_mm2=delta(summed(a, 'effective_area_m2')*1e6, summed(b, 'effective_area_m2')*1e6),
        exported_mass_g=delta(summed(a, 'transferred_mass_kg')*1e3, summed(b, 'transferred_mass_kg')*1e3),
        fracture_work_j=delta(summed(a, 'fracture_work_j'), summed(b, 'fracture_work_j')),
        temperature_k={k: delta(a['temperature_k'][k], b['temperature_k'][k]) for k in a['temperature_k']},
        max_individual_patch_fraction_difference=max(float(np.max(np.abs(
            np.asarray(a['channels'][k]['opening_fraction'])-b['channels'][k]['opening_fraction']))) for k in ids),
        conservation={k: [a[k], b[k]] for k in
            ('max_thermal_residual_j', 'max_chemical_residual_j', 'max_mass_residual_kg', 'max_oxidizer_residual_kg')})
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('coarse', type=Path); p.add_argument('fine', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = compare(args.coarse, args.fine)
    save(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

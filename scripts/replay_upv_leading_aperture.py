#!/usr/bin/env python3
"""Deterministic, fracture-first replay. Diagnostic only, no parameter fitting."""
import argparse
import hashlib
import math
from pathlib import Path

import calibrate_upv_stochastic as old
import upv_local_mass_paths as local
from calibrate_upv import save


def build(scene, duration=3., max_dt=None):
    if not math.isfinite(duration) or not 0 < duration <= 30:
        raise ValueError('Duration must be in (0,30]')
    if max_dt is not None and (not math.isfinite(max_dt) or max_dt <= 0):
        raise ValueError('Invalid timestep cap')
    # Regenerate local routes AFTER the shell remap; never reuse old patch IDs.
    s = old.refine_shells(old.geometry.attach(scene, old.geometry.initial()), None)
    for c in s['gas_network']['channels']:
        if c['kind'] != 'rupture':
            continue
        law = c['damage_opening']
        c['opening_pressure_pa'] = 50e6
        c['damage_opening'] = {k: law[k] for k in
            ('seam_width_m', 'seam_strength_factor', 'fracture_energy_j_per_m2')}
        c['damage_opening'].update(mode='deterministic_aperture', area_rate_per_s=20000., pressure_exponent=1.)
    s = local.attach(s)
    s['simulation']['duration_s'] = duration
    if max_dt is not None:
        s['simulation']['max_dt_s'] = max_dt
    s['failure_sample_times_s'] = sorted({0., duration,
        *[float(t) for t in old.np.arange(0, duration, .02)],
        *[t for t in s['failure_sample_times_s'] if t <= duration]})
    s['metadata']['leading_aperture'] = dict(
        calibration_accepted=False, physical_thermal_sha256=old.physical_signature(scene),
        grid='Each shell face matches the contents tangential divisions',
        order='Fracture and work debit; heat/chemistry/LTE; pressure-driven mass/LTE',
        geometry='Weak seam strips first, then rectangular expansion; disk on patches without seam',
        parameters='50 MPa, 20000/s inherited trial values, NOT calibrated for this law',
        frozen='Heat laws, material properties, chemistry, initial contents field, physical gas geometry',
        remap='Conservative casing-field remap; local routes regenerated for the new grid')
    return s


def summary(run, scene):
    ids = [c['id'] for c in scene['gas_network']['channels'] if c['kind'] == 'rupture']
    return dict(steps=run['steps'], wall_s=run['wall_s'],
        chamber_peak_bar=max(run['gas']['chamber']['pressure_pa'])/1e5,
        final_open_area_mm2=sum(run['channels'][k]['effective_area_m2'][-1] for k in ids)*1e6,
        fracture_work_j=sum(run['channels'][k]['fracture_work_j'][-1] for k in ids),
        exported_mass_kg=sum(run['channels'][k]['transferred_mass_kg'][-1] for k in ids),
        thermal_residual_j=run['max_thermal_residual_j'], mass_residual_kg=run['max_mass_residual_kg'])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration', type=float, default=3.)
    p.add_argument('--max-dt', type=float)
    p.add_argument('--generate-only', action='store_true')
    args = p.parse_args()
    scene, _ = old.geometry.inputs('no-cooling')
    scene = build(scene, args.duration, args.max_dt)
    fp = hashlib.sha256(Path(__file__).read_bytes()+Path(local.__file__).read_bytes()+
        Path(old.__file__).read_bytes()+old.fit.fingerprint(old.fit.PROBE).encode()).hexdigest()
    manifest = dict(fingerprint=fp, duration=args.duration, max_dt=args.max_dt, calibration_accepted=False)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists() and old.geometry.read(out/'manifest.json') != manifest:
        raise ValueError('Changed model/controls: use a fresh output directory')
    save(out/'manifest.json', manifest); save(out/'scenario.json', scene)
    print('gas nodes', len(scene['gas_network']['compartments']), 'paths', len(scene['gas_network']['channels']), flush=True)
    if args.generate_only:
        return
    runner = old.fit.native.Runner(old.fit.PROBE, out)
    try:
        run = runner.run(scene)
        save(out/'run.json', run)
        report = summary(run, scene); save(out/'summary.json', report)
        print(report, flush=True)
    finally:
        runner.close()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Package recorded diagnostics, not a new simulation or a calibration.

Mean solid temperatures are explicitly labelled as such. No invented inner
gradients or gas velocities. Original times and discrete event times retained.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def export(trial: Path, output: Path):
    source = trial / 'replay/run.json.gz'
    scenario = trial / 'scenario.json'
    with gzip.open(source, 'rt') as stream:
        run = json.load(stream)
    scene = json.loads(scenario.read_text())
    times = np.asarray(run['time_s'], dtype='<f8')
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('Non-monotone or empty recorded trajectory')
    objects = sorted(run['thermal_paths']['objects'])
    channels = sorted(run['channels'])
    gas = run['gas_fields']['node_ids']
    solids = np.zeros((len(times), len(objects), 4), dtype='<f4')
    masses = {c['id']: c.get('conversion', {}).get('mass_kg', 1.) for c in scene['cells']}
    for i, name in enumerate(objects):
        solids[:, i, 0] = run['cell_mean_temperature_k'].get(name,
            run['thermal_paths']['objects'][name]['reference_capacity_weighted_temperature_k'])
        chemistry = run['chemistry'].get(name)
        if chemistry:
            solids[:, i, 1] = np.asarray(chemistry['degraded_mass_kg']) / masses[name]
            solids[:, i, 2] = chemistry['r2_power_w']
            solids[:, i, 3] = chemistry['r2_precursor_kg']
    apertures = np.zeros((len(times), len(channels), 12), dtype='<f4')
    for i, name in enumerate(channels):
        row = run['channels'][name]
        for j, key in enumerate(('opening_fraction', 'effective_area_m2', 'mass_flow_kg_s', 'aperture_radius_m')):
            apertures[:, i, j] = row.get(key, 0.)
        if 'aperture_strips_m' in row:
            apertures[:, i, 4:8] = row['aperture_strips_m']
        for j, key in enumerate(('transferred_mass_kg', 'mixed_mass_kg', 'max_thermal_damage', 'max_bulk_damage'), 8):
            apertures[:, i, j] = row.get(key, 0.)
    gas_path = trial / 'gas-frames.bin'
    with gas_path.open('rb') as stream:
        if stream.read(8) != b'BATTGAS1' or np.frombuffer(stream.read(8), '<u8')[0] != len(gas):
            raise ValueError('Invalid gas frame header')
        record = np.dtype([('time', '<f8'), ('state', '<f4', (len(gas), 3))])
        gas_frames = np.fromfile(stream, dtype=record)
    if len(gas_frames) != len(times) or not np.allclose(gas_frames['time'], times, atol=1e-7, rtol=0):
        raise ValueError('Gas and solid recordings are not synchronized')
    gas_values = np.zeros((len(times), len(gas), 6), dtype='<f4')
    gas_values[:, :, :3] = gas_frames['state']
    for i, name in enumerate(gas):
        recorded = run['gas'].get(name, {})
        for j, key in enumerate(('mass_kg', 'pressure_pa', 'released_energy_j'), 3):
            gas_values[:, i, j] = recorded.get(key, 0.)
    frames = np.concatenate((solids.reshape(len(times), -1),
        apertures.reshape(len(times), -1), gas_values.reshape(len(times), -1)), axis=1)
    if not np.isfinite(frames).all():
        raise ValueError('Nonfinite recorded state')
    output.mkdir(parents=True, exist_ok=True)
    frames.astype('<f4').tofile(output / 'frames.bin')
    manifest = dict(format='BATTVIEW2', channel_fields=12, gas_fields=6,
        scenario_path=str(scenario.resolve()),
        source_label=trial.name + ' / recorded CUDA trajectory',
        source_run_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scenario_sha256=hashlib.sha256(scenario.read_bytes()).hexdigest(),
        gas_sha256=hashlib.sha256(gas_path.read_bytes()).hexdigest(),
        frames_sha256=hashlib.sha256((output / 'frames.bin').read_bytes()).hexdigest(),
        time_s=times.tolist(), objects=objects, channels=channels, gas=gas,
        events=[e for e in run['events'] if e['kind'] in
            ('runaway_primary', 'runaway_secondary', 'runaway_complete', 'shell_rupture')],
        solid_sampling='recorded object means, no reconstructed internal gradients',
        gas_sampling='recorded control-volume values; linear time interpolation',
        flow_sampling='only recorded channels; absent passage flows remain unknown',
        calibrated=False, validation_passed=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(frames=len(times), objects=len(objects), channels=len(channels),
                         gas=len(gas), output=str(output)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trial', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    export(a.trial, a.output)

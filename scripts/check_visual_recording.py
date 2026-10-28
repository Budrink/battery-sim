#!/usr/bin/env python3
"""Compare a read-only full-flow capture against the archived trajectory.

Compare by stable IDs, never array order. Additional dense frames are observers,
not integration constraints; common sample times must retain the trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    manifest = json.loads(path.read_text())
    fields = (4 * len(manifest['objects']) + manifest['channel_fields'] *
              len(manifest['channels']) + manifest['gas_fields'] * len(manifest['gas']))
    frames = np.memmap(path.parent / 'frames.bin', dtype='<f4', mode='r',
                       shape=(len(manifest['time_s']), fields))
    return manifest, frames


def compare(reference_path, capture_path):
    reference, old = read(reference_path)
    capture, new = read(capture_path)
    lookup = {round(t, 6): i for i, t in enumerate(capture['time_s'])}
    pairs = [(i, lookup[round(t, 6)]) for i, t in enumerate(reference['time_s'])
             if round(t, 6) in lookup]
    if len(pairs) < 2:
        raise ValueError('No comparable trajectory interval')
    old_rows, new_rows = map(np.asarray, zip(*pairs))

    def error(group, field):
        widths = [4 if group == 'objects' else m[
            'channel_fields' if group == 'channels' else 'gas_fields']
            for m in (reference, capture)]
        offsets = [0 if group == 'objects' else len(m['objects']) * 4 + (
            len(m['channels']) * m['channel_fields'] if group == 'gas' else 0)
            for m in (reference, capture)]
        indices = {name: i for i, name in enumerate(capture[group])}
        new_columns = [offsets[1] + widths[1] * indices[name] + field
                       for name in reference[group]]
        old_columns = offsets[0] + widths[0] * np.arange(len(reference[group])) + field
        return float(np.max(np.abs(old[np.ix_(old_rows, old_columns)] -
                                   new[np.ix_(new_rows, new_columns)])))

    errors = {'solid_temperature_k': error('objects', 0),
              'gas_temperature_k': error('gas', 0),
              'gas_chemical_j': error('gas', 1),
              'gas_oxygen_kg': error('gas', 2),
              'recorded_channel_cumulative_mass_kg': error('channels', 8)}
    last = reference['time_s'][pairs[-1][0]]
    old_events = {(e['kind'], e['id']): e['time_s'] for e in reference['events']
                  if e['time_s'] <= last}
    new_events = {(e['kind'], e['id']): e['time_s'] for e in capture['events']
                  if e['time_s'] <= last}
    matching_events = old_events.keys() == new_events.keys()
    event_error = max((abs(t - new_events.get(k, float('inf')))
                       for k, t in old_events.items()), default=0)
    scenario_hash = hashlib.sha256(Path(capture['scenario_path']).read_bytes()).hexdigest()
    source_unchanged = scenario_hash == reference['scenario_sha256']
    return dict(reference=str(reference_path.resolve()), capture=str(capture_path.resolve()),
                partial=capture.get('partial', False), common_samples=len(pairs),
                compared_through_s=last, maximum_absolute_errors=errors,
                source_scenario_unchanged=source_unchanged, scenario_sha256=scenario_hash,
                event_sets_match=matching_events, event_time_error_s=event_error,
                identical_float32_fields=all(v == 0 for v in errors.values()),
                trajectory_preserved=source_unchanged and matching_events and
                    event_error < 1e-6 and all(v == 0 for v in errors.values()),
                experimental_validation=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-through', type=float, default=0)
    args = parser.parse_args()
    result = compare(args.reference, args.capture)
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    print(text, end='')
    if args.output:
        args.output.write_text(text)
    if not result['trajectory_preserved'] or result['compared_through_s'] < args.require_through:
        raise SystemExit(1)

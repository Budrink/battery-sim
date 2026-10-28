#!/usr/bin/env python3
"""Join actual accepted frames at an identical full-checkpoint boundary.

No interpolation or reconstructed gas state. Prefix may be an interrupted
recording; suffix must be complete. Inputs are retained unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect(path):
    manifest = json.loads(path.read_text())
    if manifest['format'] != 'BATTVIEW2':
        raise ValueError('Expected BATTVIEW2')
    times = np.asarray(manifest['time_s'])
    if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError('Invalid frame times')
    width = 4*len(manifest['objects']) + manifest['channel_fields']*len(manifest['channels']) + manifest['gas_fields']*len(manifest['gas'])
    frames_path = path.parent/'frames.bin'
    if frames_path.stat().st_size != len(times)*width*4:
        raise ValueError('Frame count differs from the manifest; finish capture first')
    frames = np.memmap(frames_path, dtype='<f4', mode='r', shape=(len(times), width))
    return manifest, times, frames


def stitch(prefix, suffix, output):
    prefix, suffix, output = prefix.resolve(), suffix.resolve(), output.resolve()
    a, ta, fa = inspect(prefix)
    b, tb, fb = inspect(suffix)
    if b.get('partial', True):
        raise ValueError('Suffix is not complete')
    for key in ['format', 'objects', 'channels', 'gas', 'channel_fields', 'gas_fields', 'solid_sampling', 'gas_sampling']:
        if a.get(key) != b.get(key):
            raise ValueError('Recording layout differs: '+key)
    scenario_hash = digest(Path(b['scenario_path']))
    if digest(Path(a['scenario_path'])) != scenario_hash:
        raise ValueError('Scenario differs across checkpoint boundary')
    boundary = np.flatnonzero(np.abs(ta-tb[0]) < 1e-8)
    if len(boundary) != 1:
        raise ValueError('No unique checkpoint frame in prefix')
    cut = int(boundary[0])
    if fa[cut].tobytes() != fb[0].tobytes():
        raise ValueError('Checkpoint frame differs; cannot join trajectories')
    signature = lambda e: (e['kind'], e['id'], e['time_s'])
    previous = {signature(e) for e in a['events'] if e['time_s'] <= tb[0]}
    following = {signature(e) for e in b['events'] if e['time_s'] <= tb[0]}
    if previous != following:
        raise ValueError('Event history differs across restart')
    times = np.concatenate((ta[:cut], tb))
    if not np.all(np.diff(times) > 0):
        raise ValueError('Non-monotone join')
    output.mkdir(parents=True, exist_ok=False)
    with (output/'frames.bin').open('wb') as target:
        # Bounded-memory copy of original bytes. One duplicate joint frame
        # is omitted; no observations are invented or retimed.
        for source, count in [(prefix.parent/'frames.bin', cut*fa.shape[1]*4),
                              (suffix.parent/'frames.bin', fb.size*4)]:
            with source.open('rb') as stream:
                while count:
                    chunk = stream.read(min(count, 8*1024*1024))
                    if not chunk:
                        raise ValueError('Unexpected EOF')
                    target.write(chunk)
                    count -= len(chunk)
    result = deepcopy(b)
    result.pop('wall_s', None)  # A failed prefix includes work beyond the join.
    result.update(time_s=times.tolist(), partial=False,
                  scenario_sha256=scenario_hash,
                  source_label=output.parent.name,
                  frames_sha256=digest(output/'frames.bin'),
                  continuation=dict(checkpoint_time_s=float(tb[0]),
                      identical_joint_frame=True, interpolation=False,
                      prefix_manifest=str(prefix), suffix_manifest=str(suffix),
                      prefix_manifest_sha256=digest(prefix), suffix_manifest_sha256=digest(suffix)))
    (output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return dict(frames=len(times), start_s=float(times[0]), end_s=float(times[-1]),
                checkpoint_time_s=float(tb[0]), manifest=str(output/'manifest.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('suffix', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(stitch(args.prefix, args.suffix, args.output), indent=2))

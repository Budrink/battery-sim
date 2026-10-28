#!/usr/bin/env python3
"""Isolated join/rejection checks; no simulation or GPU required."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np

from stitch_visual_restart import inspect, stitch


def main():
    with tempfile.TemporaryDirectory(prefix='battery-visual-join-') as temp:
        root = Path(temp)
        scenario = root/'scenario.json'
        scenario.write_text('{}\n')
        common = dict(format='BATTVIEW2', objects=['B10'], channels=[], gas=[],
                      channel_fields=12, gas_fields=6, scenario_path=str(scenario),
                      events=[dict(kind='shell_rupture', id='B10', time_s=.5)])
        a = dict(common, time_s=[0., 1., 2.], partial=True)
        b = dict(common, time_s=[1., 2., 3.], partial=False)
        fa = np.asarray([[300,0,0,1], [400,.5,1,.5], [600,1,0,0]], dtype='<f4')
        fb = np.asarray([[400,.5,1,.5], [500,1,0,0], [450,1,0,0]], dtype='<f4')

        def capture(name, manifest, frames):
            folder = root/name
            folder.mkdir()
            (folder/'manifest.json').write_text(json.dumps(manifest))
            frames.tofile(folder/'frames.bin')
            return folder/'manifest.json'

        prefix = capture('prefix', a, fa)
        suffix = capture('suffix', b, fb)
        result = stitch(prefix, suffix, root/'joined')
        _, times, frames = inspect(Path(result['manifest']))
        assert np.array_equal(times, [0,1,2,3])
        assert np.array_equal(frames, np.concatenate((fa[:1],fb)))
        assert (root/'prefix/frames.bin').read_bytes() == fa.tobytes()

        partial = deepcopy(b); partial['partial'] = True
        wrong = fb.copy(); wrong[0,0] += 1
        changed = deepcopy(b)
        other = root/'other.json'; other.write_text('{"different":true}\n')
        changed['scenario_path'] = str(other)
        for name, manifest, data in [('partial',partial,fb),('mismatched',b,wrong),('changed',changed,fb)]:
            bad = capture(name, manifest, data)
            try:
                stitch(prefix, bad, root/('out-'+name))
            except ValueError:
                pass
            else:
                raise AssertionError('Accepted '+name)
            assert not (root/('out-'+name)).exists()
        print('Visual restart: exact bytes retained; partial, mismatched state and changed scenario rejected')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Reassess immutable CUDA histories; no simulation, no score transplantation."""
import argparse
import gzip
import json
from pathlib import Path

import ijson
import numpy as np

from fsri30_data import load, save, sha
from fsri30_wave_fit import reference, assess_fields


def temperature_history(path):
    # Streaming avoids loading all gas/channel histories just to read temperatures.
    t, values, finished = [], {}, set()
    with gzip.open(path, 'rb') as stream:
        for prefix, kind, value in ijson.parse(stream, use_float=True):
            if prefix == 'time_s.item' and kind == 'number':
                t.append(value)
            elif prefix.startswith('cell_mean_temperature_k.') and kind == 'number':
                key = prefix.split('.')[1]
                values.setdefault(key, []).append(value)
            if (prefix, kind) in [('time_s', 'end_array'), ('cell_mean_temperature_k', 'end_map')]:
                finished.add(prefix)
                if len(finished) == 2:
                    break
    ids = sorted(values)
    if len(ids) != 30 or any(len(values[i]) != len(t) for i in ids):
        raise ValueError('Incomplete temperature histories: '+str(path))
    return np.asarray(t), np.asarray([values[i] for i in ids]).T, ids


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--history',type=Path,nargs='+',required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    data,manifest=load();ref=reference(data)
    save(out/'reference-waves.json',dict(ref,csv_sha256=manifest['csv_sha256']))
    seen=set();results=[]
    for folder in args.history:
        old=json.loads((folder/'observations.json').read_text())
        for row in old:
            trial=Path(row['trial'])
            if str(trial) in seen:continue
            seen.add(str(trial))
            paths=[trial/'replay'/n for n in ('run.json.gz','incomplete-run.json.gz')]
            path=next((p for p in paths if p.exists()),None)
            if path is None:continue
            t,y,ids=temperature_history(path)
            assessed=assess_fields(t,y,ids,ref)
            key=f'history-{len(results):03d}'
            np.savez_compressed(out/(key+'.npz'),time_s=t,temperature_k=y,ids=ids)
            result=dict(trial=str(trial),history=str(path),native_sha256=row.get('native_sha256'),
                        history_sha256=sha(path),cache=key+'.npz',
                        parameters=row.get('parameters'),assessment=assessed,
                        inherited_old_acceptance=False)
            results.append(result);save(out/'reassessment.json',results)
            print(json.dumps(dict(trial=str(trial),end_s=float(t[-1]),
                    accepted=assessed['accepted'],reason=assessed['reason'])),flush=True)
    save(out/'summary.json',dict(histories=len(results),
        accepted=sum(r['assessment']['accepted'] is True for r in results),
        rejected=sum(r['assessment']['accepted'] is False for r in results),
        pending=sum(r['assessment']['accepted'] is None for r in results),
        independent_validation=False))


if __name__=='__main__':main()

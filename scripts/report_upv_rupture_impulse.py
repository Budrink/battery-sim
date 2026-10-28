#!/usr/bin/env python3
"""Read-only comparison with the explicit archived control; no new simulation."""
import argparse
import gzip
import json
from pathlib import Path

import ijson
import calibrate_upv_r123 as cal
import upv_r123_data as dataio
from resolve_upv_rupture_impulse import compact, impulse_report, plot_impulse
from upv_cooling_review import plot_cooling


def read_control(path):
    # Skip unwanted arrays in the C parser. The full replay is large enough
    # that loading every chemical zone twice can dwarf the plotting workload.
    result = {}
    for key in ('time_s','temperature_k','gas_observers'):
        with gzip.open(path,'rb') as f:
            result[key] = next(ijson.items(f,key,use_float=True))
    for key,prefix in (('channels','rupture-C2-'),('gas','gap-ISS1-')):
        result[key] = {}
        with gzip.open(path,'rb') as f:
            for name,value in ijson.kvitems(f,key,use_float=True):
                if name.startswith(prefix): result[key][name] = value
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--trials',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args();out = args.output.resolve();out.mkdir(exist_ok=True)
    data,_ = dataio.load('no-cooling');runs = {};reports = {};labels = dict(control='Исходный расчёт')
    cached = out/'control-curves.json'
    identity = str(args.reference.resolve())
    if cached.exists():
        saved = json.loads(cached.read_text())
        if saved['source'] != identity: raise ValueError('Cached control is from another source')
        runs['control'] = saved['curves'];reports['control'] = saved['impulse']
    else:
        scene = json.loads((args.reference/'scenario.json').read_text())
        run = read_control(args.reference/'replay/run.json.gz')
        runs['control'] = compact(scene,run);reports['control'] = impulse_report(scene,run,data)
        cal.save(cached,dict(source=identity,curves=runs['control'],impulse=reports['control']))
    for folder in args.trials:
        if not (folder/'curves.json').exists(): continue
        name = folder.name
        if name in runs: raise ValueError('Duplicate comparison label')
        runs[name] = json.loads((folder/'curves.json').read_text())
        reports[name] = json.loads((folder/'report.json').read_text())
        metadata = json.loads((folder/'scenario.json').read_text())['metadata']
        probe = metadata['rupture_impulse_probe']
        labels[name] = f"Оболочка ×{probe['strength_factor']:g}"
        if probe['area_rate_factor'] != 1.:
            labels[name] += f"; раскрытие ×{probe['area_rate_factor']:g}"
        if metadata.get('hot_internal_recovery_probe'):
            labels[name] += '; внутренняя S(T)'
        r3 = metadata.get('r3_activation_probe')
        if r3: labels[name] += f"; R3 {r3['activation_j_per_mol']/1000:g} кДж/моль"
    plot_impulse(out,runs,data,labels);plot_cooling(out,runs,data,labels)
    cal.save(out/'comparison.json',dict(source=identity,cases=reports,diagnostic_only=True,
        accepted_calibration=False,validation_run=False,
        caveat='The source retains explicitly diagnostic route conductances; not a measured geometry fit.'))
    print('REPORT',out,flush=True)


if __name__=='__main__':main()

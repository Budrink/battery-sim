#!/usr/bin/env python3
"""Synthetic connected hot-gas scaling; NOT a fit or cascade validation.

Replicate the actual 3-battery scene, retain all material/transport laws,
connect chamber voids by explicit tubes, seed hot reacting nonuniform gas.
Run serially, only AFTER experimental replays, to avoid competing for CUDA.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import calibrate_upv_r123 as cal
from check_upv_restart import native


def replicate(source,count,duration):
    base=copy.deepcopy(source)
    for key in ('measurement_probes','thermal_path_trace','gas_film_trace','surface_probe_trace','process_history_selection'):
        base.pop(key,None)
    base['measurement_probes']=[]
    base['metadata']={'synthetic_scaling':True,'not_experimental_validation':True}
    base['simulation']['duration_s']=duration
    base['simulation']['output_interval_s']=duration/4.
    base['failure_sample_times_s']=[j*duration/4. for j in range(5)]
    base['primary_failures']=[]
    # No artificially triggered solid chemistry: benchmark populated hot gas,
    # pressure-driven barrier opening, finite contacts and flow from t=0.
    for c in base['cells']:
        if 'electrical'in c:
            c['electrical']['current_profile_a']=[[0.,0.],[duration,0.]]
            c['electrical']['voltage_profile_v']=[[0.,0.],[duration,0.]]
    free=[g for g in base['gas_network']['compartments']if g['id'].startswith('void-')]
    volume=sum(g['volume_m3']for g in free)
    for j,g in enumerate(free):
        g['initial_temperature_k']=700.+100.*(j%3)
        g['initial_mass_kg']*=1.+.1*((j%3)-1)
        g['initial_chemical_j']=2.e4*g['volume_m3']/volume
    # Synthetic compressed inventory exercises all casings, not just the void
    # mixing operator. Finite gas energy pays fracture work normally.
    for g in base['gas_network']['compartments']:
        if g['id'].startswith('inside-'):g['initial_mass_kg']*=200.
    arrays=('cells','ports','interfaces','boundaries','shells','primary_failures')
    net_arrays=('compartments','channels','thermal_contacts')
    identifiers={item['id']for key in arrays for item in base[key]if 'id'in item}
    identifiers.update(item['id']for key in net_arrays for item in base['gas_network'][key]if 'id'in item)
    solid_ids=[c['id']for c in base['cells']]
    def renamed(value,prefix):
        if isinstance(value,dict):return {k:renamed(v,prefix)for k,v in value.items()}
        if isinstance(value,list):return [renamed(v,prefix)for v in value]
        if isinstance(value,str)and(value in identifiers or any(value.startswith(s+':')for s in solid_ids)):
            return prefix+value
        return value
    target=copy.deepcopy(base)
    for key in arrays:target[key]=[]
    for key in net_arrays:target['gas_network'][key]=[]
    for i in range(count):
        prefix=f'module-{i}:'
        for key in arrays:
            values=renamed(base[key],prefix)
            if key=='cells':
                for c in values:c['pose']['position_m'][1]+=i*1.
            # Port centroids are absolute, so translate these alongside solids.
            if key=='ports':
                for p in values:p['centroid_m'][1]+=i*1.
            target[key].extend(values)
        for key in net_arrays:target['gas_network'][key].extend(renamed(base['gas_network'][key],prefix))
        if i:
            target['gas_network']['channels'].append(dict(id=f'connecting-tube-{i}',kind='passage',
                **{'from':f'module-{i-1}:'+free[-1]['id'],'to':prefix+free[0]['id']},
                area_m2=.001,discharge_coefficient=.7,opening_pressure_pa=0.,mixing_length_m=.01))
    target['name']=f'SYNTHETIC {3*count}-battery hot-gas/casing scaling, connected chambers'
    return target


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--modules',type=int,nargs='+',default=[1,4,10])
    ap.add_argument('--duration',type=float,default=10.)
    a=ap.parse_args();out=a.output.resolve();out.mkdir();source=json.loads(a.source.read_text())
    cal.PROBE=a.probe.resolve();core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    rows=[]
    for count in a.modules:
        folder=out/f'modules-{count:02d}';folder.mkdir();scene=replicate(source,count,a.duration)
        cal.save(folder/'scenario.json',scene)
        run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=core),max(120.,a.duration*10.),enforce_speed=False)
        rows.append(dict(modules=count,batteries=3*count,gas_volumes=len(scene['gas_network']['compartments']),
            solid_volumes=run['volumes'],wall_s=run['wall_s'],simulated_s=a.duration,realtime_ratio=a.duration/run['wall_s'],
            minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
            balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k}))
        cal.save(out/'report.json',dict(native_sha256=core,synthetic=True,experimental_validation=False,results=rows,
            limitation='Replicated modules with narrow inter-chamber tubes; not evidence of full 30-battery cascade accuracy'))
        print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':main()

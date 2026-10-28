#!/usr/bin/env python3
"""Summarize recorded heat/gas routes without extrapolating a run prefix."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def nearest(times, target):
    if not times or target<times[0]-1e-7 or target>times[-1]+1e-7:
        raise ValueError('Requested interval is outside the saved trajectory')
    return min(range(len(times)),key=lambda i:abs(times[i]-target))


def audit(run, start, end, source='B10'):
    times=run['time_s'];i=nearest(times,start);j=nearest(times,end)
    if i>=j:raise ValueError('Empty interval')
    def delta(v):return v[j]-v[i]
    films=defaultdict(float)
    for value in run['gas_film_trace'].values():
        films[value['solid'].split(':')[0]]+=delta(value['heat_to_solid_j'])
    contacts={}
    for key,value in run['thermal_paths']['interfaces'].items():
        if not key.startswith('can-contact:'):continue
        a=value['from'].split(':')[0];b=value['to'].split(':')[0]
        if source not in (a,b):continue
        power=value['a_to_b_power_w']
        energy=sum(.5*(power[k]+power[k+1])*(times[k+1]-times[k]) for k in range(i,j))
        contacts[b if a==source else a]=energy if a==source else -energy
    vents=[x for name,x in run['channels'].items() if name.startswith('vent-')]
    groups=run['gas_groups']
    return dict(interval_s=[times[i],times[j]],source=source,can_contact_j=contacts,
        gas_to_solid_j=dict(films),module_r3_j=delta(groups['module']['released_energy_j']),
        outside_r3_j=delta(groups['outside']['released_energy_j']),
        vent_mass_kg=sum(delta(v['transferred_mass_kg']) for v in vents),
        vent_chemical_j=sum(delta(v['transferred_chemical_j']) for v in vents),
        note='Vent mass includes displaced gas; sensor/cell identification remains unverified. Contact heat is sampled trapezoidal integration.')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path)
    ap.add_argument('--start',type=float,required=True);ap.add_argument('--end',type=float,required=True)
    ap.add_argument('--source',default='B10');args=ap.parse_args()
    with gzip.open(args.run,'rt') as stream:run=json.load(stream)
    print(json.dumps(audit(run,args.start,args.end,args.source),indent=2))


if __name__=='__main__':main()

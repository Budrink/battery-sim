"""Fresh FSRI diagnostic: valve threshold, gas yield, R3 and no can contacts."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from fsri30_data import save


def candidate(source, valve_factor=.5, gas_yield=.7, r3=2.):
    if not all(math.isfinite(x) and x>0 for x in (valve_factor,gas_yield,r3)) or valve_factor>1 or gas_yield>1:
        raise ValueError('Invalid diagnostic multipliers')
    s=deepcopy(source);meta=s['metadata']['can_contact_hypothesis'];width=meta['width_m']
    batteries={c['id']:c for c in s['cells'] if c.get('role')=='battery'}
    shells={x['contents_id']:x for x in s['shells']};counts=Counter()
    for pair in meta['contact_pairs']:
        counts[(pair['a'],pair['face_a'])]+=1;counts[(pair['b'],pair['face_b'])]+=1
    for (name,face), count in counts.items():
        axis='xyz'.index(face[1]);dims=batteries[name]['geometry']['dimensions_m'];thick=shells[name]['thickness_m']
        panel=[thick if k==axis else dims[k]+(2*thick if k>axis else 0.) for k in range(3)]
        other=[k for k in range(3) if k!=axis];exposure=1.-count*width*.065/(panel[other[0]]*panel[other[1]])
        if not 0<exposure<=1:raise ValueError('Invalid former gas exposure')
        target=f'{name}:case:{face}:outer';films=[f for f in s['gas_network']['thermal_contacts'] if f['port_id']==target]
        if not films:raise ValueError('Missing gas film '+target)
        for film in films:
            for key in film['connectivity']:
                if key.endswith('conductance_w_per_k'):film['connectivity'][key]/=exposure
    interfaces=s['interfaces'];removed=sum(i['id'].startswith('can-contact:') for i in interfaces)
    if removed!=len(meta['contact_pairs']):raise ValueError('Inconsistent can contact count')
    s['interfaces']=[i for i in interfaces if not i['id'].startswith('can-contact:')]
    s['ports']=[p for p in s['ports'] if not p['id'].startswith('can-contact:')]
    meta.update(direct_contacts_enabled=False,removed_for_diagnostic=True,gas_film_exposure='restored area-average')
    valves=0
    for channel in s['gas_network']['channels']:
        valve=channel.get('damage_opening',{}).get('valve')
        if valve is not None:valve['pressure_factor']*=valve_factor;valves+=1
    if not valves:raise ValueError('No safety valves found')
    for cell in batteries.values():
        reaction=cell['conversion'];reaction['gas_yield']*=gas_yield
        if 'gas_yield_max' in reaction:reaction['gas_yield_max']*=gas_yield
    for compartment in s['gas_network']['compartments']:
        if 'reaction_rate_per_s' in compartment:compartment['reaction_rate_per_s']*=r3
    s['metadata']['valve_no_contact_probe']=dict(valve_factor_multiplier=valve_factor,gas_yield_multiplier=gas_yield,
        r3_rate_multiplier=r3,valves=valves,disabled_can_contacts=removed,measured=False,validation_passed=False)
    return s


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    s=candidate(json.loads(a.source.read_text()));save(a.output,s);print(s['metadata']['valve_no_contact_probe'])

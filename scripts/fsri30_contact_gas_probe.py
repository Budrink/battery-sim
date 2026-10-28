"""Fresh-scene exploratory contact width and gas transport changes."""
from copy import deepcopy
from collections import Counter
import math
from fsri30_gas_mobility import with_internal_resistance


def candidate(source, width=.001, resistance=4., mixing=2., r3_rate=1., gas_yield=1.):
    s=with_internal_resistance(source,resistance)
    meta=s['metadata']['can_contact_hypothesis']; old=meta['width_m']
    if not (0 < old <= .003 and 0 < width <= .003 and math.isfinite(mixing) and mixing > 0 and math.isfinite(r3_rate) and r3_rate > 0 and math.isfinite(gas_yield) and 0 < gas_yield <= 1):
        raise ValueError('Invalid exploratory contact/mixing setting')
    ratio=width/old; pairs=meta['contact_pairs']; counts=Counter()
    for pair in pairs:
        counts[(pair['a'],pair['face_a'])]+=1; counts[(pair['b'],pair['face_b'])]+=1
    ports={p['id']:p for p in s['ports']}
    for interface in s['interfaces']:
        if not interface['id'].startswith('can-contact:'): continue
        for key in ('port_a','port_b'):
            port=ports[interface[key]]; rect=port['rectangle_m']
            index=min(range(2),key=lambda k:abs(rect[k]-old))
            if not math.isclose(rect[index],old,abs_tol=1e-12): raise ValueError('Unexpected strip width')
            rect[index]*=ratio;port['area_m2']*=ratio
        for key in interface['law']:
            if key.endswith('conductance_w_per_k'): interface['law'][key]*=ratio
    # Reproduce the original generator's area-averaged gas exposure, applied
    # to the already aligned patches. This is still a geometric approximation.
    cells={c['id']:c for c in s['cells'] if c.get('role')=='battery'}
    shell={x['contents_id']:x for x in s['shells']}
    for (name,face), count in counts.items():
        c=cells[name];axis='xyz'.index(face[1]);dims=c['geometry']['dimensions_m'];t=shell[name]['thickness_m']
        panel=[t if k==axis else dims[k]+(2*t if k>axis else 0.) for k in range(3)]
        transverse=[k for k in range(3) if k!=axis];panel_area=panel[transverse[0]]*panel[transverse[1]]
        old_exposure=1.-count*old*.065/panel_area;new_exposure=1.-count*width*.065/panel_area
        if not (0 < new_exposure <= 1 and 0 < old_exposure <= 1): raise ValueError('Invalid gas-exposed area')
        target=f'{name}:case:{face}:outer';films=[f for f in s['gas_network']['thermal_contacts'] if f['port_id']==target]
        if not films:raise ValueError('Missing contact gas film '+target)
        for film in films:
            for key in film['connectivity']:
                if key.endswith('conductance_w_per_k'):film['connectivity'][key]*=new_exposure/old_exposure
    for pair in pairs:
        pair['area_m2']*=ratio;pair['nominal_conductance_w_k']*=ratio
    meta['width_m']=width
    gas=s['gas_network'];gas['mixing_diffusivity_m2_s']*=mixing;gas['thermal_stirring_coefficient']*=mixing
    for compartment in gas['compartments']:
        if 'reaction_rate_per_s' in compartment: compartment['reaction_rate_per_s']*=r3_rate
    for cell in cells.values():
        reaction=cell['conversion']
        reaction['gas_yield']*=gas_yield
        if 'gas_yield_max' in reaction: reaction['gas_yield_max']*=gas_yield
    s['metadata']['contact_gas_probe']=dict(width_m=width,resistance_multiplier=resistance,mixing_multiplier=mixing,r3_rate_multiplier=r3_rate,gas_yield_multiplier=gas_yield,
        qualification='diagnostic assumptions, not measured or calibrated',
        mixing_note='existing gas mixing also exchanges species and oxygen; not heat-only')
    return s


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from fsri30_data import save
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--width',type=float,default=.001);p.add_argument('--resistance',type=float,default=4.)
    p.add_argument('--mixing',type=float,default=2.);p.add_argument('--r3-rate',type=float,default=1.)
    p.add_argument('--gas-yield',type=float,default=1.)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=candidate(json.loads(a.source.read_text()),a.width,a.resistance,a.mixing,a.r3_rate,a.gas_yield)
    save(a.output,result)
    print(result['metadata']['contact_gas_probe'])

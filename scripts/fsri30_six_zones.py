"""Fresh six-zone 18650 approximation with conservative local pore remeshing.

Uses three transverse strips and two axial halves in the existing prismatic
solver. The middle strip is only a coarse core surrogate, not a radial core.
Never apply to a checkpoint or a nonuniform initial field.
"""
from copy import deepcopy
import math


def six_zone_batteries(source):
    scene = deepcopy(source)
    gas = scene['gas_network']
    old_nodes = {n['id']: n for n in gas['compartments']}
    keep_nodes = [n for n in gas['compartments'] if not n['id'].startswith('inside-')]
    keep_channels = [e for e in gas['channels'] if not e['id'].startswith(('pore-', 'rupture-'))]
    new_nodes, new_channels = [], []
    for cell in scene['cells']:
        if cell.get('role') != 'battery':
            continue
        name = cell['id']; thermal = cell['thermal']
        if thermal['divisions'] != [3, 3, 3] or 'initial_field_k' in thermal:
            raise ValueError('Expected a fresh uniform 3x3x3 battery: '+name)
        if any(s.get('surface_divisions') for s in scene['shells'] if s['contents_id'] == name):
            raise ValueError('Independent shell mesh requires explicit remapping')
        nodes = [old_nodes[f'inside-{name}-{i}'] for i in range(27)]
        def material(n):
            return {k:v for k,v in n.items() if k not in ('id','volume_m3','initial_mass_kg','contents_id','local_equilibrium')}
        if any(material(n) != material(nodes[0]) for n in nodes):
            raise ValueError('Battery pore composition is not initially uniform')
        volume = sum(n['volume_m3'] for n in nodes)
        mass = sum(n['initial_mass_kg'] for n in nodes)
        for index in range(6):
            node = deepcopy(nodes[0]); node.pop('contents_id', None)
            node.update(id=f'inside-{name}-{index}', volume_m3=volume/6., initial_mass_kg=mass/6.)
            node['local_equilibrium']['thermal_volume'] = index
            if index == 0: node['contents_id'] = name
            new_nodes.append(node)
        dims = cell['geometry']['dimensions_m']; old_h = [x/3. for x in dims]
        new_h = [dims[0]/3., dims[1], dims[2]/2.]
        original_pores = [e for e in gas['channels'] if e['id'].startswith('pore-'+name+'-')]
        templates = {a:next(e for e in original_pores if e['mixing_axis']==a) for a in (0,2)}
        for z in range(2):
            for x in range(3):
                index=x+3*z
                for axis, allowed, neighbor in ((0,x<2,index+1),(2,z<1,index+3)):
                    if not allowed: continue
                    edge=deepcopy(templates[axis]); transverse=[k for k in range(3) if k!=axis]
                    old_face=math.prod(old_h[k] for k in transverse)
                    new_face=math.prod(new_h[k] for k in transverse)
                    edge.update(id=f'pore-{name}-{index}-{axis}', **{'from':f'inside-{name}-{index}',
                        'to':f'inside-{name}-{neighbor}'}, area_m2=edge['area_m2']*new_face/old_face,
                        mixing_length_m=new_h[axis])
                    new_channels.append(edge)
        openings = [e for e in gas['channels'] if e['id'].startswith('rupture-'+name+'-')]
        if len(openings)!=9: raise ValueError('Expected nine original casing openings')
        for x in range(3):
            choices=[e for e in openings if e['shell_thermal_volume']%3==x]
            valves=[e for e in choices if 'valve' in e.get('damage_opening',{})]
            edge=deepcopy(valves[0] if valves else choices[0])
            edge.update(id=f'rupture-{name}-{x}', **{'from':f'inside-{name}-{x+3}'}, shell_thermal_volume=x)
            new_channels.append(edge)
        thermal['divisions']=[3,1,2]
        cell['conversion']['gas_compartment']=f'inside-{name}-0'
    gas['compartments']=keep_nodes+new_nodes
    gas['channels']=keep_channels+new_channels
    scene['metadata']['six_zone_battery']=dict(divisions=[3,1,2], axial_zone_m=.0325,
        approximation='three equal transverse strips, not a cylindrical radial core',
        fresh_initial_state_only=True, calibrated=False, validation_passed=False)
    return scene

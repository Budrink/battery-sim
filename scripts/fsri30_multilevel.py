"""Coarse-to-fine FSRI search meshes derived from the RETAINED physical scene.

80 mm air / 40 mm ABS / one battery zone -> 40 / 20 / three axial zones
-> original 20 / 10 / 3^3. Coarse scores are never fine-grid validation.
No batteries are merged, and apertures, material and chemical inventories,
ambient exchange and directed contact coefficients remain physical priors.
"""
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np

from fsri30_resolved_gas import build_resolved
from fsri30_open_boundary import open_boundary
from fsri30_radiation import add_exterior_radiation


LEVELS={80:dict(gas_m=.08,abs_m=.04,battery_divisions=[1,1,1]),
        40:dict(gas_m=.04,abs_m=.02,battery_divisions=[1,1,3]),
        20:dict(gas_m=.02,abs_m=.01,battery_divisions=[3,3,3])}


def inventory(scene):
    gas=scene['gas_network']['compartments']
    return dict(gas_volume_m3=sum(n['volume_m3']for n in gas),
        gas_mass_kg=sum(n['initial_mass_kg']for n in gas),
        gas_oxygen_kg=sum(n['initial_mass_kg']*n.get('initial_oxidizer_mass_fraction',0.)for n in gas),
        gas_mass_temperature=sum(n['initial_mass_kg']*n['initial_temperature_k']for n in gas),
        solid_capacity_j_k=sum(c['thermal']['heat_capacity_j_per_k']for c in scene['cells']),
        renewal_m3_s=sum(n.get('ambient_boundary',{}).get('renewal_flow_m3_s',0.)for n in gas),
        module_exit_area_m2=sum(c['area_m2']for c in scene['gas_network']['channels']if c['id'].startswith('vent-')),
        thermal_zones=sum(math.prod(c['thermal']['divisions'])for c in scene['cells']),gas_nodes=len(gas))


def _coarsen_contents(scene, divisions):
    """Aggregate nested uniform initial zones and their parallel pore faces."""
    gas=scene['gas_network'];mapping={};new_nodes=[]
    for cell in scene['cells']:
        if cell.get('role')!='battery':continue
        ident=cell['id'];old=cell['thermal']['divisions']
        if old!=[3,3,3]or any(3%n for n in divisions):
            raise ValueError('Expected nested coarsening of 3^3 battery zones')
        if any(s.get('surface_divisions')for s in scene['shells']if s['contents_id']==ident):
            raise ValueError('Independent shell mesh needs an explicit remapping')
        if 'initial_field_k'in cell['thermal']:
            raise ValueError('Cannot remesh a nonuniform state as a fresh initial condition')
        nodes=[n for n in gas['compartments']if n['id'].startswith('inside-'+ident+'-')]
        if len(nodes)!=27:raise ValueError('Unexpected battery gas topology')
        groups=defaultdict(list)
        for node in nodes:
            binding=node['local_equilibrium']
            index=binding['thermal_volume'];xyz=[index%3,(index//3)%3,index//9]
            q=[xyz[k]//(3//divisions[k])for k in range(3)]
            target=q[0]+divisions[0]*(q[1]+divisions[1]*q[2])
            name=f'inside-{ident}-{target}';mapping[node['id']]=name;groups[target].append(node)
        for i,members in sorted(groups.items()):
            def physical(n):
                return {k:v for k,v in n.items()if k not in
                        ('id','volume_m3','initial_mass_kg','contents_id','local_equilibrium')}
            if any(physical(n)!=physical(members[0])for n in members):
                raise ValueError('Cannot merge dissimilar gas materials or nonuniform initial states')
            node=deepcopy(members[0]);node.pop('contents_id',None)
            node.update(id=f'inside-{ident}-{i}',volume_m3=sum(n['volume_m3']for n in members),
                        initial_mass_kg=sum(n['initial_mass_kg']for n in members))
            node['local_equilibrium']['thermal_volume']=i
            if i==0:node['contents_id']=ident
            new_nodes.append(node)
        cell['thermal']['divisions']=list(divisions)
    gas['compartments']=[n for n in gas['compartments']if n['id']not in mapping]+new_nodes
    channels=[];pores={};ruptures=defaultdict(list)
    for edge in gas['channels']:
        if edge['id'].startswith('pore-'):
            a,b=mapping[edge['from']],mapping[edge['to']]
            if a==b:continue
            key=(a,b,edge['mixing_axis'])
            if key not in pores:
                pores[key]=deepcopy(edge);pores[key].update({'from':a,'to':b,'area_m2':0.})
            elif any(pores[key].get(k)!=edge.get(k)for k in
                     ('mixing_length_m','discharge_coefficient','opening_pressure_pa')):
                raise ValueError('Dissimilar parallel pore laws')
            pores[key]['area_m2']+=edge['area_m2']
        elif edge['id'].startswith('rupture-'):
            ruptures[edge['port_id']].append(edge)
        else:channels.append(edge)
    for edges in ruptures.values():
        if len(edges)!=9 or any(e['port_id'].split(':')[-2]!='+z'for e in edges):
            raise ValueError('Only the existing nine-patch cap is supported')
        # A single coarse cap retains the SAME physical valve radius, threshold
        # and fracture-work density. Only the surface discretization changes.
        valves=[e for e in edges if 'valve'in e['damage_opening']]
        if len(valves)!=1:raise ValueError('Expected one physical safety valve')
        def law(e):
            e=deepcopy(e)
            for k in ('id','from','shell_thermal_volume'):e.pop(k,None)
            e['damage_opening'].pop('valve',None)
            return e
        if any(law(e)!=law(edges[0])for e in edges):
            raise ValueError('Distinct rupture material laws cannot be collapsed')
        e=deepcopy(valves[0]);e['from']=mapping[e['from']];e['shell_thermal_volume']=0
        e['id']='rupture-'+e['port_id'].split(':')[0]+'-0';channels.append(e)
    gas['channels']=channels+list(pores.values())


def _contact_totals(scene):
    totals=defaultdict(lambda:defaultdict(float))
    for c in scene['gas_network']['thermal_contacts']:
        p=c['port_id']
        # A remeshed ABS face can have many gas overlaps, but their total
        # directed film conductance must remain unchanged in each direction.
        if not p.startswith('gasmesh:ABS:'):continue
        group=':'.join(p.split(':')[:4])
        for k,v in c['connectivity'].items():
            if k.endswith('conductance_w_per_k'):totals[group][k]+=v
    return {k:dict(v)for k,v in totals.items()}


def remesh(base, level_mm):
    if level_mm not in LEVELS:raise ValueError('Unknown fixed mesh level')
    if base['metadata']['air_mesh']['spacing_m']!=.02:
        raise ValueError('Derive every level from the retained 20 mm profile')
    if level_mm==20:return deepcopy(base)  # Exact target profile, no rebuilding drift.
    level=LEVELS[level_mm];before=inventory(base);scene=deepcopy(base);gas=scene['gas_network']
    external=[n for n in gas['compartments']if n['id'].startswith('air-')]
    if not external:raise ValueError('No resolved external gas')
    def material(n):return {k:v for k,v in n.items()if k not in
                           ('id','volume_m3','initial_mass_kg','ambient_boundary')}
    if any(material(n)!=material(external[0])for n in external):
        raise ValueError('Cannot remesh a nonuniform or heterogeneous initial atmosphere')
    ambient=[n['ambient_boundary']for n in external if 'ambient_boundary'in n]
    if not ambient:raise ValueError('Expected retained open far-field boundary')
    fixed=lambda b:{k:v for k,v in b.items()if k not in ('area_m2','renewal_flow_m3_s')}
    if any(fixed(b)!=fixed(ambient[0])for b in ambient):raise ValueError('Different far-field reservoirs')
    receiver=deepcopy(external[0]);receiver.pop('ambient_boundary',None)
    receiver.update(id='hood',volume_m3=sum(n['volume_m3']for n in external),
                    initial_mass_kg=sum(n['initial_mass_kg']for n in external))
    removed={n['id']for n in external}
    gas['compartments']=[n for n in gas['compartments']if n['id']not in removed]+[receiver]
    gas['channels']=[e for e in gas['channels']if e['from']not in removed and e['to']not in removed]
    gas['thermal_contacts']=[c for c in gas['thermal_contacts']if not c['port_id'].startswith('gasmesh:ABS:')]
    scene['ports']=[p for p in scene['ports']if not p['id'].startswith('gasmesh:ABS:')]
    scene['boundaries']=[b for b in scene['boundaries']if not b['id'].startswith('radiation:')]
    for key in ('air_mesh','gas_volume_geometry','ambient_boundary','exterior_radiation'):
        scene['metadata'].pop(key,None)
    scene=build_resolved(scene,level['gas_m'],wall_spacing=level['abs_m'],
        stirring=base['gas_network']['thermal_stirring_coefficient'],
        stirring_length=base['gas_network']['thermal_stirring_length_m'],
        pressure_patterns=base['gas_network']['pressure_pattern_predictor'],
        halo=base['metadata']['air_mesh']['halo_m'],controlled_heater=False)
    # Builder defaults are not a licence to reset numerical controls or gains.
    for k,v in base['gas_network'].items():
        if k not in ('compartments','channels','thermal_contacts'):scene['gas_network'][k]=deepcopy(v)
    scene['metadata']['heater_protocol']=deepcopy(base['metadata']['heater_protocol'])
    scene=open_boundary(scene,before['renewal_m3_s'],pressure_open=any(b['area_m2']>0 for b in ambient))
    for n in scene['gas_network']['compartments']:
        if 'ambient_boundary'in n:n['ambient_boundary'].update(fixed(ambient[0]))
    scene=add_exterior_radiation(scene,base['metadata']['exterior_radiation']['effective_emissivity'])
    _coarsen_contents(scene,level['battery_divisions'])
    after=inventory(scene)
    for key,value in before.items():
        if key not in ('thermal_zones','gas_nodes') and not math.isclose(after[key],value,rel_tol=2e-11,abs_tol=1e-13):
            raise ValueError('Remeshing changed physical inventory: '+key)
    old,new=_contact_totals(base),_contact_totals(scene)
    if old.keys()!=new.keys()or any(not math.isclose(old[g][k],new[g][k],rel_tol=2e-10,abs_tol=1e-13)
                                    for g in old for k in old[g]):
        raise ValueError('Remeshing changed learned ABS contact conductance')
    for key in ('heating_protocols','shells','interfaces','primary_failures'):
        if scene[key]!=base[key]:raise ValueError('Remeshing changed physical input: '+key)
    for a,b in zip(scene['cells'],base['cells']):
        for key in ('conversion','geometry','pose','electrical'):
            if a.get(key)!=b.get(key):raise ValueError('Remeshing changed '+key)
    scene['metadata']['search_resolution']=dict(level_mm=level_mm,**level,
        target_level_mm=20,screening_only=True,inventory_before=before,inventory_after=after,
        parameter_scores_transfer=False,checkpoint_transfer=False)
    return scene

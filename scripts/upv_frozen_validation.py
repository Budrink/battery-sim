"""Transfer an explicit frozen UPV scene to the cooled test, without refitting.

Metadata parameters alone are NOT the calibrated model: later contact, gas,
chemistry and rupture changes live in the actual scene. Copy those laws, while
rebuilding only experiment inputs and geometry obstructed by the cooling plate.
Unknown/nonuniform transfer groups are errors, never silent default resets.
"""
import copy
import hashlib
import json
import math

import upv_r123_data as dataio
import upv_r123_model as model
from upv_gas_access import AREA_COEFFICIENTS


def digest(scene):
    return hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _same(a,b):
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(_same(a[k],b[k])for k in a)
    if isinstance(a,list):
        return len(a)==len(b) and all(_same(x,y)for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-13)
    return a==b


def _contact_group(contact):
    port=contact['port_id']
    if port.startswith('chamber-wall:gas-partition:'):
        port='chamber-wall:gas-partition'
    elif port.startswith('opening-contact:'):
        port=port.split(':void-',1)[0]
    return port,contact.get('thermal_volume')


def _contact_groups(scene):
    ports={p['id']:p for p in scene['ports']};groups={}
    for contact in scene['gas_network']['thermal_contacts']:
        area=ports[contact['port_id']]['area_m2']
        law=copy.deepcopy(contact)
        for key in ('compartment','port_id','thermal_volume'):law.pop(key,None)
        for key in AREA_COEFFICIENTS:law['connectivity'][key]/=area
        group=_contact_group(contact)
        if group in groups and not _same(groups[group],law):
            raise ValueError('Nonuniform frozen gas/surface transfer group: '+str(group))
        groups[group]=law
    return groups


def _inventory_template(node):
    return {k:copy.deepcopy(v)for k,v in node.items()if k not in
            ('id','volume_m3','initial_mass_kg','initial_temperature_k')}


def _void_face(scene,edge):
    boxes=scene['metadata']['gas_geometry']['void_boxes']
    a,b=boxes[edge['from']],boxes[edge['to']]
    axes=[k for k in range(3)if abs(a['hi'][k]-b['lo'][k])<1e-10 or
          abs(b['hi'][k]-a['lo'][k])<1e-10]
    if len(axes)!=1:raise ValueError('Void route is not a shared face: '+edge['id'])
    axis=axes[0]
    plane=a['hi'][axis]if abs(a['hi'][axis]-b['lo'][axis])<1e-10 else b['hi'][axis]
    rectangle=[(max(a['lo'][k],b['lo'][k]),min(a['hi'][k],b['hi'][k]))for k in range(3)if k!=axis]
    return axis,plane,rectangle


def _void_restrictions(source,reference):
    baseline={e['id']:e for e in reference['gas_network']['channels']}
    result=[];axes=set()
    allowed={'area_m2','mixing_length_m','discharge_coefficient','mixing_axis'}
    for edge in source['gas_network']['channels']:
        if not edge['id'].startswith('void-face-'):continue
        axes.add(edge.get('mixing_axis'))
        base=baseline[edge['id']]
        changed={k for k in edge.keys()|base.keys()if edge.get(k)!=base.get(k)}
        if changed-allowed:raise ValueError('Unsupported modified free-space face field: '+str(changed-allowed))
        factors=tuple(edge[k]/base[k]for k in ('area_m2','mixing_length_m','discharge_coefficient'))
        if factors!=(1.,1.,1.):result.append((*_void_face(source,edge),factors))
    if len(axes)!=1:raise ValueError('Nonuniform explicit void axes need their own transfer rule')
    return result,axes.pop()


def _apply_void_restrictions(scene,edge,restrictions,legacy_axis):
    # Grid indices change when a plate adds cuts. Transfer corrections by
    # physical shared-face overlap, never by coincident void-N labels.
    axis,plane,rectangle=_void_face(scene,edge)
    nominal=math.prod(hi-lo for lo,hi in rectangle)
    covered=0.;area=flow=mixing=1.
    for other_axis,other_plane,patch,(a,l,cd)in restrictions:
        if axis!=other_axis or abs(plane-other_plane)>1e-10:continue
        overlap=math.prod(max(0.,min(hi,y)-max(lo,x))for (lo,hi),(x,y)in zip(rectangle,patch))
        fraction=overlap/nominal
        covered+=fraction;area+=fraction*(a-1.);flow+=fraction*(a*cd-1.);mixing+=fraction*(a/l-1.)
    if covered>1.+1e-9:raise ValueError('Overlapping frozen corrections on one gas face')
    # Parallel parts share p,T and S, so their Cd*A and A/L sum exactly. This
    # equivalent edge preserves both pressure flow and diffusive conductance.
    edge['area_m2']*=area
    edge['discharge_coefficient']*=flow/area
    edge['mixing_length_m']*=area/mixing
    if legacy_axis is None:edge.pop('mixing_axis',None)
    else:edge['mixing_axis']=legacy_axis


def build(source,experiment='active-cooling',duration=None,voltage_limited_setpoint=False):
    if source['metadata']['experiment']!='no-cooling':
        raise ValueError('Require a frozen training scene, never a validation-fitted scene')
    if experiment not in ('no-cooling','active-cooling'):
        raise ValueError('Unsupported UPV experiment')
    before=digest(source)
    data,record=dataio.load(experiment);event=dataio.event(data)
    training_data,_=dataio.load('no-cooling');training_event=dataio.event(training_data)
    end=duration if duration is not None else event['time_s']+source['simulation']['duration_s']-training_event['time_s']
    kwargs=dict(duration=end,gas_extensions=source['metadata']['gas_extensions'],
                gas_geometry=source['metadata']['gas_geometry']['configuration'])
    # Regeneration provides topology/reference dimensions only, never the fitted
    # reaction or transport parameters used by the replay.
    reference,_=model.build('no-cooling',source['metadata']['parameters'],**dict(kwargs,duration=source['simulation']['duration_s']))
    target,_=model.build(experiment,source['metadata']['parameters'],**kwargs)
    counts={}

    originals={c['id']:c for c in source['cells']}
    for index,current in enumerate(target['cells']):
        if current['id'] not in originals:
            # New geometry retains its documented dimensions/C/k, but an
            # existing material must not silently get an unfitted S=0 law.
            # Transfer only dimensionless material response, never heat capacity.
            keys=('surface_connectivity','bulk_connectivity','bulk_response')
            profiles=[{key:copy.deepcopy(old['thermal'][key])for key in keys if key in old['thermal']}
                      for old in source['cells']if old.get('material_id')==current.get('material_id')
                      and old.get('role')==current.get('role')]
            if profiles:
                if any(not _same(profiles[0],p)for p in profiles[1:]):
                    raise ValueError('Nonuniform frozen material profiles: '+current['material_id'])
                current['thermal'].update(copy.deepcopy(profiles[0]))
                counts.setdefault('new_objects_inheriting_material_profiles',[]).append(current['id'])
            continue
        old=originals[current['id']]
        if old['geometry']!=current['geometry'] or old['pose']!=current['pose']:
            raise ValueError('Frozen solid dimensions/positions changed: '+current['id'])
        replacement=copy.deepcopy(old)
        replacement['thermal']['initial_temperature_k']=current['thermal']['initial_temperature_k']
        if any(k.startswith('initial_')and k!='initial_temperature_k'for k in replacement['thermal']):
            raise ValueError('Do not reuse a nonuniform initialized training field')
        if 'electrical'in replacement:
            for k in ('current_profile_a','voltage_profile_v'):
                replacement['electrical'][k]=copy.deepcopy(current['electrical'][k])
            if voltage_limited_setpoint:
                if experiment!='active-cooling':raise ValueError('Never reinterpret the measured training current as a setpoint')
                replacement['electrical']['current_input']='voltage_limited_setpoint'
        target['cells'][index]=replacement
    shells={s['contents_id']:s for s in source['shells']}
    for index,current in enumerate(target['shells']):
        replacement=copy.deepcopy(shells[current['contents_id']])
        replacement['initial_temperature_k']=current['initial_temperature_k']
        if 'initial_temperature_fields_k'in replacement:raise ValueError('Nonuniform shell initialization')
        target['shells'][index]=replacement
    interfaces={e['id']:e for e in source['interfaces']}
    for index,current in enumerate(target['interfaces']):
        if current['id']in interfaces:target['interfaces'][index]=copy.deepcopy(interfaces[current['id']])
    boundaries={e['id']:e for e in source['boundaries']}
    for index,current in enumerate(target['boundaries']):
        if current['id'] not in boundaries:continue
        replacement=copy.deepcopy(boundaries[current['id']])
        replacement['temperature_k']=current['temperature_k']
        target['boundaries'][index]=replacement

    net=target['gas_network'];frozen=source['gas_network']
    for key,value in frozen.items():
        if key not in ('compartments','channels','thermal_contacts'):net[key]=copy.deepcopy(value)
    stock={n['id']:n for n in frozen['compartments']}
    free=[_inventory_template(n)for n in frozen['compartments']if n['id'].startswith('void-')]
    if not free or any(not _same(free[0],n)for n in free):
        raise ValueError('Spatially nonuniform frozen external inventory requires an explicit mapping')
    initial_pressure=2.5*float(data['U114_v'][0])*1e5
    for index,node in enumerate(net['compartments']):
        # A void-N label is a grid index, not a physical material identifier.
        template=free[0]if node['id'].startswith('void-')else _inventory_template(stock[node['id']])
        replacement=dict(copy.deepcopy(template),id=node['id'],volume_m3=node['volume_m3'],
                         initial_temperature_k=node['initial_temperature_k'])
        replacement['initial_mass_kg']=initial_pressure*node['volume_m3']/(
            net['specific_r_j_per_kg_k']*node['initial_temperature_k'])
        net['compartments'][index]=replacement

    baseline={e['id']:e for e in reference['gas_network']['channels']}
    channels={e['id']:e for e in frozen['channels']}
    restrictions,legacy_axis=_void_restrictions(source,reference)
    counts['frozen_modified_void_faces']=len(restrictions)
    for index,current in enumerate(net['channels']):
        if current['id'].startswith('void-face-'):
            _apply_void_restrictions(target,current,restrictions,legacy_axis)
            continue
        old=channels[current['id']];base=baseline[current['id']]
        replacement=copy.deepcopy(old)
        # Existing restrictions are dimensionless geometric fractions. New plate
        # obstructions/receiver positions come from the target's rebuilt grid.
        for key in ('from','to'):replacement[key]=current[key]
        for key in ('area_m2','mixing_length_m'):
            if key in old:
                replacement[key]=current[key]*(old[key]/base[key])
        if 'mixing_axis'not in old:replacement.pop('mixing_axis',None)
        net['channels'][index]=replacement
    counts['removed_blocked_nonvoid_routes']=sum(not e['id'].startswith('void-face-') and
        e['id']not in {v['id']for v in net['channels']}for e in frozen['channels'])

    groups=_contact_groups(source);ports={p['id']:p for p in target['ports']}
    for index,current in enumerate(net['thermal_contacts']):
        group=_contact_group(current)
        if group not in groups:raise ValueError('Unmapped new gas/surface law: '+str(group))
        replacement=copy.deepcopy(groups[group])
        for key in ('compartment','port_id','thermal_volume'):
            if key in current:replacement[key]=current[key]
        for key in AREA_COEFFICIENTS:
            replacement['connectivity'][key]*=ports[current['port_id']]['area_m2']
        net['thermal_contacts'][index]=replacement

    target['simulation']=dict(copy.deepcopy(source['simulation']),duration_s=end)
    target['measurement_probes']=copy.deepcopy(source['measurement_probes'])
    for key in ('thermal_path_trace','gas_film_trace','surface_probe_trace','process_history_selection'):
        if key in source:target[key]=copy.deepcopy(source[key])
    geometry_keys={'experiment','canonical_sha256','electrical','event_marker_diagnostic',
        'gas_geometry_controls','pad_openings','gas_pressure_observers','gas_geometry','gas_nodes'}
    target_meta=copy.deepcopy(target['metadata'])
    target['metadata']=copy.deepcopy(source['metadata'])
    target['metadata'].update({k:target_meta[k]for k in geometry_keys})
    target['metadata']['frozen_validation']=dict(frozen_scene_sha256=before,
        frozen_training_source='explicit scene; metadata defaults are not replay parameters',
        experiment=experiment,experimental_sha256=record['canonical_sha256'],
        parameters_refitted=False,validation_used_for_selection=False,
        changes=['Measured initial temperatures/pressure and electrical protocol',
                 '13 mm Al cooling plate + existing 10 W/K water-side prior; 2000 W/(m2 K) Al/plate contacts',
                 'Gas topology rebuilt around occupied plate; blocked paths stay blocked'],
        limitations=['Cooling contact and water-side G are unmeasured priors',
                     'Condition holdout within the same experimental campaign, not blind independent validation'],**counts)
    if digest(source)!=before:raise AssertionError('Frozen source mutated')
    target['metadata']['frozen_validation']['current_input_assumption']=('35 A protocol setpoint, voltage-limited actual current; no measured validation current log'
        if voltage_limited_setpoint else 'Prescribed current; strict UI consistency required')
    if voltage_limited_setpoint:
        target['metadata']['frozen_validation']['limitations'].append(
            'DIAGNOSTIC ONLY: measured terminal U is NOT the 80 V source compliance; reducing current violates the published constant-35 A protocol')
    return target,data

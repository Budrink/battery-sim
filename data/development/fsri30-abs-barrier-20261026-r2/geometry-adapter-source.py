"""Conservative fixture-only ABS opening hypothesis, NOT a measured material fit.

Only unambiguous existing inside/outside gas-film pairs are connected. A mixed
wall voxel is deliberately left closed rather than assigned a remote receiver
or given duplicate full-area apertures. The native aperture law owns pressure,
damage history and fracture work; there is no timed hole or new heat source.
"""
from collections import defaultdict
from copy import deepcopy
import math


def damage_abs_barrier(base, *, pressure_pa, onset_k, rate_per_s,
                       temperature_scale_k, area_rate_per_s, fracture_j_m2):
    values=(pressure_pa,onset_k,rate_per_s,temperature_scale_k,area_rate_per_s,fracture_j_m2)
    if any(not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('Declare positive finite diagnostic material parameters')
    if base.get('metadata',{}).get('abs_barrier_damage'):
        raise ValueError('ABS damage hypothesis already installed')
    scene=deepcopy(base);ports={p['id']:p for p in scene['ports']}
    thickness=scene['metadata']['geometry_budget']['abs_thickness_m']
    # These are the fixed reference-area constructions in build_resolved,
    # not fitted new film coefficients. The hot-contact adapter preserves G0.
    outer_h=1./(1./8.+thickness/(2.*.18))
    def exposed_area(contact,h):
        law=contact['connectivity'];z=law.get('geometry_reference_logit',0.)
        reference=law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-z))
        return reference/h
    contacts=defaultdict(list)
    for contact in scene['gas_network']['thermal_contacts']:
        port=ports.get(contact['port_id'])
        if port and port['cell_id'].startswith('ABS:'):
            contacts[port['cell_id'],contact['thermal_volume']].append((contact,port))
    connected=[];omitted=[];total_area=0.;connected_area=0.
    for wall in scene['cells']:
        if not wall['id'].startswith('ABS:'):continue
        if wall.get('role')!='passive' or wall.get('material_id')!='ABS':
            raise ValueError('ABS barrier must be an actual passive ABS solid')
        surface=wall['id'].split(':')[-1]
        if surface not in ('-x','+x','-y','+y','-z','+z'):
            raise ValueError('Unknown ABS wall orientation')
        axis='xyz'.index(surface[1]);sign=1. if surface[0]=='+'else -1.
        divisions=wall['thermal']['divisions'];dimensions=wall['geometry']['dimensions_m']
        if divisions[axis]!=1:raise ValueError('Barrier needs a one-volume-thick wall')
        count=math.prod(divisions);area=dimensions[(axis+1)%3]*dimensions[(axis+2)%3]/count
        total_area+=area*count
        center=list(wall['pose']['position_m']);center[axis]+=sign*dimensions[axis]/2
        normal=[0.,0.,0.];normal[axis]=sign
        port_id=f"abs-damage:{wall['id']}:outer"
        scene['ports'].append(dict(id=port_id,cell_id=wall['id'],kind='contact',surface=surface,
            centroid_m=center,normal=normal,area_m2=area*count,
            rectangle_m=[dimensions[(axis+1)%3],dimensions[(axis+2)%3]]))
        wall['degradation']=dict(aging_fraction=0.,initial_damage=0.,onset_k=onset_k,
            rate_at_onset_per_s=rate_per_s,temperature_scale_k=temperature_scale_k,
            aging_damage_multiplier=0.)
        for local in range(count):
            paired=contacts[wall['id'],local]
            outer=[c for c,p in paired if p['surface']==surface]
            inner=[c for c,p in paired if p['surface'][1]==surface[1]and p['surface'][0]!=surface[0]]
            whole=(len(outer)==len(inner)==1 and
                math.isclose(exposed_area(outer[0],outer_h),area,rel_tol=1e-9) and
                math.isclose(exposed_area(inner[0],8.),area,rel_tol=1e-9))
            if not whole:
                omitted.append(dict(wall=wall['id'],thermal_volume=local,
                    inside_contacts=len(inner),outside_contacts=len(outer),area_m2=area,
                    reason='Ambiguous neighbours or only partial gas exposure of a wall voxel'))
                continue
            source=inner[0]['compartment'];target=outer[0]['compartment']
            if source==target:raise ValueError('ABS wall contacts identify the same gas volume')
            edge=dict(id=f"abs-opening:{wall['id']}:{local}",kind='rupture',port_id=port_id,
                shell_thermal_volume=local,**{'from':source,'to':target},
                discharge_coefficient=.7,opening_pressure_pa=pressure_pa,
                mixing_axis=axis,mixing_length_m=dimensions[axis],
                damage_opening=dict(mode='deterministic_aperture',area_rate_per_s=area_rate_per_s,
                    pressure_exponent=1.,fracture_energy_j_per_m2=fracture_j_m2,max_damage_step=.005))
            scene['gas_network']['channels'].append(edge)
            connected.append(dict(wall=wall['id'],thermal_volume=local,
                source=source,target=target,area_m2=area))
            connected_area+=area
    if not connected:raise ValueError('No unambiguous physical ABS gas crossings')
    scene.setdefault('metadata',{})['abs_barrier_damage']=dict(
        status='unmeasured geometric/material hypothesis; diagnostic only',
        calibrated=False,validation_passed=False,model='existing leading deterministic aperture',
        parameters=dict(pressure_pa=pressure_pa,onset_k=onset_k,rate_per_s=rate_per_s,
            temperature_scale_k=temperature_scale_k,area_rate_per_s=area_rate_per_s,
            fracture_j_m2=fracture_j_m2),
        connected_area_m2=connected_area,total_abs_face_area_m2=total_area,
        connected=connected,omitted_ambiguous=omitted,
        exposure_area_provenance='build_resolved: inside h0=8; outside h0 includes half-ABS resistance; no hot-response fit',
        limitation='Mixed or partially gas-exposed patches remain closed; no full-area duplicates or guessed remote routes',
        frozen='Battery chemistry, inventories, solid Cp/k, existing heat links, mesh and numerical floor')
    return scene

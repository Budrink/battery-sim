"""Join only abutting ABS panels; native material halves set their resistance.

No fitted h, can-to-can bridge, gas route, new mass or energy source. The tiny
reference contact resistance is a numerical approximation to continuous ABS.
"""
from copy import deepcopy
from itertools import combinations
import math


def join_abs_panels(base, *, expected_pairs=12):
    scene=deepcopy(base)
    if 'abs_seam_continuity' in scene.get('metadata',{}):
        raise ValueError('ABS seams already reconstructed')
    panels=[c for c in scene['cells'] if c['id'].startswith('ABS:')]
    changed=[]
    for a,b in combinations(panels,2):
        for c in (a,b):
            if c['material_id']!='ABS' or c.get('pose',{}).get('orientation_xyzw',[0,0,0,1])!=[0,0,0,1]:
                raise ValueError('Only aligned same-material ABS panels supported')
        da=a['geometry']['dimensions_m'];db=b['geometry']['dimensions_m']
        pa=a['pose']['position_m'];pb=b['pose']['position_m']
        lo=[max(pa[k]-da[k]/2,pb[k]-db[k]/2) for k in range(3)]
        hi=[min(pa[k]+da[k]/2,pb[k]+db[k]/2) for k in range(3)]
        overlap=[hi[k]-lo[k] for k in range(3)]
        if all(x>1e-11 for x in overlap):
            raise ValueError('Overlapping ABS material volumes')
        normals=[k for k in range(3) if abs(overlap[k])<1e-11 and
                 all(overlap[j]>1e-11 for j in range(3) if j!=k)]
        if not normals:continue
        axis=normals[0];u,v=(axis+1)%3,(axis+2)%3
        area=overlap[u]*overlap[v];point=[(lo[k]+hi[k])/2 for k in range(3)]
        ident='abs-seam:'+a['id']+'--'+b['id']
        ids=[]
        for cell,other in ((a,b),(b,a)):
            sign=1 if other['pose']['position_m'][axis]>cell['pose']['position_m'][axis] else -1
            normal=[0.,0.,0.];normal[axis]=sign
            port=ident+':'+cell['id'];ids.append(port)
            scene['ports'].append(dict(id=port,cell_id=cell['id'],kind='contact',
                surface=('+' if sign>0 else '-')+'xyz'[axis],area_m2=area,
                rectangle_m=[overlap[u],overlap[v]],centroid_m=point,normal=normal))
        # Native mesher adds both material half-cell resistances per overlap.
        # At reference state Rc/Rmaterial=1e-6, not a calibration parameter.
        half_r=sum(c['geometry']['dimensions_m'][axis]/c['thermal']['divisions'][axis]/
                   (2*c['thermal']['conductivity_w_per_m_k'][axis]) for c in (a,b))
        if not math.isfinite(half_r) or half_r<=0:raise ValueError('Invalid ABS conductivity')
        reference=area/half_r*1e6
        law=dict(kind='directed_geometry',geometry_reference_logit=0.,
            background_conductance_w_per_k=.01*reference,
            active_conductance_w_per_k=1.98*reference,
            destroyed_background_conductance_w_per_k=.01*reference,
            destroyed_active_conductance_w_per_k=1.98*reference,endpoint_damage_weight=0.)
        scene['interfaces'].append(dict(id=ident,port_a=ids[0],port_b=ids[1],
            material_a='ABS',material_b='ABS',interface_material='continuous-ABS',law=law))
        changed.append(dict(interface=ident,objects=[a['id'],b['id']],area_m2=area,
            normal_axis=axis,reference_material_conductance_w_k=area/half_r,
            contact_to_material_resistance_ratio=1e-6))
    if len(changed)!=expected_pairs:
        raise ValueError(f'Expected {expected_pairs} actual ABS seams, found {len(changed)}')
    scene.setdefault('metadata',{})['abs_seam_continuity']=dict(added=changed,
        interpretation='Continuous same-material casing; mechanical assembly quality not measured',
        material_and_chemistry_frozen=True,can_contacts_added=False,
        gas_geometry_changed=False,calibrated=False,validation_passed=False)
    return scene

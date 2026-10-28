"""Exploratory fixed metal-can contact strips for the 30-cell FSRI fixture.

The drawing does not measure contact pressure, width, or lateral spacing.  The
existing positions are a digitisation, not CAD: adjacency is interpreted as a
packed-contact hypothesis.  Effective planar ports map those cylindrical
contacts onto the existing equal-volume prismatic shell thermal mesh.
"""
from copy import deepcopy
import math

from generate_fsri30 import connectivity


def add_can_contacts(base, width_m=0.001, contact_h=1000.):
    if not 0 < width_m <= .003 or not 10 <= contact_h <= 10000:
        raise ValueError('Exploratory strip width or contact conductance out of bounds')
    scene = deepcopy(base)
    if 'can_contact_hypothesis' in scene['metadata']:
        raise ValueError('Can contacts already present')
    cells = {c['id']: c for c in scene['cells'] if c['role'] == 'battery'}
    shells = {s['contents_id']: s for s in scene['shells']}
    heater = scene['metadata']['heated_component']
    records, occupied = [], {}
    for ia, ib in scene['metadata']['neighbour_pairs']:
        a, b = f'B{ia:02d}', f'B{ib:02d}'
        ca, cb = cells[a], cells[b]
        pa, pb = ca['pose']['position_m'], cb['pose']['position_m']
        # Adjacent columns use x-facing patches, same-column neighbours y.
        axis = 0 if abs(pb[0]-pa[0]) > 1e-9 else 1
        sign = 1 if pb[axis] > pa[axis] else -1
        face_a = ('+' if sign > 0 else '-') + 'xy'[axis]
        face_b = ('-' if sign > 0 else '+') + 'xy'[axis]
        if (a == heater and face_a == '-x') or (b == heater and face_b == '-x'):
            # Heater is a physical layer, not a naked metal-metal contact.
            continue
        u, v = (axis+1)%3, (axis+2)%3
        def panel(name, face):
            c, t = cells[name], shells[name]['thickness_m']
            size = [t if k == axis else c['geometry']['dimensions_m'][k] + (2*t if k > axis else 0.) for k in range(3)]
            pos = list(c['pose']['position_m'])
            pos[axis] += (1 if face[0] == '+' else -1)*(c['geometry']['dimensions_m'][axis]+t)/2
            return size, pos
        da, sa = panel(a, face_a); db, sb = panel(b, face_b)
        # Common projected rectangle is required by the native contact mesher.
        center = [0., 0., 0.]; extent = [0., 0., 0.]
        for k in (u, v):
            lo = max(sa[k]-da[k]/2, sb[k]-db[k]/2)
            hi = min(sa[k]+da[k]/2, sb[k]+db[k]/2)
            if hi <= lo: raise ValueError('No projected overlap for '+a+'--'+b)
            extent[k] = min(hi-lo, width_m if k != 2 else .065)
            center[k] = (lo+hi)/2
        area = extent[u]*extent[v]
        ports = []
        for name, face, dims, pos in ((a, face_a, da, sa), (b, face_b, db, sb)):
            point = list(center); normal = [0., 0., 0.]
            normal[axis] = 1 if face[0] == '+' else -1
            point[axis] = pos[axis]+normal[axis]*dims[axis]/2
            port_id = f'can-contact:{a}:{b}:{name}'
            scene['ports'].append(dict(id=port_id, cell_id=f'{name}:case:{face}', kind='contact',
                surface=face, area_m2=area, rectangle_m=[extent[u],extent[v]], centroid_m=point, normal=normal))
            ports.append(port_id)
            key = (name,face)
            for old in occupied.get(key, []):
                if all(min(center[k]+extent[k]/2,old[0][k]+old[1][k]/2) >
                       max(center[k]-extent[k]/2,old[0][k]-old[1][k]/2)+1e-12 for k in (u,v)):
                    raise ValueError('Overlapping contact strips on '+str(key))
            occupied.setdefault(key, []).append((list(center),list(extent),area,dims[u]*dims[v]))
        scene['interfaces'].append(dict(id=f'can-contact:{a}:{b}',port_a=ports[0],port_b=ports[1],
            material_a='steel-can-PROXY',material_b='steel-can-PROXY',
            interface_material='unmeasured-metal-metal-strip',law=connectivity(area,contact_h)))
        records.append(dict(a=a,b=b,face_a=face_a,face_b=face_b,area_m2=area,
                            nominal_conductance_w_k=area*contact_h))
    # Exclude metal-to-metal strips from the gas-facing film area. The native
    # full-face film remains distributed over its face, so this is area-averaged.
    for (name,face), patches in occupied.items():
        exposed = 1. - sum(p[2] for p in patches)/patches[0][3]
        if not 0 < exposed <= 1: raise ValueError('Invalid exposed film area')
        key = f'{name}:case:{face}:outer'
        films = [f for f in scene['gas_network']['thermal_contacts'] if f['port_id'] == key]
        if not films: raise ValueError('Missing gas film '+key)
        for film in films:
            for attr in film['connectivity']:
                if attr.endswith('conductance_w_per_k'): film['connectivity'][attr] *= exposed
    packed='packed_cylinder_geometry' in scene['metadata']
    scene['metadata']['can_contact_hypothesis'] = dict(width_m=width_m,contact_h_w_m2_k=contact_h,
        width_measured=False,contact_h_measured=False,contact_pairs=records,
        excluded_heater_side=True,gas_film_exposure='area-averaged strip exclusion',
        positions='ideal tangential close packing; unmeasured' if packed else 'digitised drawing; assumed close packed; no physical gaps validated',
        gas_passages=('tangent pockets resolved; each pocket well mixed' if 'packed_pocket_geometry' in scene['metadata'] else
                      'rebuilt for packed positions; body pockets lumped') if packed else 'retained prior geometry; not resolved for zero-gap contacts',
        validation_passed=False)
    return scene

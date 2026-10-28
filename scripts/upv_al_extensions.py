"""Bottom Al spreader extensions, with conservative cut-cell free volume.

Figure 2 shows protruding plates but does not specify their length. This is an
explicit geometry hypothesis, not a material fit. Reuse the current scenario's
constitutive laws; never rebuild its chemistry or gas network from defaults.
New solid mass requires a fresh run from ambient, not a gas-only checkpoint.
"""
import copy
import math

from gas_void_geometry import cell_box
from upv_gas_access import AREA_COEFFICIENTS

EPS = 1e-10


def overlap_volume(a, b):
    return math.prod(max(0., min(a[1][k], b[1][k]) - max(a[0][k], b[0][k]))
                     for k in range(3))


def face_overlap(a, b, axis):
    """Tangential intersection of two boxes, or None."""
    uv = ((axis + 1) % 3, (axis + 2) % 3)
    lo = [max(a[0][k], b[0][k]) for k in uv]
    hi = [min(a[1][k], b[1][k]) for k in uv]
    if min(h-l for l, h in zip(lo, hi)) <= EPS:
        return None
    return uv, lo, hi


def port(ident, owner, axis, sign, plane, uv, lo, hi):
    center = [0.] * 3
    center[axis] = plane
    for k, l, h in zip(uv, lo, hi):
        center[k] = .5 * (l + h)
    size = [h-l for l, h in zip(lo, hi)]
    normal = [0.] * 3
    normal[axis] = sign
    return dict(id=ident, cell_id=owner, kind='contact',
        surface=('-' if sign < 0 else '+') + 'xyz'[axis], area_m2=math.prod(size),
        rectangle_m=size, centroid_m=center, normal=normal)


def cut_free_volume(scene, obstacles):
    """Only subtract new solid: keep ambient gas density and pressure unchanged.

    A plate ends inside an old bottom-band volume. The remaining volume connects
    below its tip; shared-face area is reduced wherever a plate blocks a face.
    No completely blocked nodes or new arbitrary gas bypasses are introduced.
    """
    meta = scene['metadata']['gas_geometry']
    boxes = {key: (v['lo'], v['hi']) for key, v in meta['void_boxes'].items()}
    occupied, removed = {}, []
    for node in scene['gas_network']['compartments']:
        if node['id'] not in boxes:
            continue
        volume = sum(overlap_volume(boxes[node['id']], box) for box in obstacles)
        if volume <= 1e-18:
            continue
        old = node['volume_m3']
        if volume >= old * (1. - 1e-9):
            raise ValueError('Extension completely blocks a gas volume; remesh required')
        node['volume_m3'] = old - volume
        node['initial_mass_kg'] *= (old - volume) / old
        occupied[node['id']] = volume
    expected = sum(math.prod(h-l for l, h in zip(*b)) for b in obstacles)
    if not math.isclose(sum(occupied.values()), expected, rel_tol=1e-8, abs_tol=1e-15):
        raise ValueError('New solid is not wholly inside existing free space')
    channels = []
    narrowed = []
    for edge in scene['gas_network']['channels']:
        if not edge['id'].startswith('void-face-'):
            channels.append(edge)
            continue
        a, b = boxes[edge['from']], boxes[edge['to']]
        shared = [(k, a[1][k]) for k in range(3) if abs(a[1][k]-b[0][k]) < EPS]
        shared += [(k, a[0][k]) for k in range(3) if abs(a[0][k]-b[1][k]) < EPS]
        if len(shared) != 1:
            raise ValueError('Non-face adjacent free volumes')
        axis, plane = shared[0]
        face = ([max(a[0][k], b[0][k]) for k in range(3)],
                [min(a[1][k], b[1][k]) for k in range(3)])
        blocked = 0.
        for box in obstacles:
            if box[0][axis]-EPS <= plane <= box[1][axis]+EPS:
                patch = face_overlap(face, box, axis)
                if patch:
                    _, lo, hi = patch
                    blocked += math.prod(h-l for l, h in zip(lo, hi))
        if blocked:
            old = edge['area_m2']
            if blocked > old + 1e-12:
                raise ValueError('Blocked area exceeds gas face')
            narrowed.append(dict(channel=edge['id'], blocked_area_m2=blocked))
            if old-blocked <= 1e-15:
                removed.append(edge['id'])
                continue
            edge['area_m2'] = old-blocked
        channels.append(edge)
    scene['gas_network']['channels'] = channels
    meta['external_volume_m3'] -= expected
    meta['solid_cutouts'] = [dict(lo=a, hi=b) for a, b in obstacles]
    return dict(displaced_volume_m3=expected, occupied_voids_m3=occupied,
                narrowed_faces=narrowed, removed_channels=removed)


def candidate(source, length_m):
    if not math.isfinite(length_m) or not 0. <= length_m <= .015:
        raise ValueError('Al extension outside [0, 15] mm')
    scene = copy.deepcopy(source)
    if length_m == 0.:
        return scene
    if 'al_bottom_extensions' in scene['metadata']:
        raise ValueError('Extensions already applied')
    cells = {c['id']: c for c in scene['cells']}
    ports = {p['id']: p for p in scene['ports']}
    gas = scene['gas_network']
    exposures = {r['port']: r['gas_exposed_area_m2']
                 for r in scene['metadata']['gas_access_geometry']['backing_exposure']}
    template = next(c for c in gas['thermal_contacts']
        if ports[c['port_id']]['cell_id'] == 'Al1'
        and c['port_id'] in exposures and 'thermal_volume' not in c)
    template_area = exposures[template['port_id']]
    boxes = scene['metadata']['gas_geometry']['void_boxes']
    extensions, new_contacts = [], []
    gas['thermal_contacts'] = [c for c in gas['thermal_contacts']
        if not (ports[c['port_id']]['cell_id'] in ('Al1', 'Al2', 'Al3')
                and ports[c['port_id']]['surface'] == '-z')]
    for owner in ('Al1', 'Al2', 'Al3'):
        original = cells[owner]
        old_box = cell_box(original)
        flange = copy.deepcopy(original)
        flange['id'] = owner + ':bottom-extension'
        flange['geometry']['dimensions_m'][2] = length_m
        flange['pose']['position_m'][2] = old_box[0][2] - .5*length_m
        flange['thermal']['heat_capacity_j_per_k'] *= length_m / original['geometry']['dimensions_m'][2]
        flange['thermal']['divisions'] = [1, original['thermal']['divisions'][1], 1]
        scene['cells'].append(flange)
        box = cell_box(flange)
        extensions.append(box)
        # A monolithic bond; half-cell aluminium resistances remain in series.
        bond_ports = [port(owner+':extension-bond', owner, 2, -1, old_box[0][2],
                           (0, 1), box[0][:2], box[1][:2]),
                      port(flange['id']+':bond', flange['id'], 2, 1, box[1][2],
                           (0, 1), box[0][:2], box[1][:2])]
        scene['ports'].extend(bond_ports)
        conductance = 1e8 * bond_ports[0]['area_m2']
        scene['interfaces'].append(dict(id=owner+'--extension',
            port_a=bond_ports[0]['id'], port_b=bond_ports[1]['id'],
            law=dict(kind='directed_geometry', geometry_reference_logit=0.,
                background_conductance_w_per_k=conductance, active_conductance_w_per_k=0.,
                endpoint_damage_weight=1., destroyed_geometry_reference_logit=0.,
                destroyed_background_conductance_w_per_k=conductance,
                destroyed_active_conductance_w_per_k=0.)))
        for axis in range(3):
            for sign in (-1, 1):
                if axis == 2 and sign == 1:
                    continue  # Bonded face is not gas exposed.
                plane = box[0 if sign < 0 else 1][axis]
                total = 0.
                for node, bounds in boxes.items():
                    outside = plane + sign*EPS
                    if not bounds['lo'][axis] < outside < bounds['hi'][axis]:
                        continue
                    patch = face_overlap(box, (bounds['lo'], bounds['hi']), axis)
                    if patch is None:
                        continue
                    uv, lo, hi = patch
                    p = port(f'{flange["id"]}:{axis}:{sign}:{node}', flange['id'],
                             axis, sign, plane, uv, lo, hi)
                    scene['ports'].append(p)
                    contact = copy.deepcopy(template)
                    contact.update(compartment=node, port_id=p['id'])
                    for key in AREA_COEFFICIENTS:
                        contact['connectivity'][key] *= p['area_m2'] / template_area
                    new_contacts.append(contact)
                    total += p['area_m2']
                expected = math.prod(box[1][k]-box[0][k] for k in range(3) if k != axis)
                if not math.isclose(total, expected, rel_tol=1e-8, abs_tol=1e-15):
                    raise ValueError('Incomplete or overlapping flange surface coverage')
    gas['thermal_contacts'].extend(new_contacts)
    cutouts = cut_free_volume(scene, extensions)
    scene['metadata']['al_bottom_extensions'] = dict(length_m=length_m, **cutouts,
        new_gas_contacts=len(new_contacts), material_properties_unchanged=True,
        gas_contact_law_source=template['port_id'], requires_fresh_from_ambient=True,
        length_measured=False, free_volume_discretization='partial bottom-band volumes',
        source='UPV data description, Fig. 2: protrusion shown, extent unspecified')
    return scene

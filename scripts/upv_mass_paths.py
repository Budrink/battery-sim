"""Explicit reduced pressure geometry for the frozen UPV stack.

No solid material/contact parameters are fitted here. Surface-gap nodes are
subgrid gas adjacent to the existing pad volumes, not pores through silicone.
Their local LTE is the previously declared ideal-contact approximation.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'data/calibration/upv-training-refit-v13'
SOURCE = 'https://zenodo.org/records/22273467/files/TR_Experimental_Data_Description.pdf'
PARAMETERS = ('energy_j', 'gas_fraction', 'solid_rate', 'gap_m')
BOUNDS = dict(energy_j=(150000., 600000.), gas_fraction=(.2, .85), solid_rate=(.5, 15.), gap_m=(.00005, .001))
FIXED = dict(near_extent_m=.02, internal_open_fraction=.03, internal_cd=.7,
             gap_cd=.35, near_cd=.7, reactant_density=2000., residue_density=2000.)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def thermal_signature(scene):
    return digest(dict(cells=[{k: v for k, v in c.items() if k != 'conversion'} for c in scene['cells']],
                       **{k: scene[k] for k in ('shells', 'interfaces', 'ports', 'boundaries', 'measurement_probes')}))


def initial():
    p = read(PRIOR / 'profile.json')['parameters']
    total = p['solid_energy_j'] + p['gas_energy_j']
    return dict(energy_j=total, gas_fraction=p['gas_energy_j']/total, solid_rate=p['solid_rate'], gap_m=.0002)


def inputs(name):
    folder = PRIOR if name == 'no-cooling' else PRIOR / 'cooling-check/nominal'
    return (read(folder / ('no-cooling-fit-scenario.json' if name == 'no-cooling' else 'scenario.json')),
            read(folder / ('fit.json' if name == 'no-cooling' else 'report.json')))


def field(cell):
    t = cell['thermal']
    return t.get('initial_temperature_field_k', [t['initial_temperature_k']] * math.prod(t['divisions']))


def index(x, y, z, dims):
    return x + dims[0]*(y + dims[1]*z)


def attach(scene, parameters):
    if set(parameters) != set(PARAMETERS):
        raise ValueError('Only declared destruction/mass-path parameters may change')
    for k, (lo, hi) in BOUNDS.items():
        if not math.isfinite(parameters[k]) or not lo <= parameters[k] <= hi:
            raise ValueError('Out of bounds: '+k)
    s = copy.deepcopy(scene); frozen = thermal_signature(s)
    cells = {c['id']: c for c in s['cells']}; ports = {p['id']: p for p in s['ports']}
    gas = s['gas_network']; old = {g['id']: g for g in gas['compartments']}
    inside, chamber = old['inside-C2'], copy.deepcopy(old['chamber'])
    gas['transport'] = 'split_local_equilibrium'
    nodes, paths = [], []
    r, cv, tref = gas['specific_r_j_per_kg_k'], gas['cv_j_per_kg_k'], gas['reference_temperature_k']
    p0 = chamber['initial_mass_kg']*r*chamber['initial_temperature_k']/chamber['volume_m3']

    def node(name, volume, temperature, binding=None, external=True):
        n = dict(id=name, volume_m3=volume, initial_mass_kg=p0*volume/(r*temperature), initial_temperature_k=temperature)
        if binding is not None:
            n['local_equilibrium'] = binding
        if external:
            for k in ('initial_oxidizer_mass_fraction', 'reaction_rate_per_s', 'oxidizer_heat_j_per_kg'):
                n[k] = chamber[k]
        nodes.append(n)
        return n

    def path(name, a, b, area, port=None, cd=.7):
        q = dict(id=name, kind='vent' if port else 'internal', **{'from': a, 'to': b},
                 area_m2=area, discharge_coefficient=cd, opening_pressure_pa=0.)
        if port:
            q['port_id'] = port
        paths.append(q)

    # 3^3 source pores: conversion products stay local before pressure transport.
    source = cells['C2']; n = source['thermal']['divisions']; sizes = [d/k for d, k in zip(source['geometry']['dimensions_m'], n)]
    count = math.prod(n); temperatures = field(source)
    source_ids = ['inside-C2' if i == 0 else f'inside-C2-{i}' for i in range(count)]
    for i, temp in enumerate(temperatures):
        q = node(source_ids[i], inside['volume_m3']/count, temp,
                 dict(cell_id='C2', thermal_volume=i, reactant_density_kg_m3=FIXED['reactant_density'],
                      residue_density_kg_m3=FIXED['residue_density']), external=False)
        q['initial_mass_kg'] = inside['initial_mass_kg']/count
        if i == 0:
            q['contents_id'] = 'C2'
    for z in range(n[2]):
        for y in range(n[1]):
            for x in range(n[0]):
                xyz = [x, y, z]; i = index(*xyz, n)
                for axis in range(3):
                    if xyz[axis]+1 == n[axis]:
                        continue
                    other = xyz.copy();other[axis] += 1;j = index(*other, n)
                    area = sizes[(axis+1)%3]*sizes[(axis+2)%3]*FIXED['internal_open_fraction']
                    path(f'core-{i}-{j}', source_ids[i], source_ids[j], area, cd=FIXED['internal_cd'])

    # Four exterior near-field sectors. No path traverses an intact plate/pad.
    # Source-centred x span of the original pressure plates, not the cooling plate.
    left, right = cells['steel-L'], cells['steel-R']
    span = right['pose']['position_m'][0]+right['geometry']['dimensions_m'][0]/2 - (left['pose']['position_m'][0]-left['geometry']['dimensions_m'][0]/2)
    width, length = cells['ISS2']['geometry']['dimensions_m'][1:]; extent = FIXED['near_extent_m']
    for axis, transverse in [('y', length), ('z', width+2*extent)]:
        for sign in ('-', '+'):
            face = sign+axis; name = 'near-'+face
            node(name, span*transverse*extent, chamber['initial_temperature_k'])
            # The declared port anchors the passage direction. Its area is
            # conservatively limited to this existing source-side opening.
            port = 'C2:'+face
            if face == '-z' and 'cooling-plate' in cells:
                # The plate closes the bottom of the stack. The reduced bottom
                # plenum drains around its two lateral edges, not through metal.
                for side in ('-y', '+y'):
                    edge_port = 'C2:'+side
                    path(name+'-around-'+side, name, 'near-'+side,
                         min(span*extent, ports[edge_port]['area_m2']), edge_port, FIXED['near_cd'])
            else:
                path(name+'-chamber', name, 'chamber', ports[port]['area_m2'], port, FIXED['near_cd'])

    gap = parameters['gap_m']; gap_nodes = {}
    for pad, face in [(f'{material}{i}', face) for i in (1, 2, 3)
                      for material, face in [('ISS', '-x'), ('TIM', '+x')]]:
        c = cells[pad]; div = c['thermal']['divisions']; dims = c['geometry']['dimensions_m']
        if div[0] != 1:
            raise ValueError('Surface gas mapping requires one thickness volume in the pad')
        area = dims[1]*dims[2]; inlet = f'gap-{pad}-inlet' if pad.endswith('2') else None
        # Small entrance manifold; it is not thermally bound and does not
        # bypass the receiving gap patches to reach the chamber.
        if inlet:
            node(inlet, area*gap*.1, chamber['initial_temperature_k'])
            gap_nodes[face] = inlet
        ids = []
        for i, temp in enumerate(field(c)):
            name = f'gap-{pad}-{i}';ids.append(name)
            node(name, area*gap*(.9 if inlet else 1.)/math.prod(div), temp, dict(cell_id=pad, thermal_volume=i))
            if inlet:
                receiving_port = pad+(':+x' if face == '-x' else ':-x')
                path(f'feed-{pad}-{i}', inlet, name, area/math.prod(div), receiving_port, FIXED['gap_cd'])
            y, z = i % div[1], i // div[1]
            for axis, coord, limit, cross in [('y', y, div[1], dims[2]/div[2]), ('z', z, div[2], dims[1]/div[1])]:
                for sign, boundary in [('-', 0), ('+', limit-1)]:
                    if coord == boundary:
                        port = pad+':'+sign+axis
                        path(f'escape-{pad}-{i}-{sign}{axis}', name, 'near-'+sign+axis,
                             min(gap*cross, ports[port]['area_m2']), port, FIXED['gap_cd'])
        # In-plane paths, NEVER through the pad into the next layer.
        for z in range(div[2]):
            for y in range(div[1]):
                i = y + div[1]*z
                if y+1 < div[1]:
                    path(f'gap-{pad}-{i}-y', ids[i], ids[i+1], gap*dims[2]/div[2], cd=FIXED['gap_cd'])
                if z+1 < div[2]:
                    path(f'gap-{pad}-{i}-z', ids[i], ids[i+div[1]], gap*dims[1]/div[1], cd=FIXED['gap_cd'])

    for old_channel in gas['channels']:
        ch = copy.deepcopy(old_channel);ch.pop('surface_link', None)
        face = ch['port_id'].split(':')[2];axis = 'xyz'.index(face[1]);ijk = [k//2 for k in n];ijk[axis] = 0 if face[0] == '-' else n[axis]-1
        ch['from'] = source_ids[index(*ijk, n)]
        ch['to'] = gap_nodes[face] if face[1] == 'x' else 'near-'+face
        paths.append(ch)

    # Split the external reservoir rather than adding another inventory on top.
    ext = nodes[count:]; added_volume = sum(q['volume_m3'] for q in ext)
    added_mass = sum(q['initial_mass_kg'] for q in ext)
    added_u = sum(q['initial_mass_kg']*cv*(q['initial_temperature_k']-tref) for q in ext)
    old_u = chamber['initial_mass_kg']*cv*(chamber['initial_temperature_k']-tref)
    chamber['volume_m3'] -= added_volume;chamber['initial_mass_kg'] -= added_mass
    chamber['initial_temperature_k'] = tref+(old_u-added_u)/(chamber['initial_mass_kg']*cv)
    if min(chamber['volume_m3'], chamber['initial_mass_kg'], chamber['initial_temperature_k']) <= 0:
        raise ValueError('Local gas volumes exceed chamber inventory')
    nodes.append(chamber)
    gas['compartments'], gas['channels'] = nodes, paths
    # Existing h values stay frozen; only the gas-side neighbour is local now.
    for contact in gas['thermal_contacts']:
        face = ports[contact['port_id']]['surface']
        if contact['compartment'] == 'chamber' and face[1] in 'yz' and contact['port_id'] != 'wall:inner':
            contact['compartment'] = 'near-'+face
    conversion = source['conversion']
    conversion['net_heat_j_per_kg'] = parameters['energy_j']/conversion['mass_kg']
    conversion['gas_chemical_fraction'] = parameters['gas_fraction']
    conversion['activation_rate_per_s'] = conversion['reaction_rate_per_s'] = parameters['solid_rate']
    if thermal_signature(s) != frozen:
        raise ValueError('Frozen thermal geometry, laws, sensors or initial fields changed')
    s['metadata']['mass_paths'] = dict(source=SOURCE, thermal_sha256=frozen, parameters=parameters, fixed_priors=FIXED,
        gas_nodes=len(nodes), pressure_paths=len(paths), source_local_volumes=count, local_external_volume_m3=added_volume,
        chamber_volume_conservation_m3=chamber['volume_m3']+added_volume,
        bottom_route='around cooling plate edges' if 'cooling-plate' in cells else 'open perimeter',
        limitations=['Gap heights, holder blockage and actual rupture location are not measured',
                     'Gap gas uses local equilibrium with adjacent pad voxel, not permeability through silicone',
                     'Near-field sectors are reduced gas volumes, not CFD jets',
                     'Legacy sensor numbering hypothesis and thermal fields are frozen, not re-fitted'])
    return s

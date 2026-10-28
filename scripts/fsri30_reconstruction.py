"""Source-constrained FSRI geometry and explicitly scoped UPV R2 transfer.

The solver, inventories, R1/R3 and can rupture laws are not changed here.
Polygon partitions describe connected free space, not a CFD velocity field.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from fsri30_data import ROOT, sha
from generate_fsri30 import BOX, DIAMETER, HEIGHT, WIDTH, connectivity

UPV = ROOT/'data/calibration/upv-separated-chemistry-v6/final-training-calibration-20261022-r3/reverse-cooling-fresh-20261023-r1/trial-000'
UPV_SHA = '747c5aefc5932b1759ee48f6042692907f3992d0719190ba44c3135404b3b210'
R2_KEYS = ('reaction_rate_per_s', 'reaction_reference_k', 'reaction_activation_j_per_mol')
ARTICLE = 'https://doi.org/10.1016/j.dib.2022.108712'


def transfer_r2(base, source=UPV):
    """Copy only k_ref, T_ref, Ea from the accepted ACTUAL three-cell scene."""
    source = Path(source)
    if sha(source/'scenario.json') != UPV_SHA:
        raise ValueError('Accepted UPV scene checksum changed')
    assessment = json.loads((source/'assessment.json').read_text())
    if not assessment.get('complete') or not assessment.get('accepted'):
        raise ValueError('UPV source is not a completed accepted training replay')
    old = json.loads((source/'scenario.json').read_text())
    laws = [{k: c['conversion'][k] for k in R2_KEYS}
            for c in old['cells'] if c['role'] == 'battery']
    if len(laws) != 3 or any(law != laws[0] for law in laws):
        raise ValueError('Expected one shared R2 law in all three UPV batteries')
    scene = deepcopy(base)
    for cell in scene['cells']:
        if cell['role'] == 'battery':
            cell['conversion'].update(laws[0])
    scene['metadata']['assumptions'] = [
        'No imposed ignition or target-temperature forcing; R2 comes only from the explicitly pinned UPV training scene'
        if 'UPV fitted coefficients' in note else note
        for note in scene['metadata'].get('assumptions', [])]
    scene['metadata'].pop('chemistry_heater_materials_frozen', None)
    scene['metadata']['r2_transfer'] = dict(source=str(source.relative_to(ROOT)),
        source_scenario_sha256=UPV_SHA, parameters=laws[0],
        transferred_fields=list(R2_KEYS), source_training_accepted=True,
        source_independent_validation_passed=False,
        chemistry_equivalence_confirmed=False,
        interpretation='Cross-cell-family hypothesis, not measured 18650 chemistry; no inventory/energy/yield/R1/R3/shell transfer')
    return scene


def clip(poly, normal, offset):
    """Convex polygon intersected with n.x <= b, no Qhull infinite vertices."""
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = float(a@normal-offset), float(b@normal-offset)
        if da <= 1e-14:
            out.append(a)
        if (da < 0.) != (db < 0.):
            out.append(a+(b-a)*(da/(da-db)))
    if len(out) < 3:
        raise ValueError('Empty gas control polygon')
    return np.array(out)


def polygon_area(poly):
    return float(abs(np.sum(poly[:, 0]*np.roll(poly[:, 1], -1)-
                            poly[:, 1]*np.roll(poly[:, 0], -1)))/2)


def partition(points, dimensions):
    """Bounded nearest-cylinder Voronoi cells, with exact common/wall faces."""
    points = np.asarray(points, dtype=float)[:, :2]
    half = np.asarray(dimensions[:2])/2
    rectangle = np.array([[-half[0], -half[1]], [half[0], -half[1]],
                          [half[0], half[1]], [-half[0], half[1]]])
    polygons, shared, walls = [], {}, []
    for j, point in enumerate(points):
        if np.any(abs(point)+DIAMETER/2 > half+1e-12):
            raise ValueError('Physical cylinder extends outside ABS interior')
        poly = rectangle.copy()
        for k, other in enumerate(points):
            if k == j:
                continue
            if np.linalg.norm(point-other) < DIAMETER-1e-12:
                raise ValueError('Physical cylinders overlap')
            poly = clip(poly, other-point, (other@other-point@point)/2)
        polygons.append(poly)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            length = float(np.linalg.norm(b-a))
            if length < 1e-10:
                continue
            mid = (a+b)/2
            at_wall = False
            for axis in range(2):
                for sign in (-1, 1):
                    if abs(mid[axis]-sign*half[axis]) < 1e-10:
                        walls.append(dict(component=j, face=('-' if sign < 0 else '+')+'xy'[axis],
                            length_m=length, endpoints=[a.tolist(), b.tolist()]))
                        at_wall = True
            if not at_wall:
                distances = np.sum((points-mid)**2, axis=1)
                distances[j] = np.inf
                k = int(np.argmin(distances))
                if abs(distances[k]-np.sum((point-mid)**2)) > 1e-10:
                    raise ValueError('Unidentified interior gas face')
                if j < k:
                    shared[(j, k)] = length
    return polygons, shared, walls


def exterior_area(scene):
    """Convert the old 30 parallel two-orifice paths to one equivalent Cd*A.

    Keep the learned UNMEASURED hydraulic prior; do not reset its area simply
    because the source constrains where outlets are, but not their diameter.
    """
    groups = {}
    for edge in scene['gas_network']['channels']:
        if edge['id'].startswith('exhaust-path-'):
            group = edge['id'].split('-')[2]
            groups.setdefault(group, []).append(edge)
        elif edge['id'].startswith('exhaust-'):
            groups[edge['id']] = [edge]
    if len(groups) != 30:
        raise ValueError('Expected the archived thirty-route reconstruction')
    return sum(1/math.sqrt(sum(1/(e['area_m2']*e['discharge_coefficient'])**2
                    for e in group)) for group in groups.values())/.7


def source_geometry(base):
    """Replace synthetic all-cell exits and all-cell wall contacts by geometry.

    Table3 envelope wins over the inconsistent Fig17 width. Pore/shell state
    laws remain untouched; every new thermal contact uses directed geometry.
    """
    scene = deepcopy(base); gas = scene['gas_network']
    meta = scene['metadata']; thickness = meta['geometry_budget']['abs_thickness_m']
    inner = np.array(BOX)-2*thickness
    points = np.array(list(meta['approximate_positions_m'].values()))
    polygons, neighbours, walls = partition(points, inner)
    areas = np.array([polygon_area(p) for p in polygons])
    cylinder = math.pi*DIAMETER**2/4
    cap_height = (inner[2]-HEIGHT)/2
    if min(areas-cylinder) <= 0. or cap_height <= 0.:
        raise ValueError('Nonpositive connected gas space')
    nodes = {n['id']: n for n in gas['compartments']}
    original_total_volume = sum(n['volume_m3'] for n in nodes.values())
    conserved = lambda: np.array([sum(n['initial_mass_kg'] for n in gas['compartments']),
        sum(n['initial_mass_kg']*n['initial_oxidizer_mass_fraction'] for n in gas['compartments'])])
    initial_inventory = conserved()
    old_outer = [n for n in nodes.values() if n['id'] == 'hood' or n['id'].startswith('plume-')]
    reference = deepcopy(nodes['hood'])
    near_volume = sum(n['volume_m3'] for n in old_outer if n['id'] != 'hood')
    total_area = exterior_area(base)
    aperture_side = math.sqrt(total_area/2)
    if not 0 < aperture_side < min(inner[0], HEIGHT):
        raise ValueError('Exit aperture does not fit its face')
    # External partition is a finite oxygen/mass-conservative open-air surrogate,
    # NOT the separate 82L nitrogen gas-characterisation pressure vessel.
    gas['compartments'] = [n for n in gas['compartments'] if not n['id'].startswith('plume-')]
    density = reference['initial_mass_kg']/reference['volume_m3']
    for j, area in enumerate(areas):
        for band, volume in (('body', (area-cylinder)*HEIGHT),
                             ('top', area*cap_height), ('base', area*cap_height)):
            node = nodes[f'{band}-{j:02d}']
            node['volume_m3'] = float(volume); node['initial_mass_kg'] = float(volume*density)
    for label in ('front', 'back'):
        n = deepcopy(reference); n.update(id='plume-'+label, volume_m3=near_volume/2,
            initial_mass_kg=density*near_volume/2)
        gas['compartments'].append(n)
    def passage(ident, a, b, area, length, axis):
        return dict(id=ident, kind='passage', **{'from': a, 'to': b}, area_m2=float(area),
            mixing_length_m=float(length), mixing_axis=axis, discharge_coefficient=.7, opening_pressure_pa=0.)
    # Keep rupture/internal-pore laws exactly. Cap passages must use the free
    # cross-section, not a neighbour's tiny inter-cylinder lateral gap.
    gas['channels'] = [e for e in gas['channels'] if e['kind'] in ('rupture', 'internal')]
    for j, area in enumerate(areas):
        for band in ('top', 'base'):
            gas['channels'].append(passage(f'cap-{band}-{j}', f'body-{j:02d}', f'{band}-{j:02d}',
                area-cylinder, (HEIGHT+cap_height)/2, 2))
    for (a, b), length in sorted(neighbours.items()):
        distance = float(np.linalg.norm(points[b]-points[a]))
        axis = int(np.argmax(abs(points[b, :2]-points[a, :2])))
        for band, height in (('body', HEIGHT), ('top', cap_height), ('base', cap_height)):
            gas['channels'].append(passage(f'free-{band}-{a}-{b}', f'{band}-{a:02d}',
                f'{band}-{b:02d}', length*height, distance, axis))
    apertures = []
    for face, label in (('-y', 'back'), ('+y', 'front')):
        # Source specifies central front/back holes, but not width/height.
        # Square centred at mid-height is an EXPLICIT shape prior.
        for wall in walls:
            if wall['face'] != face:
                continue
            endpoints = np.array(wall['endpoints'])[:, 0]
            overlap = max(0., min(max(endpoints), aperture_side/2)-
                               max(min(endpoints), -aperture_side/2))
            if not overlap:
                continue
            area = overlap*aperture_side
            gas['channels'].append(passage(f'vent-{label}-{wall["component"]}',
                f'body-{wall["component"]:02d}', 'plume-'+label, area, thickness, 1))
            apertures.append(dict(face=face, component=wall['component'], area_m2=area))
        gas['channels'].append(passage('outside-'+label, 'plume-'+label, 'hood',
            inner[0]*inner[2]/2, .05, 1))
    if not math.isclose(sum(x['area_m2'] for x in apertures), total_area, rel_tol=1e-10):
        raise ValueError('Vent clipping lost open area')
    # Do not keep synthetic remote wall films or empirical shell-to-neighbour
    # shortcuts. Each shell touches its own connected void. Its heat can then
    # reach neighbouring voids through the geometric common faces above.
    old_contacts = gas['thermal_contacts']; gas['thermal_contacts'] = []
    for j in range(30):
        ident = f'B{j:02d}'
        for face in ('-x', '+x', '-y', '+y', '-z', '+z'):
            suffix = f'{ident}:case:{face}:outer'
            old = [e for e in old_contacts if e['port_id'] == suffix]
            if not old:
                raise ValueError('Retained full-face contact missing: '+suffix)
            contact = deepcopy(old[0]); contact['compartment'] = (
                f'top-{j:02d}' if face == '+z' else f'base-{j:02d}' if face == '-z' else f'body-{j:02d}')
            # Sum existing partitions first: a copied .30 shared slice is NOT
            # the complete contact. No directed-response law is symmetrised.
            for key in ('background_conductance_w_per_k', 'active_conductance_w_per_k',
                        'destroyed_background_conductance_w_per_k', 'destroyed_active_conductance_w_per_k'):
                contact['connectivity'][key] = sum(e['connectivity'][key] for e in old)
            if face[1] != 'z':
                for key in ('background_conductance_w_per_k', 'active_conductance_w_per_k',
                            'destroyed_background_conductance_w_per_k', 'destroyed_active_conductance_w_per_k'):
                    contact['connectivity'][key] *= math.pi*DIAMETER/(4*WIDTH)
            gas['thermal_contacts'].append(contact)
    wall_areas = {}
    for wall in walls:
        face, j, length = wall['face'], wall['component'], wall['length_m']
        for band, height in (('body', HEIGHT), ('top', cap_height), ('base', cap_height)):
            removed = sum(v['area_m2'] for v in apertures if v['face'] == face and v['component'] == j) if band == 'body' else 0.
            wall_areas[(face, band, j)] = length*height-removed
    for j, area in enumerate(areas):
        wall_areas[('-z', 'base', j)] = area; wall_areas[('+z', 'top', j)] = area
    for (face, band, j), area in sorted(wall_areas.items()):
        inside = ('-' if face[0] == '+' else '+')+face[1]
        gas['thermal_contacts'].append(dict(compartment=f'{band}-{j:02d}',
            port_id=f'ABS:{face}:{inside}', connectivity=connectivity(area, 8.),
            gas_surface_connectivity=deepcopy(gas['surface_connectivity'])))
    # A perforated panel is represented by effective capacity/exposed area;
    # not an opaque full panel PLUS a fictitious through-solid shortcut.
    removed_abs_mass = 0.
    for face in ('-y', '+y'):
        area = total_area/2
        cell = next(c for c in scene['cells'] if c['id'] == 'ABS:'+face)
        removed = area*thickness*1050.; removed_abs_mass += removed
        cell['thermal']['heat_capacity_j_per_k'] -= removed*1400.
        boundary = next(b for b in scene['boundaries'] if b['id'] == 'room-'+face)
        boundary['conductance_w_per_k'] -= area*8.
        boundary['connectivity'] = connectivity(boundary['conductance_w_per_k']/8., 8.)
    np.testing.assert_allclose(conserved(), initial_inventory, rtol=2e-13, atol=1e-15)
    np.testing.assert_allclose(sum(n['volume_m3'] for n in gas['compartments']), original_total_volume, rtol=2e-14)
    # Preserve old search provenance, but do not label its now-incompatible
    # 30-route parameters as the active geometry of this new scene.
    historical = {k:meta.pop(k) for k in ('transport_parameters', 'geometry_hypothesis',
        'search_direction', 'near_field', 'chemistry_heater_materials_frozen') if k in meta}
    meta['predecessor_geometry_metadata'] = historical
    meta['near_field'] = dict(outlets=2, total_volume_m3=near_volume, measured=False,
        inventory_removed_from='hood', note='Two end-vent receiving regions; finite total external inventory conserved')
    meta.update(geometry_status='source-constrained bounded void graph; cylinder geometry / equal-volume thermal box surrogate',
        geometry_reconstruction=dict(version='fsri30-source-geometry-v2', article=ARTICLE,
            report_figure=17, report_table=3, enclosure_dimensions_m=BOX,
            figure_width_m=.07874, dimension_conflict='Table3 envelope retained; Figure17 width is inconsistent',
            polygon_xy_m=[p.tolist() for p in polygons], shared_faces=[dict(a=a,b=b,length_m=l) for (a,b),l in sorted(neighbours.items())],
            wall_faces=walls, central_front_back_apertures=apertures,
            unmeasured_total_aperture_area_m2=total_area, aperture_shape_prior='square at x=0,z=0 on each y-end face',
            aperture_area_prior_source='retained r8 equivalent parallel Cd*A; NOT measured vent size',
            removed_abs_mass_kg=removed_abs_mass,
            retained_nuisance_parameters=['thermal contact laws and hot range', 'flow mixing', 'equivalent total exterior Cd*A'],
            incompatible_old_geometry=['30 independent top exits', 'all-volume contacts to every ABS wall',
                'equal void allocation', 'shared shell-to-neighbour patch fit', 'tiny cylinder gap used in unobstructed upper cavity'],
            unresolved=['actual vent sizes/height', 'can contacts/spacers', 'ABS wall thickness inferred from assembly mass',
                'true cylinder thermal mesh', 'buoyancy and radiation', 'ABS burning', 'open-air far-field renewal',
                'heater feedback (still the archived finite-power prior)', 'nine thermocouple locations']),
        conditioned_transport_diagnostic=False, full_warmup_validation=False,
        comparison_scope='Fresh-from-zero cross-chemistry/geometry diagnostic, NOT experimental validation')
    return scene

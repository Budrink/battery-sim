"""Geometry-only FSRI search: expose shared cavities, not invented heat sources.

The old search could improve the initiating cell's clock without delivering
heat to its neighbours. Here all condensed chemistry, heater drive, shell
mechanics and material properties are immutable. Only unmeasured gas-space
partition, exposed shell areas and passage/contact scales may change.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import numpy as np

from fsri30_objective import cascade, loss, model_fields
from fsri30_search_model import Parameter, validate_controls


SPACE = (
    Parameter('shared_gap_fraction', 0., 1., 0., False,
              'Fraction of a facing shell patch exposed to the adjacent local gas volume'),
    Parameter('upper_sleeve_fraction', 0., .5, 0., False,
              'Fraction of lateral height belonging to the upper connected gas cavity'),
    Parameter('exhaust_area_scale', .05, 1., 1., True,
              'Unmeasured exit area; exploratory prior, not a measured confidence bound'),
    Parameter('passage_area_scale', .25, 4., 1., True,
              'Shared scale of interstitial gas passages, with sleeve cross-sections accounted for'),
    Parameter('hot_contact_span', 1., 30., 1., True,
              'Shared directed-contact active range; renormalized to preserve cold conductance exactly'),
    Parameter('mixing_flow_fraction', .05, .9, .25, False,
              'Shared existing flow-associated gas mixing coefficient'),
)


def specification():
    return [asdict(p) for p in SPACE]


def initial_point():
    return np.array([p.encode(p.initial) for p in SPACE])


def decode(point):
    if len(point) != len(SPACE):
        raise ValueError('Wrong transport-search dimension')
    return {p.name: p.decode(float(x)) for p, x in zip(SPACE, point)}


def point(**changes):
    return np.array([p.encode(changes.get(p.name, p.initial)) for p in SPACE])


def experiments():
    """One mechanism at a time before any GP; the last point tests interaction."""
    return [
        ('shared-facing-cavities', point(shared_gap_fraction=1.)),
        ('upper-cavity-shell-exposure', point(upper_sleeve_fraction=1/3)),
        ('longer-local-residence', point(exhaust_area_scale=.05)),
        ('hot-directed-contact-only', point(hot_contact_span=10.)),
        ('combined-transport-hypothesis', point(shared_gap_fraction=.75,
            upper_sleeve_fraction=1/3, exhaust_area_scale=.1,
            hot_contact_span=10., mixing_flow_fraction=.55)),
    ]


def inventory(scene):
    nodes = scene['gas_network']['compartments']
    cv = scene['gas_network']['cv_j_per_kg_k']
    return np.array([sum(n['volume_m3'] for n in nodes),
        sum(n['initial_mass_kg'] for n in nodes),
        sum(n['initial_mass_kg']*n['initial_oxidizer_mass_fraction'] for n in nodes),
        sum(n['initial_mass_kg']*cv*n['initial_temperature_k'] for n in nodes)])


def frozen(scene):
    """Everything that cannot become a back door for fitting runaway timing."""
    gas = scene['gas_network']
    rates = ('reaction_rate_per_s', 'reaction_reference_k',
             'reaction_activation_j_per_mol', 'oxidizer_heat_j_per_kg')
    return dict(cells=scene['cells'], shells=scene['shells'], interfaces=scene['interfaces'],
        boundaries=scene['boundaries'], failures=scene['primary_failures'], simulation=scene['simulation'],
        ruptures=[e for e in gas['channels'] if e['kind'] == 'rupture'],
        pores=[e for e in gas['channels'] if e['kind'] == 'internal'],
        reaction_laws={n['id']: {k: n.get(k) for k in rates} for n in gas['compartments']},
        internal_nodes=[n for n in gas['compartments'] if 'local_equilibrium' in n])


def facing_neighbours(scene, ident, face):
    """Only declared Delaunay neighbours in the outward half-space; no teleports.

    These lumped interstitial volumes have no measured CAD boundary. Projected
    solid angle supplies shared weights, not a fitted per-battery contact map.
    """
    axis = 'xyz'.index(face[1]); sign = 1 if face[0] == '+' else -1
    if axis == 2:
        return []
    positions = scene['metadata']['approximate_positions_m']
    index = int(ident[1:]); own = np.array(positions[ident])
    result = []
    for a, b in scene['metadata']['neighbour_pairs']:
        other = b if a == index else a if b == index else None
        if other is None:
            continue
        delta = np.array(positions[f'B{other:02d}'])-own
        distance = float(np.linalg.norm(delta)); cosine = sign*delta[axis]/distance
        if cosine >= .5-1e-12:
            result.append((other, cosine/(distance*distance)))
    total = sum(w for _, w in result)
    return [(other, w/total) for other, w in result]


def scaled_contact(contact, compartment, fraction):
    value = deepcopy(contact)
    value['compartment'] = compartment
    for key in ('background_conductance_w_per_k', 'active_conductance_w_per_k',
                'destroyed_background_conductance_w_per_k', 'destroyed_active_conductance_w_per_k'):
        value['connectivity'][key] *= fraction
    return value


def hot_contact(contact, span):
    """Enlarge the EXISTING sigmoid range without changing its cold value.

    A global h multiplier changed heat-up losses too, encouraging the old
    optimizer to repair B10's clock instead of hot-gas delivery. Adjusting
    active span and reference logit together keeps G(S=0) unchanged in BOTH
    directions. Temperature response tables and directed donor selection are
    unchanged; this is a shared unmeasured gas-contact prior, not radiation.
    """
    c = deepcopy(contact)
    if span == 1.:
        return c
    law = c['connectivity']
    law['active_conductance_w_per_k'] *= span
    law['geometry_reference_logit'] = math.log(.5/(span-.5))
    return c


def sleeve_port(scene, contents, face, fraction, upper):
    """Actual nonoverlapping rectangles on the OUTER steel, never bare core."""
    axis = 'xyz'.index(face[1]); sign = 1 if face[0] == '+' else -1
    shell = next(x for x in scene['shells'] if x['contents_id'] == contents['id'])
    thickness = shell['thickness_m']; core = contents['geometry']['dimensions_m']
    dims = [thickness if k == axis else core[k]+(2*thickness if k > axis else 0.) for k in range(3)]
    rectangle = [dims[(axis+1)%3], dims[(axis+2)%3]]
    rectangle[[(axis+1)%3, (axis+2)%3].index(2)] *= fraction if upper else 1-fraction
    center = list(contents['pose']['position_m'])
    center[axis] += sign*(core[axis]/2+thickness)
    center[2] += dims[2]*(1-fraction)/2 if upper else -dims[2]*fraction/2
    normal = [0., 0., 0.]; normal[axis] = sign
    ident = f"{contents['id']}:case:{face}:gas-{'upper' if upper else 'body'}"
    scene['ports'].append(dict(id=ident, cell_id=f"{contents['id']}:case:{face}",
        kind='contact', surface=face, area_m2=math.prod(rectangle),
        rectangle_m=rectangle, centroid_m=center, normal=normal))
    return ident


def candidate(base, x):
    s = deepcopy(base); p = decode(x); gas = s['gas_network']
    nodes = {n['id']: n for n in gas['compartments']}
    sleeve = p['upper_sleeve_fraction']
    ratios = {}
    geometry = s['metadata']['geometry_budget']
    outer_height = s['metadata']['observed']['cell_dimensions_m'][1]
    cap_height = (s['metadata']['observed']['case_dimensions_m'][2]
                  -2*geometry['abs_thickness_m']-outer_height)/2
    if cap_height <= 0.:
        raise ValueError('No physical upper gas cavity')
    for j in range(30):
        body, top = nodes[f'body-{j:02d}'], nodes[f'top-{j:02d}']
        moved_v = sleeve*body['volume_m3']; moved_m = sleeve*body['initial_mass_kg']
        top['volume_m3'] += moved_v; body['volume_m3'] -= moved_v
        top['initial_mass_kg'] += moved_m; body['initial_mass_kg'] -= moved_m
        # Preserve the assumed total free cross-section rather than increasing
        # both body and upper channels independently.
        # Upper lateral passages now extend down the real sleeve height.
        # Body void area differs from cap footprint: a volume ratio would
        # therefore be the WRONG cross-sectional correction here.
        ratios[j] = 1+sleeve*outer_height/cap_height
    for edge in gas['channels']:
        if edge['id'].startswith('free-'):
            factor = (1-sleeve) if edge['id'].startswith('free-body-') else 1.
            if edge['id'].startswith('free-top-'):
                a, b = int(edge['from'].split('-')[-1]), int(edge['to'].split('-')[-1])
                factor = (ratios[a]+ratios[b])/2
            edge['area_m2'] *= factor*p['passage_area_scale']
        elif edge['id'].startswith(('cap-up-', 'cap-down-')):
            edge['area_m2'] *= p['passage_area_scale']
        elif edge['id'].startswith('exhaust-'):
            edge['area_m2'] *= p['exhaust_area_scale']
    gas['mixing_flow_fraction'] = p['mixing_flow_fraction']
    cells = {c['id']: c for c in s['cells'] if c['role'] == 'battery'}
    contacts = []
    for original in gas['thermal_contacts']:
        port = original['port_id']
        if not port.startswith('B'):
            contacts.append(deepcopy(original)); continue
        ident, _, face, _ = port.split(':')
        reference = hot_contact(original, p['hot_contact_span'])
        if face[1] == 'z':
            contacts.append(reference); continue
        neighbours = facing_neighbours(s, ident, face)
        shared = p['shared_gap_fraction'] if neighbours else 0.
        for band, portion in (('body', 1-sleeve), ('top', sleeve)):
            if not portion:
                continue
            patch = deepcopy(reference)
            if sleeve:
                patch['port_id'] = sleeve_port(s, cells[ident], face, sleeve, band == 'top')
            own = f'{band}-{int(ident[1:]):02d}'
            if shared < 1.:
                contacts.append(scaled_contact(patch, own, portion*(1-shared)))
            for other, weight in neighbours:
                if shared:
                    contacts.append(scaled_contact(patch, f'{band}-{other:02d}', portion*shared*weight))
    gas['thermal_contacts'] = contacts
    np.testing.assert_allclose(inventory(s), inventory(base), rtol=5e-14, atol=1e-12)
    if frozen(s) != frozen(base):
        raise ValueError('Transport proposal changed frozen chemistry/materials/heater/shell/numerics')
    validate_controls(s)
    s['metadata'].update(search_direction='neighbour-energy-delivery',
        transport_parameters=p, chemistry_heater_materials_frozen=True,
        geometry_hypothesis='Shared facing gas cavities and upper sleeve; conserved volume/air/O2/energy and exposed area; unmeasured CAD',
        external_plume_surface_shortcut=False)
    return s


def diagnostics(scene, run):
    t, ids, values = model_fields(run)
    source = scene['metadata']['heated_component']
    outside = [i for i, key in enumerate(ids) if key != source]
    peaks = sorted((float(np.max(values[:, i])), ids[i]) for i in outside)[::-1]
    amounts = {c['id']: c['conversion']['mass_kg'] for c in scene['cells'] if c['role'] == 'battery'}
    converted = sorted((float(np.max(np.array(run['chemistry'][key]['residue_kg'])+
        np.array(run['chemistry'][key]['r2_generated_gas_kg'])))/amounts[key], key)
        for key in ids if key != source)[::-1]
    gas = run.get('gas', {}); hood = gas.get('hood', {})
    return dict(source_component=source, source_peak_k=float(np.max(values[:, ids.index(source)])),
        hottest_neighbours=[dict(component=k, peak_k=temp) for temp, k in peaks[:6]],
        max_neighbour_converted_fraction=converted[0][0],
        top_three_neighbour_mean_peak_k=float(np.mean([v for v, _ in peaks[:3]])),
        recorded_r3_j=sum(g['released_energy_j'][-1] for g in gas.values()),
        hood_unburned_chemical_j=hood.get('chemical_j', [None])[-1],
        hood_temperature_k=hood.get('temperature_k', [None])[-1],
        global_r3_history_complete=False,
        note='Neighbour heating/conversion are model diagnostics, NOT invented thermocouple locations')


def transport_loss(scene, run, data):
    reference = loss(scene, run, data); metrics = diagnostics(scene, run)
    ambient = 298.15
    # 450 K is the FIXED R2 Arrhenius reference, not a new ignition trigger.
    # Smooth search shaping gives credit for neighbour energy before a second
    # component crosses the unchanged 10% reporting threshold.
    preheat = np.clip((metrics['top_three_neighbour_mean_peak_k']-ambient)/(450.-ambient), 0., 1.)
    conversion = np.clip(metrics['max_neighbour_converted_fraction']/.1, 0., 1.)
    process = cascade(scene, run)
    parts = dict(neighbour_preheat=20.*float(1-preheat)**2,
        neighbour_conversion=20.*float(1-conversion)**2,
        minimum_cascade=8.*max(0, 2-process['reacted_components'])**2,
        experimental_envelope=.5*sum(reference['phase_loss'].values()))
    return dict(reference, objective=sum(parts.values()),
        original_envelope_objective=reference['objective'], transport_loss_parts=parts,
        transfer=metrics, search_direction='neighbour-energy-delivery')


def useful_response(baseline, result):
    """Do not spend hours fitting a direction that cannot deliver neighbour heat."""
    a, b = baseline['transfer'], result['transfer']
    source = b['source_component']
    source_reacted = any(e['component'] == source for e in result['cascade']['events'])
    rise = b['top_three_neighbour_mean_peak_k']-a['top_three_neighbour_mean_peak_k']
    return source_reacted and (rise >= 10. or b['max_neighbour_converted_fraction'] >= .01)

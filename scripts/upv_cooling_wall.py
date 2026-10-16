"""UPV geometry adapter: ordinary passive volumes + contacts + one heat sink.

No temperature integration here; the standard CUDA graph evolves every volume.
The liquid loop is represented only by a constant boundary conductance.
"""
import copy
import math

WALL_ID = 'cooling-plate'
SINK_ID = 'coolant-reservoir'


def response():
    return dict(reference_temperature_k=298.15, temperature_scale_k=100.,
                linear=[0., 0., 0.], quadratic=[0., 0., 0.], damage=[0., 0., 0.],
                temperature_contrast=[0., 0., 0.], damage_contrast=[0., 0., 0.])


def contact_law(area, reference_h):
    # Same directed state closure as other contacts; reference h is a prior,
    # not a fitted material constant. At zero endpoint response G=area*h_ref.
    lo, hi = .01*reference_h, 2.*reference_h
    return dict(kind='directed_geometry', l0_m=1e-4, activation_width_m=1e-5,
                geometry_reference_logit=math.log((reference_h-lo)/(hi-reference_h)),
                background_conductance_w_per_k=area*lo,
                active_conductance_w_per_k=area*(hi-lo), endpoint_damage_weight=1.,
                damage_rate_at_onset_per_s=0., destruction_damage=1.,
                destroyed_l0_m=1e-4, destroyed_l_m=1e-4,
                destroyed_background_conductance_w_per_k=area*lo,
                destroyed_active_conductance_w_per_k=0.)


def attach(scene, *, coolant_temperature_k=298.15, sink_conductance_w_per_k=10.,
           contact_h_w_per_m2_k=10000., thickness_m=.013, divisions=(7, 3, 1),
           surface_response=None, bulk_response=None, initial_temperature_k=None):
    """Replace only the three old Al-tip sinks; do not mutate the input scene.

    Thickness/k/rho come from the UPV description. Footprint, cp, contact h and
    sink G are explicit engineering priors. A post-event call must subsequently
    restore the wall's field from a consistent warmup, not reset it to water T.
    """
    for name, value in dict(coolant_temperature_k=coolant_temperature_k,
                            contact_h=contact_h_w_per_m2_k, thickness=thickness_m).items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid '+name)
    if not math.isfinite(sink_conductance_w_per_k) or sink_conductance_w_per_k < 0:
        raise ValueError('Invalid sink conductance')
    if len(divisions) != 3 or any(type(n) is not int or not 1 <= n <= 256 for n in divisions):
        raise ValueError('Invalid wall divisions')
    initial = coolant_temperature_k if initial_temperature_k is None else initial_temperature_k
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError('Invalid initial wall temperature')
    cells = {c['id']: c for c in scene['cells']}
    if WALL_ID in cells or SINK_ID in {b['id'] for b in scene['boundaries']}:
        raise ValueError('Cooling wall already attached')
    sinks = {b['id']: b for b in scene['boundaries'] if b['id'].startswith('cooling-Al')}
    if set(sinks) != {'cooling-Al1', 'cooling-Al2', 'cooling-Al3'}:
        raise ValueError('Expected exactly three legacy Al-tip boundaries')
    ports = {p['id']: p for p in scene['ports']}
    tips = [ports[sinks['cooling-'+name]['port_id']] for name in ('Al1', 'Al2', 'Al3')]
    z = tips[0]['centroid_m'][2]
    for tip in tips:
        cell = cells[tip['cell_id']]
        if tip['surface'] != '-z' or tip['normal'] != [0, 0, -1] or abs(tip['centroid_m'][2]-z) > 1e-9:
            raise ValueError('Cooling tips must share the -z contact plane')
        if cell['pose'].get('orientation_xyzw', [0, 0, 0, 1]) != [0, 0, 0, 1]:
            raise ValueError('UPV wall adapter requires aligned solids')
    # Use the stack's clamping-plate footprint, never the distant chamber-wall
    # proxy. This planar extent is a declared assumption, not measured CAD.
    stack = [cells[k] for k in ('steel-L', 'steel-R')]
    low = [min(c['pose']['position_m'][a]-c['geometry']['dimensions_m'][a]/2 for c in stack) for a in (0, 1)]
    high = [max(c['pose']['position_m'][a]+c['geometry']['dimensions_m'][a]/2 for c in stack) for a in (0, 1)]
    dims = [high[0]-low[0], high[1]-low[1], thickness_m]
    position = [(high[0]+low[0])/2, (high[1]+low[1])/2, z-thickness_m/2]
    # Explicit positive-volume overlap check; no hidden face contact is added.
    for c in cells.values():
        d, p = c['geometry']['dimensions_m'], c['pose']['position_m']
        if all(min(position[a]+dims[a]/2, p[a]+d[a]/2)-max(position[a]-dims[a]/2, p[a]-d[a]/2) > 1e-9 for a in range(3)):
            raise ValueError('Cooling wall overlaps '+c['id'])
    s = copy.deepcopy(scene)
    surface = copy.deepcopy(surface_response if surface_response is not None else
                            cells['Al2']['thermal'].get('surface_connectivity', response()))
    bulk = copy.deepcopy(bulk_response if bulk_response is not None else response())
    # D never evolves for this material. Temperature-dependent/anisotropic
    # surface and bulk responses remain independent ordinary material laws.
    surface['damage'] = surface['damage_contrast'] = [0., 0., 0.]
    bulk['damage'] = bulk['damage_contrast'] = [0., 0., 0.]
    thermal = dict(heat_capacity_j_per_k=math.prod(dims)*2700.*900.,
                   initial_temperature_k=initial, divisions=list(divisions),
                   conductivity_w_per_m_k=[235., 235., 235.],
                   surface_connectivity=surface, bulk_response=bulk,
                   bulk_connectivity=dict(kind='directed_geometry', l0_m=1e-4,
                     activation_width_m=1e-5, geometry_reference_logit=0.,
                     background_factor=.1, active_factor=1.8, endpoint_damage_weight=0.,
                     damage_rate_at_onset_per_s=0., destroyed_l0_m=1e-4, destroyed_l_m=1e-4))
    s['cells'].append(dict(id=WALL_ID, role='passive', material_id='aluminium',
                          geometry=dict(kind='prismatic', dimensions_m=dims),
                          pose=dict(position_m=position), thermal=thermal,
                          degradation=dict(initial_damage=0., rate_at_onset_per_s=0.),
                          runaway=dict(energy_j=0., duration_s=1.)))
    s['boundaries'] = [b for b in s['boundaries'] if b['id'] not in sinks]
    for tip in tips:
        cell = cells[tip['cell_id']]; dx, dy = cell['geometry']['dimensions_m'][:2]
        if not math.isclose(tip['area_m2'], dx*dy, rel_tol=1e-9):
            raise ValueError('Expected a full rectangular spreader tip')
        partner = WALL_ID+':'+tip['cell_id']
        s['ports'].append(dict(id=partner, cell_id=WALL_ID, kind='contact', surface='+z',
                              area_m2=tip['area_m2'], rectangle_m=[dx, dy],
                              centroid_m=tip['centroid_m'].copy(), normal=[0, 0, 1]))
        edge_id = tip['cell_id']+'--'+WALL_ID
        s['interfaces'].append(dict(id=edge_id, port_a=tip['id'], port_b=partner,
                                   law=contact_law(tip['area_m2'], contact_h_w_per_m2_k)))
        if tip['cell_id'] == 'Al2':
            s.setdefault('measurement_probes', []).append(dict(id='110', cell_id='Al2',
                interface_id=edge_id, port_id=tip['id'], neighbor_port_id=partner,
                neighbor_material='aluminium'))
    sink_port = WALL_ID+':water'
    s['ports'].append(dict(id=sink_port, cell_id=WALL_ID, kind='contact', surface='-z',
        area_m2=dims[0]*dims[1], centroid_m=[position[0], position[1], z-thickness_m], normal=[0, 0, -1]))
    s['boundaries'].append(dict(id=SINK_ID, port_id=sink_port,
                              temperature_k=coolant_temperature_k,
                              conductance_w_per_k=sink_conductance_w_per_k))
    meta = s.setdefault('metadata', {})
    meta['cooling_wall'] = dict(model='ordinary finite-volume passive solid, directed contacts, constant external G',
        thickness_m=thickness_m, sink_conductance_w_per_k=sink_conductance_w_per_k,
        coolant_temperature_k=coolant_temperature_k, contact_h_w_per_m2_k=contact_h_w_per_m2_k,
        source_measured='Al k=235 W/m/K, rho=2700 kg/m3, thickness=13 mm',
        unmeasured_priors='Stack footprint; cp=900 J/kg/K; contact h and sink G; inherited Al surface response; bulk response defaults to reference k',
        temperature_observer='110: Al2 tip face; diagnostic only, no shared temperature driver',
        defects='Disabled: D(0)=0, damage rate=0, no chemical source',
        status='Implementation prior, not recalibrated; do not claim previous validation metrics')
    meta['drivers'] = [v for v in meta.get('drivers', []) if v != 'ch110 cooling contact']
    meta['assumptions'] = [v for v in meta.get('assumptions', []) if not v.startswith('cooled case: ch110')]
    meta['calibration_status'] = 'Explicit cooling-wall trial; changed cooling model, not calibrated'
    return s

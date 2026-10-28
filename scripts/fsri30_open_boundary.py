"""Open far field for the FROZEN 20 mm FSRI case, without remeshing.

The article reports an open-air cell test under an exhaust hood. Its 420 m3/min
is the facility exhaust, NOT measured flow through the local 1 m3 domain.
Local renewal is therefore an explicit apparatus prior, not material data.
"""
from copy import deepcopy
import math


def exterior_faces(scene):
    geometry = scene['metadata']['gas_volume_geometry']
    if not geometry or any(not key.startswith('air-') for key in geometry):
        raise ValueError('Expected resolved external air boxes only')
    low = [min(box['low_m'][i] for box in geometry.values()) for i in range(3)]
    high = [max(box['high_m'][i] for box in geometry.values()) for i in range(3)]
    areas = {}
    for key, box in geometry.items():
        widths = [b-a for a,b in zip(box['low_m'],box['high_m'])]
        if min(widths) <= 0.: raise ValueError('Nonpositive air box')
        area = 0.
        for axis in range(3):
            face = math.prod(widths[i] for i in range(3) if i != axis)
            for field, plane in (('low_m',low),('high_m',high)):
                if abs(box[field][axis]-plane[axis]) < 1e-12: area += face
        if area: areas[key] = area
    expected = 2.*sum((high[i]-low[i])*(high[j]-low[j]) for i,j in ((0,1),(0,2),(1,2)))
    if not math.isclose(sum(areas.values()),expected,rel_tol=1e-10):
        raise ValueError('Outer faces do not tile the far field')
    return areas


def open_boundary(base, renewal_m3_s, *, pressure_open=True):
    if not math.isfinite(renewal_m3_s) or renewal_m3_s < 0.:
        raise ValueError('Renewal must be finite and nonnegative')
    scene = deepcopy(base)
    if any('ambient_boundary' in c for c in scene['gas_network']['compartments']):
        raise ValueError('Boundary already configured')
    if not pressure_open and renewal_m3_s == 0.: return scene
    areas = exterior_faces(scene)
    total = sum(areas.values())
    configured = 0
    for node in scene['gas_network']['compartments']:
        area = areas.get(node['id'],0.)
        if not area: continue
        node['ambient_boundary'] = dict(pressure_pa=101325.,temperature_k=298.15,
            oxidizer_mass_fraction=.232,area_m2=area if pressure_open else 0.,
            discharge_coefficient=.7,renewal_flow_m3_s=renewal_m3_s*area/total)
        configured += 1
    if configured != len(areas): raise ValueError('Missing boundary gas nodes')
    scene['name'] += f' / open far field, renewal {renewal_m3_s:g} m3/s'
    scene['metadata']['air_mesh']['boundary'] = 'open far field; explicit mass/enthalpy/fuel/oxygen ledgers'
    scene['metadata']['ambient_boundary'] = dict(renewal_m3_s=renewal_m3_s,
        outer_area_m2=total,boundary_nodes=configured,pressure_open=pressure_open,
        source='https://pmc.ncbi.nlm.nih.gov/articles/PMC9679666/',
        facility_exhaust_m3_s=7.,local_flow_measured=False,
        scope='Apparatus-boundary diagnostic; chemistry, materials, contacts and 20 mm mesh frozen')
    scene['metadata'].update(validation_passed=False,fresh_from_zero=True,
        calibration_status='UNVALIDATED open-boundary diagnostic')
    return scene

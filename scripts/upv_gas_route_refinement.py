"""Existing slit routes: independent source exhaust and witness entrance.

These are unresolved effective apertures, not measured fixture dimensions.
No new edge crosses a solid. Pressure-flow and diffusive access change together;
fracture work/opening law, gas volume, materials and chemistry stay untouched.
"""
import copy
import math

from upv_gas_access import AREA_COEFFICIENTS
from upv_gas_response_refinement import assess


def candidate(source, source_exhaust=1., witness_mouth=None, pad_footprint=None):
    scene = copy.deepcopy(source)
    geometry = scene['metadata']['gas_access_geometry']
    old_mouth = geometry['iss3_accessible_fraction']
    old_pad = geometry.get('open_iss_pad_wetted_fraction', 1.)
    mouth = old_mouth if witness_mouth is None else witness_mouth
    pad = old_pad if pad_footprint is None else pad_footprint
    for value, lo, hi, label in [(source_exhaust, .3, 10., 'source exhaust'),
                                (mouth, .001, .05, 'ISS3 mouth'),
                                (pad, .2, .75, 'common ISS pad footprint')]:
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Invalid geometry ' + label)
    counts = dict(source_exhaust=0, witness_mouth=0, pad_footprint=0)
    for edge in scene['gas_network']['channels']:
        if edge['id'].startswith('escape-ISS2-'):
            edge['area_m2'] *= source_exhaust
            edge['mixing_length_m'] /= source_exhaust
            counts['source_exhaust'] += 1
        if edge['id'].startswith('escape-ISS3-'):
            edge['area_m2'] *= mouth / old_mouth
            counts['witness_mouth'] += 1
        if edge['kind'] == 'rupture' and edge['to'].startswith('gap-ISS3-'):
            edge['discharge_coefficient'] *= mouth / old_mouth
            edge['mixing_length_m'] /= mouth / old_mouth
            counts['witness_mouth'] += 1
    ports = {p['id']: p['cell_id'] for p in scene['ports']}
    for contact in scene['gas_network']['thermal_contacts']:
        owner = ports[contact['port_id']]
        if owner in ('ISS1', 'ISS2') and contact['compartment'].startswith('gap-' + owner + '-'):
            for key in AREA_COEFFICIENTS:
                contact['connectivity'][key] *= pad / old_pad
            counts['pad_footprint'] += 1
    if not all(counts.values()):
        raise ValueError('Expected existing exhaust/mouth/pad contacts are missing')
    geometry['iss3_accessible_fraction'] = mouth
    geometry['open_iss_pad_wetted_fraction'] = pad
    scene['metadata']['gas_route_refinement'] = dict(source_exhaust=source_exhaust,
        witness_mouth=mouth, pad_footprint=pad, counts=counts,
        source_is_explicit_current_profile=True, constitutive_laws_unchanged=True,
        no_new_solid_crossing=True, unmeasured_effective_geometry=True,
        requires_fresh_verification=True)
    return scene


def assessment(run, data, end):
    result = assess(run, data, end)
    result['cooling_objective'] = result['objective']
    # A 2 s delayed, sharp pulse is still delayed. Fit physical arrival times,
    # never align the traces afterwards. Peak temperature remains softened.
    timing = {}
    for key, row in result['front']['features'].items():
        a, b = row['predicted'], row['observed']
        if a is None:
            timing[key] = dict(error_s=None, loss=100.)
        else:
            error = .5 * (a['t10_s'] + a['t90_s'] - b['t10_s'] - b['t90_s'])
            timing[key] = dict(error_s=error, loss=(error / .5) ** 2)
    result['arrival_times'] = timing
    result['shape_penalty'] = 20. * result['shape']['shape_with_front_objective']
    result['timing_penalty'] = sum(row['loss'] for row in timing.values())
    result['arrival_accepted'] = all(row['error_s'] is not None and abs(row['error_s']) <= .5
                                     for row in timing.values())
    result['accepted'] = bool(result['accepted'] and result['arrival_accepted'])
    result['objective'] += result['shape_penalty'] + result['timing_penalty']
    result['geometry_objective_version'] = 'cooling-shape-and-physical-arrival-v1'
    return result

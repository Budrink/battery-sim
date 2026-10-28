"""One common unresolved gas-access fraction for all three TIM bays.

The same geometric rule applies to source and witnesses. This does not alter
thermal contact area, material properties, aperture damage, or fracture work.
The existing downstream-throat representation scales passage area and rupture
Cd/A-over-L access together. Dimensions are not measured fixture dimensions.
"""
import copy
import math


def candidate(source, fraction, fracture_access_fraction=None):
    if not math.isfinite(fraction) or not 1e-5 <= fraction <= 1.:
        raise ValueError('TIM gas access outside [0.00001, 1]')
    scene = copy.deepcopy(source)
    geometry = scene['metadata']['gas_access_geometry']
    previous = geometry['protected_accessible_fraction']
    previous_entry=geometry.get('protected_fracture_fraction',previous)
    entry=previous_entry*(fraction/previous) if fracture_access_fraction is None else fracture_access_fraction
    if not math.isfinite(entry) or not 1e-5<=entry<=1.:
        raise ValueError('TIM fracture access outside [0.00001, 1]')
    if not 0. < previous <= 1.:
        raise ValueError('Invalid explicit reference TIM access')
    ratio = fraction / previous
    entry_ratio=entry/previous_entry
    counts = {pad:dict(passages=0, ruptures=0) for pad in ('TIM1','TIM2','TIM3')}
    for edge in scene['gas_network']['channels']:
        for pad, count in counts.items():
            if edge['id'].startswith('escape-'+pad+'-'):
                edge['area_m2'] *= ratio
                count['passages'] += 1
            if edge['kind'] == 'rupture' and edge['to'].startswith('gap-'+pad+'-'):
                edge['discharge_coefficient'] *= entry_ratio
                edge['mixing_length_m'] /= entry_ratio
                count['ruptures'] += 1
    if not all(all(c.values()) for c in counts.values()):
        raise ValueError('Expected all three TIM gas routes')
    geometry['protected_accessible_fraction'] = fraction
    geometry['protected_fracture_fraction'] = entry
    scene['metadata']['common_tim_gas_access'] = dict(fraction=fraction,
        fracture_access_fraction=entry,reference_fraction=previous,
        reference_fracture_fraction=previous_entry,counts=counts, measured=False,
        constitutive_laws_unchanged=True, common_for_source_and_witnesses=True,
        requires_fresh_verification=True)
    return scene

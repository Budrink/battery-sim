"""Common Al-backing gas exposure, without changing any constitutive law."""
import copy
import math

from upv_gas_access import AREA_COEFFICIENTS


def candidate(source, fraction, lip_fraction=None):
    if not math.isfinite(fraction) or not .001 <= fraction <= 1.:
        raise ValueError('Backing gas exposure outside [0.001, 1]')
    scene = copy.deepcopy(source)
    geometry = scene['metadata']['gas_access_geometry']
    previous = geometry['backing_open_fraction']
    old_lip = geometry['holder_lip_open_fraction']
    new_lip = old_lip if lip_fraction is None else lip_fraction
    if not math.isfinite(new_lip) or not 1e-5 <= new_lip <= 1.:
        raise ValueError('Holder rim exposure outside [0.00001, 1]')
    if not 0. < previous <= 1.:
        raise ValueError('Invalid explicit reference exposure')
    ports = {p['id']: p for p in scene['ports']}
    def rim(port, lip):
        values = {p: lip for p in ('TIM1', 'TIM2', 'TIM3', 'ISS3')}
        values['ISS4'] = geometry['outer_insulation_open_fraction']
        return next((f for p, f in values.items()
                     if port.startswith('opening-contact:' + p + ':')), 1.)
    def exposure(port, backing, lip):
        value = rim(port, lip)
        owner = ports[port]['cell_id']
        if owner in ('Al1', 'Al2', 'Al3'):
            value = min(value, backing)
        if owner.split(':')[0] in ('C1', 'C3') and ':case:' in owner and owner[-1] in ('y', 'z'):
            value = min(value, geometry['witness_edge_open_fraction'])
        return value
    changed = 0
    for contact in scene['gas_network']['thermal_contacts']:
        port = ports[contact['port_id']]
        ratio = exposure(contact['port_id'], fraction, new_lip) / exposure(contact['port_id'], previous, old_lip)
        for key in AREA_COEFFICIENTS:
            contact['connectivity'][key] *= ratio
        changed += ratio != 1.
    if not any(p['cell_id'] in ('Al1', 'Al2', 'Al3') for p in ports.values()):
        raise ValueError('No Al backing ports')
    for row in geometry['backing_exposure']:
        row['gas_exposed_area_m2'] = row['nominal_area_m2'] * exposure(row['port'], fraction, new_lip)
    for row in geometry['holder_lip_exposure']:
        row['gas_exposed_area_m2'] = row['nominal_area_m2'] * rim(row['port'], new_lip)
    for row in geometry['witness_edge_exposure']:
        row['gas_exposed_patch_area_m2'] = row['nominal_patch_area_m2'] * exposure(row['port'], fraction, new_lip)
    geometry['backing_open_fraction'] = fraction
    geometry['holder_lip_open_fraction'] = new_lip
    scene['metadata']['backing_response_probe'] = dict(
        reference_fraction=previous, common_fraction=fraction,
        reference_lip_fraction=old_lip, common_lip_fraction=new_lip,
        contact_count=changed, measured=False, material_and_response_laws_unchanged=True,
        conditional_until_fresh_verification=True)
    return scene

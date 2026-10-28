"""One shared residual-conductance parameter for battery/TIM contacts.

This is the existing destroyed branch of the directed contact law, NOT a
material conductivity, shell strength, reaction, or thermocouple correction.
Changing it requires a fresh simulation; gas-only restart is not authorized.
"""
import copy
import math


def candidate(source, retained_fraction, *, reference='cold'):
    if not math.isfinite(retained_fraction) or not .005 <= retained_fraction <= 1.:
        raise ValueError('Residual battery/TIM contact fraction outside [0.005, 1]')
    if reference not in ('cold', 'hot_ceiling'):
        raise ValueError('Contact reference must be cold or hot_ceiling')
    scene = copy.deepcopy(source)
    ports = {p['id']: p['cell_id'] for p in scene['ports']}
    materials = {c['id']: c['material_id'] for c in scene['cells']}
    def contact_role(owner):
        # Shell panels are generated from `shells`; they are deliberately NOT
        # ordinary root `cells`. Their outer port belongs to the battery's
        # casing, not to its reacting interior material.
        root = owner.split(':case:', 1)[0]
        if ':case:' in owner and root not in {s['contents_id'] for s in scene.get('shells', [])}:
            raise ValueError('Unknown generated casing owner: ' + owner)
        return materials.get(root)
    changed = []
    for interface in scene['interfaces']:
        pair = {contact_role(ports[interface['port_a']]), contact_role(ports[interface['port_b']])}
        if pair != {'battery', 'TIM'}:
            continue
        law = interface['law']
        if law['kind'] != 'directed_geometry' or law['destroyed_active_conductance_w_per_k'] != 0.:
            raise ValueError('Require the explicit existing constant destroyed branch')
        cold = law['background_conductance_w_per_k'] + law['active_conductance_w_per_k'] / (
            1. + math.exp(-law['geometry_reference_logit']))
        # The intact directed law already bounds the hot contact by background
        # plus active conductance. Retaining that ceiling after damage does not
        # alter any material property or exceed the existing contact capacity.
        ceiling = law['background_conductance_w_per_k'] + law['active_conductance_w_per_k']
        reference_conductance = cold if reference == 'cold' else ceiling
        old = law['destroyed_background_conductance_w_per_k']
        target = retained_fraction * reference_conductance
        if not math.isclose(old, target, rel_tol=1e-14, abs_tol=0.):
            law['destroyed_background_conductance_w_per_k'] = target
        changed.append(dict(interface=interface['id'], reference_w_per_k=reference_conductance,
                            cold_reference_w_per_k=cold, hot_ceiling_w_per_k=ceiling,
                            old_destroyed_w_per_k=old, new_destroyed_w_per_k=target))
    if not changed:
        raise ValueError('No battery/TIM contact pair')
    scene['metadata']['damaged_contact_probe'] = dict(retained_fraction=retained_fraction,
        reference=reference,
        contacts=changed, fresh_start_required=True, material_constants_unchanged=True,
        shell_and_chemistry_unchanged=True, measured=False)
    return scene


def hot_contact_candidate(source, multiplier):
    """Bracket an unmeasured *contact* G, not the conductivity of TIM.

    The intact-law ceiling is a calibration reference, not a physical upper
    bound on a changed contact geometry. Keep the same factor at every TIM
    interface, bound it to [0.25, 4], and retain all bulk series resistances.
    This hypothesis requires a fresh full simulation, including the heat tail.
    """
    if not math.isfinite(multiplier) or not .25 <= multiplier <= 4.:
        raise ValueError('Hot battery/TIM contact multiplier outside [0.25, 4]')
    scene = candidate(source, 1., reference='hot_ceiling')
    probe = scene['metadata']['damaged_contact_probe']
    changes = {row['interface']: row for row in probe['contacts']}
    for interface in scene['interfaces']:
        if interface['id'] not in changes:
            continue
        target = multiplier * changes[interface['id']]['hot_ceiling_w_per_k']
        interface['law']['destroyed_background_conductance_w_per_k'] = target
        changes[interface['id']]['new_destroyed_w_per_k'] = target
    probe.pop('retained_fraction')
    probe.update(hot_contact_multiplier=multiplier, reference='intact_hot_ceiling',
                 unchanged_bulk_series_resistance=True, contact_geometry_unmeasured=True)
    return scene

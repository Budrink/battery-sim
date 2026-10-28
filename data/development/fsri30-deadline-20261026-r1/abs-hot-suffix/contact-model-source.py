"""Shared hot gas/ABS contact hypothesis; material Cp/k and cold G unchanged."""
from copy import deepcopy
import math


def hot_abs_contact(base, ceiling=60., strength=1.):
    if not math.isfinite(ceiling) or not 2.<=ceiling<=60.:
        raise ValueError('Contact saturation prior outside [2,60] times cold reference')
    if not math.isfinite(strength) or not 0.<=strength<=1.:
        raise ValueError('Contact hypothesis strength must be within [0,1]')
    s=deepcopy(base);changed=[]
    if strength==0.:return s
    for c in s['gas_network']['thermal_contacts']:
        if not c['port_id'].startswith('gasmesh:ABS:'):continue
        law=c['connectivity'];z=law.get('geometry_reference_logit',0.)
        reference=law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-z))
        old_ceiling=(law['background_conductance_w_per_k']+law['active_conductance_w_per_k'])/reference
        effective_ceiling=old_ceiling+strength*(ceiling-old_ceiling)
        law['background_conductance_w_per_k']=.01*reference
        law['active_conductance_w_per_k']=(effective_ceiling-.01)*reference
        fraction=.99/(effective_ceiling-.01)
        law['geometry_reference_logit']=math.log(fraction/(1.-fraction))
        response=deepcopy(c.get('gas_surface_connectivity',s['gas_network']['surface_connectivity']))
        # The hotter donor uses its own ordered response in either direction.
        # No averaging of Sij/Sji or replacement with a scalar film.
        c['gas_surface_connectivity']=response
        solid=deepcopy(response)
        for point in solid.get('temperature_log_response',[]):point[1]*=strength
        c['solid_surface_connectivity']=solid
        changed.append(c['port_id'])
    if not changed:raise ValueError('No reconstructed ABS surfaces')
    s['metadata']['abs_hot_contact_hypothesis']=dict(reference_conductance_unchanged=True,
        saturation_ceiling_relative_to_reference=ceiling,strength=strength,changed_contacts=len(changed),
        response='Existing temperature table on the ordered hotter donor, gas or solid',
        measured=False,calibrated=False,validation_passed=False,
        scope='Shared gas/ABS hot contact only; material Cp/k, mass, chemistry and geometry frozen')
    return s

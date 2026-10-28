"""Shared hot gas/ABS contact hypothesis; material Cp/k and cold G unchanged."""
from copy import deepcopy
import math


def response_at(response,temperature):
    table=response.get('temperature_log_response',[])
    if not table:return 0.
    if temperature<=table[0][0]:return table[0][1]
    for (a,x),(b,y)in zip(table,table[1:]):
        if temperature<b:return x+(y-x)*(temperature-a)/(b-a)
    return table[-1][1]


def conductance_at(law,response,temperature):
    z=law.get('geometry_reference_logit',0.)+response_at(response,temperature)
    return law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-z))


def hot_abs_contact(base, ceiling=60., strength=1.):
    if not math.isfinite(ceiling) or not 2.<=ceiling<=60.:
        raise ValueError('Contact saturation prior outside [2,60] times cold reference')
    if not math.isfinite(strength) or not 0.<=strength<=1.:
        raise ValueError('Contact hypothesis strength must be within [0,1]')
    s=deepcopy(base);changed=[]
    if strength==0.:return s
    for c in s['gas_network']['thermal_contacts']:
        if not c['port_id'].startswith('gasmesh:ABS:'):continue
        law=c['connectivity'];old=deepcopy(law);z=law.get('geometry_reference_logit',0.)
        reference=law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-z))
        law['background_conductance_w_per_k']=.01*reference
        law['active_conductance_w_per_k']=(ceiling-.01)*reference
        fraction=.99/(ceiling-.01)
        law['geometry_reference_logit']=math.log(fraction/(1.-fraction))
        response=deepcopy(c.get('gas_surface_connectivity',s['gas_network']['surface_connectivity']))
        # The hotter donor uses its own ordered response in either direction.
        # No averaging of Sij/Sji or replacement with a scalar film.
        c['gas_surface_connectivity']=response
        old_solid=deepcopy(c.get('solid_surface_connectivity',{}))
        c['solid_surface_connectivity']=deepcopy(response)
        if strength<1.:
            # Interpolate TWO SAME-DIRECTION contact hypotheses, not Sij/Sji.
            # Eight derived knots encode ONE shared strength; they are not
            # eight fitted degrees of freedom. Endpoints 0/1 remain exact.
            knots=(298.15,400.,500.,600.,750.,900.,1050.,1200.)
            floor=law['background_conductance_w_per_k'];span=law['active_conductance_w_per_k']
            for endpoint,previous in (('gas_surface_connectivity',response),('solid_surface_connectivity',old_solid)):
                points=[]
                for temperature in knots:
                    g=(1.-strength)*conductance_at(old,previous,temperature)+strength*conductance_at(law,response,temperature)
                    opening=(g-floor)/span
                    value=math.log(opening/(1.-opening))-law['geometry_reference_logit']
                    points.append([temperature,0. if temperature==298.15 else value])
                c[endpoint]=dict(reference_temperature_k=298.15,temperature_log_response=points)
        changed.append(c['port_id'])
    if not changed:raise ValueError('No reconstructed ABS surfaces')
    s['metadata']['abs_hot_contact_hypothesis']=dict(reference_conductance_unchanged=True,
        saturation_ceiling_relative_to_reference=ceiling,strength=strength,changed_contacts=len(changed),
        response='Existing temperature table on the ordered hotter donor, gas or solid',
        measured=False,calibrated=False,validation_passed=False,
        scope='Shared gas/ABS hot contact only; material Cp/k, mass, chemistry and geometry frozen')
    return s

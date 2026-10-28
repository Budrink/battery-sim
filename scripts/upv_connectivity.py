"""Shared directed thermal-contact law, including gas/surface contacts."""
import math


def contact(area, h, destroyed=.02):
    floor = area*.01*h
    span = area*1.98*h
    return dict(kind='directed_geometry', geometry_reference_logit=0.,
                background_conductance_w_per_k=floor,
                active_conductance_w_per_k=span,
                endpoint_damage_weight=1., destroyed_geometry_reference_logit=0.,
                destroyed_background_conductance_w_per_k=(floor if destroyed is None else area*h*destroyed),
                destroyed_active_conductance_w_per_k=(span if destroyed is None else 0.))


def explicit_gas_contacts(scene):
    """Normalize archived reference h values; no fitted parameter changes."""
    ports = {p['id']: p for p in scene['ports']}
    for c in scene['gas_network']['thermal_contacts']:
        if 'h_w_per_m2_k' in c:
            c['connectivity'] = contact(ports[c['port_id']]['area_m2'], c.pop('h_w_per_m2_k'), destroyed=None)


def reference_gas_conductance(c, port_area):
    """S=0 reference scale for reports; NOT the instantaneous conductance."""
    if 'connectivity' not in c:
        return c['h_w_per_m2_k']*port_area
    law = c['connectivity']
    return law['background_conductance_w_per_k'] + law['active_conductance_w_per_k']/(1.+math.exp(-law.get('geometry_reference_logit', 0.)))

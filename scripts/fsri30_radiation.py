"""Visible exterior ABS radiation, without touching condensed inventories.

Use geometry, not fitted film coefficients, for emitting area. Radiation and
gas convection are distinct parallel edges; the native solver owns their heat.
"""
from copy import deepcopy
import math
from free_gas_mesh import rectangle_overlap


def add_exterior_radiation(base,effective_emissivity=.1):
    if not math.isfinite(effective_emissivity) or not 0.<=effective_emissivity<=1.:
        raise ValueError('Effective emissivity must be in [0,1]')
    if any('radiation'in b for b in base.get('boundaries',[])):
        raise ValueError('Radiation already configured')
    if base['metadata'].get('exhaust_aperture_diagnostic',{}).get('scale',1.)!=1.:
        raise ValueError('Effective aperture-only diagnostic is not exact wall geometry')
    scene=deepcopy(base);ports={p['id']:p for p in scene['ports']}
    exposed={c['port_id'] for c in scene['gas_network']['thermal_contacts']
        if c['port_id'].startswith('gasmesh:ABS:') and ':outside:' in c['port_id']}
    side=math.sqrt(scene['metadata']['geometry_reconstruction']['unmeasured_total_aperture_area_m2']/2.)
    hole=[-side/2,side/2,-side/2,side/2];areas={};count=0
    for ident in sorted(exposed):
        p=ports[ident];axis='xyz'.index(p['surface'][1]);u,v=(axis+1)%3,(axis+2)%3
        center=p['centroid_m'];width,height=p['rectangle_m'];area=p['area_m2']
        if axis==1:
            rectangle=[center[u]-width/2,center[u]+width/2,center[v]-height/2,center[v]+height/2]
            area-=rectangle_overlap(rectangle,hole)
        if area<=1e-15:continue
        fraction=area/p['area_m2']
        if fraction>1.+1e-10:raise ValueError('Visible area exceeds its material patch')
        scene['boundaries'].append(dict(id='radiation:'+ident,port_id=ident,temperature_k=298.15,
            conductance_w_per_k=0.,radiation=dict(effective_emissivity=effective_emissivity,
                area_fraction=min(1.,fraction),temperature_limit_k=4000.)))
        face=p['surface'];areas[face]=areas.get(face,0.)+area;count+=1
    expected=scene['metadata']['air_mesh']['actual_abs_exposure']
    if len(areas)!=6 or any(not math.isclose(area,expected[face]['outside_m2'],rel_tol=1e-9) for face,area in areas.items()):
        raise ValueError('Radiating areas do not close reconstructed exterior geometry')
    scene['metadata']['exterior_radiation']=dict(effective_emissivity=effective_emissivity,
        emissivity_measured=False,visible_area_m2=areas,edges=count,
        source='Grey-body exchange to 298.15 K exterior; geometry removes vent openings',
        scope='Explicit unmeasured exterior boundary prior, NOT a chemistry/material fit',
        condensed_inventory_unchanged=True,calibrated=False,validation_passed=False)
    return scene

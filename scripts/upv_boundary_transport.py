"""Common unresolved aperture path and jet footprint; no material refitting.

The aperture path affects counterexchange, not the pressure-orifice law. Slit
occlusion factors already present in each path are retained. Jet coverage uses
the same fraction on both walls of both open ISS bays, including unobserved C2.
"""
import copy
import math

from upv_gas_access import AREA_COEFFICIENTS, OPEN


def candidate(source,path_scale=1.,jet_fraction=None,*,relative_to_current=False):
    if not math.isfinite(path_scale) or not .5<=path_scale<=10.:
        raise ValueError('Aperture path scale outside [0.5,10]')
    old_jet=source['metadata']['gas_access_geometry']['jet_wetted_fraction']
    jet=old_jet if jet_fraction is None else jet_fraction
    if not math.isfinite(jet) or not .1<=jet<=1.:
        raise ValueError('Common jet coverage outside [0.1,1]')
    if 'boundary_transport_fit' in source['metadata'] and not relative_to_current:
        raise ValueError('Use the explicit unmodified reference, not a previous candidate')
    scene=copy.deepcopy(source);paths=surfaces=0
    for edge in scene['gas_network']['channels']:
        if edge['kind']=='rupture':
            edge['mixing_length_m']*=path_scale;paths+=1
    for contact in scene['gas_network']['thermal_contacts']:
        if any(contact['compartment'].startswith('gap-'+pad+'-')for pad in OPEN):
            for key in AREA_COEFFICIENTS:contact['connectivity'][key]*=jet/old_jet
            surfaces+=1
    if not paths or not surfaces:raise ValueError('No aperture paths or common open ISS surfaces')
    scene['metadata']['gas_access_geometry']['jet_wetted_fraction']=jet
    scene['metadata']['boundary_transport_fit']=dict(path_scale=path_scale,jet_fraction=jet,
        nominal_unoccluded_path_m=.001*path_scale,apertures=paths,surfaces=surfaces,
        measured=False,common_for_all_cells=True,pressure_orifice_law_unchanged=True,
        chemical_and_material_properties_unchanged=True,requires_fresh_verification=True)
    if relative_to_current:
        scene['metadata']['boundary_transport_fit']['relative_to_current_explicit_scene']=True
        scene['metadata']['boundary_transport_fit']['previous_fit']=copy.deepcopy(source['metadata'].get('boundary_transport_fit'))
    return scene

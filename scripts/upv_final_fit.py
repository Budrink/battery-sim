"""Explicit current-scene calibration overlays; no regeneration/default reset.

One shared law per material/contact. C/k, current protocol, chemistry energy,
and observation mapping never change. Solid/electrical/kinetic edits require
a fresh run. Gas/fracture suffixes remain conditional until fresh verification.
"""
import copy
import math
import numpy as np

from upv_tim_gas_access import candidate as tim_candidate
from upv_boundary_transport import candidate as aperture_candidate
from upv_cooling_fit import partition_candidate
from upv_gas_route_refinement import candidate as route_candidate
from upv_backing_response import candidate as backing_candidate

# Multipliers/additive logits are relative to the ACTUAL source scene. The
# absolute fractions below are inherited from that scene unless specified.
BOUNDS = dict(
    entry=(1e-5, .05),
    path=(.5, 10.),
    opening_rate=(.001, 10.),
    strength=(1., 10.),
    case_h=(.5, 8.),
    resistance_charge_delta=(-6., 2.),
    resistance_scale=(.5, 1.22),
    resistance_temperature_delta=(-.005, .015),
    product_conductivity_scale=(.25, 5.),
    r2_rate=(.1, 100.),
    r2_activation_delta=(-40000., 40000.),
    bulk_metric=(.4, 1.2),
    damaged_span=(.25, 2.),
    damaged_logit=(-2., 4.),
    mobile_energy_fraction=(.5, .9),
    jet_fraction=(.1, 1.),
    core_contact_damage=(0., 8.),
    casing_contact_damage=(-4., 0.),
    witness_mouth=(.001, .05),
    wall_hot_scale=(.6, 2.),
    wall_hot_knee_k=(550., 900.),
    pad_footprint=(.2, .75),
    reverse_hot_logit=(-8., 2.),
    casing_damage_contrast=(-8., 0.),
    backing_fraction=(.001, .1),
    reverse_profile_power=(0., 6.),
    casing_gas_damage=(-8., 8.),
    damaged_iss_scale=(.05, 2.),
    core_surface_power=(0., 12.),
    intact_iss_scale=(.5, 8.),
    iss_temperature_contrast=(-8., 0.),
    damaged_tim_scale=(.05, 2.),
    al_surface_softening=(0., 6.),
    tim_al_scale=(.5, 8.),
)
DEFAULT = dict(
    entry=.001, path=1., opening_rate=1., strength=1., case_h=1.,
    resistance_charge_delta=0., r2_rate=1., bulk_metric=1.,
    resistance_scale=1., resistance_temperature_delta=0., r2_activation_delta=0.,
    product_conductivity_scale=1.,
    damaged_span=1., damaged_logit=0., core_contact_damage=0.,
    casing_contact_damage=0., wall_hot_scale=1., reverse_hot_logit=0.,
    wall_hot_knee_k=900.,
    casing_damage_contrast=0.,
    reverse_profile_power=0.,
    casing_gas_damage=0.,
    damaged_iss_scale=1.,
    core_surface_power=0.,
    intact_iss_scale=1.,
    iss_temperature_contrast=0.,
    damaged_tim_scale=1.,
    al_surface_softening=0.,
    tim_al_scale=1.,
)
FRESH_AXES = (
    'case_h', 'resistance_charge_delta', 'resistance_scale',
    'resistance_temperature_delta', 'r2_rate', 'r2_activation_delta', 'bulk_metric',
    'product_conductivity_scale',
    'core_contact_damage', 'casing_contact_damage', 'casing_damage_contrast',
    'damaged_iss_scale',
    'core_surface_power',
    'intact_iss_scale', 'iss_temperature_contrast',
    'damaged_tim_scale',
    'al_surface_softening',
    'tim_al_scale',
)


def only_gas_contacts_changed(reference, trial):
    """Strict proof for using a different overlay pivot on a saved state.

    Metadata is descriptive. Every solid, source, time control, gas inventory
    and gas route must remain byte-value identical; only film laws/areas vary.
    The native restart contract is checked independently as well.
    """
    def frozen(scene):
        value=copy.deepcopy(scene)
        value.pop('metadata',None)
        value['gas_network'].pop('thermal_contacts')
        return value
    return frozen(reference)==frozen(trial)


def hot_wall_response(scene,scale):
    """Fit gas->chamber hot transfer without changing the <=450 K plateau.

    Capacity, geometry, ambient loss and wall-donor reference stay fixed.
    One multiplier applies to every wall partition. This is the already
    existing directed S law, not an extra h bypass or an energy sink.
    """
    for contact in scene['gas_network']['thermal_contacts']:
        if not contact['port_id'].startswith('chamber-wall:'):continue
        law=contact['connectivity'];profile=contact['gas_surface_connectivity']
        floor=law['background_conductance_w_per_k'];old_span=law['active_conductance_w_per_k'];old_z=law['geometry_reference_logit']
        rows=profile['temperature_log_response']
        if any(abs(y)>1e-12 for t,y in rows if t<=450.):raise ValueError('Require the existing constant cold wall response')
        cold=floor+old_span/(1.+math.exp(-old_z));span=(floor+old_span)*scale-floor
        f=(cold-floor)/span;z=math.log(f/(1.-f));updated=[]
        for t,y in rows:
            target=floor+old_span/(1.+math.exp(-old_z-y))
            if t>450.:target*=scale
            f=(target-floor)/span;updated.append([t,math.log(f/(1.-f))-z])
        for prefix in ('','destroyed_'):
            law[prefix+'active_conductance_w_per_k']=span
            law[prefix+'geometry_reference_logit']=z
        profile['temperature_log_response']=updated


def wall_hot_knee(scene,temperature):
    """Fit the existing monotone gas/wall S(T) ramp, not a new heat path.

    Cold transfer through450K and the high-T plateau remain EXACTLY fixed.
    Only the transition endpoint moves, identically on every wall partition.
    It controls warm-gas cooling independently of the peak hot conductance.
    """
    for contact in scene['gas_network']['thermal_contacts']:
        if not contact['port_id'].startswith('chamber-wall:'):continue
        rows=contact['gas_surface_connectivity']['temperature_log_response']
        hot=[row for row in rows if row[0]>450.]
        if len(hot)!=1:raise ValueError('Require one existing hot gas-wall ramp endpoint')
        hot[0][0]=temperature


def hot_solid_to_gas_response(scene,delta):
    """Change only the hot SOLID donor S; gas heating and <=375 K stay fixed."""
    for contact in scene['gas_network']['thermal_contacts']:
        if contact['port_id'].startswith('chamber-wall:'):
            continue
        profile = contact['solid_surface_connectivity']
        old = np.asarray(profile['temperature_log_response'])
        grid = sorted(set(old[:, 0]) | {375., 450.})
        profile['temperature_log_response'] = [
            [t, float(np.interp(t, old[:, 0], old[:, 1]))
             + delta * float(np.interp(t, [375., 450.], [0., 1.]))]
            for t in grid
        ]


def power_solid_to_gas_response(scene, power):
    """Smooth reverse S(T)=power*log(T/Tref), with the cold reference fixed.

    In the unsaturated range this is a power-law effective conductance. Unlike
    subtracting a hot-only logit, cooling never gets stronger as T decreases.
    All non-wall solids share this closure; each keeps its own reference area
    and conductance. Incoming gas heating is completely untouched.
    """
    for contact in scene['gas_network']['thermal_contacts']:
        if contact['port_id'].startswith('chamber-wall:'):
            continue
        profile = contact['solid_surface_connectivity']
        tref = profile['reference_temperature_k']
        grid = sorted({tref, 320., 350., 375., 400., 450., 600., 900.}
                      | {t for t, _ in profile['temperature_log_response']})
        profile['temperature_log_response'] = [
            [t, power * math.log(t / tref)] for t in grid
        ]


def scale_casing_pad_contacts(scene, pad_prefix, *, intact=1., destroyed=1.):
    """One material-pair rule, independently for intact and failed contacts."""
    owners = {port['id']: port['cell_id'] for port in scene['ports']}
    changed = 0
    for edge in scene['interfaces']:
        pair = [owners[edge[key]] for key in ('port_a', 'port_b')]
        if not (any(name.startswith(pad_prefix) for name in pair)
                and any(':case:' in name for name in pair)):
            continue
        for key in ('background_conductance_w_per_k', 'active_conductance_w_per_k'):
            edge['law'][key] *= intact
            edge['law']['destroyed_' + key] *= destroyed
        changed += 1
    if not changed:
        raise ValueError('Expected casing/' + pad_prefix + ' contacts are missing')
    return changed


def soften_aluminium_surface(scene, amplitude):
    """Shared reversible contact geometry, NOT melting or bulk-k degradation.

    A bounded smooth donor S(T) represents unresolved expansion/contact loss.
    The 150 C / 20 K transition is an explicit hypothesis; only its amplitude
    is fitted. All Al objects use the same normalized profile. Measured bulk
    conductivity, capacity and damage are untouched.
    """
    for cell in scene['cells']:
        if cell.get('material_id') != 'aluminium':
            continue
        thermal = cell['thermal']
        surface = thermal['surface_connectivity']
        if any(surface.get(key, [0., 0., 0.]) != [0., 0., 0.]
               for key in ('linear', 'quadratic')):
            raise ValueError('Require the current isotropic Al surface baseline')
        thermal.setdefault('bulk_connectivity', dict(
            kind='directed_geometry', background_factor=.01,
            active_factor=1.98, destroyed_background_factor=.01,
            destroyed_active_factor=1.98, endpoint_damage_weight=1.,
        ))
        thermal.setdefault('bulk_response', copy.deepcopy(surface))
        tref = surface['reference_temperature_k']
        old = np.asarray(surface.get('temperature_log_response', [[tref, 0.]]))
        grid = sorted(set(old[:, 0]) | {tref, 350., 375., 400.,
                                       423.15, 450., 500., 900.})
        if len(grid) > 8:
            raise ValueError('Al profile exceeds the native eight-knot table')
        def soft(t):
            return 1. / (1. + math.exp(-(t-423.15)/20.))
        cold = soft(tref)
        surface['temperature_log_response'] = [
            [t, float(np.interp(t, old[:, 0], old[:, 1]))
             - amplitude * (soft(t)-cold)]
            for t in grid
        ]


def scale_tim_al_contacts(scene, scale):
    """Unmeasured TIM/Al contact h, common to all identical material pairs."""
    owners = {port['id']: port['cell_id'] for port in scene['ports']}
    aluminium = {cell['id'] for cell in scene['cells']
                 if cell.get('material_id') == 'aluminium'}
    count = 0
    for edge in scene['interfaces']:
        pair = [owners[edge[key]] for key in ('port_a', 'port_b')]
        if not (any(name.startswith('TIM') for name in pair)
                and any(name in aluminium for name in pair)):
            continue
        for key in ('background_conductance_w_per_k', 'active_conductance_w_per_k'):
            edge['law'][key] *= scale
        count += 1
    if not count:
        raise ValueError('Expected TIM/aluminium contacts are missing')


def candidate(source,values):
    unknown=set(values)-set(BOUNDS)
    if unknown:raise ValueError('Unknown calibration axes: '+str(unknown))
    fractions={c['conversion']['gas_chemical_fraction']for c in source['cells']if 'conversion'in c}
    if len(fractions)!=1:raise ValueError('Require one common chemistry energy partition')
    baseline_fraction=fractions.pop()
    base_jet=source['metadata']['gas_access_geometry']['jet_wetted_fraction']
    base_mouth=source['metadata']['gas_access_geometry']['iss3_accessible_fraction']
    base_pad=source['metadata']['gas_access_geometry'].get('open_iss_pad_wetted_fraction',1.)
    base_backing=source['metadata']['gas_access_geometry']['backing_open_fraction']
    wall_knots={c['gas_surface_connectivity']['temperature_log_response'][-1][0]
        for c in source['gas_network']['thermal_contacts']if c['port_id'].startswith('chamber-wall:')}
    if len(wall_knots)!=1:raise ValueError('Require one common gas-wall temperature law')
    base_wall_knee=wall_knots.pop()
    p=dict(DEFAULT,mobile_energy_fraction=baseline_fraction,jet_fraction=base_jet,witness_mouth=base_mouth,pad_footprint=base_pad,backing_fraction=base_backing,wall_hot_knee_k=base_wall_knee);p['entry']=source['metadata']['gas_access_geometry'].get('protected_fracture_fraction',source['metadata']['gas_access_geometry']['protected_accessible_fraction']);p.update(values)
    for key,(lo,hi)in BOUNDS.items():
        if not math.isfinite(p[key])or not lo<=p[key]<=hi:raise ValueError('Outside calibration bounds: '+key)
    if p['reverse_profile_power']and p['reverse_hot_logit']:
        raise ValueError('Choose one reverse temperature parametrization, not both')
    if source['metadata']['experiment']!='no-cooling':raise ValueError('Never fit the held-out experiment')
    for cell in source['cells']:
        if cell.get('electrical',{}).get('current_input','measured')!='measured':raise ValueError('Calibration requires the original measured current')
    exit_fraction=source['metadata']['gas_access_geometry']['protected_accessible_fraction']
    scene=tim_candidate(source,exit_fraction,p['entry'])
    if p['path']!=1. or p['jet_fraction']!=base_jet:
        scene=aperture_candidate(scene,p['path'],p['jet_fraction'],relative_to_current=True)
    if p['witness_mouth']!=base_mouth or p['pad_footprint']!=base_pad:
        scene=route_candidate(scene,witness_mouth=p['witness_mouth'],pad_footprint=p['pad_footprint'])
    if p['backing_fraction']!=base_backing:scene=backing_candidate(scene,p['backing_fraction'])
    if p['wall_hot_scale']!=1.:hot_wall_response(scene,p['wall_hot_scale'])
    if p['wall_hot_knee_k']!=base_wall_knee:wall_hot_knee(scene,p['wall_hot_knee_k'])
    if p['reverse_hot_logit']!=0.:hot_solid_to_gas_response(scene,p['reverse_hot_logit'])
    if p['reverse_profile_power']!=0.:power_solid_to_gas_response(scene,p['reverse_profile_power'])
    if p['mobile_energy_fraction']!=baseline_fraction:
        # Redistribute, do not invent energy. This also proposes a rate that
        # keeps local R2 power at the initial frozen (T, precursor) unchanged.
        # The changed gas-production/kinetic history still demands a fresh run.
        scene=partition_candidate(scene,p['mobile_energy_fraction'])
    if p['damaged_iss_scale'] != 1. or p['intact_iss_scale'] != 1.:
        scale_casing_pad_contacts(scene, 'ISS', intact=p['intact_iss_scale'],
                                  destroyed=p['damaged_iss_scale'])
    if p['damaged_tim_scale'] != 1.:
        scale_casing_pad_contacts(scene, 'TIM', destroyed=p['damaged_tim_scale'])
    if p['al_surface_softening']:
        soften_aluminium_surface(scene, p['al_surface_softening'])
    if p['tim_al_scale'] != 1.:
        scale_tim_al_contacts(scene, p['tim_al_scale'])
    altered=0
    for edge in scene['gas_network']['channels']:
        if edge['kind']!='rupture':continue
        edge['opening_pressure_pa']*=p['strength']
        edge['damage_opening']['area_rate_per_s']*=p['opening_rate'];altered+=1
    for shell in scene['shells']:
        shell['contents_contact_h']*=p['case_h']
        if p['casing_contact_damage']!=0.:
            # The donating shell uses its own irreversible damage in S(D).
            # Intact casing stays unchanged. This does not modify the
            # core-donor response, measured shell k/Cp or explicit gas films.
            profile=shell['surface_connectivity']
            profile['damage']=[x+p['casing_contact_damage']for x in profile.get('damage',[0.,0.,0.])]
        if p['casing_damage_contrast']!=0.:
            # Same material rule for every casing: a damaged wall facing an
            # intact core need not couple like a damaged wall facing products.
            # Existing directed_log_response uses this as beta*(D_wall-D_receiver).
            profile=shell['surface_connectivity']
            profile['damage_contrast']=[x+p['casing_damage_contrast']for x in profile.get('damage_contrast',[0.,0.,0.])]
    for cell in scene['cells']:
        if cell['id'].startswith('ISS') and p['iss_temperature_contrast']:
            thermal = cell['thermal']
            surface = thermal['surface_connectivity']
            # Surface state is independent of the measured porous bulk k.
            # Preserve the old internal law explicitly before changing only
            # the donating ISS surface. No battery-index-dependent response.
            thermal.setdefault('bulk_connectivity', dict(
                kind='directed_geometry', background_factor=.01,
                active_factor=1.98, destroyed_background_factor=.01,
                destroyed_active_factor=1.98, endpoint_damage_weight=1.,
            ))
            thermal.setdefault('bulk_response', copy.deepcopy(surface))
            surface['temperature_contrast'] = [
                value + p['iss_temperature_contrast']
                for value in surface.get('temperature_contrast', [0., 0., 0.])
            ]
        if 'electrical'in cell:
            electrical=cell['electrical']
            electrical['charge_log_slope']+=p['resistance_charge_delta']
            electrical['resistance_ohm']*=p['resistance_scale']
            electrical['temperature_log_slope_per_k']+=p['resistance_temperature_delta']
            electrical['product_conductivity_ratio']*=p['product_conductivity_scale']
            # Same engineering bounds as the fresh R123 builder; never hide a
            # bad fit behind a larger R ceiling, clipped heat or altered drive.
            if not .001 <= electrical['resistance_ohm'] <= .0195:
                raise ValueError('Reference resistance outside declared bounds')
            if not -.02 <= electrical['temperature_log_slope_per_k'] <= .02:
                raise ValueError('Resistance temperature slope outside declared bounds')
            if not 0. <= electrical['charge_log_slope'] <= 20.:
                raise ValueError('Resistance charge slope outside declared bounds')
            if not .01 <= electrical['product_conductivity_ratio'] <= 1.:
                raise ValueError('Electrical product conductivity outside declared bounds')
        if 'conversion'in cell:
            conversion=cell['conversion']
            conversion['reaction_rate_per_s']*=p['r2_rate']
            conversion['reaction_activation_j_per_mol']+=p['r2_activation_delta']
            if not 80000. <= conversion['reaction_activation_j_per_mol'] <= 180000.:
                raise ValueError('R2 activation outside declared bounds')
            cold_rate=conversion['reaction_rate_per_s']*math.exp(
                conversion['reaction_activation_j_per_mol']/8.31446261815324*
                (1./conversion['reaction_reference_k']-1./298.15))
            # Retain the original kinetic-search storage prior, not only its
            # rectangular parameter bounds. This is not a fitted ignition gate.
            if cold_rate>1e-7*(1.+1e-9):
                raise ValueError('R2 consumes stored charge too quickly at 25 C')
        if cell.get('role')=='battery'and p['bulk_metric']!=1.:
            for row in cell['thermal']['bulk_response']['temperature_log_response']:row[1]*=p['bulk_metric']
        if cell.get('role')=='battery'and p['core_contact_damage']!=0.:
            # Existing directed S(T,D): only the donating core uses this
            # response. It does NOT increase the reverse casing->core law.
            # Explicit bulk_response keeps internal diffusion unchanged.
            if 'bulk_response'not in cell['thermal']:raise ValueError('Require independent bulk and boundary state profiles')
            surface=cell['thermal']['surface_connectivity']
            surface['damage']=[x+p['core_contact_damage']for x in surface.get('damage',[0.,0.,0.])]
        if cell.get('role') == 'battery' and p['core_surface_power']:
            # Only the donating core changes. A hotter casing does not inherit
            # the core's T-response, nor does this change internal bulk k(T).
            if 'bulk_response' not in cell['thermal']:
                raise ValueError('Require independent bulk and boundary profiles')
            surface = cell['thermal']['surface_connectivity']
            old = np.asarray(surface['temperature_log_response'])
            tref = surface['reference_temperature_k']
            grid = sorted(set(old[:, 0]) | {tref, 350., 425., 500., 900.})
            surface['temperature_log_response'] = [
                [t, float(np.interp(t, old[:, 0], old[:, 1]))
                 + p['core_surface_power'] * math.log(t / tref)]
                for t in grid
            ]
    changed_contacts=0
    if p['damaged_span']!=1. or p['damaged_logit']!=0. or p['casing_gas_damage']!=0.:
        owners={port['id']:port['cell_id']for port in scene['ports']}
        for contact in scene['gas_network']['thermal_contacts']:
            if ':case:'not in owners[contact['port_id']]:continue
            law=contact['connectivity']
            law['destroyed_active_conductance_w_per_k']*=p['damaged_span']
            law['destroyed_geometry_reference_logit']+=p['damaged_logit'];changed_contacts+=1
            if p['casing_gas_damage']:
                # Healthy D=0 keeps the already fitted heating/cooling curve.
                # Only the SOLID donor changes as its casing is damaged.
                profile=contact['solid_surface_connectivity']
                profile['damage']=[x+p['casing_gas_damage']for x in profile.get('damage',[0.,0.,0.])]
    scene['metadata']['final_calibration_trial']=dict(parameters=p,rupture_channels=altered,
        fresh_required=any(p[k]!=DEFAULT[k]for k in FRESH_AXES)or p['mobile_energy_fraction']!=baseline_fraction,reference_is_actual_scene=True,
        measured_materials_unchanged=True,current_unchanged=True,chemical_energy_unchanged=True,
        validation_used_for_selection=False,changed_damaged_casing_contacts=changed_contacts)
    return scene,p

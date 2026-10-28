"""Relative gas-response calibration on an explicit current geometry.

No changes to geometry, chemistry, measured material constants, forward gas
heating, wall cooling, mechanical channels or numerical integration controls.
Cold free-gas conductance is preserved; molecular/internal mixing is untouched.
"""
import copy
import math
import numpy as np

from upv_cooling_fit import front, score
from upv_gas_access import shape_report, PROTECTED_TC, feature_windows
import upv_r123_data as dataio
from upv_observation_policy import VERSION as OBSERVATION_VERSION, CALIBRATION_CHANNELS

CRITERIA_VERSION='cooling-with-protected-pulses-'+OBSERVATION_VERSION


def front_constraints(result):
    """Do not improve cooling by erasing a real witness pulse.

    The lower bound is deliberately broad: peak T remains uncertain, but a
    missing rapid event is a different waveform. Pressure is not a jet TC.
    """
    reasons=list(result['front']['reasons'])
    row=result['shape']['channels']['101']
    ratio=row['calculated_max_rise_1_5s_k']/max(1e-9,row['observed_max_rise_1_5s_k'])
    if ratio<.5:reasons.append('C1/ISS rapid pulse below half the observed jump')
    # A jet-contaminated peak is uncertain, but inventing a violent impulse
    # on a measured smooth/protected face is not a cooling improvement.
    # These exploration bounds are deliberately broad, not exact peak fits.
    if ratio>4.:reasons.append('C1/ISS rapid pulse above four times the observed jump')
    protected={}
    for key in ('102','107'):
        item=result['shape']['channels'][key]
        measured=item['observed_max_rise_1_5s_k'];predicted=item['calculated_max_rise_1_5s_k']
        limit=max(10.,5.*measured)
        protected[key]=dict(predicted_jump_k=predicted,observed_jump_k=measured,maximum_jump_k=limit)
        if predicted>limit:reasons.append(key+': unobserved rapid impulse on protected TIM face')
    pressure=result['front']['features']['pressure']
    pred=pressure['predicted'];obs=pressure['observed']
    error=abs(pred['peak']/obs['peak']-1.)if pred else None
    if error is None or error>.3:reasons.append('Chamber pressure peak outside 30% exploration bracket')
    if result['shape']['no_heat_rejected']:reasons.append('Missing witness heating')
    return dict(accepted=not reasons,reasons=reasons,witness_pulse_ratio=ratio,
        minimum_witness_pulse_ratio=.5,pressure_peak_relative_error=error,
        maximum_witness_pulse_ratio=4.,protected_impulses=protected,
        pressure_peak_exploration_limit=.3,pressure_peak_acceptance_limit=.05)


def gain(law, profile, temperature):
    knots=np.asarray(profile['temperature_log_response'])
    z=law['geometry_reference_logit']+float(np.interp(temperature,knots[:,0],knots[:,1]))
    return law['background_conductance_w_per_k']+law['active_conductance_w_per_k']/(1.+math.exp(-z))


def candidate(source, mixing_scale=1., damage_delta=0., end=1100.):
    if not math.isfinite(mixing_scale)or not .03<=mixing_scale<=3.:
        raise ValueError('Free-gas mixing ratio outside [0.03, 3]')
    if not math.isfinite(damage_delta)or not -6.<=damage_delta<=6.:
        raise ValueError('Casing reverse damage increment outside [-6, 6]')
    scene=copy.deepcopy(source);net=scene['gas_network']
    if not net.get('free_gas_response_only'):
        raise ValueError('Require explicit free-gas-only response; do not change molecular edges')
    if mixing_scale!=1.:
        law=net['mixing_connectivity'];profile=net['surface_connectivity']
        cold=profile['reference_temperature_k'];cold_gain=gain(law,profile,cold)
        old=[(t,gain(law,profile,t))for t,_ in profile['temperature_log_response']]
        floor=law['background_conductance_w_per_k']
        span=(floor+law['active_conductance_w_per_k'])*mixing_scale-floor
        fraction=(cold_gain-floor)/span
        if not 0.<fraction<1.:raise ValueError('Scaled gas law cannot retain the cold reference')
        z=math.log(fraction/(1.-fraction))
        rows=[]
        for t,value in old:
            target=value if t<=cold else value*mixing_scale
            f=(target-floor)/span
            if not 0.<f<1.:raise ValueError('Gas response knot outside finite law range')
            rows.append([t,math.log(f/(1.-f))-z])
        law.update(active_conductance_w_per_k=span,geometry_reference_logit=z,
            destroyed_active_conductance_w_per_k=span,destroyed_geometry_reference_logit=z)
        profile['temperature_log_response']=rows
    if damage_delta!=0.:
        owners={p['id']:p['cell_id']for p in scene['ports']};count=0
        for contact in net['thermal_contacts']:
            if ':case:'not in owners[contact['port_id']]:continue
            profile=contact['solid_surface_connectivity']
            profile['damage']=[v+damage_delta for v in profile.get('damage',[0.,0.,0.])]
            count+=1
        if not count:raise ValueError('No casing reverse contacts')
    scene['simulation']['duration_s']=end
    data,_=dataio.load('no-cooling');event=dataio.event(data)
    scene['failure_sample_times_s']=dataio.observation_times(data,event,end)
    scene['metadata']['gas_response_refinement']=dict(mixing_scale=mixing_scale,
        casing_reverse_damage_delta=damage_delta,reference_is_explicit_current_profile=True,
        gas_geometry_unchanged=True,forward_heating_and_wall_unchanged=True,
        chemical_and_material_parameters_unchanged=True,conditional_until_fresh_verification=True)
    return scene


def assess(run,data,end):
    result=score(run,data,end)
    result['shape']=shape_report(run,data)
    result['time_shift_s']=0.;result['temperature_gain']=1.
    result['preserved_front']=front_constraints(result)
    result['criteria_version']=CRITERIA_VERSION
    result['accepted']=bool(result['accepted']and result['preserved_front']['accepted']
        and result['preserved_front']['pressure_peak_relative_error']<=.05)
    return result


def front_prune_reason(run,data):
    windows=feature_windows(dataio.event(data)['time_s'])
    if run['time_s'][-1]<windows['short_end_s']:return None
    result=dict(front=front(run,data),shape=shape_report(run,data))
    if run['time_s'][-1] < windows['protected_end_s']:
        # Protected faces have a 930 s excitation window. A prefix ending at
        # 916 s is not evidence that they will never warm by 930 s. Hot-face
        # excitation, resolved impulses and pressure timing are already known.
        result['shape']['no_heat_rejected'] = any(
            row.get('excitation_penalty', 0.) > 0.
            for key, row in result['shape']['channels'].items()
            if key in CALIBRATION_CHANNELS and key not in PROTECTED_TC
        )
    check=front_constraints(result)
    return '; '.join(check['reasons'])if not check['accepted']else None

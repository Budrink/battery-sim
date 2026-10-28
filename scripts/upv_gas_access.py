"""Inverse-geometry hypotheses for gas access; not material/TC corrections.

Keep the thermal stack, gas volumes and area-normalized S(T,D), chemistry and fracture
laws unchanged. A common holder/contact occlusion narrows entrances to the
protected slots. A fractured face backed by such a slot has the same hydraulic
access restriction: fracture area/work and gas-accessible area are distinct.
"""
import copy
import math
import numpy as np
import upv_r123_data as dataio
from upv_observation_policy import CALIBRATION_CHANNELS, VERSION as OBSERVATION_VERSION, metadata as observation_policy

PROTECTED = ('TIM1', 'TIM2', 'TIM3', 'ISS3')
OPEN = ('ISS1', 'ISS2')
CHANNELS = ('101','102','C2_ISS','C2_TIM','108','107')
PROTECTED_TC = ('101','102','C2_TIM','107')
# Preserve the original training windows exactly, but locate them relative to
# each experiment's observed event. Raw simulation/data clocks are never moved.
FEATURE_REFERENCE_EVENT_S = 886.6389999166131


def feature_windows(event_s):
    shift = float(event_s) - FEATURE_REFERENCE_EVENT_S
    return dict(baseline_start_s=875.+shift, front_start_s=881.+shift,
                short_end_s=915.+shift, protected_end_s=930.+shift)


AREA_COEFFICIENTS = ('background_conductance_w_per_k','active_conductance_w_per_k',
    'destroyed_background_conductance_w_per_k','destroyed_active_conductance_w_per_k')


def _exposed_area(contact,fraction):
    for key in AREA_COEFFICIENTS:contact['connectivity'][key]*=fraction


def candidate(source, protected_fraction, end=950., lip_open_fraction=1., plume_exit_fraction=1., backing_open_fraction=1., witness_edge_open_fraction=1., jet_wetted_fraction=1., outer_insulation_open_fraction=1., iss3_accessible_fraction=None, open_iss_pad_wetted_fraction=1.):
    if not math.isfinite(protected_fraction) or not 1e-5 <= protected_fraction <= 1.:
        raise ValueError('Protected access fraction outside diagnostic bounds')
    if not math.isfinite(lip_open_fraction) or not 1e-5<=lip_open_fraction<=1.:
        raise ValueError('Holder lip opening outside diagnostic bounds')
    if not math.isfinite(plume_exit_fraction) or not 1e-3<=plume_exit_fraction<=1.:
        raise ValueError('Plume collar opening outside diagnostic bounds')
    if not math.isfinite(backing_open_fraction) or not 1e-5<=backing_open_fraction<=1.:
        raise ValueError('Backing exposure outside diagnostic bounds')
    if not math.isfinite(witness_edge_open_fraction) or not 1e-5<=witness_edge_open_fraction<=1.:
        raise ValueError('Witness edge exposure outside diagnostic bounds')
    if not math.isfinite(jet_wetted_fraction) or not .1<=jet_wetted_fraction<=1.:
        raise ValueError('Jet wetted area outside diagnostic bounds')
    if not math.isfinite(open_iss_pad_wetted_fraction) or not .1<=open_iss_pad_wetted_fraction<=1.:
        raise ValueError('Open ISS pad wetted area outside diagnostic bounds')
    if not math.isfinite(outer_insulation_open_fraction) or not 1e-5<=outer_insulation_open_fraction<=1.:
        raise ValueError('Outer insulation rim exposure outside diagnostic bounds')
    iss3_fraction=protected_fraction if iss3_accessible_fraction is None else iss3_accessible_fraction
    if not math.isfinite(iss3_fraction) or not 1e-5<=iss3_fraction<=1.:
        raise ValueError('Near-witness insulation mouth outside diagnostic bounds')
    def access_fraction(pad):return iss3_fraction if pad=='ISS3'else protected_fraction
    protected_slots=PROTECTED+(('ISS4',)if outer_insulation_open_fraction!=1. else ())
    rim_coverage={p:lip_open_fraction for p in PROTECTED}
    rim_coverage['ISS4']=outer_insulation_open_fraction
    def rim_fraction(port_id):
        return next((f for p,f in rim_coverage.items()if port_id.startswith('opening-contact:'+p+':')),1.)
    s=copy.deepcopy(source)
    if 'gas_access_geometry' in s.get('metadata',{}):
        raise ValueError('Apply geometry to the explicit baseline, not another candidate')
    changed=[]
    for edge in s['gas_network']['channels']:
        pad=next((p for p in protected_slots if edge['id'].startswith('escape-'+p+'-')),None)
        if pad:
            before=edge['area_m2'];edge['area_m2']*=access_fraction(pad)
            changed.append(dict(channel=edge['id'],kind='entrance_throat',slot=pad,
                old_area_m2=before,new_area_m2=edge['area_m2'],length_m=edge['mixing_length_m']))
        backed=next((p for p in protected_slots if edge['to'].startswith('gap-'+p+'-')),None)
        if edge['kind']=='rupture' and backed:
            # Rupture area belongs to the shell solver. Represent an unresolved
            # downstream access throat: hydraulic Cd*A and diffusive A/L must
            # BOTH shrink; changing only Cd would leave a mixing bypass.
            before=edge['discharge_coefficient'];length=edge['mixing_length_m']
            fraction=access_fraction(backed)
            edge['discharge_coefficient']*=fraction
            edge['mixing_length_m']/=fraction
            changed.append(dict(channel=edge['id'],kind='backed_fracture_access',
                receiver=edge['to'],old_cd=before,new_cd=edge['discharge_coefficient'],
                old_length_m=length,new_length_m=edge['mixing_length_m']))
    if not changed:raise ValueError('No access throats selected')
    # Optional porous holder/collar around the C2 -> C3 side. Narrow only
    # existing shared gas faces, never jump across a solid or create a hub.
    # This retains access to the hot-side corridor while limiting immediate
    # dilution into the large outer reservoir. The collar is an unmeasured
    # geometrical hypothesis, not a prescribed jet/temperature source.
    collar=[];bounds=None
    if plume_exit_fraction!=1.:
        objects={c['id']:c for c in s['cells']}
        left,right=objects['ISS1'],objects['TIM2']
        xlo=left['pose']['position_m'][0]-left['geometry']['dimensions_m'][0]/2
        xhi=right['pose']['position_m'][0]+right['geometry']['dimensions_m'][0]/2
        core=objects['C2'];extent=s['metadata']['gas_geometry']['configuration']['outer_extent_m']
        shell=next(sh for sh in s['shells']if sh['contents_id']=='C2')['thickness_m']
        half_y=core['geometry']['dimensions_m'][1]/2+shell+extent
        half_z=core['geometry']['dimensions_m'][2]/2+shell+extent
        y,z=core['pose']['position_m'][1:]
        bounds=dict(lo=[xlo,y-half_y,z-half_z],hi=[xhi,y+half_y,z+half_z])
        positions=s['metadata']['gas_geometry']['node_positions_m']
        inside={name for name,point in positions.items()if name.startswith('void-') and
            all(bounds['lo'][k]-1e-12<=point[k]<=bounds['hi'][k]+1e-12 for k in range(3))}
        for edge in s['gas_network']['channels']:
            if not edge['id'].startswith('void-face-'):continue
            if (edge['from']in inside)==(edge['to']in inside):continue
            area=edge['area_m2'];edge['area_m2']*=plume_exit_fraction
            collar.append(dict(channel=edge['id'],nominal_face_area_m2=area,
                open_area_m2=edge['area_m2']))
        if not collar:raise ValueError('No hot-side collar boundary faces found')
    # The rim galleries additionally expose the casing directly. A holder lip
    # can cover these already-resolved surface patches without covering the
    # whole interior face of the slit. A_eff/A_nom is an unresolved geometric
    # coverage, NOT a change in h, k or the directed temperature/damage law.
    # Homogenize the covered patch on its existing panel: every reference G
    # is extensive in area; keep all logits and endpoint response tables.
    lips=[];ports={p['id']:p for p in s['ports']}
    if lip_open_fraction!=1. or outer_insulation_open_fraction!=1.:
        for contact in s['gas_network']['thermal_contacts']:
            pad=next((p for p in protected_slots if contact['port_id'].startswith('opening-contact:'+p+':')),None)
            if pad is None:continue
            area=ports[contact['port_id']]['area_m2']
            fraction=rim_fraction(contact['port_id'])
            _exposed_area(contact,fraction)
            lips.append(dict(port=contact['port_id'],slot=pad,nominal_area_m2=area,
                gas_exposed_area_m2=area*fraction))
        if not lips:raise ValueError('No resolved rim exposures found for holder lip')
    # Backing plates have low mass and are an independent route: gas -> Al ->
    # TIM -> casing. One common holder-coverage hypothesis covers ALL Al plates,
    # including the outer ISS4 side of Al3, not just instrumented contacts.
    backing=[]
    if backing_open_fraction!=1.:
        for contact in s['gas_network']['thermal_contacts']:
            port=ports[contact['port_id']]
            if port['cell_id']not in ('Al1','Al2','Al3'):continue
            previous=rim_fraction(contact['port_id'])
            fraction=min(previous,backing_open_fraction)
            _exposed_area(contact,fraction/previous)
            backing.append(dict(port=contact['port_id'],nominal_area_m2=port['area_m2'],
                gas_exposed_area_m2=port['area_m2']*fraction))
        if not backing:raise ValueError('No physical gas exposure of Al backings found')
    # The same holder can cover the narrow lateral edges of BOTH witnesses.
    # Leave C2 rupture outlets and the broad open ISS faces untouched. Coverage
    # is common for C1/C3; no thermocouple-specific temperature multiplier.
    witness_edges=[]
    if witness_edge_open_fraction!=1.:
        objects={c['id']:c for c in s['cells']};shells={sh['contents_id']:sh for sh in s['shells']}
        for contact in s['gas_network']['thermal_contacts']:
            port=ports[contact['port_id']];owner=port['cell_id']
            cell=owner.split(':')[0]
            if cell not in ('C1','C3') or ':case:'not in owner or owner[-1]not in ('y','z'):continue
            previous=rim_fraction(contact['port_id'])
            fraction=min(previous,witness_edge_open_fraction);_exposed_area(contact,fraction/previous)
            area=port['area_m2']
            if contact.get('thermal_volume',-1)>=0:
                dims=shells[cell].get('surface_divisions',objects[cell]['thermal']['divisions'])
                axis='xyz'.index(port['surface'][1]);area/=math.prod(dims[k]for k in range(3)if k!=axis)
            witness_edges.append(dict(port=contact['port_id'],thermal_volume=contact.get('thermal_volume',-1),
                nominal_patch_area_m2=area,gas_exposed_patch_area_m2=area*fraction))
        if not witness_edges:raise ValueError('No lateral witness casing contacts found')
    # A hot jet need not wet the entire nominal open slit face. Use ONE common
    # exposed-area fraction for both ISS hot sides, on BOTH casing and pad.
    # This changes deposited energy, not measured Cp/k, gas fuel or TC gains.
    # It is homogenized on current patches; no resolved footprint is claimed.
    jet=[]
    if jet_wetted_fraction!=1.:
        for contact in s['gas_network']['thermal_contacts']:
            if not any(contact['compartment'].startswith('gap-'+p+'-')for p in OPEN):continue
            _exposed_area(contact,jet_wetted_fraction)
            jet.append(dict(compartment=contact['compartment'],port=contact['port_id'],
                wetted_fraction=jet_wetted_fraction))
        if not jet:raise ValueError('No open ISS slit surfaces found')
    # A slit plume need not wet its two opposing walls equally. Test ONE
    # common pad-side footprint on both open ISS slots; leave the facing
    # casing, all solid contacts and every area-normalized S law unchanged.
    # This is an unresolved surface-coverage hypothesis, not measured geometry
    # or a new heat-transfer coefficient. Never apply it to protected ISS3.
    pad_footprint=[]
    if open_iss_pad_wetted_fraction!=1.:
        for contact in s['gas_network']['thermal_contacts']:
            owner=ports[contact['port_id']]['cell_id']
            if owner not in OPEN or not contact['compartment'].startswith('gap-'+owner+'-'):continue
            _exposed_area(contact,open_iss_pad_wetted_fraction)
            pad_footprint.append(dict(compartment=contact['compartment'],port=contact['port_id'],
                relative_to_common_footprint=open_iss_pad_wetted_fraction))
        if not pad_footprint:raise ValueError('No open ISS pad-side surfaces found')
    # The outer ISS4 has no open slit in this stack, but its exposed rim is
    # a real gas -> ISS4 -> Al3 -> TIM3 bypass. Its holder coverage is a separate
    # geometric exposure of the OUTER edge, not a changed ISS conductivity or
    # thermocouple gain. Do not assume that an outer rim and an internal
    # compressed slot have the same opening. Fraction=1 reproduces r8.
    s['metadata']['gas_access_geometry']=dict(protected_slots=list(protected_slots),open_slots=list(OPEN),
        outer_insulation_open_fraction=outer_insulation_open_fraction,
        protected_accessible_fraction=protected_fraction,changes=changed,measured=False,
        holder_lip_open_fraction=lip_open_fraction,holder_lip_exposure=lips,
        plume_exit_fraction=plume_exit_fraction,plume_collar_bounds_m=bounds,plume_collar_faces=collar,
        backing_open_fraction=backing_open_fraction,backing_exposure=backing,
        witness_edge_open_fraction=witness_edge_open_fraction,witness_edge_exposure=witness_edges,
        jet_wetted_fraction=jet_wetted_fraction,jet_wetted_surfaces=jet,
        hypothesis='Common obstruction of gas access by holder/compressed contact; no per-TC gains',
        fracture_laws_unchanged=True,surface_response_and_h_unchanged=True,chemical_energy_unchanged=True,
        inherited_open_ISS1_saturation='Baseline diagnostic widening retained, not a measured dimension',
        requires_fresh_start_before_acceptance=True)
    if iss3_accessible_fraction is not None:
        # The ISS3 entrance may differ from compressed TIM mouths. Change
        # only hydraulic/diffusive access, not its already resolved S law.
        s['metadata']['gas_access_geometry']['iss3_accessible_fraction']=iss3_fraction
    if open_iss_pad_wetted_fraction!=1.:
        s['metadata']['gas_access_geometry']['open_iss_pad_wetted_fraction']=open_iss_pad_wetted_fraction
        s['metadata']['gas_access_geometry']['open_iss_pad_wetted_surfaces']=pad_footprint
    s['simulation']['duration_s']=end
    s['failure_sample_times_s']=[t for t in s['failure_sample_times_s']if t<=end]
    if s['failure_sample_times_s'][-1]!=end:s['failure_sample_times_s'].append(end)
    s['gas_film_trace']=True
    s['process_history_selection']=dict(gas_prefixes=['inside-C2-','gap-'],channel_prefixes=['rupture-','escape-'])
    return s


def rise_front(times,values,baseline,rise):
    """Own-rise timing diagnostic only; never bridge a missing measured scan."""
    t=np.asarray(times);y=np.asarray(values)-baseline
    def cross(fraction):
        level=fraction*rise
        if np.isfinite(y[0]) and y[0]>=level:return float(t[0])
        for i in range(1,len(t)):
            if np.isfinite(y[i-1:i+1]).all() and y[i-1]<level<=y[i]:
                return float(t[i-1]+(t[i]-t[i-1])*(level-y[i-1])/(y[i]-y[i-1]))
        return None
    result={f't{p}_s':cross(p/100.)for p in [10,50,90]}
    result['rise_10_90_s']=(result['t90_s']-result['t10_s']
        if result['t10_s']is not None and result['t90_s']is not None else None)
    return result


def front_concentration_error(calculated_jump,calculated_rise,observed_jump,observed_rise):
    """Pulse presence/sharpness independent of each curve's peak amplitude."""
    return abs(calculated_jump/calculated_rise-observed_jump/observed_rise)


def shape_report(run,data):
    """Shape-first objective. No fitted time shift, gains or cooling loss.

    Own-amplitude normalization is ONLY a shape diagnostic, never applied to
    displayed curves. Explicit excitation floors reject a non-heating model.
    Protected-face short-window changes penalize a false impulse. The rising
    hot-face curves stop at each first peak; cooling is deliberately unscored.
    """
    t=np.asarray(run['time_s']);event=dataio.event(data)['time_s']
    windows=feature_windows(event)
    observed=dataio.observed(data,t);channels={};loss=front_loss=0.
    for key in CHANNELS:
        selected=key in CALIBRATION_CHANNELS
        pred=np.asarray(run['temperature_k'][key]);obs=observed[key]
        baseline=(t>=windows['baseline_start_s'])&(t<=windows['front_start_s'])&np.isfinite(obs)
        if not np.any(baseline):
            if selected:raise ValueError('Missing pre-burst reference')
            channels[key]=dict(available=False,used_for_calibration=False);continue
        p0=float(np.mean(pred[baseline]));o0=float(np.mean(obs[baseline]))
        protected=key in PROTECTED_TC
        limit=windows['protected_end_s'] if protected else event+8.
        phase=(t>=windows['front_start_s'])&(t<=limit)
        mask=phase&np.isfinite(obs)
        if not np.any(mask):
            if selected:raise ValueError('Missing burst observations: '+key)
            channels[key]=dict(available=False,used_for_calibration=False);continue
        tt=t[mask];pp=pred[mask]-p0;oo=obs[mask]-o0
        # Missing experimental scans must not erase an existing model peak
        # from the shape normalizer (C2/ISS has missing scans around rupture).
        pa=max(float(np.max(pred[phase]-p0)),1.);oa=max(float(np.max(oo)),1.)
        if not protected:
            # Shape includes onset/rising front but not the cooling segment.
            stop=max(float(t[phase][np.argmax(pred[phase]-p0)]),float(tt[np.argmax(oo)]))
            keep=tt<=stop;tt,pp,oo=tt[keep],pp[keep],oo[keep]
        normalized_error=float(np.sqrt(np.mean((pp/pa-oo/oa)**2)))
        excitation_floor=max(0.,.2-pa/oa)
        # Quantify a sharp impulse independently of its subsequent cooling.
        dense=np.linspace(windows['front_start_s'],windows['short_end_s'],137)
        pscan=np.interp(dense,t,pred)
        oscan=dataio.observed(data,dense)[key]
        def jump(x):
            dx=x[6:]-x[:-6];return float(np.nanmax(dx))if np.isfinite(dx).any()else 0.
        pj,oj=jump(pscan),jump(oscan)
        concentration=front_concentration_error(pj,pa,oj,oa)
        if selected:front_loss+=concentration
        excess=max(0.,pj-max(20.,2.*oj))/max(20.,abs(oj)) if protected else 0.
        value=normalized_error+3.*excitation_floor+excess
        if selected:loss+=value
        channels[key]=dict(used_for_calibration=selected,available=True,
            shape_rmse=normalized_error,calculated_rise_k=pa,observed_rise_k=oa,
            short_window_s=1.5,calculated_max_rise_1_5s_k=pj,observed_max_rise_1_5s_k=oj,
            false_impulse_penalty=excess,excitation_penalty=excitation_floor,
            normalized_front_error=concentration,
            pre_burst_offset_k=p0-o0,loss=value,
            peak_c=float(np.max(pred[(t>=windows['front_start_s'])&(t<=windows['protected_end_s'])]))-273.15,
            calculated_rise_front=rise_front(t[phase],pred[phase],p0,pa),
            observed_rise_front=rise_front(t[phase],obs[phase],o0,oa))
    return dict(shape_objective=loss,front_concentration_objective=front_loss,
        shape_with_front_objective=loss+front_loss,objective_version='curve-plus-normalized-front-v2-'+OBSERVATION_VERSION,
        channels=channels,cooling_scored=False,peak_amplitude_fitted=False,
        time_shift_s=0.,no_heat_rejected=any(channels[k]['excitation_penalty']>0. for k in CALIBRATION_CHANNELS),
        feature_windows_s=windows,
        observation_policy=observation_policy())


def received_gas_heat(run,start=880.,end=900.):
    """Actual accepted heat per object. No power quadrature or heat rescaling."""
    result={};t=run['time_s']
    for row in run['gas_film_trace'].values():
        q=float(np.interp(end,t,row['heat_to_solid_j'])-np.interp(start,t,row['heat_to_solid_j']))
        result[row['solid']]=result.get(row['solid'],0.)+q
    return result

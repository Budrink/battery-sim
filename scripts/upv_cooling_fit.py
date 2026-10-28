"""Cooling loss and common gas/surface response parameters.

R2 mass yield is distinct from fixed chemical energy. Gas-donor surface heat
laws and measured materials stay fixed. Free-gas mixing may vary, while
internal molecular edges retain their own unchanged reference. No TC gains.
"""
import copy
import math
import numpy as np
import upv_r123_data as dataio
import upv_destruction_objective as guards
from upv_observation_policy import CALIBRATION_CHANNELS, metadata as observation_policy

AXES = dict(gas_yield_scale=(.35,1.15,'log'), casing_damage_metric=(0.,6.,'linear'),
            warm_mixing_scale=(1.,30.,'log'), warm_surface_scale=(1.,50.,'log'),
            gas_transport_response_scale=(1.,100.,'log'))
BASE = dict(gas_yield_scale=1.,casing_damage_metric=0.,warm_mixing_scale=1.,warm_surface_scale=1.,gas_transport_response_scale=1.)
PADS = {'101':'ISS3','102':'TIM3','107':'TIM1','108':'ISS1','C2_ISS':'ISS2','C2_TIM':'TIM2'}


def rejection_reason(run,data,marker):
    """Do not mistake the still-hot source thermocouple for a cold tail.

    Source readings above 150 C retain a finite, explicitly softened abort
    margin. Witnesses, pressure, physical bounds and all raw error reports stay.
    This guard does not change the objective or accept a calibration.
    """
    policy=dict(guards.POLICY)
    if run['time_s'][-1]>marker['time_s']+20.:
        obs=dataio.observed(data,[run['time_s'][-1]])
        hot=any(np.isfinite(obs[k][0])and float(obs[k][0])>423.15
                for k in CALIBRATION_CHANNELS if k.startswith('C2_'))
        policy.update(source_post_margin_k=500. if hot else 75.,source_post_relative=.9 if hot else .5)
    return guards.reject_reason(run,dict(data=data,event=marker),policy)


def partition_candidate(original,fraction):
    """Transfer a fixed energy stock from local R2 heat to mobile R3 fuel.

    The rate proposal preserves initial local power at the SAME (T, B), not
    an asserted trajectory. Full fresh simulation is required: chemistry changes.
    """
    if not math.isfinite(fraction)or not .5<=fraction<=.9:
        raise ValueError('Energy partition outside conservative search bounds')
    scene=copy.deepcopy(original)
    for cell in scene['cells']:
        law=cell.get('conversion')
        if not law:continue
        before=law['gas_chemical_fraction']
        law['reaction_rate_per_s']*=(1.-before)/(1.-fraction)
        law['gas_chemical_fraction']=fraction
    scene.setdefault('metadata',{})['energy_partition_probe']=dict(
        mobile_chemical_fraction=fraction,total_specific_energy_unchanged=True,
        fresh_start_required=True,rate_note='Initial local power proposal at fixed T and precursor mass; not trajectory equivalence')
    return scene


def validate(p):
    if set(p)!=set(AXES):raise ValueError('Unexpected cooling parameters')
    for k,(lo,hi,_) in AXES.items():
        if not math.isfinite(p[k]) or not lo <= p[k] <= hi:raise ValueError('Outside cooling bound: '+k)


def candidate(original,p,end):
    validate(p);s=copy.deepcopy(original)
    for cell in s['cells']:
        law=cell.get('conversion')
        if law:
            for k in ('gas_yield','gas_yield_max'):law[k]*=p['gas_yield_scale']
            if not 0. <= law['gas_yield'] <= law['gas_yield_max'] <= 1.:raise ValueError('Invalid mass yield')
    owners={port['id']:port['cell_id']for port in s['ports']}
    selected=0
    for contact in s['gas_network']['thermal_contacts']:
        owner=owners.get(contact['port_id'],contact['port_id'].rsplit(':outer',1)[0])
        if ':case:' not in owner:continue
        profile=contact['solid_surface_connectivity']
        profile['damage']=[v+p['casing_damage_metric'] for v in profile.get('damage',[0.,0.,0.])]
        selected+=1
    if not selected:raise ValueError('No casing gas contacts were selected; check port ownership')
    # The reverse (solid donor) law was never constrained by the heating fit.
    # Fit its warm response jointly for ALL materials, not per thermocouple.
    # Gas-donor heating, chamber-wall law, room reference and >=400 K stay fixed.
    for contact in s['gas_network']['thermal_contacts']:
        if p['warm_surface_scale']==1.:break
        if contact['port_id'].startswith('chamber-wall:')or owners.get(contact['port_id'],'').startswith('chamber-wall'):continue
        law=contact['connectivity'];profile=contact['solid_surface_connectivity']
        knots=profile['temperature_log_response'];ts=np.array([row[0]for row in knots]);ys=np.array([row[1]for row in knots])
        floor=law['background_conductance_w_per_k'];span=law['active_conductance_w_per_k'];z=law['geometry_reference_logit']
        def gain(t):return floor+span/(1.+math.exp(-z-float(np.interp(t,ts,ys))))
        cap=gain(400.)
        updates={}
        for t,power in ((320.,.5),(350.,1.)):
            g=min(cap,gain(t)*p['warm_surface_scale']**power);f=(g-floor)/span
            updates[t]=math.log(f/(1.-f))-z
        profile['temperature_log_response']=[[t,updates.get(t,float(np.interp(t,ts,ys)))]for t in sorted(set(ts)|set(updates))]
    net=s['gas_network'];law=net['mixing_connectivity'];profile=net['surface_connectivity']
    floor=law['background_conductance_w_per_k'];span=law['active_conductance_w_per_k'];z=law['geometry_reference_logit']
    knots=profile['temperature_log_response'];gains={t:floor+span/(1.+math.exp(-z-v))for t,v in knots}
    # Keep 298.15 K and >=400 K EXACTLY unchanged. Only warm eddy decay changes.
    cap=gains[400.]
    for row in knots:
        if row[0] not in (320.,350.):continue
        g=min(cap,gains[row[0]]*p['warm_mixing_scale']);f=(g-floor)/span
        row[1]=math.log(f/(1.-f))-z
    # Independent, non-saturating free-gas response probe. Molecular internal
    # edges ignore this law in gas_mixing.hpp. Cold G(S=0) remains exactly 1.
    multiplier=p['gas_transport_response_scale']
    if multiplier!=1.:
        old=[(t,floor+span/(1.+math.exp(-z-v)))for t,v in knots]
        new_floor=floor;new_span=(floor+span)*multiplier-new_floor
        fraction=(1.-new_floor)/new_span;new_z=math.log(fraction/(1.-fraction))
        law.update(active_conductance_w_per_k=new_span,geometry_reference_logit=new_z,
            destroyed_background_conductance_w_per_k=new_floor,
            destroyed_active_conductance_w_per_k=new_span,destroyed_geometry_reference_logit=new_z)
        for row,(t,g)in zip(knots,old):
            target=g if t<=298.15 else g*multiplier;fraction=(target-new_floor)/new_span
            row[1]=math.log(fraction/(1.-fraction))-new_z
    s['simulation']['duration_s']=end
    s['metadata']['zone_diagnostics']=False
    s['metadata']['cooling_trial']=dict(p,chemical_energy_unchanged=True,measured_cp_k_unchanged=True,casing_contacts_changed=selected)
    s['gas_film_trace']=False;s['thermal_path_trace']=True;s['surface_probe_trace']=True
    s['process_history_selection']=dict(gas_prefixes=['inside-C2-','gap-'],channel_prefixes=['rupture-'])
    # Observe BOTH sides without replacing any experimental channel's model.
    for probe in list(s['measurement_probes']):
        if probe['id'] in PADS:
            s['measurement_probes'].append(dict(probe,id='diagnostic-pad-'+probe['id'],cell_id=PADS[probe['id']]))
    return s


def encode(p):
    validate(p);v=[]
    for k,(lo,hi,kind)in AXES.items():
        x=p[k]
        if kind=='log':x,lo,hi=map(math.log,(x,lo,hi))
        v.append((x-lo)/(hi-lo))
    return np.asarray(v)


def decode(x):
    if len(x)!=len(AXES)or not np.isfinite(x).all()or np.any(np.asarray(x)<0.)or np.any(np.asarray(x)>1.):
        raise ValueError('Cooling coordinate outside unit cube')
    return {k:float(np.clip(math.exp(math.log(lo)+a*math.log(hi/lo))if kind=='log'else lo+a*(hi-lo),lo,hi))
            for a,(k,(lo,hi,kind))in zip(x,AXES.items())}


def rise(t,y,event):
    t=np.asarray(t);y=np.asarray(y);mask=(t>=event-10.)&(t<=event+20.)&np.isfinite(y)
    t=t[mask];y=y[mask]
    if len(t)<4:return None
    before=y[t<event-5.]
    baseline=float(np.median(before))if len(before)else float(y[0])
    peak=int(np.argmax(y));amplitude=float(y[peak]-baseline)
    if amplitude<=1e-8:return None
    def cross(f):
        level=baseline+f*amplitude;ids=np.flatnonzero(y[:peak+1]>=level)
        if not len(ids):return None
        i=int(ids[0])
        if not i:return float(t[0])
        return float(t[i-1]+(level-y[i-1])/(y[i]-y[i-1])*(t[i]-t[i-1]))
    a,b=cross(.1),cross(.9)
    return dict(amplitude=amplitude,peak=float(y[peak]),peak_s=float(t[peak]),
                width_10_90_s=b-a,t10_s=a,t90_s=b)


def front(run,data):
    event=dataio.event(data)['time_s'];t=run['time_s'];raw=data['time_s'];obs=dataio.observed(data,raw)
    pressure=np.asarray(dataio.chamber_pressure_pa(run))/1e5
    result={};reasons=[]
    for key,values,observed in [('pressure',pressure,obs['pressure_bar']),('108',run['temperature_k']['108'],obs['108'])]:
        a,b=rise(t,values,event),rise(raw,observed,event)
        if b is None:raise ValueError('Experimental burst is unresolved: '+key)
        limit=max(2.,2.*b['width_10_90_s'])
        result[key]=dict(predicted=a,observed=b,maximum_width_s=limit)
        if a is None:reasons.append(key+' has no resolved rise');continue
        if a['width_10_90_s']>limit:reasons.append(key+' front too broad')
        if abs(a['peak_s']-b['peak_s'])>3.:reasons.append(key+' peak timing outside 3 s')
        if a['amplitude']<(.5 if key=='pressure'else .3)*b['amplitude']:
            reasons.append(key+' burst missing')
    return dict(accepted=not reasons,reasons=reasons,features=result)


def score(run,data,end):
    """Fixed full denominators; a partial score is a genuine lower bound.

    Five selected cooling traces count; C2/ISS remains informational.
    Peak amplitude is soft, but rise width and timing are a separate hard gate.
    """
    t=np.asarray(run['time_s']);event=dataio.event(data)['time_s']
    grid=np.r_[np.arange(math.ceil(event+60.),end,1.),end]
    obs=dataio.observed(data,grid);known=(grid<=t[-1]+1e-8)&(grid>=t[0])
    total=0.;channels={}
    for key in dataio.CHANNELS:
        full=np.isfinite(obs[key]);mask=full&known
        e=np.interp(grid[mask],t,run['temperature_k'][key])-obs[key][mask]
        scale=10. if key.startswith('C2_')else 5.
        selected=key in CALIBRATION_CHANNELS
        if selected:total+=float(np.sum((e/scale)**2))/max(1,int(full.sum()))
        channels[key]=dict(rmse_k=float(np.sqrt(np.mean(e*e)))if len(e)else None,
            bias_k=float(np.mean(e))if len(e)else None,tolerance_k=scale,
            evaluated_from_s=float(grid[mask][0])if len(e)else None,covered_samples=int(mask.sum()),
            used_for_calibration=selected)
    full=np.isfinite(obs['pressure_bar']);mask=full&known
    ep=np.interp(grid[mask],t,np.asarray(dataio.chamber_pressure_pa(run))/1e5)-obs['pressure_bar'][mask]
    total+=float(np.sum((ep/.1)**2))/max(1,int(full.sum()))
    f=front(run,data)if t[-1]>=event+20.else None
    if f:
        for key,item in f['features'].items():
            a,b=item['predicted'],item['observed']
            if a:total+=(.1 if key=='108'else .3)*((a['amplitude']/b['amplitude']-1.)**2)
    complete=bool(not run.get('partial',False)and t[-1]>=end-1e-7)
    accepted=complete and f is not None and f['accepted'] and all(channels[k]['rmse_k']is not None and channels[k]['rmse_k']<=channels[k]['tolerance_k']for k in CALIBRATION_CHANNELS) and len(ep)>0 and float(np.sqrt(np.mean(ep*ep)))<=.1
    return dict(objective=total,complete=complete,accepted=bool(accepted),known_until_s=float(t[-1]),
        requested_end_s=end,channels=channels,front=f,pressure_rmse_bar=float(np.sqrt(np.mean(ep*ep)))if len(ep)else None,
        observation_policy=observation_policy())

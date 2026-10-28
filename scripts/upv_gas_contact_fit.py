"""Gas contact calibration only: no chemistry, solid material or routing changes."""
import copy
import math
import numpy as np
import upv_connectivity as links
import upv_destruction_objective as objective
from upv_observation_policy import CALIBRATION_CHANNELS, metadata as observation_policy
import upv_r123_data as dataio

AXES=dict(gas_ceiling=(1.99,300.,'log'),metric_450=(0.,12.,'linear'),
          metric_900=(0.,18.,'linear'),wall_h=(10.,300.,'log'))
KEYS=tuple(AXES)
BASE=dict(gas_ceiling=1.99,metric_450=0.,metric_900=0.,wall_h=80.)
COORDINATES='monotone-gas-temperature-v1'


def gas_mixing_profile(scene,p):
    """Use the EXISTING gas material response also between gas volumes.

    This is a relative reference law: at 298.15 K its response is one.
    Geometry/rho/D and pressure-flow entrainment supply the physical rate.
    No second sensible-heat packet, extra fuel or fitted multiplier is added.
    """
    validate(p)
    scene['gas_network']['surface_connectivity']=dict(reference_temperature_k=298.15,temperature_scale_k=100.,
        temperature_log_response=[[298.15,0.],[350.,0.],[450.,p['metric_450']],[900.,p['metric_900']]])
    ceiling=p['gas_ceiling'];z=math.log(.99/(ceiling-1.))
    scene['gas_network']['mixing_connectivity']=dict(kind='directed_geometry',geometry_reference_logit=z,
        background_conductance_w_per_k=.01,active_conductance_w_per_k=ceiling-.01,
        endpoint_damage_weight=0.,destroyed_geometry_reference_logit=z,
        destroyed_background_conductance_w_per_k=.01,destroyed_active_conductance_w_per_k=ceiling-.01)


def validate(p):
    if set(p)!=set(KEYS):raise ValueError('Unexpected gas-contact parameters')
    for k,(lo,hi,_)in AXES.items():
        if not math.isfinite(p[k])or not lo-1e-12<=p[k]<=hi+1e-12:raise ValueError('Invalid '+k)
    if p['metric_900']<p['metric_450']:
        raise ValueError('Gas connectivity must not decrease with temperature')


def encode(p):
    validate(p)
    values=[]
    for k,(lo,hi,kind)in AXES.items():
        v=p[k]
        # Triangular coordinates cover ONLY ordered temperature knots. Every
        # proposal in the unit cube is admissible; do not clip a falling law.
        if k=='metric_900':lo=p['metric_450']
        if kind=='log':v,lo,hi=map(math.log,(v,lo,hi))
        values.append((v-lo)/(hi-lo))
    return np.asarray(values)


def decode(x):
    if len(x)!=len(KEYS) or not np.isfinite(x).all() or np.any(np.asarray(x)<0.) or np.any(np.asarray(x)>1.):
        raise ValueError('Gas contact proposal outside hard bounds')
    p={}
    for value,(k,(lo,hi,kind))in zip(x,AXES.items()):
        if k=='metric_900':lo=p['metric_450']
        p[k]=math.exp(math.log(lo)+value*math.log(hi/lo))if kind=='log'else lo+value*(hi-lo)
    return p


def candidate(template,p):
    validate(p)
    scene=copy.deepcopy(template);links.explicit_gas_contacts(scene)
    ports={v['id']:v for v in scene['ports']}
    profile=dict(reference_temperature_k=298.15,temperature_scale_k=100.,
        temperature_log_response=[[298.15,0.],[350.,0.],[450.,p['metric_450']],[900.,p['metric_900']]])
    for c in scene['gas_network']['thermal_contacts']:
        wall=c['port_id'].startswith('chamber-wall:gas-partition:')
        area=ports[c['port_id']]['area_m2']
        g=p['wall_h']*area if wall else links.reference_gas_conductance(c,area)
        ceiling=1.99 if wall else p['gas_ceiling']
        floor=.01*g;span=(ceiling-.01)*g;z=math.log(.99/(ceiling-1.))
        # Preserve the cold reference G exactly. Temperature changes S through
        # the EXISTING law, not an extra h(T) bypass. Destruction scales unchanged.
        c['connectivity']=dict(kind='directed_geometry',geometry_reference_logit=z,
            background_conductance_w_per_k=floor,active_conductance_w_per_k=span,
            endpoint_damage_weight=1.,destroyed_geometry_reference_logit=z,
            destroyed_background_conductance_w_per_k=floor,destroyed_active_conductance_w_per_k=span)
        c['gas_surface_connectivity']=copy.deepcopy(profile)
    scene.pop('gas_film_trace',None)
    gas_mixing_profile(scene,p)
    scene['metadata']['gas_contact_trial']=p
    return scene


def suffix_loss(run,meta,start=750.):
    """Use fixed full-suffix denominators; never extrapolate a 750 s state to 0 s."""
    t=np.asarray(run['time_s']);event=meta['event']['time_s'];end=meta['fit_end_s']
    grid=np.unique(np.r_[np.arange(start,end,1.),end]);obs=dataio.observed(meta['data'],grid)
    known=grid<=t[-1]+1e-8;total=0.;phases={};rmse={}
    for phase,lo,hi in [('pre',start,event-5.),('burst',event-5.,event+20.),
                        ('early',event+20.,event+60.),('tail',event+60.,end+.001)]:
        subtotal=0.;rmse[phase]={}
        for key in CALIBRATION_CHANNELS:
            o=obs[key];rise=abs(o-obs[key][0]);weight=1.
            scale=np.full_like(grid,5. if phase=='pre'else 10.)
            if phase=='burst':scale=np.maximum(40.,.35*rise)
            elif phase=='early':scale=np.maximum(15.,.2*rise)
            if key.startswith('C2_')and phase!='pre':
                weight=.35 if phase=='burst'else .2;scale=np.maximum(scale,30.)
            full=(grid>=lo)&(grid<hi)&np.isfinite(o);mask=full&known
            if np.any(mask):
                err=np.interp(grid[mask],t,run['temperature_k'][key])-o[mask]
                subtotal+=weight*float(np.sum(objective.huber_square(err/scale[mask])))/max(1,int(full.sum()))
                rmse[phase][key]=float(np.sqrt(np.mean(err*err)))
        subtotal*=objective.TEMPERATURE_PHASE_WEIGHTS[dict(pre='warmup',burst='burst',early='early',tail='tail')[phase]]
        full=(grid>=lo)&(grid<hi)&np.isfinite(obs['pressure_bar']);mask=full&known
        if np.any(mask):
            err=np.interp(grid[mask],t,np.asarray(dataio.chamber_pressure_pa(run))/1e5)-obs['pressure_bar'][mask]
            scale=.05 if phase=='pre'else .7 if phase=='burst'else .25
            subtotal+=float(np.sum(objective.huber_square(err/scale)))/max(1,int(full.sum()))
        phases[phase]=subtotal;total+=subtotal
    # Reuse the selected nonzero peak-amplitude/impulse penalties. Other terms from
    # the 0-based loss are deliberately NOT used on a checkpoint suffix.
    features=objective.loss(run,meta)['peak_features'] if t[-1]>=event+20. else {}
    total+=sum(features.values())
    return dict(objective=total,phase_loss=phases,rmse_k=rmse,peak_features=features,
        temperature_phase_weights=objective.TEMPERATURE_PHASE_WEIGHTS,
        temperature_peak_weight=objective.TEMPERATURE_PEAK_WEIGHT,
        started_at_s=start,full_horizon_s=end,known_until_s=float(t[-1]),observation_policy=observation_policy())


def critical_history(run,meta):
    t=np.asarray(run['time_s'])
    for at in np.arange(t[0]+25.,t[-1]+5.,5.):
        n=np.searchsorted(t,min(at,t[-1]),side='right')
        slim=dict(time_s=t[:n],temperature_k={k:np.asarray(v)[:n]for k,v in run['temperature_k'].items()},
            gas_observers={'chamber':{'pressure_pa':np.asarray(dataio.chamber_pressure_pa(run))[:n]}})
        reason=objective.reject_reason(slim,meta)
        if reason:return dict(time_s=float(t[n-1]),reason=reason)
    return None


def ranking(row,run,meta):
    if row['status']not in ('complete','pruned'):return None
    if row['feasible']:return (0,row['objective'])
    critical=row.get('critical_violation')
    progress=critical['time_s']if critical else float(run['time_s'][-1])
    # Compare recovery candidates on exactly the SAME completed time interval,
    # not on however far the 20 s checkpoint polling happened to run ahead.
    stop=float(np.floor(progress/10.)*10.);t=np.asarray(run['time_s'])
    n=int(np.searchsorted(t,stop,side='right'))
    slim=dict(time_s=t[:n],temperature_k={k:np.asarray(v)[:n]for k,v in run['temperature_k'].items()},
        gas_observers={'chamber':{'pressure_pa':np.asarray(dataio.chamber_pressure_pa(run))[:n]}})
    return (1,-stop,suffix_loss(slim,meta)['objective'])


def meaningful_gain(rank,old):
    if rank is None:return False
    if old is None:return True
    before=tuple(old['rank'])
    if tuple(rank)>=before:return False
    if rank[:-1]!=before[:-1]:return True
    return before[-1]-rank[-1]>=max(.01,.002*abs(before[-1]))

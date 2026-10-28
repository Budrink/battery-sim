"""Phase-aware observation loss and persistent-divergence guards; no physics.

Five selected thermocouples remain in the loss, including burst peak and impulse.
Brief jet-contaminated readings receive softer scales, never zero weight.
"""
import numpy as np
import upv_r123_data as dataio
from upv_observation_policy import CALIBRATION_CHANNELS, metadata as observation_policy

POLICY = dict(pre_margin_k=20., pre_relative=.30, pre_hold_s=25.,
              burst_before_s=5., burst_after_s=20.,
              early_margin_k=80., early_relative=.35, early_hold_s=20.,
              tail_margin_k=45., tail_relative=.35, tail_hold_s=25.,
              source_post_margin_k=200., source_post_relative=.70,
              max_pressure_bar=40., max_solid_temperature_k=5000.)

# Thermocouple jets are uncertain observations, not equally precise targets.
# Chamber pressure and physical safety are NOT downweighted with temperature.
TEMPERATURE_PHASE_WEIGHTS = dict(warmup=1., burst=.15, early=.5, tail=1.)
TEMPERATURE_PEAK_WEIGHT = .1


def huber_square(x):
    a=np.abs(x)
    return np.where(a<=2.,x*x,4.*a-4.)


def persistent(t,error,margin,hold):
    if len(t)<4 or t[-1]-t[0]<.9*hold:return False
    # One scan is not a critical zone. Require a persistent SIGNED departure.
    ratio=error/margin
    return np.mean(ratio>1.)>=.9 or np.mean(ratio< -1.)>=.9


def reject_reason(run,meta,policy=POLICY):
    t=np.asarray(run['time_s']);event=meta['event']['time_s']
    if not len(t):return None
    end=float(t[-1])
    p=np.asarray(dataio.chamber_pressure_pa(run))/1e5
    if np.max(p)>policy['max_pressure_bar']:return 'critical pressure above 40 bar'
    live=run.get('live_state',{}).get('objects',{})
    if any(v.get('max_temperature_k',0.)>policy['max_solid_temperature_k'] for v in live.values()):
        return 'critical solid temperature above 5000 K'
    if end<60.:return None
    if event-policy['burst_before_s']<=end<=event+policy['burst_after_s']:return None
    if end<event:
        start=60.;hold=policy['pre_hold_s'];floor=policy['pre_margin_k'];relative=policy['pre_relative']
    elif end<event+60.:
        start=event+policy['burst_after_s'];hold=policy['early_hold_s'];floor=policy['early_margin_k'];relative=policy['early_relative']
    else:
        start=event+60.;hold=policy['tail_hold_s'];floor=policy['tail_margin_k'];relative=policy['tail_relative']
    obs=dataio.observed(meta['data'],t)
    for key in CALIBRATION_CHANNELS:
        y=np.asarray(run['temperature_k'][key]);o=obs[key]
        margin=np.maximum(floor,relative*np.abs(o-o[0]))
        if end>event and key.startswith('C2_'):
            margin=np.maximum(margin,np.maximum(policy['source_post_margin_k'],policy['source_post_relative']*np.abs(o-o[0])))
        mask=(t>=max(start,end-hold))&np.isfinite(o)&np.isfinite(y)
        if persistent(t[mask],(y-o)[mask],margin[mask],hold):
            return f'temperature critical zone {key}: persistent error {y[-1]-o[-1]:+.1f} K at {end:.3f} s'
    return None


def loss(run,meta,include_features=True):
    """Fixed FULL-window denominators make the observed prefix a lower bound.

    Feature penalties are added only after their entire event window is known.
    A prefix is never presented as a completed destruction objective.
    """
    t=np.asarray(run['time_s']);event=meta['event']['time_s'];end=meta['fit_end_s']
    grid=np.unique(np.r_[np.arange(0.,end,1.),end]);obs=dataio.observed(meta['data'],grid)
    known=grid<=t[-1]+1e-8;total=0.;parts={}
    phases=[('warmup',0.,event-5.),('burst',event-5.,event+20.),
            ('early',event+20.,event+60.),('tail',event+60.,end+.001)]
    for phase,lo,hi in phases:
        subtotal=0.
        for key in CALIBRATION_CHANNELS:
            o=obs[key];rise=np.abs(o-o[0]);weight=1.
            if phase=='warmup':scale=np.full_like(grid,5.)
            elif phase=='burst':scale=np.maximum(40.,.35*rise)
            elif phase=='early':scale=np.maximum(15.,.20*rise)
            else:scale=np.full_like(grid,10.)
            if phase!='warmup' and key.startswith('C2_'):
                weight=.35 if phase=='burst' else .20
                scale=np.maximum(scale,30.)
            full=(grid>=lo)&(grid<hi)&np.isfinite(o);mask=full&known
            if np.any(mask):
                y=np.interp(grid[mask],t,run['temperature_k'][key])
                subtotal+=weight*float(np.sum(huber_square((y-o[mask])/scale[mask])))/max(1,int(full.sum()))
        subtotal*=TEMPERATURE_PHASE_WEIGHTS[phase]
        scale=.05 if phase=='warmup' else (.7 if phase=='burst' else .25)
        full=(grid>=lo)&(grid<hi)&np.isfinite(obs['pressure_bar']);mask=full&known
        if np.any(mask):
            y=np.interp(grid[mask],t,np.asarray(dataio.chamber_pressure_pa(run))/1e5)
            subtotal+=float(np.sum(huber_square((y-obs['pressure_bar'][mask])/scale)))/max(1,int(full.sum()))
        parts[phase]=subtotal;total+=subtotal
    features={}
    if include_features and t[-1]>=event+20.:
        raw=meta['data']['time_s'];times=raw[(raw>=event-5.)&(raw<=event+20.)]
        if len(times)>1:
            targets=dataio.observed(meta['data'],times)
            pairs=[(key,run['temperature_k'][key],targets[key],80.,.35 if key.startswith('C2_') else 1.)for key in CALIBRATION_CHANNELS]
            pairs.append(('pressure',np.asarray(dataio.chamber_pressure_pa(run))/1e5,targets['pressure_bar'],1.,1.))
            for key,values,o,floor,weight in pairs:
                y=np.interp(times,t,values);valid=np.isfinite(o)
                if valid.sum()<2:continue
                x=times[valid];a=np.maximum(0.,o[valid]-o[valid][0]);b=np.maximum(0.,y[valid]-o[valid][0])
                amplitude=(np.max(b)-np.max(a))/max(floor,.5*np.max(a))
                area_a=float(np.trapz(a,x));area_b=float(np.trapz(b,x))
                impulse=(area_b-area_a)/max(floor*(x[-1]-x[0])*.25,.5*area_a)
                confidence=1. if key=='pressure' else TEMPERATURE_PEAK_WEIGHT
                value=confidence*weight*float(huber_square(amplitude)+huber_square(impulse))
                features[key]=value;total+=value
    return dict(objective=float(total),phase_loss=parts,peak_features=features,
                temperature_phase_weights=TEMPERATURE_PHASE_WEIGHTS,
                temperature_peak_weight=TEMPERATURE_PEAK_WEIGHT,
                full_window_s=end,known_until_s=float(t[-1]),observation_policy=observation_policy())


def score(run,meta):
    if run.get('incomplete') or run.get('partial') or run['time_s'][-1]<meta['fit_end_s']-1e-7:return None
    result=loss(run,meta)
    # Raw phase RMSE remains visible alongside the robust fitting loss.
    import calibrate_upv_r123 as cal
    result['phases']=cal.metrics(run,meta)
    result['objective_note']='Five T channels; C2/ISS informational; cooling prioritized, burst/peaks softened but nonzero; pressure and safety unchanged; no fitted time shift'
    return result

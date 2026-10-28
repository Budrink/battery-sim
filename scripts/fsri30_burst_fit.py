"""All-wave gate plus the legacy late-peak diagnostic, before cooling.

The median peak alone can NEVER qualify a cascade. Individual-channel wave
checks are mandatory; all times remain on the original clock.
"""
from dataclasses import asdict,dataclass
import numpy as np
from scipy.ndimage import median_filter
from fsri30_objective import model_fields,targets


@dataclass(frozen=True)
class BurstPolicy:
    timing_target_s:float=10.
    timing_limit_s:float=100.
    phase_end_s:float=3600.
    confirmation_s:float=40.
    smoothing_seconds:int=11
    ascent_lookback_s:float=300.
    peak_absolute_tolerance_k:float=250.
    peak_relative_kelvin_tolerance:float=.25
    maximum_rise_duration_factor:float=4.
    rise_duration_slack_s:float=30.
    minimum_converted_components:int=2


POLICY=BurstPolicy()


def features(t,temperature,policy=POLICY):
    """Last upward half/90% crossings of the principal peak's local ascent."""
    t=np.asarray(t);temperature=np.asarray(temperature)
    i=int(np.argmax(temperature));peak=float(temperature[i])
    left=int(np.searchsorted(t,t[i]-policy.ascent_lookback_s))
    j=left+int(np.argmin(temperature[left:i+1]));base=float(temperature[j])
    out=dict(peak_time_s=float(t[i]),peak_temperature_k=peak,local_base_time_s=float(t[j]),
        local_base_temperature_k=base)
    for fraction,label in ((.5,'rise50_time_s'),(.9,'rise90_time_s')):
        level=base+fraction*(peak-base);values=temperature[j:i+1]
        crossing=np.flatnonzero((values[:-1]<level)&(values[1:]>=level))
        if len(crossing):
            k=j+int(crossing[-1]);out[label]=float(np.interp(level,temperature[k:k+2],t[k:k+2]))
        else:out[label]=None
    a,b=out['rise50_time_s'],out['rise90_time_s']
    out['rise50_90_duration_s']=b-a if a is not None and b is not None else None
    return out


def reference(data,policy=POLICY):
    target=targets(data);mask=(target['time']>=target['event'])&(target['time']<=policy.phase_end_s)
    curve=target['q'][mask,1];times=target['time'][mask]
    out=features(times,curve,policy)
    out.update(window_start_s=target['event'],window_end_s=policy.phase_end_s,
        evaluation_after_s=out['peak_time_s']+policy.timing_limit_s+policy.confirmation_s,
        scope='Collective median thermal front; 9 TC versus 30 component means, no sensor mapping')
    return out


def conversion_count(scene,run):
    count=0
    for c in scene['cells']:
        if c['role']!='battery':continue
        law=c['conversion']
        if law.get('gas_yield_temperature_per_k',0.) or law.get('gas_yield_density_exponent',0.) or law['gas_yield']<=0.:
            raise ValueError('Burst reporting requires direct R2 mass or a constant positive gas yield')
        generated=run['chemistry'][c['id']]['r2_generated_gas_kg'][-1]
        count+=generated/law['gas_yield']>=.1*law['mass_kg']
    return int(count)


def _late_peak_assessment(scene,run,data,policy=POLICY):
    ref=reference(data,policy);t,_,v=model_fields(run);end=float(t[-1])
    result=dict(evaluated=False,accepted=None,phase_window_complete=end>=policy.phase_end_s-1e-7,
        last_observed_s=end,reference=ref,policy=asdict(policy),scope=ref['scope'])
    # Do NOT reject an early local peak while a valid later collective peak
    # can still occur. Do NOT fill a future model phase by interpolation.
    if end<ref['evaluation_after_s']-1e-7:return result
    target=targets(data);grid=target['time'][(target['time']>=ref['window_start_s'])&
        (target['time']<=min(end,policy.phase_end_s)+1e-7)]
    median=np.interp(grid,t,np.median(v,axis=1))
    median=median_filter(median,size=policy.smoothing_seconds,mode='nearest')
    measured=features(grid,median,policy);count=conversion_count(scene,run)
    offsets={key:measured[key]-ref[key] if measured[key]is not None and ref[key]is not None else None
        for key in ('peak_time_s','rise50_time_s','rise90_time_s')}
    timing=max((abs(x)for x in offsets.values()if x is not None),default=1e9)
    amplitude=abs(measured['peak_temperature_k']-ref['peak_temperature_k'])
    amplitude_limit=max(policy.peak_absolute_tolerance_k,
        policy.peak_relative_kelvin_tolerance*ref['peak_temperature_k'])
    duration=measured['rise50_90_duration_s'];ref_duration=ref['rise50_90_duration_s']
    duration_limit=max(policy.maximum_rise_duration_factor*ref_duration,
        ref_duration+policy.rise_duration_slack_s)if ref_duration is not None else 30.
    reasons=[];violation=0.
    if count<policy.minimum_converted_components:
        reasons.append(f'No autonomous cascade: {count} R2 components');violation=max(violation,1.)
    if any(x is None for x in offsets.values()):
        reasons.append('No resolved collective heating front');violation=max(violation,1.)
    if timing>policy.timing_limit_s+1e-7:
        reasons.append(f'Burst/front timing error {timing:.1f} s exceeds {policy.timing_limit_s:g} s')
        violation=max(violation,timing/policy.timing_limit_s-1.)
    if amplitude>amplitude_limit:
        reasons.append(f'Collective peak error {amplitude:.1f} K exceeds softened {amplitude_limit:.1f} K')
        violation=max(violation,amplitude/amplitude_limit-1.)
    if duration is None or duration>duration_limit:
        reasons.append('Collective heating front is absent or too gradual')
        violation=max(violation,1. if duration is None else duration/duration_limit-1.)
    result.update(evaluated=True,accepted=not reasons,measured=measured,
        converted_components=count,timing_offsets_s=offsets,timing_error_s=timing,
        timing_target_met=timing<=policy.timing_target_s+1e-7,
        peak_error_k=amplitude,peak_error_limit_k=amplitude_limit,
        rise_duration_limit_s=duration_limit,constraint=max(.001,violation)if reasons else 0.,
        reason='; '.join(reasons)if reasons else None)
    return result


def assess(scene,run,data,policy=POLICY):
    from fsri30_wave_fit import assess as assess_waves, WavePolicy
    result=_late_peak_assessment(scene,run,data,policy)
    waves=assess_waves(scene,run,data,WavePolicy(phase_end_s=policy.phase_end_s,
        timing_limit_s=policy.timing_limit_s,timing_target_s=policy.timing_target_s,
        confirmation_s=policy.confirmation_s))
    result['sequence']=waves
    result['all_waves_required']=True
    if waves['accepted'] is False:
        result.update(evaluated=True,accepted=False,
            constraint=max(result.get('constraint',0.),waves['constraint']),
            reason='; '.join(x for x in (result.get('reason'),waves['reason'])if x))
    elif result['accepted'] is True and waves['accepted'] is None:
        result.update(evaluated=False,accepted=None)
    result['timing_target_met']=bool(result.get('timing_target_met')and waves['timing_target_met'])
    if 'timing_error_s'in result:
        result['timing_error_s']=max(result['timing_error_s'],waves['timing_error_s'])
    return result


def objective(burst):
    """Completed BURST-window objective; never an invented full-tail score."""
    if not burst.get('phase_window_complete')or not burst.get('accepted'):return None
    return max(0.,burst['timing_error_s']-POLICY.timing_target_s)/10.

#!/usr/bin/env python3
"""Fresh GPU-only UPV calibration: R123 chemistry + I²R, from ambient to failure.

No old scenarios/profiles are read. Every candidate keeps measured material
properties fixed. The held-out experiment is never used to select parameters.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
from scipy.optimize import lsq_linear, minimize, LinearConstraint
import upv_r123_model as model
import upv_r123_data as dataio
from realtime_guard import RealtimeFloor, completed_performance

OUT=dataio.ROOT/'data/calibration/upv-r123-kirchhoff-v4'
PROBE=Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe')
HEAT=('resistance_ohm','resistance_temperature','resistance_charge','product_conductivity_ratio','charge_formation_gain','r1_activation','r2_rate','r2_activation','energy_j','gas_fraction','gas_yield_temperature',
      'precursor_yield','precursor_temperature',
      'iss_h','tim_h','battery_iss_factor','battery_tim_factor','case_h','environment_h','surface_iss','surface_tim','bulk_slope','case_warm_response','case_hot_response')
FAILURE=('r1_activation','charge_formation_gain','precursor_yield','r2_rate','r2_activation','energy_j','gas_fraction','gas_yield','gas_yield_temperature','gas_rate','gas_activation','shell_strength_pa','opening_rate','chamber_gas_h')
LINEAR={'resistance_temperature','resistance_charge','product_conductivity_ratio','r1_activation','r2_activation','gas_activation','gas_yield_temperature','precursor_yield',
        'precursor_temperature','surface_iss','surface_tim','bulk_slope','gas_fraction','gas_yield',
        'case_warm_response','case_hot_response'}
# R2 reference is 450 K. Below ~330 K the current prior is already sluggish;
# the early window is for I²R, not for φ or Ea,R2.
R2_GATE_K=330.
ONSET_WINDOW_S=10.


def set_onset_window(window):
    global ONSET_WINDOW_S
    ONSET_WINDOW_S=float(window)
    if not 1.<=ONSET_WINDOW_S<=900.:
        raise ValueError('onset window must be in [1, 900] s')
PROTOCOL=dict(
    joule=('resistance_ohm','resistance_temperature','resistance_charge','product_conductivity_ratio'),
    phi_r2=('precursor_yield','r2_rate','r2_activation','energy_j'),
    contacts=('iss_h','tim_h','battery_iss_factor','battery_tim_factor','case_h','environment_h','surface_iss','surface_tim','bulk_slope',
              'case_warm_response','case_hot_response'),
    # Charge-onset window: R2 is Arrhenius-off, R1 is heat-neutral. Identify
    # I²R, including charge-dependent effective R, and S-links. Do not move
    # φ, Ea,R2, or energy.
    onset10=('resistance_ohm','resistance_charge','iss_h','tim_h','case_h','environment_h','surface_iss','surface_tim','bulk_slope'))
SOURCE_FACES=('C2_TIM','C2_ISS')
INNER_FACES=('C2_TIM','C2_ISS','101','107')
OUTER_FACES=('102','108')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def fingerprint():
    files=[Path(__file__),Path(model.__file__),Path(dataio.__file__),PROBE,
           dataio.ROOT/'scripts/upv_gas_geometry.py',dataio.ROOT/'scripts/gas_void_geometry.py',
           dataio.ROOT/'scripts/realtime_guard.py']
    files+=sorted((dataio.ROOT/'native').rglob('*.hpp'))+sorted((dataio.ROOT/'native').rglob('*.cuh'))
    files+=sorted((dataio.ROOT/'native').rglob('*.cpp'))+sorted((dataio.ROOT/'native').rglob('*.cu'))
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


def coordinate(p,key):
    return p[key] if key in LINEAR else np.log(p[key])


def value(x,key):
    return float(x if key in LINEAR else np.exp(x))


def range_for(key):
    lo,hi=model.BOUNDS[key]
    return (lo,hi) if key in LINEAR else (np.log(lo),np.log(hi))


def measured_c2(data, times):
    obs=dataio.observed(data,np.asarray(times))
    return .5*(obs['C2_TIM']+obs['C2_ISS'])


def stack_capacity_j_per_k(scene):
    shells={s['contents_id']:s for s in scene['shells']}
    total=0.
    for cell in scene['cells']:
        if cell['id'] in ('chamber-wall','cooling-plate'):continue
        cap=cell['thermal']['heat_capacity_j_per_k']
        shell=shells.get(cell['id'])
        if shell:
            dims=cell['geometry']['dimensions_m']
            outer=[v+2*shell['thickness_m'] for v in dims]
            cap+=(math.prod(outer)-math.prod(dims))*shell['density_kg_m3']*shell['specific_heat_j_kg_k']
        total+=cap
    return total


def onset_balance(name='no-cooling'):
    """0-D heat and R1/R2 rates on the first ONSET_WINDOW_S of charging.

    R2 is treated as off: its Arrhenius factor at ~300 K is ~1e-7 of the 450 K
    reference. R1 still produces B (heat-neutral). Stored heat is C*ΔT; I²R is
    the only source in this window. Two C*ΔT totals bound the unobserved layers.
    """
    d,_=dataio.load(name);mark=dataio.event(d)
    onset=dataio.charge_onset_s(name,d,mark);stop=onset+ONSET_WINDOW_S
    scene,_=model.build(name,warmup=True,duration=stop,dense_after=onset,dense_span=ONSET_WINDOW_S)
    cap=stack_capacity_j_per_k(scene)
    t0,t1=dataio.observed(d,[onset]),dataio.observed(d,[stop])
    delta={ch:float(t1[ch][0]-t0[ch][0]) for ch in dataio.CHANNELS}
    mean_c2=.5*(delta['C2_TIM']+delta['C2_ISS'])
    mean_all=float(np.mean(list(delta.values())))
    cells=next(c for c in scene['cells'] if c['id']=='C2')
    c2_cap=cells['thermal']['heat_capacity_j_per_k']
    if name=='no-cooling':
        raw,_=dataio.electrical_record()
        hold=raw[:,0]<=ONSET_WINDOW_S
        i2=float(np.trapz(raw[hold,2]**2,raw[hold,0]))
        work=float(np.trapz(np.maximum(0.,raw[hold,1]*raw[hold,2]),raw[hold,0]))
        current=float(raw[0,2]);voltage=float(raw[0,1])
    else:
        current=35.;voltage=float(np.interp(onset,d['time_s'],d['U115_v']))
        i2=current*current*ONSET_WINDOW_S;work=voltage*current*ONSET_WINDOW_S
    mixed=float(np.clip(cap*mean_c2/max(i2,1e-12),*model.BOUNDS['resistance_ohm']))
    isolated=float(np.clip(c2_cap*mean_c2/max(i2,1e-12),*model.BOUNDS['resistance_ohm']))
    temperature=.5*(float(t0['C2_TIM'][0])+float(t0['C2_ISS'][0]))
    rgas=8.31446261815324
    p=model.DEFAULT
    ka=p['r1_rate']*math.exp(p['r1_activation']/rgas*(1./400.-1./temperature))
    charge_rate=p['charge_formation_gain']*current/(35.*3600.)
    kb=p['r2_rate']*math.exp(p['r2_activation']/rgas*(1./450.-1./temperature))
    r1_fraction=1.-math.exp(-ka*ONSET_WINDOW_S)
    return dict(experiment=name,onset_s=onset,window_s=ONSET_WINDOW_S,temperature_k=temperature,
                delta_k=delta,mean_c2_k=mean_c2,mean_all_k=mean_all,
                stack_capacity_j_per_k=cap,c2_capacity_j_per_k=c2_cap,
                stored_if_mixed_j=cap*mean_c2,stored_if_c2_isolated_j=c2_cap*mean_c2,
                stored_if_mean_all_j=cap*mean_all,integral_i2_a2s=i2,supplied_ui_j=work,
                well_mixed_resistance_ohm=mixed,isolated_c2_resistance_ohm=isolated,
                r1_thermal_rate_per_s=ka,r1_converted_fraction=r1_fraction,charging_rate_per_s=charge_rate,
                r2_rate_per_s=kb,r2_converted_fraction=1.-math.exp(-kb*ONSET_WINDOW_S),
                r2_treated_as_off=True,r1_heat_neutral=True,
                notes='Observed ΔT~0.05 K. Neighbours barely move, so heat is still in C2: isolated C*ΔT matches ~2 mΩ. Mixed-stack 15 mΩ would require the whole module to rise. R2/R1 heat is not a 10 s source.')


def r2_gate_s(data,event):
    """First training time the measured C2 mean reaches R2_GATE_K.

    Before that the Arrhenius tail is not the explanation for a missing watt.
    """
    time=np.asarray(data['time_s']);c2=measured_c2(data,time)
    hit=time[(c2>=R2_GATE_K)&(time<=event-20.)]
    return float(hit[0]) if len(hit) else float(event-20.)


def objective_windows(stage,event,stop,protocol=None,gate=None,onset=None):
    if protocol=='onset10':
        start=float(onset or 0.)
        return [(start,min(start+ONSET_WINDOW_S,stop),.1)]
    if stage!='heat':
        return [(0.,event-5.,3.),(event-5.,event+3.,100.),(event+3.,event+30.,30.),
                (event+30.,event+300.,10.),(event+300.,event+1800.,5.)]
    split=gate if gate is not None else 300.
    early=(0.,min(split,stop),2.)
    late=(min(split,stop),min(event-20.,stop),2.)
    if protocol=='joule':
        return [early]
    if protocol=='phi_r2':
        # Keep the early window as a guard: R2 must not leak into the Joule era.
        return [(early[0],early[1],4.),late]
    return [early,late]


def electrical_envelope_residual(run,meta,stop=None):
    """Explicit effective-loss approximation, NOT an independently known DCIR.

    The unexplained terminal voltage is allowed in [3.65, 4.2] V. At the
    lower R edge this assigns quasi-steady polarization losses to effective R;
    the interval is a sensitivity prior, not measured OCV bounds. Native also
    enforces the upper electrical-work limit on every accepted charging step.
    Use diagnostic snapshot timestamps, not the following observation time.
    """
    state=run['electrical']['C2'];t=np.asarray(state['snapshot_time_s'])
    i=np.asarray(state['current_a']);r=np.asarray(state['resistance_ohm'])
    raw,_=dataio.electrical_record();marker=meta['event']['time_s']
    measured=np.interp(t,raw[:,0]+marker-raw[-1,0],raw[:,1])
    unexplained=measured-i*r
    distance=np.maximum(0.,unexplained-4.2)-np.maximum(0.,3.65-unexplained)
    blocks=[]
    for lo,hi,scale in [(0.,marker-20.,.15),(marker-20.,marker,.5)]:
        if stop is not None:hi=min(hi,float(stop))
        mask=(t>=lo)&(t<hi)&(i>1.)&(r>0.)&np.isfinite(distance)
        # Keep objective dimension invariant for full and finite-difference
        # trajectories: sampling times are fixed, but a sample may have I=0.
        block=np.zeros_like(t);block[mask]=distance[mask]/(scale*np.sqrt(max(1,mask.sum())))
        blocks.append(block)
    return np.concatenate(blocks)


def residual(run,meta,stage,electrical_envelope=False,protocol=None,stop=None):
    at=np.asarray(run['time_s']);obs=dataio.observed(meta['data'],at);event=meta['event']['time_s'];blocks=[]
    end=at[-1] if stop is None else min(float(stop),at[-1])
    onset=dataio.charge_onset_s(meta.get('experiment','no-cooling'),meta['data'],meta['event']) if protocol=='onset10' else None
    gate=r2_gate_s(meta['data'],event) if stage=='heat' and protocol!='onset10' else None
    windows=objective_windows(stage,event,end,protocol,gate,onset)
    for lo,hi,scale in windows:
        mask_time=(at>=lo)&(at<=hi)
        for name in dataio.CHANNELS:
            # Source thermocouple attachment after ejection is not established;
            # keep its curves diagnostic rather than identifying bulk chemistry
            # from a jet-contaminated broken sensor.
            if stage!='heat' and name.startswith('C2_') and hi>event:continue
            error=np.asarray(run['temperature_k'][name])-obs[name]
            mask=mask_time&np.isfinite(error)
            selected=error[mask]
            # First 10 s of charge: identify the pulse, not the rest offset at onset.
            if protocol=='onset10' and len(selected):selected=selected-selected[0]
            if len(selected):blocks.append(selected/(scale*np.sqrt(len(selected))))
    if protocol!='onset10':
        dp=np.asarray(dataio.chamber_pressure_pa(run))/1e5-obs['pressure_bar']
        if stage=='heat':
            mask=np.isfinite(dp)
            if mask.any():blocks.append(dp[mask]/(.02*np.sqrt(mask.sum())))
        else:
            for lo,hi,scale in [(0.,meta['event']['previous_s'],.03),(event,event+30.,.25),(event+30.,event+1800.,.1)]:
                mask=(at>=lo)&(at<=hi)&np.isfinite(dp)
                if mask.any():blocks.append(dp[mask]/(scale*np.sqrt(mask.sum())))
    if electrical_envelope:blocks.append(electrical_envelope_residual(run,meta,end))
    return np.concatenate(blocks)


def first_divergence(run,meta,margin_k=8.,hold_s=30.,ignore_s=60.):
    """First time a channel stays off by more than margin_k for hold_s."""
    t=np.asarray(run['time_s']);obs=dataio.observed(meta['data'],t)
    dt=float(np.median(np.diff(t))) if len(t)>1 else hold_s
    need=max(3,int(round(hold_s/max(dt,1e-6))))
    start=None;channel=None
    for i,time in enumerate(t):
        if time<ignore_s:continue
        worst=None
        for name in dataio.CHANNELS:
            err=float(run['temperature_k'][name][i]-obs[name][i])
            if not np.isfinite(err):continue
            if abs(err)>=margin_k and (worst is None or abs(err)>abs(worst[1])):
                worst=(name,err)
        if worst is None:
            start=None;channel=None;continue
        if start is None:
            start=i;channel=worst[0]
        if t[i]-t[start]>=hold_s and i-start+1>=need:
            return dict(time_s=float(t[i]),channel=channel,
                        error_k=float(run['temperature_k'][channel][i]-obs[channel][i]))
    return None


def diagnose_heat(run,meta):
    """Decide joule vs R2 vs transport from early/late C2 and spatial skew.

    R2 is not an early-window explanation. Uniform early cold is I²R headroom.
    Late cold after a good early prefix is φ / Ea,R2. Mixed hot/cold faces are
    contacts. A hot C2 mean with cold surface probes is internal conduction.
    """
    t=np.asarray(run['time_s']);obs=dataio.observed(meta['data'],t)
    event=meta['event']['time_s'];gate=r2_gate_s(meta['data'],event)
    def bias(names,mask):
        vals=[]
        for name in names:
            err=np.asarray(run['temperature_k'][name])-obs[name]
            sel=err[mask&np.isfinite(err)]
            if len(sel):vals.append(float(np.mean(sel)))
        return float(np.mean(vals)) if vals else 0.
    early=t<=gate;late=t>gate
    faces=tuple(dataio.CHANNELS)
    early_c2=bias(SOURCE_FACES,early);late_c2=bias(SOURCE_FACES,late)
    inner=bias(INNER_FACES,t>=0);outer=bias(OUTER_FACES,t>=0)
    end=np.zeros(len(t),bool);end[-1]=True
    end_bias={name:bias((name,),end) for name in faces}
    skew=float(max(end_bias.values())-min(end_bias.values())) if end_bias else 0.
    core=None
    means=run.get('cell_mean_temperature_k') or {}
    if 'C2' in means:
        core=float(means['C2'][-1])-.5*(run['temperature_k']['C2_TIM'][-1]+run['temperature_k']['C2_ISS'][-1])
    if early_c2<=-5. and abs(skew)<12.:
        action='joule'
    elif late_c2<=-5. and early_c2>-5.:
        action='phi_r2'
    elif core is not None and core>=8. and early_c2<=-5.:
        action='internal'
    elif skew>=8. and max(end_bias.values())>=4. and min(end_bias.values())<=-4.:
        action='contacts'
    elif late_c2<=-5.:
        action='phi_r2'
    else:
        action='contacts' if abs(skew)>=8. else 'hold'
    return dict(action=action,gate_s=gate,early_c2_bias_k=early_c2,late_c2_bias_k=late_c2,
                inner_bias_k=inner,outer_bias_k=outer,skew_k=skew,core_minus_surface_k=core,
                divergence=first_divergence(run,meta))


def observed_runaway_rejection(run,meta,margin_k=40.,after_s=90.,protocol=None):
    t=np.asarray(run['time_s'])
    if not len(t) or t[-1]<after_s:return None
    obs=dataio.observed(meta['data'],t)
    window=(t>=max(after_s,t[-1]-30.))
    for name in dataio.CHANNELS:
        err=np.asarray(run['temperature_k'][name])-obs[name]
        mask=window&np.isfinite(err)
        if mask.sum()>=3 and float(np.sqrt(np.mean(err[mask]**2)))>margin_k:
            return f'{name}: trailing RMSE {float(np.sqrt(np.mean(err[mask]**2))):.1f} K vs observation at {t[-1]:.3f} s'
    return None


def partial_objective_lower_bound(run,scene,meta,stage,protocol=None):
    """Known squared residuals / FULL-window denominators; unseen errors >=0."""
    t=np.asarray(run['time_s']);full=np.asarray(scene['failure_sample_times_s'])
    obs=dataio.observed(meta['data'],t);all_obs=dataio.observed(meta['data'],full)
    event=meta['event']['time_s'];bound=0.
    onset=dataio.charge_onset_s(meta.get('experiment','no-cooling'),meta['data'],meta['event']) if protocol=='onset10' else None
    gate=r2_gate_s(meta['data'],event) if stage=='heat' and protocol!='onset10' else None
    for lo,hi,scale in objective_windows(stage,event,full[-1],protocol,gate,onset):
        for name in dataio.CHANNELS:
            if stage!='heat' and name.startswith('C2_') and hi>event:continue
            denominator=np.count_nonzero((full>=lo)&(full<=hi)&np.isfinite(all_obs[name]))
            error=np.asarray(run['temperature_k'][name])-obs[name]
            mask=(t>=lo)&(t<=hi)&np.isfinite(error)
            selected=error[mask]
            if protocol=='onset10' and len(selected):selected=selected-selected[0]
            if denominator and len(selected):bound+=float(selected@selected)/(scale*scale*denominator)
    pressure_windows=[] if protocol=='onset10' else ([(0.,full[-1],.02)] if stage=='heat' else [
        (0.,meta['event']['previous_s'],.03),(event,event+30.,.25),(event+30.,event+1800.,.1)])
    error=np.asarray(dataio.chamber_pressure_pa(run))/1e5-obs['pressure_bar']
    for lo,hi,scale in pressure_windows:
        denominator=np.count_nonzero((full>=lo)&(full<=hi)&np.isfinite(all_obs['pressure_bar']))
        mask=(t>=lo)&(t<=hi)&np.isfinite(error)
        if denominator:bound+=float(error[mask]@error[mask])/(scale*scale*denominator)
    return bound


def heat_sources(run):
    """Actual cumulative ledger entries, not inferred from thermocouple slopes."""
    count=len(run['chemistry']);times=len(run['time_s'])
    local=np.asarray(run['released_energy_j_history']).reshape(times,count)
    r2=float(np.sum(local[-1]));joule=float(sum(v[-1] for v in run['joule_energy_j'].values()))
    r3=float(sum(v['released_energy_j'][-1] for v in run['gas'].values()))
    total=joule+r2+r3
    stored_input=sum(v.get('electrical_chemical_input_j',[0.])[-1] for v in run['chemistry'].values())
    precursor_charge_energy=sum(v.get('precursor_charge_energy_j',[0.])[-1] for v in run['chemistry'].values())
    return dict(joule_j=joule,r2_local_j=r2,r3_j=r3,
                electrical_chemical_input_j=float(stored_input),precursor_charge_energy_j=float(precursor_charge_energy),
                joule_share=joule/total if total>0. else 0.)


def source_prior_residual(run,minimum):
    # A declared hypothesis, NOT another measured signal. Two percentage
    # points below its lower bound contribute one normalized residual unit.
    return max(0.,minimum-heat_sources(run)['joule_share'])/.02


def fit_increment(jacobian,residual,lower,upper,share_gradient=None,share_gap=None):
    """Small host-side proposal only; every physical trajectory is CUDA.

    A linearized source-share constraint guides the proposal. Acceptance below
    checks the ACTUAL integrated share, not this local approximation.
    """
    if share_gradient is None:return lsq_linear(jacobian,-residual,bounds=(lower,upper)).x
    result=minimize(lambda x:float(np.sum((jacobian@x+residual)**2)),np.zeros(len(lower)),
        jac=lambda x:2.*jacobian.T@(jacobian@x+residual),method='SLSQP',bounds=list(zip(lower,upper)),
        constraints=[LinearConstraint(np.atleast_2d(share_gradient),share_gap,np.inf)],
        options=dict(ftol=1e-9,maxiter=150))
    return result.x if result.success else np.zeros(len(lower))


def local_trajectory_rejection(run,reference,meta,stage,factor=1.,protocol=None):
    """Reject sustained local blow-ups, not one noisy thermocouple sample.

    Compare a trailing 30-s interval against the same interval of the best
    completed trajectory. Absolute margins are K / bar, not relative Kelvin.
    This is a search heuristic, unlike the rigorous prefix objective bound.
    """
    t=np.asarray(run['time_s']);rt=np.asarray(reference['time_s'])
    if len(t)<3:return None
    event=meta['event']['time_s'];obs=dataio.observed(meta['data'],t)
    # Never mix the sharp event and its immediate aftermath with warmup/cooling.
    # The 10 s charge window is a 0.05 K signal; 2 K RMSE is already hopeless.
    phases=[(0.,float('inf'),2.)] if protocol=='onset10' else [(0.,event-5.,20.),(event-5.,event+3.,100.),
            (event+3.,event+30.,60.),(event+30.,float('inf'),20.)]
    lo,hi,margin=next(p for p in phases if p[0]<=t[-1]<=p[1])
    window=(t>=max(lo,t[-1]-30.))&(t<=hi)
    for name in dataio.CHANNELS:
        if stage!='heat' and name.startswith('C2_') and t[-1]>=event-5.:continue
        error=np.asarray(run['temperature_k'][name])-obs[name]
        ref_error=np.interp(t,rt,reference['temperature_k'][name])-obs[name]
        mask=window&np.isfinite(error)&np.isfinite(ref_error)
        if mask.sum()<3:continue
        rms=float(np.sqrt(np.mean(error[mask]**2)))
        ref_rms=float(np.sqrt(np.mean(ref_error[mask]**2)))
        limit=factor*max(margin,2.*ref_rms+margin/2.)
        if rms>limit and np.count_nonzero(np.abs(error[mask])>limit)>=3:
            return f'{name}: trailing interval RMSE {rms:.2f} K > {limit:.2f} K at {t[-1]:.3f} s'
    return None


def metrics(run,meta,onset=None):
    at=np.asarray(run['time_s']);obs=dataio.observed(meta['data'],at);event=meta['event']['time_s'];result={}
    phases=[('warmup',0.,event-20.),('peak',event-3.,event+3.),
            ('early',event+3.,event+30.),('propagation',event+30.,event+300.),('cooldown',event+300.,event+1800.)]
    if onset is not None:
        phases=[('onset',float(onset),float(onset)+ONSET_WINDOW_S)]+phases
    for phase,lo,hi in phases:
        rows={}
        for channel in dataio.CHANNELS:
            error=np.asarray(run['temperature_k'][channel])-obs[channel]
            selected=error[(at>=lo)&(at<=hi)&np.isfinite(error)]
            if len(selected):rows[channel]=dict(rmse_k=float(np.sqrt(np.mean(selected**2))),bias_k=float(np.mean(selected)))
        if rows:result[phase]=rows
    pressure=np.asarray(dataio.chamber_pressure_pa(run))/1e5
    result['pressure']=dict(peak_bar=float(np.max(pressure)),observed_peak_bar=float(np.nanmax(obs['pressure_bar'])),
        peak_time_s=float(at[np.argmax(pressure)]),rmse_bar=float(np.sqrt(np.mean((pressure-obs['pressure_bar'])**2))))
    result['joule_j']={key:values[-1] for key,values in run['joule_energy_j'].items()}
    result['heat_sources']=heat_sources(run)
    if 'electrical' in run and 'C2' in run['electrical']:
        e=run['electrical']['C2'];i=np.asarray(e['current_a']);r=np.asarray(e['resistance_ohm']);mask=i>1.
        result['electrical_response']=dict(max_resistance_ohm=float(np.max(r)),
            max_power_w=float(np.max(e['power_w'])),
            charging_last_resistance_ohm=float(r[mask][-1]) if mask.any() else None,
            voltage_decomposition='Effective R includes unresolved polarization; full UI budget enforced; a voltage-envelope prior is opt-in')
    result['wall_s']=run['wall_s'];result['steps']=run['steps']
    result['balance']={key:run[key] for key in ('max_thermal_residual_j','max_chemical_residual_j','max_mass_residual_kg')}
    return result


def replay(scene,directory,label,timeout,refinement=1.,reject_partial=None):
    check_initial_voltage(scene)
    # Native output is transient. Retain only the best compressed replay and
    # small metrics; do not regenerate gigabytes of discarded trial fields.
    with tempfile.TemporaryDirectory(prefix='battery-r123-trial-') as tmp:
        root=Path(tmp);path=root/'scene.json';save(path,scene)
        def retain_checkpoint():
            checkpoint=root/'partial.json'
            if checkpoint.exists():
                partial=json.loads(checkpoint.read_text());partial['incomplete']=True
                with gzip.open(directory/'last-incomplete-run.json.gz','wt') as stream:
                    json.dump(partial,stream,allow_nan=False)
        env=dict(os.environ,BATTERY_SIM_PROGRESS='1')
        with (root/'run.json').open('w') as output,(directory/(label+'.log')).open('w') as progress:
            child=subprocess.Popen([str(PROBE),str(path)],stdin=subprocess.PIPE,stdout=output,stderr=progress,text=True,env=env)
            try:
                child.stdin.write(json.dumps(dict(refinement=refinement,checkpoint_path=str(root/'partial.json')))+'\n')
                child.stdin.close();started=time.monotonic();inspected=-1.
                performance=RealtimeFloor()
                while child.poll() is None:
                    remaining=timeout-(time.monotonic()-started)
                    if remaining<=0:raise subprocess.TimeoutExpired(child.args,timeout)
                    try:child.wait(timeout=min(5.,remaining))
                    except subprocess.TimeoutExpired:pass
                    speed=performance.inspect((directory/(label+'.log')).read_text())
                    if speed and not speed['accepted']:
                        save(directory/'last-speed-rejected.json',dict(label=label,**speed,
                            reason='Below rolling 0.25x realtime floor; timestep/physics were NOT relaxed'))
                        raise RuntimeError('Performance rejected: %.4fx realtime < %.2fx rolling floor'%
                                           (speed['realtime_ratio'],performance.minimum))
                    checkpoint=root/'partial.json'
                    if reject_partial and checkpoint.exists() and checkpoint.stat().st_mtime!=inspected:
                        inspected=checkpoint.stat().st_mtime
                        partial=json.loads(checkpoint.read_text())
                        reason=reject_partial(partial)
                        if reason:
                            save(directory/'last-pruned.json',dict(label=label,last_sample_s=partial['time_s'][-1],reason=reason))
                            raise RuntimeError('Pruned: '+reason)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                retain_checkpoint()
                raise
        if child.returncode:
            retain_checkpoint()
            raise RuntimeError((directory/(label+'.log')).read_text()[-3000:])
        run=json.loads((root/'run.json').read_text())
        if 'error' in run:
            retain_checkpoint()
            raise RuntimeError(run['error'])
    speed=completed_performance(run['advanced_to_s']-run.get('started_at_s',0.),run['wall_s'])
    if not speed['accepted']:
        with gzip.open(directory/'last-slow-complete-run.json.gz','wt')as stream:json.dump(run,stream,allow_nan=False)
        save(directory/'last-speed-rejected.json',dict(label=label,**speed,
            reason='Completed trajectory slower than 0.5x realtime'))
        raise RuntimeError('Completed trajectory below required 0.5x realtime')
    if run['max_thermal_residual_j']>1e-4 or run['max_chemical_residual_j']>1e-4 or run['max_mass_residual_kg']>1e-10:
        raise ValueError('Candidate failed energy/mass conservation')
    actual={c['id']:c for c in run['applied_objects']}
    for c in scene['cells']:
        a=actual[c['id']];t=c['thermal']
        if a['capacity_j_k']!=t['heat_capacity_j_per_k'] or a['conductivity_w_m_k']!=t['conductivity_w_per_m_k']:
            raise ValueError('Measured material properties changed')
    if scene['metadata']['experiment']=='no-cooling':
        electrical,_=dataio.electrical_record();offset=scene['metadata']['electrical']['offset_s']
        electrical_t=electrical[:,0]+offset;power=np.maximum(0.,electrical[:,1]*electrical[:,2])
        # Bound cumulative dissipated Joule heat by actually supplied electrical
        # work. The remaining work is not silently treated as another heat source.
        accumulated=np.r_[0.,np.cumsum(.5*(power[:-1]+power[1:])*np.diff(electrical_t))]
        available=np.interp(run['time_s'],electrical_t,accumulated,left=0.,right=accumulated[-1])
        joule=np.asarray(run['joule_energy_j']['C2'])
        stored=np.asarray(run['chemistry']['C2'].get('electrical_chemical_input_j',np.zeros_like(joule)))
        if np.any(joule+stored>available*1.01+1.):raise ValueError('I²R plus chemical storage exceeds supplied U*I work')
    return run


def check_initial_voltage(scene):
    if scene['metadata']['experiment']!='no-cooling':return
    electrical,_=dataio.electrical_record()
    source=next(c for c in scene['cells'] if c['id']=='C2');law=source['electrical']
    temperature=source['thermal']['initial_temperature_k']
    resistance=np.clip(law['resistance_ohm']*np.exp(law['temperature_log_slope_per_k']*
        (temperature-law['reference_temperature_k'])),law['min_resistance_ohm'],law['max_resistance_ohm'])
    # Honor only the scene's explicit reserve; primary UPV calibration uses zero.
    # Charging work is budgeted separately, not hidden in an assumed OCV floor.
    floor=law.get('reversible_voltage_floor_v',0.)
    if floor+electrical[0,2]*resistance>electrical[0,1]+.05:
        raise ValueError('Initial I*R exceeds measured terminal voltage minus explicit scene reserve')


def write_best(directory,stage,scene,run,p,meta,score,prior=None,onset=None):
    save(directory/(stage+'-parameters.json'),dict(parameters=p,objective=score,metrics=metrics(run,meta,onset),
        fingerprint=fingerprint(),validation_used_for_selection=False,status='candidate, not independently validated',
        calibration_prior=prior))
    save(directory/(stage+'-scenario.json'),scene)
    with gzip.open(directory/(stage+'-run.json.gz'),'wt') as stream:json.dump(run,stream,allow_nan=False)


def fit(args):
    stage=args.stage;directory=args.output;directory.mkdir(parents=True,exist_ok=True)
    allowed=HEAT if stage=='heat' else FAILURE
    if args.protocol:
        if stage!='heat' or args.protocol not in PROTOCOL:raise ValueError('Protocol is heat-only: joule, phi_r2, contacts, onset10')
        names=PROTOCOL[args.protocol]
        if args.keys:
            names=tuple(args.keys.split(','))
            allowed_protocol=PROTOCOL[args.protocol]
            if not names or any(key not in allowed_protocol for key in names) or len(set(names))!=len(names):
                raise ValueError('Keys must be a unique subset of protocol '+args.protocol)
    else:
        names=tuple(args.keys.split(',')) if args.keys else allowed
    if not names or any(key not in allowed for key in names) or len(set(names))!=len(names):
        raise ValueError('Invalid parameter subset for '+stage)
    dense_after=None
    if args.protocol=='onset10':
        d,_=dataio.load('no-cooling');mark=dataio.event(d)
        dense_after=dataio.charge_onset_s('no-cooling',d,mark)
        if args.duration is None:args.duration=dense_after+ONSET_WINDOW_S
        save(directory/'onset-balance.json',onset_balance('no-cooling'))
    p=dict(model.DEFAULT)
    if args.protocol=='onset10' and not args.resume and not args.start:
        p['resistance_ohm']=json.loads((directory/'onset-balance.json').read_text())['isolated_c2_resistance_ohm']
    if args.resume:
        saved=json.loads((directory/(stage+'-parameters.json')).read_text())
        if (saved.get('calibration_prior') or {}).get('joule_share_min')!=args.joule_share_min:
            raise ValueError('Resume must preserve the declared Joule-share hypothesis')
        if bool((saved.get('calibration_prior') or {}).get('electrical_envelope'))!=args.electrical_envelope:
            raise ValueError('Resume must preserve the electrical decomposition prior')
        p.update(saved['parameters'])
    if args.start:
        # Analytic stack-balance seed. It fixes nothing: the objective is still
        # measured by an actual CUDA trajectory, on the same data and windows.
        seed=json.loads(args.start.read_text());p.update((seed.get('fit') or seed)['parameters'])
    if stage=='failure' and not args.resume and not args.start:
        p.update(json.loads((directory/'heat-parameters.json').read_text())['parameters'])
    history_path=directory/(stage+'-history.json')
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    best=float('inf');best_run=None;count=0;started=time.monotonic();run_fingerprint=fingerprint()
    def evaluate(candidate,label,derivative=False):
        nonlocal best,best_run,count
        count+=1;scene,meta=model.build('no-cooling',candidate,warmup=stage=='heat',duration=args.duration,
                                         dense_after=dense_after,dense_span=ONSET_WINDOW_S if args.protocol=='onset10' else 10.)
        save(directory/'status.json',dict(stage=stage,candidate=label,parameters=candidate,evaluations=count))
        print('START',label,flush=True)
        # Slightly worse finite differences are useful derivatives, not failed
        # proposals. Keep them, but still cut off grossly divergent probes.
        allowance=4. if derivative else 1.
        def reject(partial):
            if partial_objective_lower_bound(partial,scene,meta,stage,args.protocol)>allowance*best*(1+1e-9):
                return 'Partial objective already exceeds allowed complete-candidate objective'
            runaway=observed_runaway_rejection(partial,meta,protocol=args.protocol)
            if runaway:return runaway
            return local_trajectory_rejection(partial,best_run,meta,stage,allowance,args.protocol) if best_run else None
        try:run=replay(scene,directory,label,args.timeout,reject_partial=reject)
        except (subprocess.TimeoutExpired,ValueError,RuntimeError) as error:
            status='pruned' if str(error).startswith('Pruned:') else 'failed'
            history.append(dict(label=label,parameters=candidate,status=status,error=str(error)))
            save(directory/(stage+'-history.json'),history);print(status.upper(),label,str(error),flush=True);return None
        r=residual(run,meta,stage,args.electrical_envelope,args.protocol)
        if args.joule_share_min is not None:r=np.r_[r,source_prior_residual(run,args.joule_share_min)]
        score=float(r@r);share=heat_sources(run)['joule_share']
        feasible=args.joule_share_min is None or share>=args.joule_share_min
        improved=score<best and feasible
        diagnosis=diagnose_heat(run,meta) if stage=='heat' else None
        summary=dict(label=label,parameters=candidate,status='complete',objective=score,metrics=metrics(run,meta,dense_after),fingerprint=run_fingerprint,
                     joule_share_min=args.joule_share_min,electrical_envelope=args.electrical_envelope,source_hypothesis_satisfied=feasible,
                     protocol=args.protocol,diagnosis=diagnosis)
        history.append(summary);save(directory/(stage+'-history.json'),history)
        if improved:
            best=score;best_run=run;write_best(directory,stage,scene,run,candidate,meta,score,
                dict(joule_share_min=args.joule_share_min,share_residual_scale=.02,
                     electrical_envelope=args.electrical_envelope,unexplained_voltage_prior_v=[3.65,4.2],
                     protocol=args.protocol,onset_window_s=ONSET_WINDOW_S if args.protocol=='onset10' else None,
                     r2_treated_as_off=args.protocol=='onset10'),onset=dense_after)
        print('DONE',label,'objective',round(score,5),'best',round(best,5),'native_s',round(run['wall_s'],2),
              'Joule_share',round(heat_sources(run)['joule_share'],4),'elapsed_s',round(time.monotonic()-started),end='',flush=True)
        if diagnosis:
            print(' action',diagnosis['action'],'early_C2',round(diagnosis['early_c2_bias_k'],2),
                  'late_C2',round(diagnosis['late_c2_bias_k'],2),'skew',round(diagnosis['skew_k'],2),flush=True)
        else:
            print(flush=True)
        return r,score,feasible,share
    base=evaluate(p,'baseline-'+stage)
    if base is None:raise RuntimeError('Baseline could not run; diagnose before optimizing')
    if args.protocol=='onset10':
        diag=next((row.get('diagnosis') or {} for row in reversed(history) if row.get('label')=='baseline-'+stage),{})
        c2=max(abs(float(diag.get('early_c2_bias_k') or 0.)),abs(float(diag.get('late_c2_bias_k') or 0.)))
        if c2>=8.:
            print('DIVERGED baseline |C2|',round(c2,2),'K; skip local rounds',flush=True)
            save(directory/'status.json',dict(stage=stage,status='diverged',best_objective=base[1],evaluations=count,
                 wall_s=time.monotonic()-started,validation_used_for_selection=False))
            print(json.dumps(json.loads((directory/(stage+'-parameters.json')).read_text())['metrics'],indent=2),flush=True)
            return
    # Broad electrical/kinetic regimes first, not a long local optimizer trapped
    # on a flat objective. Every proposal is actually evaluated on CUDA.
    regimes=[('local-heat-rich',dict(energy_j=500000.,gas_fraction=.3)),
             ('precursor-rich',dict(precursor_yield=.9,gas_fraction=.3)),
             ('delayed-r2',dict(r2_rate=3.,r2_activation=120000.,energy_j=500000.,gas_fraction=.3,precursor_yield=.8)),
             ('stored-precursor',dict(r2_rate=.5,r2_activation=100000.,energy_j=500000.,gas_fraction=.3,precursor_yield=.8)),
             ('steep-r2',dict(r2_rate=10.,r2_activation=150000.,energy_j=500000.,gas_fraction=.3,precursor_yield=.8))] if stage=='heat' else [
             ('energy-high',dict(energy_j=500000.)),('gas-fast',dict(gas_rate=100.)),
             ('strong-shell',dict(shell_strength_pa=1e7)),('gas-rich',dict(gas_fraction=.85))]
    if stage=='heat' and args.joule_share_min is not None:
        regimes=[
            ('joule-hotter',dict(resistance_ohm=.024,resistance_temperature=.003,product_conductivity_ratio=.7,charge_formation_gain=.5,r2_rate=100.,r2_activation=100000.,gas_fraction=.7)),
            ('joule-gentle',dict(resistance_ohm=.024,resistance_temperature=.002,product_conductivity_ratio=.5,charge_formation_gain=.4,r2_rate=30.,r2_activation=90000.,gas_fraction=.6)),
            ('joule-late',dict(resistance_ohm=.023,resistance_temperature=.003,product_conductivity_ratio=.5,charge_formation_gain=.4,r2_rate=100.,r2_activation=120000.,gas_fraction=.6)),
            ('resistance-dominant',dict(resistance_ohm=.024,resistance_temperature=.004,product_conductivity_ratio=.4,charge_formation_gain=.25,r2_rate=100.,r2_activation=100000.,gas_fraction=.8)),
            ('joule-reservoir',dict(resistance_ohm=.023,resistance_temperature=.001,product_conductivity_ratio=.3,charge_formation_gain=.4,r2_rate=300.,r2_activation=120000.,gas_fraction=.75))]
    if not args.no_regimes:
        for label,changes in regimes:
            result=evaluate(dict(p,**changes),label)
            if result and result[2] and result[1]<base[1]:p=dict(p,**changes);base=result
    lo=np.array([range_for(k)[0] for k in names]);hi=np.array([range_for(k)[1] for k in names])
    for iteration in range(args.rounds):
        x=np.array([coordinate(p,k) for k in names]);columns=[];share_columns=[]
        for j,key in enumerate(names):
            direction=-1. if x[j]>.5*(lo[j]+hi[j]) else 1.
            step=direction*min(args.difference_fraction*(hi[j]-lo[j]),(hi[j]-x[j]) if direction>0 else (x[j]-lo[j]))
            if abs(step)<.001*(hi[j]-lo[j]):step=-direction*args.difference_fraction*(hi[j]-lo[j])
            q=dict(p);q[key]=value(x[j]+step,key);result=evaluate(q,f'{stage}-r{iteration}-{key}',derivative=True)
            columns.append((result[0]-base[0])/step if result else np.zeros_like(base[0]))
            share_columns.append((result[3]-base[3])/step if result else 0.)
        jac=np.column_stack(columns);span=hi-lo
        # One bounded log-secant proposal per round, with a fixed trust region.
        increment=fit_increment(jac*span,base[0],np.maximum((lo-x)/span,-args.trust_fraction),np.minimum((hi-x)/span,args.trust_fraction),
            np.asarray(share_columns)*span if args.joule_share_min is not None else None,
            args.joule_share_min-base[3] if args.joule_share_min is not None else None)
        proposed=np.clip(x+increment*span,lo,hi)
        q=dict(p,**{key:value(proposed[j],key) for j,key in enumerate(names)})
        result=evaluate(q,f'{stage}-r{iteration}-joint')
        if result is None or not result[2] or result[1]>=base[1]:
            for damping in (.5,.25):
                reduced=x+damping*(proposed-x)
                q=dict(p,**{key:value(reduced[j],key) for j,key in enumerate(names)})
                result=evaluate(q,f'{stage}-r{iteration}-joint-damped-{damping}')
                if result and result[2] and result[1]<base[1]:break
        best_row=min((v for v in history if v['status']=='complete' and v.get('fingerprint')==run_fingerprint and
                      v.get('joule_share_min')==args.joule_share_min and bool(v.get('electrical_envelope'))==args.electrical_envelope and
                      v.get('source_hypothesis_satisfied',True)),key=lambda v:v['objective'])
        if best_row['objective']<base[1]:
            p=best_row['parameters'];base=evaluate(p,f'{stage}-r{iteration}-accepted')
        else:break
    final=json.loads((directory/(stage+'-parameters.json')).read_text())
    save(directory/'status.json',dict(stage=stage,status='finished',best_objective=final['objective'],evaluations=count,
         wall_s=time.monotonic()-started,validation_used_for_selection=False))
    print(json.dumps(final['metrics'],indent=2),flush=True)


def main():
    cli=argparse.ArgumentParser(__doc__);cli.add_argument('stage',choices=['heat','failure','trial','validate'])
    cli.add_argument('--output',type=Path,default=OUT);cli.add_argument('--timeout',type=float,default=600.)
    cli.add_argument('--rounds',type=int,default=1);cli.add_argument('--no-regimes',action='store_true')
    cli.add_argument('--resume',action='store_true',help='Start from the best current R123/Joule candidate, never a legacy profile')
    cli.add_argument('--start',type=Path,help='Seed from scripts/analyze_upv_heat_balance.py output, evaluated like any other candidate')
    cli.add_argument('--keys',help='Comma-separated subset of parameters for this stage')
    cli.add_argument('--protocol',choices=tuple(PROTOCOL),help='Heat: onset10 (first seconds of charge), then later joule/phi_r2/contacts. Do not mix.')
    cli.add_argument('--onset-window',type=float,help='Charge-onset window in seconds for onset10 (default 10)')
    cli.add_argument('--difference-fraction',type=float,default=.025)
    cli.add_argument('--trust-fraction',type=float,default=.12)
    cli.add_argument('--joule-share-min',type=float,help='Declared pre-event I²R fraction hypothesis (not a measured observation)')
    cli.add_argument('--electrical-envelope',action='store_true',help='Fit effective voltage-loss response with an explicit 3.65–4.2 V non-ohmic-voltage prior')
    cli.add_argument('--duration',type=float);args=cli.parse_args()
    if args.joule_share_min is not None and (args.stage!='heat' or not 0.<args.joule_share_min<1.):
        cli.error('--joule-share-min requires heat stage and a fraction strictly between 0 and 1')
    if args.onset_window is not None:set_onset_window(args.onset_window)
    if args.protocol:
        args.no_regimes=True
        if args.protocol in ('joule','onset10'):args.electrical_envelope=True
    if args.stage in ('heat','failure'):fit(args);return
    args.output.mkdir(parents=True,exist_ok=True);p=dict(model.DEFAULT)
    name='active-cooling' if args.stage=='validate' else 'no-cooling'
    warmup=False;protocol=args.protocol;dense_after=None
    if args.stage=='validate':
        failure=args.output/'failure-parameters.json';heat=args.output/'heat-parameters.json'
        profile=json.loads((failure if failure.exists() else heat).read_text())
        if profile['fingerprint']!=fingerprint():raise ValueError('Model changed after freeze')
        if (args.output/'validate.json').exists():raise ValueError('Held-out replay already exists')
        p=profile['parameters'];warmup=not failure.exists()
        protocol=(profile.get('calibration_prior') or {}).get('protocol') or protocol
        prior_window=(profile.get('calibration_prior') or {}).get('onset_window_s')
        if args.onset_window is None and prior_window:set_onset_window(prior_window)
    if protocol=='onset10':
        d,_=dataio.load(name);mark=dataio.event(d)
        dense_after=dataio.charge_onset_s(name,d,mark)
        if args.duration is None:args.duration=dense_after+ONSET_WINDOW_S
        save(args.output/f'onset-balance-{name}.json',onset_balance(name))
    scene,meta=model.build(name,p,warmup=warmup,duration=args.duration,dense_after=dense_after,
                           dense_span=ONSET_WINDOW_S if protocol=='onset10' else 10.);run=replay(scene,args.output,args.stage,args.timeout)
    score_stage='heat' if warmup else 'failure'
    envelope=bool(args.electrical_envelope) and name=='no-cooling'
    score=float(residual(run,meta,score_stage,envelope,protocol)@residual(run,meta,score_stage,envelope,protocol))
    save(args.output/(args.stage+'.json'),dict(parameters=p,metrics=metrics(run,meta,dense_after),
         validation_used_for_selection=False,heat_stage_only=warmup,protocol=protocol))
    write_best(args.output,args.stage,scene,run,p,meta,score,onset=dense_after)
    print(json.dumps(metrics(run,meta,dense_after),indent=2))


if __name__=='__main__':main()

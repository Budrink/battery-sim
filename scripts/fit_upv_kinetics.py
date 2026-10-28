#!/usr/bin/env python3
"""Bounded GPU-only R1/R2 trials with explicitly opt-in electrical fitting.

Rates at 75 C and activation energies are search coordinates, NOT a trigger.
The existing exact continuous chemistry and all energy guards stay unchanged.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'scripts/upv_r123_model.py').exists():ROOT=Path('/home/konstantin/work/battery-sim')
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_upv_r123 as cal
import upv_r123_model as model
import upv_r123_data as dataio

RGAS=8.31446261815324
ANCHOR_K=348.15
FREE={'r1_rate','r1_activation','r2_rate','r2_activation'}
ELECTRICAL={'resistance_ohm','resistance_temperature','resistance_charge'}
CHARGE_FORMATION={'charge_formation_gain'}
DESTRUCTION={'shell_strength_pa','opening_rate','energy_j','gas_fraction','gas_yield',
             'gas_yield_temperature','gas_rate','gas_activation','chamber_gas_h'}


def rate_at(rate,activation,reference,temperature):
    return rate*math.exp(activation/RGAS*(1/reference-1/temperature))


def parameters(base,proposal,allow_electrical=False,allow_charge_formation=False,allow_destruction=False):
    """Explicit engineering trial ranges, not measured material bounds."""
    for key,lo,hi in [('k1_75',1e-11,1.),('ea1',60000.,300000.),
                      ('k2_75',1e-9,3.),('ea2',80000.,300000.)]:
        if not math.isfinite(proposal[key]) or not lo<=proposal[key]<=hi:
            raise ValueError('Out of declared kinetic trial range: '+key)
    p=dict(base,r1_rate=rate_at(proposal['k1_75'],proposal['ea1'],ANCHOR_K,400.),
        r1_activation=proposal['ea1'],r2_rate=rate_at(proposal['k2_75'],proposal['ea2'],ANCHOR_K,450.),
        r2_activation=proposal['ea2'])
    for key in ELECTRICAL:
        if key in proposal:
            if not allow_electrical:raise ValueError('Electrical fitting requires explicit opt-in')
            lo,hi=model.BOUNDS[key]
            if not math.isfinite(proposal[key]) or not lo<=proposal[key]<=hi:raise ValueError('Invalid electrical trial: '+key)
            p[key]=proposal[key]
    if 'charge_formation_gain' in proposal:
        if not allow_charge_formation:raise ValueError('Charge-formation fitting requires explicit opt-in')
        value=proposal['charge_formation_gain'];lo,hi=model.BOUNDS['charge_formation_gain']
        if not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid charge-formation coefficient')
        p['charge_formation_gain']=value
    for key in DESTRUCTION:
        if key not in proposal:continue
        if not allow_destruction:raise ValueError('Destruction fitting requires explicit opt-in')
        lo,hi=model.BOUNDS[key]
        if not math.isfinite(proposal[key]) or not lo<=proposal[key]<=hi:raise ValueError('Invalid destruction trial: '+key)
        p[key]=proposal[key]
    known={'label','k1_75','ea1','k2_75','ea2'}|ELECTRICAL|CHARGE_FORMATION|DESTRUCTION
    if proposal.keys()-known:raise ValueError('Unknown proposal fields: '+str(sorted(proposal.keys()-known)))
    free=FREE|(ELECTRICAL if allow_electrical else set())|(CHARGE_FORMATION if allow_charge_formation else set())|(DESTRUCTION if allow_destruction else set())
    assert all(p[k]==v for k,v in base.items()if k not in free)
    if rate_at(p['r2_rate'],p['r2_activation'],450.,298.15)>1e-7*(1.+1e-9):
        raise ValueError('R2 consumes stored charge too quickly at 25 C for this search')
    if rate_at(p['r1_rate'],p['r1_activation'],400.,298.15)>1e-7*(1.+1e-9):
        raise ValueError('R1 degrades healthy material too quickly at 25 C for this search')
    return p


def score(run,meta):
    if meta.get('objective_mode')=='explosion':
        # A good prefix NEVER qualifies as a destruction fit. Include the peak,
        # pressure unloading and subsequent heating on the same continuous run.
        if run['time_s'][-1]<meta['fit_end_s']-1e-7 or run.get('incomplete') or run.get('partial'):return None
        residual=cal.residual(run,meta,'failure')
        return dict(objective=float(residual@residual),fit_end_s=meta['fit_end_s'],
            phases=cal.metrics(run,meta),
            objective_note='Full pressure curve and witness temperatures through event+300 s minimum; post-rupture C2 sensors are diagnostics, not bulk-temperature targets')
    t=np.array(run['time_s']);event=meta['event']['time_s'];stop=min(event-3.,meta.get('fit_end_s',event-3.))
    if t[-1]<stop:return None
    grid=np.arange(0.,stop+.001,5.);obs=dataio.observed(meta['data'],grid)
    parts=[];channels={}
    for key in dataio.CHANNELS:
        y=np.interp(grid,t,run['temperature_k'][key]);err=y-obs[key]
        weight=2. if key.startswith('C2_') else 1.
        for left,right,scale in ((0.,300.,2.),(300.,600.,2.),(600.,stop,3.)):
            mask=(grid>=left)&(grid<=right)&np.isfinite(err)
            if mask.any():parts.extend((weight*err[mask]/(scale*np.sqrt(mask.sum()))).tolist())
        inc=(y[12:]-y[:-12])-(obs[key][12:]-obs[key][:-12]);mask=np.isfinite(inc)&(grid[12:]>=600.)
        if mask.any():parts.extend((weight*inc[mask]/(3.*np.sqrt(mask.sum()))).tolist())
        channels[key]=dict(rmse_fit_window_k=float(np.sqrt(np.nanmean(err**2))),end_error_k=float(err[-1]))
    q2=np.array(run['released_energy_j_history']).reshape(len(t),len(run['chemistry'])).sum(axis=1)
    q3=sum(np.array(v['released_energy_j'])for v in run['gas'].values())
    joule=sum(np.array(v)for v in run['joule_energy_j'].values())
    onset=meta['onset_s'];at=[onset,onset+300.]
    j=float(np.diff(np.interp(at,t,joule))[0]);q=float(np.diff(np.interp(at,t,q2+q3))[0]);share=j/max(1e-30,j+q)
    prior=max(0.,.5-share)/.05;parts.append(prior)
    return dict(objective=float(np.dot(parts,parts)),channels=channels,fit_end_s=stop,first300_joule_share=share,
                objective_note='No chamber-pressure target while shell strength is uncalibrated')


def reject_reason(run,meta):
    t=np.array(run['time_s']);end=t[-1];event=meta['event']['time_s']
    live=run.get('live_state',{});at=live.get('time_s',end);source=live.get('objects',{}).get('C2',{})
    explosion=meta.get('objective_mode')=='explosion'
    # Core temperature is unmeasured: do not prune destruction merely because
    # internal R2 precedes the observed chamber-pressure event.
    if source and not explosion and 120.<at<event-60.:
        observed=dataio.observed(meta['data'],[at])
        hottest=max(observed['C2_TIM'][0],observed['C2_ISS'][0])
        if source['max_temperature_k']>hottest+150.:
            return f'C2: gross premature internal runaway at {at:.6f} s (live Tmax {source["max_temperature_k"]:.1f} K)'
    if end<120. or end>=event-15.:return None
    mask=(t>=max(120.,end-30.))&(t<=end)
    obs=dataio.observed(meta['data'],t)
    if explosion:
        pressure=np.array(dataio.chamber_pressure_pa(run))/1e5
        v=(pressure-obs['pressure_bar'])[mask]
        if end<event-30. and len(v)>=4 and np.all(v>2.):
            return f'Chamber: persistent premature overpressure >2 bar at {end:.2f} s'
    for key in ('C2_TIM','C2_ISS'):
        err=np.array(run['temperature_k'][key])-obs[key];v=err[mask&np.isfinite(err)]
        margin=30. if explosion else 15.
        if len(v)>=4 and (np.all(v>margin) or np.all(v< -margin)):
            return f'{key}: persistent 30 s error exceeds {margin:g} K at {end:.2f} s'
        if end<event-40. and len(v) and v[-1]>50.:
            return f'{key}: gross premature heating {v[-1]:.1f} K at {end:.2f} s'
    return None


def snapshots(run,meta):
    t=np.array(run['time_s']);chem=run['chemistry']['C2'];end=t[-1];result={}
    for target in (300.,500.,meta['onset_s']+600.,750.,850.,meta['event']['time_s']):
        if target>end:continue
        i=int(np.argmin(abs(t-target)))
        obs=dataio.observed(meta['data'],[float(t[i])])
        result[str(round(float(t[i]),3))]=dict(c2_mean_c=run['cell_mean_temperature_k']['C2'][i]-273.15,
            c2_tim_error_k=run['temperature_k']['C2_TIM'][i]-obs['C2_TIM'][0],
            c2_iss_error_k=run['temperature_k']['C2_ISS'][i]-obs['C2_ISS'][0],
            b_mg=chem['r2_precursor_kg'][i]*1e6,r2_local_power_w=chem['r2_power_w'][i],
            thermal_formed_g=chem['precursor_from_degradation_kg'][i]*1000,
            charge_formed_g=chem['precursor_from_charge_kg'][i]*1000,
            chamber_bar=dataio.chamber_pressure_pa(run)[i]/1e5)
    return result


def trial_scene(parameters,end,gas_extensions=None,numerical_controls=None):
    scene,meta=model.build('no-cooling',parameters,warmup=False,duration=end,gas_extensions=gas_extensions)
    controls=dict(numerical_controls or {})
    if controls.keys()-{'coupled_gas_films','coupled_transport_interval_s','minimum_dt_s'}:
        raise ValueError('Unknown gas integrator controls')
    if 'coupled_gas_films'in controls and type(controls['coupled_gas_films']) is not bool:
        raise ValueError('coupled_gas_films must be boolean')
    if 'coupled_transport_interval_s'in controls:
        h=controls['coupled_transport_interval_s']
        if not math.isfinite(h) or not 0.<h<=1.:
            raise ValueError('Invalid coupled transport interval')
    minimum=controls.get('minimum_dt_s',.005 if controls.get('coupled_gas_films') else 0.)
    if not math.isfinite(minimum) or minimum<0. or (minimum>0. and minimum<.005):
        raise ValueError('Coarse integration minimum_dt_s must be >=0.005 s')
    scene['simulation']['minimum_dt_s']=minimum
    scene['gas_network'].update({k:v for k,v in controls.items()if k!='minimum_dt_s'})
    if minimum>0.:controls['minimum_dt_s']=minimum
    scene['metadata']['numerical_controls']=controls
    return scene,meta


def evaluate(proposal,base,out,fingerprint,timeout,emf_floor=None,allow_electrical=False,allow_charge_formation=False,duration=None,explosion=False,gas_temperature_step=None,objective_fn=None,prune_fn=None,gas_extensions=None,numerical_controls=None):
    p=parameters(base,proposal,allow_electrical,allow_charge_formation,explosion);folder=out/proposal['label'];folder.mkdir()
    data,_=dataio.load('no-cooling');event=dataio.event(data);end=duration if duration is not None else event['time_s']+(300. if explosion else 10.)
    if explosion and end<event['time_s']+300.-1e-7:raise ValueError('Destruction fitting must include at least 300 s after the measured event')
    scene,meta=trial_scene(p,end,gas_extensions,numerical_controls)
    if gas_temperature_step is not None:
        scene['gas_network']['max_temperature_step_k']=gas_temperature_step
        scene['metadata']['numerical_controls'].update(gas_max_temperature_step_k=gas_temperature_step,
            physical_parameters_unchanged=True,requires_tighter_tolerance_check=True)
    if emf_floor is not None:
        for cell in scene['cells']:
            if 'electrical' in cell:cell['electrical']['reversible_voltage_floor_v']=emf_floor
        scene['metadata']['diagnostic_scene_overrides']=dict(reversible_voltage_floor_v=emf_floor,
            full_UI_budget_and_paid_chemical_storage_retained=True,not_a_calibrated_profile=True)
    meta['onset_s']=dataio.charge_onset_s('no-cooling',data,event)
    meta['fit_end_s']=end
    meta['objective_mode']='explosion' if explosion else 'warmup'
    samples=np.r_[scene['failure_sample_times_s'],meta['onset_s']+600.,np.arange(650.,end,1.)]
    scene['failure_sample_times_s']=np.unique(samples[(samples>=0.)&(samples<=end)]).tolist()
    assert not scene['primary_failures'] and all(c['runaway']['energy_j']==0. for c in scene['cells'])
    shell=json.loads(json.dumps(model.build('no-cooling',base,warmup=False,duration=end)[0]['shells']))
    assert scene['shells']==shell
    cal.save(folder/'scenario.json',scene)
    cal.save(folder/'parameters.json',dict(parameters=p,proposal=proposal,fingerprint=fingerprint,
        changes={k:[base[k],p[k]]for k in base if base[k]!=p[k]},validation_used_for_selection=False,
        shell_material_contacts_and_phi_frozen=not explosion,solid_material_contacts_and_phi_frozen=True,
        objective_mode=meta['objective_mode'],electrical_fit_authorized=allow_electrical,
        fit_charge_formation=allow_charge_formation,diagnostic_emf_floor_v=emf_floor,
        gas_temperature_step_k=gas_temperature_step,gas_extensions=gas_extensions,
        numerical_controls=scene['metadata']['numerical_controls']))
    print('START',proposal['label'],proposal,flush=True)
    started=time.monotonic();error=None
    try:
        run=cal.replay(scene,folder,'cuda',timeout,reject_partial=lambda r:(prune_fn or reject_reason)(r,meta));status='complete'
    except (RuntimeError,ValueError,subprocess.TimeoutExpired) as exc:
        error=str(exc);status='pruned'if error.startswith('Pruned:')else'failed'
        path=folder/'last-incomplete-run.json.gz'
        if not path.exists():
            row=dict(label=proposal['label'],status=status,error=error,parameters=p)
            cal.save(folder/'status.json',row);print('FAILED',row['label'],error,flush=True);return row
        run=json.load(gzip.open(path,'rt'))
    if cal.fingerprint()!=fingerprint:raise RuntimeError('Core changed while fitting')
    with gzip.open(folder/'trial-run.json.gz','wt')as stream:json.dump(run,stream,allow_nan=False)
    diag=(objective_fn or score)(run,meta)
    row=dict(label=proposal['label'],status=status,error=error,parameters=p,proposal=proposal,
        last_sample_s=run['time_s'][-1],objective=None if diag is None else diag['objective'],
        diagnostics=diag,snapshots=snapshots(run,meta),fingerprint=fingerprint,validation_used_for_selection=False,
        wall_s=time.monotonic()-started,balances={k:run[k]for k in ('max_thermal_residual_j','max_chemical_residual_j','max_mass_residual_kg')})
    row['diagnostic_emf_floor_v']=emf_floor
    row['gas_temperature_step_k']=gas_temperature_step
    cal.save(folder/'status.json',row)
    print('DONE',json.dumps(row),flush=True)
    if status=='complete':
        subprocess.run([sys.executable,str(ROOT/'scripts/report_upv_r123.py'),str(folder),'--stage','trial'],check=True)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--proposals',type=Path);ap.add_argument('--timeout',type=float,default=120.)
    ap.add_argument('--seed',type=Path,help='Explicit full revised seed; no implicit legacy migration')
    ap.add_argument('--duration',type=float,help='Optional warmup-only fit horizon; not explosion calibration')
    ap.add_argument('--diagnostic-emf-floor',type=float,default=None)
    ap.add_argument('--fit-electrical',action='store_true',help='User-approved joint R(T,Q) fit; mixture conductivity stays fixed')
    ap.add_argument('--fit-charge-formation',action='store_true',help='Fit ordinary charging, separately from R1; B formation still pays measured UI')
    ap.add_argument('--explosion',action='store_true',help='Fit destruction physics against pressure and witness temperatures through at least event+300 s; never accept a short warmup')
    ap.add_argument('--gas-temperature-step',type=float,help='Numerical gas-only temperature tolerance in K; compare a tighter setting before accepting the fit')
    args=ap.parse_args()
    if args.gas_temperature_step is not None and not .25<=args.gas_temperature_step<=10.:
        raise ValueError('Gas-only numerical temperature tolerance must be in [.25, 10] K')
    if args.diagnostic_emf_floor is not None and not 0.<=args.diagnostic_emf_floor<=3.65:
        raise ValueError('Invalid diagnostic EMF floor')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    ptr=ROOT/'data/calibration/upv-charge-degradation-v5/current-profile.json';raw=ptr.read_bytes();pointer=json.loads(raw)
    source=args.seed.resolve() if args.seed else ROOT/pointer['parameters_file']
    seed=json.loads(source.read_text());base=seed['parameters'];fp=cal.fingerprint()
    if seed.get('model_fingerprint')!=fp:raise ValueError('Seed/core mismatch: explicitly migrate and verify the revised model first')
    if set(base)!=set(model.DEFAULT):raise ValueError('A complete current-schema seed is required; no implicit defaults/compatibility')
    if args.duration is not None and not 330.<=args.duration<=3000.:raise ValueError('Invalid fit horizon')
    # Explicit experimental-timescale storage prior, not calendar-life validation.
    # At this upper bound 1-exp(-k*3600) = 0.036% of a fixed B stock per hour.
    k25=1e-7
    proposals=json.loads(args.proposals.read_text())if args.proposals else [
        dict(label='cold-store-200',k1_75=3e-5,ea1=160000.,k2_75=rate_at(k25,200000.,298.15,ANCHOR_K),ea2=200000.),
        dict(label='cold-store-220',k1_75=3e-5,ea1=160000.,k2_75=rate_at(k25,220000.,298.15,ANCHOR_K),ea2=220000.),
        dict(label='cold-store-240',k1_75=3e-5,ea1=160000.,k2_75=rate_at(k25,240000.,298.15,ANCHOR_K),ea2=240000.)]
    # Reject invalid coordinates BEFORE spending time on any CUDA candidate.
    for proposal in proposals:parameters(base,proposal,args.fit_electrical,args.fit_charge_formation,args.explosion)
    cal.save(out/'search.json',dict(source=str(source.relative_to(ROOT)),parameters=base,fingerprint=fp,
        search_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),proposals=proposals,
        free=sorted(FREE|(ELECTRICAL if args.fit_electrical else set())|(CHARGE_FORMATION if args.fit_charge_formation else set())|(DESTRUCTION if args.explosion else set())),
        calibration_only=True,validation_used_for_selection=False,
        rate_anchor_k=ANCHOR_K,source_pointer_sha256=hashlib.sha256(raw).hexdigest(),
        previous_fingerprint=pointer['model_fingerprint'],revision=seed.get('revision'),fit_duration_s=args.duration,
        cold_r1_rate_upper_at_25c_per_s=k25,cold_r2_rate_upper_at_25c_per_s=k25,cold_storage_prior_is_not_calendar_life_calibration=True,
        diagnostic_emf_floor_v=args.diagnostic_emf_floor,objective_mode='explosion' if args.explosion else 'warmup',
        gas_temperature_step_k=args.gas_temperature_step,
        early_pruning='Surface error >15 K for 30 s; live core > observed faces +150 K >60 s before event is a search heuristic, not measured core truth; no forced ignition'))
    rows=[]
    for proposal in proposals:
        rows.append(evaluate(proposal,base,out,fp,args.timeout,args.diagnostic_emf_floor,args.fit_electrical,args.fit_charge_formation,args.duration,args.explosion,args.gas_temperature_step));cal.save(out/'summary.json',rows)
        eligible=[r for r in rows if r['status']=='complete'and r.get('objective')is not None]
        if eligible:cal.save(out/'best-candidate.json',min(eligible,key=lambda r:r['objective']))
    if ptr.read_bytes()!=raw:raise RuntimeError('Current-profile pointer changed during trials')
    print('FINISHED',out,flush=True)


if __name__=='__main__':main()

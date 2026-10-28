"""Frozen-boundary analytical sensitivity, NOT a replacement for CUDA validation."""
import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from fsri30_wall_damage import candidate, ONSET_K, RATE, SCALE_K


def damage_increment(a,b,dt):
    if max(a,b)<ONSET_K:return 0.
    if a<ONSET_K:
        dt*= (b-ONSET_K)/(b-a);a=ONSET_K
    elif b<ONSET_K:
        dt*= (a-ONSET_K)/(a-b);b=ONSET_K
    z=(b-a)/SCALE_K
    return RATE*math.exp(min(50.,(a-ONSET_K)/SCALE_K))*dt*(math.expm1(z)/z if abs(z)>1e-8 else 1.+z/2.)


def estimate(folder):
    scene=json.loads((folder/'scenario.json').read_text())
    r=json.load(gzip.open(folder/'replay/incomplete-run.json.gz','rt'))
    exp=json.loads((folder/'experiment.json').read_text())
    t=np.asarray(r['time_s']);keep=t<=1700.;t=t[keep]
    base=np.asarray(r['cell_mean_temperature_k']['B10'])[keep]
    battery=next(c for c in scene['cells'] if c['id']=='B10')
    shell=next(s for s in scene['shells'] if s['contents_id']=='B10')
    law=battery['conversion'];chem=r['chemistry']['B10'];capacity=np.full(len(t),battery['thermal']['heat_capacity_j_per_k'])
    capacity-=law['mass_kg']*law['reactant_cp']
    capacity+=np.asarray(chem['virgin_kg'])[keep]*law['reactant_cp']
    capacity+=np.asarray(chem['r2_precursor_kg'])[keep]*law['activated_cp']
    capacity+=np.asarray(chem['residue_kg'])[keep]*law['residue_cp']
    capacity-=(law['volatile_phase']['initial_mass_kg']-np.asarray(chem['liquid_kg'])[keep])*law['volatile_phase']['liquid_cp']
    paths=[]
    for name,e in r['thermal_paths']['interfaces'].items():
        if e['from']!='B10' or not e['to'].startswith('B10:case:'):continue
        ta=np.asarray(e['a_adjacent_temperature_k'])[keep];tb=np.asarray(e['b_adjacent_temperature_k'])[keep]
        old=np.asarray(e['conductance_w_per_k'])[keep]
        axis='xyz'.index(e['to'][-1]);dims=battery['geometry']['dimensions_m']
        area=np.prod([dims[k] for k in range(3) if k!=axis])
        series=np.maximum(0.,1/old-1/(shell['contents_contact_h']*area))
        D=np.zeros(len(t));T=.5*(ta+tb)
        for i in range(1,len(t)):D[i]=min(1.,D[i-1]+damage_increment(T[i-1],T[i],t[i]-t[i-1]))
        paths.append((name,ta,tb,old,area,series,D))
    assert len(paths)==6
    def response(hcold,hhot=950.):
        # Linearized core energy balance around the saved trajectory. Shell
        # temperatures and heater controller remain frozen, so this cannot
        # establish cascade timing or a final calibrated material parameter.
        power=np.zeros(len(t));conductance=np.zeros(len(t))
        z0=math.log((hcold-5)/(995-hcold));z1=math.log((hhot-5)/(995-hhot))
        for _,ta,tb,old,area,series,D in paths:
            h=5+990/(1+np.exp(-(z0+(z1-z0)*D)))
            new=1/(1/(h*area)+series)
            power-=(new-old)*(ta-tb);conductance+=new
        delta=np.zeros(len(t))
        for i in range(1,len(t)):
            dt=t[i]-t[i-1];g=.5*(conductance[i]+conductance[i-1]);p=.5*(power[i]+power[i-1]);c=.5*(capacity[i]+capacity[i-1])
            decay=math.exp(-g*dt/c)
            delta[i]=delta[i-1]*decay+(p/g)*(1-decay)
        return delta,power
    truth=np.interp(t,exp['time_s'],exp['T5'])+273.15
    mask=(t>=300)&(t<=1550)
    score=lambda h:float(np.mean((base[mask]+response(h)[0][mask]-truth[mask])**2))
    fit=minimize_scalar(score,bounds=(75.,350.),method='bounded',options={'xatol':.01})
    h=float(fit.x);delta,power=response(h)
    hottest=np.max(np.stack([v[1] for v in paths]),axis=0)
    # Arrhenius compensation at equal saved states + the archived event-time
    # derivative. Quoting its spread prevents pretending this is unique.
    timing_factor=math.exp((1752.-1763.)/219.3857)
    factors=[]
    for ts in [1600.,1650.,1700.]:
        T=float(np.interp(ts,t,hottest));d=float(np.interp(ts,t,delta))
        factors.append(timing_factor*math.exp(-law['reaction_activation_j_per_mol']/8.31446261815324*(1/T-1/(T+d))))
    factor=float(np.median(factors))
    out=candidate(scene)
    for s in out['shells']:
        if 'contents_connectivity_per_m2' not in s:continue
        c=s['contents_connectivity_per_m2'];cold=math.log((h-5)/(995-h))
        c['geometry_reference_logit']=cold;c['interface_damage_log_response']=math.log(945/45)-cold
    for c in out['cells']:
        if c.get('role')=='battery':c['conversion']['reaction_rate_per_s']*=factor
    report=dict(method='frozen-boundary energy perturbation plus Arrhenius/timing sensitivity',
        source=str(folder),fit_window_s=[300,1550],cold_h_bounds=[75,350],cold_h_w_m2_k=h,
        hot_h_w_m2_k=950.,warmup_rmse_unmodified_k=float(np.sqrt(np.mean((base[mask]-truth[mask])**2))),
        warmup_rmse_frozen_prediction_k=math.sqrt(score(h)),
        predicted_core_temperature_changes_k={str(ts):float(np.interp(ts,t,delta)) for ts in [1000,1400,1600,1650,1700]},
        r2_multipliers_by_evaluation_time=factors,r2_multiplier=factor,
        r2_rate_per_s=law['reaction_rate_per_s']*factor,r3_change_from_last_run=1.,
        r3_reason='No independent identification from this frozen boundary estimate; retain previous 30% energy reduction',
        caution='Mean-temperature correction applied to hotspot kinetics; ignores heater and gas feedback. Initial guess only.',
        calibrated=False,validation_passed=False)
    out['metadata']['wall_contact_hypothesis']['intact_h_w_m2_k']=h
    out['metadata']['analytical_insulation_seed']=report
    out['metadata']['r2_gas_shape_probe']=dict(report,objective='Test irreversible insulation; primary and secondary timing',fresh_initial_state=True)
    out['metadata']['secondary_timing_guard']=True
    return out,report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();s,r=estimate(a.source);a.output.mkdir(exist_ok=False)
    (a.output/'scenario.json').write_text(json.dumps(s,indent=2));(a.output/'estimate.json').write_text(json.dumps(r,indent=2))
    (a.output/'experiment.json').write_text((a.source/'experiment.json').read_text())
    print(json.dumps(r,indent=2))

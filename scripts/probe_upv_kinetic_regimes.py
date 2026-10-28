#!/usr/bin/env python3
"""CUDA-only controlled kinetic experiments with frozen thermal contacts.

The voltage floor is an explicit prior, not measured OCV or a DCIR estimate.
An electrical budget violation rejects a trajectory; it never clips its heat.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import calibrate_upv_r123 as calibration
import upv_r123_model as model
import upv_r123_data as dataio

ALLOWED={'resistance_ohm','resistance_temperature','product_conductivity_ratio',
         'charge_formation_gain','r1_activation','r2_rate','r2_activation','energy_j','gas_fraction',
         'precursor_yield','gas_yield_temperature'}


def electrical_work_limit(scene,times,voltage_floor=None):
    """Exact integral of I*(U-U_floor) for the two linear drive profiles."""
    law=next(c['electrical'] for c in scene['cells'] if c['id']=='C2')
    if voltage_floor is None:voltage_floor=law.get('reversible_voltage_floor_v',0.)
    current=np.asarray(law['current_profile_a']);voltage=np.asarray(law['voltage_profile_v'])
    times=np.asarray(times);stop=times[-1]
    knots=np.unique(np.r_[times,current[(current[:,0]>times[0])&(current[:,0]<stop),0],
                           voltage[(voltage[:,0]>times[0])&(voltage[:,0]<stop),0]])
    i=np.interp(knots,current[:,0],current[:,1]);u=np.interp(knots,voltage[:,0],voltage[:,1])
    if np.any(i<0.) or np.any(u<voltage_floor):raise ValueError('Budget requires nonnegative charge and U >= floor')
    dt=np.diff(knots)
    work=dt*(2*i[:-1]*u[:-1]+i[:-1]*u[1:]+i[1:]*u[:-1]+2*i[1:]*u[1:])/6.
    charge=dt*(i[:-1]+i[1:])/2.
    return np.interp(times,knots,np.r_[0.,np.cumsum(work-voltage_floor*charge)])


def electrical_violation(run,scene):
    upper=electrical_work_limit(scene,run['time_s'])
    measured=np.diff(run['joule_energy_j']['C2']);allowed=np.diff(upper)
    bad=np.flatnonzero(measured>allowed*1.01+1.)
    if len(bad):
        j=int(bad[0]);return f'I²R exceeds electrical dissipation budget at {run["time_s"][j+1]:.3f} s'
    stored=run.get('chemistry',{}).get('C2',{}).get('electrical_chemical_input_j')
    if stored is not None:
        full=electrical_work_limit(scene,run['time_s'],0.)
        bad=np.flatnonzero(measured+np.diff(stored)>np.diff(full)*1.01+1.)
        if len(bad):return f'I²R plus chemical storage exceeds full UI budget at {run["time_s"][int(bad[0])+1]:.3f} s'
    return None


def chain_summary(run):
    t=np.asarray(run['time_s']);c=run['chemistry']['C2']
    # Probe stores one cumulative R2 heat entry per battery per observation.
    released=np.asarray(run['released_energy_j_history']).reshape(len(t),3)[:,1]
    rows={}
    for when in [300.,500.,700.,float(t[-1])]:
        if when>t[-1]:continue
        heat=float(np.interp(when,t,released));joule=float(np.interp(when,t,run['joule_energy_j']['C2']))
        rows[str(when)]=dict(virgin_kg=float(np.interp(when,t,c['virgin_kg'])),
            precursor_kg=float(np.interp(when,t,c['r2_precursor_kg'])),r2_heat_j=heat,joule_j=joule,
            r2_power_w=float(np.interp(when,t,c['r2_power_w'])),r2_local_heat_share=heat/max(1e-12,heat+joule))
    return rows


def main():
    cli=argparse.ArgumentParser(__doc__)
    cli.add_argument('trials',type=Path);cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--profile',type=Path,default=calibration.OUT/'heat-parameters.json')
    cli.add_argument('--timeout',type=float,default=120.)
    cli.add_argument('--electrical-envelope',action='store_true')
    args=cli.parse_args();directory=args.output;directory.mkdir(parents=True,exist_ok=False)
    original=args.profile.read_bytes();profile=json.loads(original);frozen=calibration.fingerprint()
    if profile['fingerprint']!=frozen:raise ValueError('Frozen heat profile does not match current model')
    baseline=profile['parameters'];trials=json.loads(args.trials.read_text())
    calibration.save(directory/'protocol.json',dict(fingerprint=frozen,source_profile_sha256=hashlib.sha256(original).hexdigest(),
        baseline=baseline,trials=trials,voltage_floor_v=0.,validation_used_for_selection=False,
        no_thermal_contact_or_material_fitting=True,pruning_score_ceiling=4.*profile['objective']))
    reference_scene,_=model.build('no-cooling',baseline,warmup=True)
    reference_path=args.profile.parent/'heat-run.json.gz'
    with gzip.open(reference_path,'rt') as stream:reference_run=json.load(stream)
    rows=[];best=float('inf');winner=None
    for trial in trials:
        label=trial['label'];changes=trial['changes']
        if not changes.keys()<=ALLOWED:raise ValueError('A trial changes a frozen parameter')
        p=dict(baseline,**changes)
        for key,val in changes.items():
            lo,hi=model.BOUNDS[key]
            if not lo<=val<=hi:raise ValueError('Outside declared prior: '+key)
        scene,meta=model.build('no-cooling',p,warmup=True)
        assert scene['interfaces']==reference_scene['interfaces']
        assert [c['thermal'] for c in scene['cells']]==[c['thermal'] for c in reference_scene['cells']]
        row=dict(label=label,parameters=p);print('START',label,flush=True)
        def reject(partial):
            reason=electrical_violation(partial,scene)
            if not reason and calibration.partial_objective_lower_bound(partial,scene,meta,'heat')>min(best,4.*profile['objective']):
                reason='Prefix temperature/pressure objective exceeds allowed full objective'
            if not reason:reason=calibration.local_trajectory_rejection(partial,reference_run,meta,'heat')
            if reason:row['partial_chain']=chain_summary(partial)
            return reason
        try:
            run=calibration.replay(scene,directory,label,args.timeout,reject_partial=reject)
            violation=electrical_violation(run,scene)
            if violation:raise ValueError(violation)
            r=calibration.residual(run,meta,'heat',args.electrical_envelope);score=float(r@r)
            row.update(status='complete',objective=score,metrics=calibration.metrics(run,meta),chain=chain_summary(run))
            if score<best:
                best=score;winner=label
                calibration.write_best(directory,'heat',scene,run,p,meta,score,
                    dict(joule_share_min=None,electrical_envelope=args.electrical_envelope,
                         unexplained_voltage_prior_v=[3.65,4.2]))
            print('DONE',label,'objective',round(score,4),'best',round(best,4),'wall_s',round(run['wall_s'],2),flush=True)
        except (RuntimeError,ValueError,subprocess.TimeoutExpired) as error:
            row.update(status='pruned' if str(error).startswith('Pruned:') else 'failed',error=str(error))
            print(row['status'].upper(),label,str(error),flush=True)
        rows.append(row);calibration.save(directory/'history.json',rows)
        if calibration.fingerprint()!=frozen or args.profile.read_bytes()!=original:raise RuntimeError('Model changed during controlled trial')
    calibration.save(directory/'status.json',dict(status='finished',best_label=winner,best_objective=best if winner else None,
                                                 validation_used_for_selection=False))


if __name__=='__main__':main()

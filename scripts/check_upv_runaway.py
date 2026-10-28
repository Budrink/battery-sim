#!/usr/bin/env python3
"""Frozen GPU forward prediction; no ignition prescription or optimization."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'scripts/upv_r123_model.py').exists():ROOT=Path('/home/konstantin/work/battery-sim')
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_upv_r123 as cal
import upv_r123_data as dataio
import upv_r123_model as model


def uniform_metrics(run,meta):
    end=float(run['time_s'][-1]);t=np.array(run['time_s']);event=meta['event']['time_s']
    at=np.unique(np.r_[np.arange(0.,end,1.),end]);obs=dataio.observed(meta['data'],at)
    phases=[('before_event',0.,event-20.),('event',event-3.,event+3.),
            ('early',event+3.,event+30.),('propagation',event+30.,event+300.),
            ('later',event+300.,event+600.)]
    result={}
    for label,left,right in phases:
        mask=(at>=left)&(at<=right)
        if not mask.any():continue
        rows={}
        for key in dataio.CHANNELS:
            err=np.interp(at,t,run['temperature_k'][key])-obs[key]
            v=err[mask&np.isfinite(err)]
            if len(v):rows[key]=dict(rmse_k=float(np.sqrt(np.mean(v*v))),bias_k=float(np.mean(v)))
        result[label]=rows
    return result


def peak(t,y):
    i=int(np.nanargmax(y));return dict(value=float(y[i]),time_s=float(t[i]))


def diagnostics(run,scene,meta):
    t=np.array(run['time_s']);event=meta['event']['time_s'];end=float(t[-1])
    native=np.array(meta['data']['time_s']);mask=(native<=end)
    obs=dataio.observed(meta['data'],native[mask])
    chamber=np.array(dataio.chamber_pressure_pa(run))/1e5
    inventory=run['chemistry']['C2'];core=np.array(run['cell_mean_temperature_k']['C2'])-273.15
    inside=np.max([v['pressure_pa']for k,v in run['gas'].items()if k.startswith('inside-C2-')],axis=0)/1e5
    rupture=[v for k,v in run['channels'].items()if k.startswith('rupture-C2-')]
    area=sum(np.array(v['effective_area_m2'])for v in rupture)
    net_mass=sum(np.array(v['transferred_mass_kg'])for v in rupture)
    q2=np.array(run['released_energy_j_history']).reshape(len(t),len(run['chemistry'])).sum(axis=1)
    q3=sum(np.array(v['released_energy_j'])for v in run['gas'].values())
    qj=sum(np.array(v)for v in run['joule_energy_j'].values())
    r=dict(observed_marker=meta['event'],last_sample_s=end,
        reached_event_plus_600_s=bool(end>=event+600.-1e-5),
        chamber_pressure_bar=peak(t,chamber),observed_chamber_pressure_bar=peak(native[mask],obs['pressure_bar']),
        inside_c2_max_pressure_bar=peak(t,inside),c2_core_temperature_c=peak(t,core),
        measured_surface_peaks_c={k:peak(native[mask],obs[k]-273.15)for k in dataio.CHANNELS},
        predicted_surface_peaks_c={k:peak(t,np.array(v)-273.15)for k,v in run['temperature_k'].items()},
        c2_aperture_max_m2=float(np.max(area)),c2_net_vented_mass_g=float(net_mass[-1]*1000),
        c2_first_positive_aperture_s=float(t[np.flatnonzero(area>1e-12)[0]])if np.any(area>1e-12)else None,
        c2_first_1mm2_total_aperture_s=float(t[np.flatnonzero(area>=1e-6)[0]])if np.any(area>=1e-6)else None,
        chemical_inventory_final_g={k:float(inventory[k][-1]*1000)for k in ('virgin_kg','r2_precursor_kg','residue_kg')},
        precursor_from_charge_g=float(inventory['precursor_from_charge_kg'][-1]*1000),
        precursor_from_degradation_g=float(inventory['precursor_from_degradation_kg'][-1]*1000),
        cumulative_heat_j=dict(joule=float(qj[-1]),r2_local=float(q2[-1]),r3=float(q3[-1])),
        events=run.get('events',[]),uniform_1s_phase_metrics=uniform_metrics(run,meta),
        balances={k:run[k]for k in ('max_thermal_residual_j','max_chemical_residual_j','max_mass_residual_kg')},
        wall_s=run['wall_s'],steps=run['steps'],snapshots={})
    ids=[c['id']for c in scene['cells']if 'conversion' in c];idx=ids.index('C2')
    localmax=np.array(run['source_max_temperature_k']).reshape(len(t),len(ids))[:,idx]-273.15
    r['c2_max_local_temperature_c']=peak(t,localmax)
    for when in (600.+scene['metadata']['electrical']['onset_s'],event-20.,event,event+30.,event+300.,event+600.):
        if when>end:continue
        i=int(np.argmin(abs(t-when)))
        r['snapshots'][str(round(float(t[i]),3))]=dict(c2_mean_c=float(core[i]),c2_max_c=float(localmax[i]),
            chamber_bar=float(chamber[i]),inside_c2_max_bar=float(inside[i]),aperture_mm2=float(area[i]*1e6),
            b_mg=float(inventory['r2_precursor_kg'][i]*1e6),
            formed_charge_g=float(inventory['precursor_from_charge_kg'][i]*1000),
            formed_degradation_g=float(inventory['precursor_from_degradation_kg'][i]*1000),
            c2_r2_local_power_w=float(inventory['r2_power_w'][i]),joule_j=float(qj[i]),r2_local_j=float(q2[i]),r3_j=float(q3[i]))
    return r


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=420.)
    ap.add_argument('--refinement',type=float,default=1.)
    ap.add_argument('--diagnostic-emf-floor',type=float,default=None,
                    help='Explicit diagnostic override, never updates the current profile')
    ap.add_argument('--experiments',default='no-cooling,active-cooling')
    args=ap.parse_args()
    if args.diagnostic_emf_floor is not None and not 0.<=args.diagnostic_emf_floor<=3.65:
        raise ValueError('Diagnostic EMF floor must be between 0 and 3.65 V')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    ptr=ROOT/'data/calibration/upv-charge-degradation-v5/current-profile.json';raw=ptr.read_bytes();pointer=json.loads(raw)
    source=ROOT/pointer['parameters_file'];frozen=json.loads(source.read_text());p=frozen['parameters']
    fp=cal.fingerprint();phash=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    if phash!=pointer['parameters_sha256'] or fp!=pointer['model_fingerprint']:raise ValueError('Current profile hash/fingerprint mismatch')
    cal.save(out/'frozen-parameters.json',dict(parameters=p,parameters_sha256=phash,model_fingerprint=fp,
        source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        optimization=False,forced_ignition=False,outer_plate_diagnostic_not_used=True,refinement=args.refinement,
        diagnostic_emf_floor_v=args.diagnostic_emf_floor,
        electrical_boundary_note='Measured/declared I(t), U(t), cutoff retained; not an autonomous external supply model'))
    results={}
    for name in args.experiments.split(','):
        folder=out/name;folder.mkdir()
        data,_=dataio.load(name);marker=dataio.event(data);end=marker['time_s']+600.
        scene,meta=model.build(name,p,warmup=False,duration=end)
        assert not scene['primary_failures']
        assert all(c['runaway']['energy_j']==0. for c in scene['cells'])
        assert scene['metadata']['parameters']==p
        if args.diagnostic_emf_floor is not None:
            for cell in scene['cells']:
                if 'electrical' in cell:
                    cell['electrical']['reversible_voltage_floor_v']=args.diagnostic_emf_floor
            scene['metadata']['diagnostic_scene_overrides']={
                'reversible_voltage_floor_v':args.diagnostic_emf_floor,
                'total_UI_work_and_chemical_payment_guards_retained':True,
                'not_a_calibrated_profile':True}
        prefix_end=scene['metadata']['electrical']['onset_s']+600.
        grid=np.r_[scene['failure_sample_times_s'],prefix_end,np.arange(np.ceil(prefix_end),end,1.)]
        scene['failure_sample_times_s']=np.unique(grid[(grid>=0.)&(grid<=end)]).tolist()
        cal.save(folder/'scenario.json',scene)
        cal.save(folder/'status.json',dict(status='running',requested_end_s=end,parameters_sha256=phash))
        print('START',name,'end',end,'measured_event',marker['time_s'],flush=True)
        error=None
        try:
            run=cal.replay(scene,folder,'cuda',args.timeout,refinement=args.refinement);status='complete'
        except (RuntimeError,ValueError,subprocess.TimeoutExpired) as exc:
            error=str(exc);partial=folder/'last-incomplete-run.json.gz'
            if not partial.exists():
                failure=dict(status='failed',error=error,parameters_sha256=phash,model_fingerprint=fp,
                    refinement=args.refinement,no_complete_checkpoint=True,
                    diagnostic_emf_floor_v=args.diagnostic_emf_floor)
                cal.save(folder/'status.json',failure);results[name]=failure
                cal.save(out/'summary.json',dict(experiments=results,parameters_sha256=phash,model_fingerprint=fp,
                    optimization=False,forced_ignition=False,refinement=args.refinement,current_profile_unchanged=True))
                print('FAILED',name,error,flush=True);continue
            run=json.load(gzip.open(partial,'rt'));status='incomplete'
        prefix='validate' if name=='active-cooling' else 'trial'
        with gzip.open(folder/(prefix+'-run.json.gz'),'wt')as stream:json.dump(run,stream,allow_nan=False)
        if cal.fingerprint()!=fp:raise RuntimeError('Model changed during frozen replay')
        summary=dict(status=status,error=error,parameters_sha256=phash,model_fingerprint=fp,
            optimization=False,forced_ignition=False,refinement=args.refinement,
            diagnostic_emf_floor_v=args.diagnostic_emf_floor,metrics=diagnostics(run,scene,meta))
        cal.save(folder/'status.json',summary);results[name]=summary
        cal.save(out/'summary.json',dict(experiments=results,parameters_sha256=phash,model_fingerprint=fp,
            optimization=False,forced_ignition=False,current_profile_unchanged=True))
        report=[sys.executable,str(ROOT/'scripts/report_upv_r123.py'),str(folder),'--stage',prefix]
        if status=='incomplete':report.append('--partial')
        subprocess.run(report,check=True)
        print('DONE',name,json.dumps(summary),flush=True)
    if ptr.read_bytes()!=raw:raise RuntimeError('Current-profile pointer changed during check')


if __name__=='__main__':main()

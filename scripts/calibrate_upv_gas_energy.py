#!/usr/bin/env python3
"""Conditional, closed gas/solid energy trial; no prescribed hot ambient after event.

One effective gas species, opt-in already-ignited gas chemistry, finite oxygen.
Only no-cooling data selects parameters. Cooling is a held-out replay.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
import calibrate_upv_casing as t
from calibrate_upv import save

OUT=t.ROOT/'data/calibration/upv-gas-energy-v7'
PRIOR=t.OUT
NAMES=('solid_energy_j','gas_energy_j','gas_yield','wall_h','exposure_scale','solid_rate')
INITIAL=dict(solid_energy_j=80000.,gas_energy_j=360000.,gas_yield=.35,wall_h=150.,exposure_scale=1.,solid_rate=20.)
BOUNDS=np.log([[1000.,10000.,.05,1.,.05,.1],[400000.,700000.,.8,3000.,100.,100.]])
PROTOCOL=dict(revision='finite-oxygen-closed-gas-v7-step-consistent',calibration='no-cooling',validation='active-cooling',
    gas='Common effective ideal-gas R=287 and cv=1000 before/after reaction; no species/mole-count, CFD, radiation or acoustic wave solver',
    reaction='Gas assumed ignited when fuel arrives; rate 20/s fixed. Oxygen heat 13.1 MJ/kg is a conventional-fuel proxy, NOT a battery-gas measurement',
    oxidizer='Initially .232 mass fraction in chamber; absent from initial cell gas and generated gas; finite inventory advected and consumed, total gas mass unchanged',
    source='Frozen event-conditioned initial fields; tied solid reaction rates fitted; finite net heat split into solid and gas chemical potential, no extra energy source',
    wall='Lumped stainless-wall proxy, volume/area of equivalent 120 L cube, 2 mm steel, rho=8000, cp=500, outer h=5 W/m2K to initial chamber T; geometry/thickness NOT measured',
    thermal='All ambient-* boundaries REPLACED by conservative chamber-to-surface contacts. Measured hot ch109 is diagnostic only, not an energy input. Cooling sink ch110 retained',
    contacts='Frozen intact material/capacity laws and residue factors; one common exposure multiplier, no per-sensor offsets',
    objective='Equal-size-normalized blocks: p(0..30s)/0.25bar, p(30..1800s)/0.05bar; four late witness residuals/5K each. Original scans; no onset-time fit',
    status='Exploratory joint fit, not validated ignition/combustion physics; held-out experiment was inspected in earlier development')


def prepare(name,heat_directory=None,runner=None):
    provenance={}
    if heat_directory is None:
        scene=json.loads((PRIOR/'failure-fast'/(name+'-fitted-scenario.json')).read_text())
        provenance['prior_scene_sha256']=t.old.digest(PRIOR/'failure-fast'/(name+'-fitted-scenario.json'))
    else:
        if runner is None:raise ValueError('Fresh heat profile requires a native warmup runner')
        heat_path=Path(heat_directory)/'profile.json';heat=json.loads(heat_path.read_text())
        scene,_,_=t.failure_prepare(name,heat['parameters'],runner)
        prior_path=PRIOR/'failure-fast/profile.json';prior=json.loads(prior_path.read_text())
        scene=t.fail_bake(scene,prior['parameters'])
        scene['gas_network'].update(max_fraction_per_step=.1,max_temperature_step_k=5.)
        provenance.update(heat_profile_sha256=t.old.digest(heat_path),fixed_failure_prior_sha256=t.old.digest(prior_path))
    scene['metadata']['gas_energy_protocol']=PROTOCOL
    gas=scene['gas_network'];chamber=gas['compartments'][1]
    chamber.update(initial_oxidizer_mass_fraction=.232,reaction_rate_per_s=20.,oxidizer_heat_j_per_kg=13.1e6)
    gas['thermal_contacts']=[]
    ports={p['id']:p for p in scene['ports']};kept=[]
    for b in scene['boundaries']:
        if b['id'].startswith('ambient-'):
            gas['thermal_contacts'].append(dict(compartment='chamber',port_id=b['port_id'],h_w_per_m2_k=b['conductance_w_per_k']/ports[b['port_id']]['area_m2']))
        elif b['id'].startswith('cooling-') or b['id']=='coolant-reservoir':kept.append(b)
        else:raise ValueError('Unaccounted post-event boundary '+b['id'])
    scene['boundaries']=kept
    # One finite-capacity wall node, not an infinite-temperature sink. The flat
    # rectangle represents aggregate area, not a claimed chamber CAD geometry.
    area=6*chamber['volume_m3']**(2/3);side=math.sqrt(area);thickness=.002
    temp=chamber['initial_temperature_k'];cap=area*thickness*8000*500
    scene['cells'].append(dict(id='chamber-wall',role='passive',material_id='unmeasured-stainless-wall-proxy',
        geometry=dict(kind='prismatic',dimensions_m=[thickness,side,side]),pose=dict(position_m=[1.,0.,0.]),
        thermal=dict(heat_capacity_j_per_k=cap,initial_temperature_k=temp,divisions=[1,1,1],conductivity_w_per_m_k=[15.,15.,15.]),
        runaway=dict(energy_j=0.,duration_s=1.)))
    for sign,tag in [(-1,'inner'),(1,'outer')]:
        scene['ports'].append(dict(id='wall:'+tag,cell_id='chamber-wall',kind='contact',surface=('-' if sign<0 else '+')+'x',
            area_m2=area,centroid_m=[1.+sign*thickness/2,0.,0.],normal=[sign,0.,0.]))
    gas['thermal_contacts'].append(dict(compartment='chamber',port_id='wall:inner',h_w_per_m2_k=150.))
    scene['boundaries'].append(dict(id='wall-room',port_id='wall:outer',temperature_k=temp,conductance_w_per_k=5*area))
    data,record=t.old.load_record(name);event=t.old.event_marker(data)
    _,obs=t.old.observations(data,event['last_pre_event_s'],1800.)
    at=np.asarray(scene['failure_sample_times_s']);pressure=t.old.resample(data['time_s'],2.5*data['U114_v'],at+event['last_pre_event_s'])
    return scene,obs,pressure,dict(event=event,source_csv_sha256=record['canonical_sha256'],**provenance)


def bake(scene,p):
    s=copy.deepcopy(scene);source=next(c for c in s['cells'] if c['id']=='C2')['conversion']
    total=p['solid_energy_j']+p['gas_energy_j'];source['net_heat_j_per_kg']=total/source['mass_kg']
    source['gas_chemical_fraction']=p['gas_energy_j']/total;source['gas_yield']=p['gas_yield']
    source['activation_rate_per_s']=source['reaction_rate_per_s']=p['solid_rate']
    for contact in s['gas_network']['thermal_contacts']:
        if contact['port_id']=='wall:inner':contact['h_w_per_m2_k']=p['wall_h']
        else:contact['h_w_per_m2_k']*=p['exposure_scale']
    return s


def errors(run,obs,pressure,event):
    at=np.asarray(run['time_s']);pred=np.asarray(run['gas']['chamber']['pressure_pa'])/1e5
    blocks=[]
    for mask,scale in [((at>0)&(at<=30),.25),((at>30),.05)]:
        mask &= np.isfinite(pressure)
        e=(pred[mask]-pressure[mask])/scale;blocks.append(e/math.sqrt(len(e)))
    for e in t.late.errors(run,obs,event).values():blocks.append(e/(5*math.sqrt(len(e))))
    return np.concatenate(blocks)


def metrics(run,obs,pressure,event):
    at=np.asarray(run['time_s']);pred=np.asarray(run['gas']['chamber']['pressure_pa'])/1e5
    result=dict(witness=t.late.metrics(run,obs,event),objective_sum_squares=float(np.sum(errors(run,obs,pressure,event)**2)))
    for name,mask in [('early',(at>0)&(at<=30)),('late',at>30)]:
        mask &= np.isfinite(pressure);e=pred[mask]-pressure[mask];result['pressure_'+name+'_rmse_bar']=float(np.sqrt(np.mean(e*e)))
    result.update(predicted_peak_bar=float(np.max(pred)),observed_peak_bar=float(np.nanmax(pressure)),
        predicted_peak_time_s=float(at[np.argmax(pred)]),observed_peak_time_s=float(at[np.nanargmax(pressure)]),
        gas_released_j=run['gas']['chamber']['released_energy_j'][-1],gas_fuel_remaining_j=run['gas']['chamber']['chemical_j'][-1],
        oxygen_remaining_kg=run['gas']['chamber']['oxidizer_kg'][-1],gas_final_temperature_k=run['gas']['chamber']['temperature_k'][-1])
    return result


def fingerprint(probe,heat_directory=None):
    heat_path=(Path(heat_directory) if heat_directory else PRIOR/'heat')/'profile.json'
    return hashlib.sha256((t.fingerprint(probe)+t.old.digest(Path(__file__))+t.old.digest(heat_path)).encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('mode',choices=['verify-heat','baseline','fit','validate']);ap.add_argument('--output',type=Path,default=OUT)
    ap.add_argument('--probe',type=Path,default=Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'));ap.add_argument('--max-evaluations',type=int,default=10)
    ap.add_argument('--heat-directory',type=Path,help='Fresh frozen heating fit; regenerate post-event fields using that fit')
    ap.add_argument('--initial-profile',type=Path,help='Previous training-only fit used as initialization, not as a frozen solution')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);runner=t.Runner(args.probe,args.output);fp=fingerprint(args.probe,args.heat_directory)
    try:
        if args.heat_directory:
            heat=json.loads((args.heat_directory/'profile.json').read_text())
            if heat['fingerprint']!=t.fingerprint(args.probe):raise ValueError('Changed model since heating fit')
        if args.mode=='verify-heat':
            result={}
            heat_directory=args.heat_directory or PRIOR/'heat'
            for mode,name in [('fit','no-cooling'),('validate','active-cooling')]:
                scene=json.loads((heat_directory/(name+'-fitted-scenario.json')).read_text());old=json.loads((heat_directory/(mode+'.json')).read_text())['result'];r=runner.run(scene)
                change=max(float(np.max(np.abs(np.asarray(r['temperature_k'][k])-v))) for k,v in old['temperature_k'].items())
                if change>1e-10 or r['applied_objects']!=old['applied_objects']:raise ValueError('Frozen heat model changed')
                result[name]=dict(max_change_k=change,steps=r['steps'])
            save(args.output/'heat-replay.json',dict(results=result,fingerprint=fp));print(result,flush=True);return
        proof=json.loads((args.output/'heat-replay.json').read_text())
        if proof['fingerprint']!=fp:raise ValueError('Replay frozen heat model after changing code')
        profile_path=args.output/'profile.json';name='active-cooling' if args.mode=='validate' else 'no-cooling'
        scene,obs,pressure,meta=prepare(name,args.heat_directory,runner)
        initial=INITIAL
        if args.initial_profile:
            prior=json.loads(args.initial_profile.read_text())
            if prior.get('validation_used_for_selection') is not False:raise ValueError('Initialization must be training-only')
            initial=prior['parameters'];meta['initial_profile_sha256']=t.old.digest(args.initial_profile)
        if args.mode=='validate':
            profile=json.loads(profile_path.read_text())
            if profile['fingerprint']!=fp:raise ValueError('Frozen model changed')
            p=profile['parameters']
        elif args.mode=='baseline':p=initial
        else:
            if profile_path.exists():raise ValueError('Use a fresh output directory; profile is frozen')
            best=[float('inf'),None];count=[0];start=time.monotonic()
            def residual(x):
                p=dict(zip(NAMES,map(float,np.exp(x))));r=runner.run(bake(scene,p));v=errors(r,obs,pressure,meta['event']);cost=float(v@v);count[0]+=1
                if r.get('max_oxidizer_residual_kg',0)>1e-10:raise ValueError('Oxygen balance failed')
                if cost<best[0]:best[:]=[cost,p];save(args.output/'checkpoint.json',dict(parameters=p,cost=cost,evaluations=count[0]));save(args.output/'best-run.json',r)
                print(f'candidate {count[0]}: cost={cost:.5f}, best={best[0]:.5f}, peak={max(r["gas"]["chamber"]["pressure_pa"])/1e5:.3f} bar, wall={time.monotonic()-start:.0f}s',flush=True)
                return v
            fit=least_squares(residual,np.log([initial[k] for k in NAMES]),bounds=BOUNDS,diff_step=.005,x_scale='jac',max_nfev=args.max_evaluations,ftol=1e-4,xtol=1e-4,gtol=1e-4)
            p=best[1];save(profile_path,dict(parameters=p,fingerprint=fp,protocol=PROTOCOL,meta=meta,validation_used_for_selection=False,
                heat_directory=str(args.heat_directory.resolve()) if args.heat_directory else None,
                search=dict(success=bool(fit.success),message=fit.message,nfev=fit.nfev,evaluations=count[0],jacobian_singular_values=np.linalg.svd(fit.jac,compute_uv=False).tolist())))
        s=bake(scene,p);r=runner.run(s)
        # Missing/invalid thermocouple scans remain missing; JSON null preserves
        # the gap without emitting NaN or inventing observations for plotting.
        observed_temperature={k:[float(x) if np.isfinite(x) else None for x in v] for k,v in obs.items()}
        report=dict(experiment=name,parameters=p,protocol=PROTOCOL,meta=meta,metrics=metrics(r,obs,pressure,meta['event']),observed_pressure_bar=pressure.tolist(),observed_temperature_k=observed_temperature,result=r)
        if args.mode!='baseline':
            fs=copy.deepcopy(s);fs['simulation']['stability_factor']*=.5;fine=runner.run(fs,.5)
            report['refined_metrics']=metrics(fine,obs,pressure,meta['event'])
            report['refinement']=dict(max_pressure_change_bar=float(np.max(np.abs(np.asarray(r['gas']['chamber']['pressure_pa'])-fine['gas']['chamber']['pressure_pa']))/1e5),
                max_witness_change_k=max(float(np.max(np.abs(np.asarray(r['temperature_k'][k])-fine['temperature_k'][k]))) for k in t.late.WITNESSES),coarse_steps=r['steps'],fine_steps=fine['steps'])
        save(args.output/(name+'-'+args.mode+'-scenario.json'),s);save(args.output/(args.mode+'.json'),report);print(json.dumps(report['metrics'],indent=2),flush=True)
    finally:runner.close()


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Fixed-hole ablation of CUDA gas transport, not a physical UPV replay or fit."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
import numpy as np
from calibrate_upv import save

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SOURCE=ROOT/'data/calibration/upv-leading-aperture-v31/feedback-dt50us'
PROBE=Path('/tmp/battery-sim-cuda-v1/battery_sim_gas_transport_audit')


def empty_scene():
    return dict(name='isolated fixed gas network diagnostic',
        simulation=dict(duration_s=1.,max_dt_s=.2,output_interval_s=1.,stability_factor=.4),
        cells=[dict(id='inert-placeholder',role='passive',geometry=dict(kind='prismatic',dimensions_m=[1.,1.,1.]),
                    thermal=dict(heat_capacity_j_per_k=1.,initial_temperature_k=300.,divisions=[1,1,1]),
                    runaway=dict(energy_j=0.,duration_s=1.))],ports=[],interfaces=[],boundaries=[],
        gas_network=dict(transport='split_local_equilibrium',reference_temperature_k=300.,cv_j_per_kg_k=1000.,
                         specific_r_j_per_kg_k=287.,max_fraction_per_step=.1,max_temperature_step_k=50.,
                         compartments=[],channels=[],thermal_contacts=[]))


def chain():
    s=empty_scene();g=s['gas_network']
    for i,p in enumerate((100100.,100000.,100000.)):
        g['compartments'].append(dict(id=f'v{i}',volume_m3=1e-6,initial_mass_kg=p*1e-6/(287.*300.),initial_temperature_k=300.))
    for i in range(2):
        g['channels'].append(dict(id=f'path{i}',kind='passage',**{'from':f'v{i}','to':f'v{i+1}'},
                                  area_m2=1e-4,discharge_coefficient=.7,opening_pressure_pa=0.))
    s['metadata']=dict(scope='Three identical gas volumes, fixed passages; no solids/chemistry/rupture',
                       eventual_common_pressure_pa=100000.+100./3.)
    return s


def snapshot(source,time):
    old=json.loads((source/'scenario.json').read_text());run=json.loads((source/'run.json').read_text())
    i=min(range(len(run['time_s'])),key=lambda k:abs(run['time_s'][k]-time))
    if abs(run['time_s'][i]-time)>1e-9:raise ValueError('Snapshot time must be a stored sample')
    s=empty_scene();g=s['gas_network'];previous=old['gas_network']
    for key in ('reference_temperature_k','cv_j_per_kg_k','specific_r_j_per_kg_k',
                'max_fraction_per_step','max_temperature_step_k'):
        g[key]=previous[key]
    for old_node in previous['compartments']:
        key=old_node['id'];v=run['gas'][key]
        n=dict(id=key,volume_m3=v['free_volume_m3'][i],initial_mass_kg=v['mass_kg'][i],initial_temperature_k=v['temperature_k'][i])
        pressure=n['initial_mass_kg']*g['specific_r_j_per_kg_k']*n['initial_temperature_k']/n['volume_m3']
        if not math.isclose(pressure,v['pressure_pa'][i],rel_tol=1e-10):raise ValueError('Snapshot pressure mismatch')
        g['compartments'].append(n)
    closed=[];ruptures=[]
    for c in previous['channels']:
        area=run['channels'][c['id']]['effective_area_m2'][i]
        if area==0:closed.append(c['id']);continue
        if c['to']=='outside_domain':raise ValueError('Closed-network audit cannot have an external sink')
        g['channels'].append(dict(id=c['id'],kind='passage',**{'from':c['from'],'to':c['to']},area_m2=area,
                                 discharge_coefficient=c.get('discharge_coefficient',.7),opening_pressure_pa=0.))
        if c['kind']=='rupture':ruptures.append(c['id'])
    s['metadata']=dict(source=str(source),snapshot_time_s=run['time_s'][i],closed_paths=closed,rupture_paths=ruptures,
        scope='Gas-only ablation; identical snapshot for every dt, NOT a restart of the full multiphase model',
        preserved='Gas masses, temperatures, free volumes, R/cv, graph endpoints, open areas and discharge coefficients',
        disabled='Further rupture, solid/gas LTE, solid heat transfer, gas production, combustion; chemical tracer omitted')
    return s


def summarize(run,reference):
    ids=list(run['gas']);keys=list(run['channels'])
    if len(run['time_s'])!=len(reference['time_s']) or not np.allclose(run['time_s'],reference['time_s'],rtol=0,atol=1e-12):
        raise ValueError('Output times differ')
    def field(r,key):return np.array([r['gas'][k][key] for k in ids])
    p,q=field(run,'pressure_pa'),field(reference,'pressure_pa')
    m,n=field(run,'mass_kg'),field(reference,'mass_kg')
    temperature_delta=abs(field(run,'temperature_k')-field(reference,'temperature_k'))
    worst=np.unravel_index(np.argmax(abs(p-q)),p.shape)
    initial_mass=float(m[:,0].sum())
    flow_error=max(abs(run['channels'][k]['transferred_mass_kg'][-1]-reference['channels'][k]['transferred_mass_kg'][-1]) for k in keys)
    return dict(max_dt_s=run['max_dt_s'],mass_budget=run['mass_budget'],steps=run['steps'],wall_s=run['wall_s'],
        max_actual_dt_s=run['max_actual_dt_s'],max_pressure_difference_pa=float(abs(p-q).max()),
        max_final_pressure_difference_pa=float(abs(p[:,-1]-q[:,-1]).max()),
        worst_pressure_node=ids[worst[0]],worst_pressure_time_s=run['time_s'][worst[1]],
        final_pressure_spread_pa=float(np.ptp(p[:,-1])),
        max_mass_L1_difference_kg=float(abs(m-n).sum(axis=0).max()),
        max_mass_L1_fraction_of_inventory=float(abs(m-n).sum(axis=0).max()/initial_mass),
        max_final_path_transfer_difference_kg=flow_error,
        max_temperature_difference_k=float(temperature_delta.max()),
        max_final_temperature_difference_k=float(temperature_delta[:,-1].max()),
        fraction_steps_with_clipping=run['steps_with_clipping']/run['steps'],
        max_mass_residual_kg=run['max_mass_residual_kg'],max_thermal_residual_j=run['max_thermal_residual_j'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('case',choices=['chain','snapshot'])
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--snapshot-time',type=float,default=1.1)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--probe',type=Path,default=PROBE)
    p.add_argument('--caps',type=float,nargs='+');p.add_argument('--duration',type=float)
    p.add_argument('--sample-interval',type=float)
    p.add_argument('--pressure-fraction',type=float,default=.2)
    p.add_argument('--pressure-atol',type=float,default=1.)
    p.add_argument('--pressure-rtol',type=float,default=1e-5)
    p.add_argument('--mass-budget',type=float,help='Numerical inventory budget only; no physical parameter changes')
    args=p.parse_args();s=chain() if args.case=='chain' else snapshot(args.source,args.snapshot_time)
    controls=dict(pressure_relaxation_fraction=args.pressure_fraction,
                  pressure_absolute_tolerance_pa=args.pressure_atol,pressure_relative_tolerance=args.pressure_rtol)
    if not all(math.isfinite(x) for x in controls.values()) or not 0<args.pressure_fraction<=.5 or args.pressure_atol<=0 or not 0<=args.pressure_rtol<=1:
        p.error('Invalid pressure timestep controls')
    s['gas_network'].update(controls)
    if args.mass_budget is not None:
        if not math.isfinite(args.mass_budget) or not 0<args.mass_budget<=.1:p.error('Invalid mass budget')
        s['gas_network']['max_fraction_per_step']=args.mass_budget
    duration=args.duration or (2e-5 if args.case=='chain' else .002)
    caps=args.caps or ([2e-5,2e-6,2e-7,2e-8,5e-9] if args.case=='chain' else [2e-4,5e-5,1.25e-5,3.125e-6,7.8125e-7])
    if any(not math.isfinite(x) or x<=0 for x in [duration,*caps]):p.error('Invalid controls')
    interval=args.sample_interval if args.sample_interval is not None else (duration if args.case=='chain' else .0005)
    if not math.isfinite(interval) or interval<=0:p.error('Invalid sample interval')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+args.probe.read_bytes()).hexdigest()
    manifest=dict(fingerprint=fingerprint,case=args.case,metadata=s['metadata'],caps=caps,duration_s=duration,sample_interval_s=interval,
                  mass_budget=s['gas_network']['max_fraction_per_step'],
                  pressure_controls=controls,calibration_accepted=False,simulation='Production CUDA kernels, no physical parameter fitting')
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:
        raise ValueError('Different audit: use a fresh directory')
    save(out/'manifest.json',manifest);save(out/'scenario.json',s)
    reports=[]
    for k,cap in enumerate(caps):
        path=out/f'cap-{k}.json'
        if path.exists():r=json.loads(path.read_text())
        else:
            print('starting',args.case,'dt cap',cap,flush=True)
            completed=subprocess.run([str(args.probe),str(out/'scenario.json'),str(cap),str(s['gas_network']['max_fraction_per_step']),
                                      str(duration),str(interval)],text=True,stdout=subprocess.PIPE,check=True)
            r=json.loads(completed.stdout);save(path,r)
        reports.append(r);print('steps',r['steps'],'wall_s',r['wall_s'],flush=True)
    reference=reports[-1]
    rows=[summarize(r,reference) for r in reports]
    save(out/'comparison.json',dict(reference='Finest run, not an analytic solution',rows=rows,calibration_accepted=False))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()

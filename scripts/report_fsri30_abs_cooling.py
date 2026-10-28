#!/usr/bin/env python3
"""Accepted-film and solid-storage audit; does not modify the simulation."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from fsri30_data import save,sha


def outside_state(scene,run,frames_path,time):
    """Infer film-weighted wall/gas temperatures from actual instantaneous Q/G.

    Full exterior gas temperatures are observed, not an ambient substitution.
    Float32 gas export is diagnostic only and never modifies solver state.
    """
    if time<run['time_s'][0]-1e-7 or time>run['time_s'][-1]+1e-7:
        raise ValueError('Requested surface state has not been observed')
    path=Path(frames_path)
    with path.open('rb')as f:
        if f.read(8)!=b'BATTGAS1':raise ValueError('Unknown gas frame format')
        count=int(np.frombuffer(f.read(8),dtype='<u8')[0])
    nodes=scene['gas_network']['compartments']
    if count!=len(nodes):raise ValueError('Gas frame topology does not match scene')
    dtype=np.dtype([('time','<f8'),('values','<f4',(count,3))])
    length=(path.stat().st_size-16)//dtype.itemsize
    if not length:raise ValueError('No complete gas frames')
    frames=np.memmap(path,dtype=dtype,mode='r',offset=16,shape=(length,))
    j=int(np.argmin(abs(np.asarray(run['time_s'])-time)))
    t=run['time_s'][j];n=int(np.argmin(abs(frames['time']-t)))
    if abs(frames['time'][n]-t)>1e-7:raise ValueError('Gas and solid observations do not align')
    temperatures={node['id']:float(frames['values'][n,i,0])for i,node in enumerate(nodes)}
    conductance=gas_g_t=wall_to_gas=0.
    for contact in run['gas_film_trace'].values():
        if not contact['solid'].startswith('ABS:') or not contact['gas'].startswith('air-'):continue
        g=contact['instantaneous_conductance_w_per_k'][j]
        conductance+=g;gas_g_t+=g*temperatures[contact['gas']]
        wall_to_gas-=contact['instantaneous_power_w'][j]
    if conductance<=0:raise ValueError('No exterior ABS film conductance')
    gas=gas_g_t/conductance;wall=gas+wall_to_gas/conductance
    return dict(time_s=t,conductance_w_per_k=conductance,
        instantaneous_wall_to_gas_w=wall_to_gas,
        conductance_weighted_gas_temperature_k=gas,
        conductance_weighted_wall_temperature_k=wall,
        film_resistance_k_per_w=1/conductance,
        apparent_wall_to_298K_resistance_k_per_w=(wall-298.15)/wall_to_gas if wall_to_gas>0. else None,
        diagnostic_gas_precision='float32',scope='Observed ordinary exterior ABS gas films only')


def audit(run,start,end):
    t=np.asarray(run['time_s'])
    if start>=end or start<t[0]-1e-7 or end>t[-1]+1e-7:
        raise ValueError('Requested window has not been observed')
    a=int(np.argmin(abs(t-start)));b=int(np.argmin(abs(t-end)));dt=t[b]-t[a]
    if dt<=0:raise ValueError('Empty observation window')
    rows={}
    for ident,c in run['gas_film_trace'].items():
        if not c['solid'].startswith('ABS:'):continue
        row=rows.setdefault(c['solid'],dict(gas_inside_to_wall_j=0.,wall_to_external_gas_j=0.))
        delta=c['heat_to_solid_j'][b]-c['heat_to_solid_j'][a]
        row['wall_to_external_gas_j'if c['gas'].startswith('air-')else'gas_inside_to_wall_j']+=-delta if c['gas'].startswith('air-')else delta
    capacities={c['id']:c['capacity_j_k']for c in run['applied_objects']}
    for ident,row in rows.items():
        temp=run['thermal_paths']['objects'][ident]['reference_capacity_weighted_temperature_k']
        row['stored_energy_change_j']=capacities[ident]*(temp[b]-temp[a])
        for key in list(row):row[key.removesuffix('_j')+'_w']=row[key]/dt
    total={key:sum(row[key]for row in rows.values())for key in (
        'gas_inside_to_wall_j','wall_to_external_gas_j','stored_energy_change_j')}
    # All ABS seam flows are internal to the union of the six panels and must
    # cancel. Constant panel cp makes this an independent accepted-heat closure.
    residual=total['stored_energy_change_j']-total['gas_inside_to_wall_j']+total['wall_to_external_gas_j']
    result=dict(start_s=float(t[a]),end_s=float(t[b]),panels=rows,
        total_power_w={k.removesuffix('_j'):v/dt for k,v in total.items()},
        abs_union_energy_closure_j=residual,closure_accepted=abs(residual)<1e-4,
        ambient_export_w=(run['process_totals']['escaped_energy_j'][b]-run['process_totals']['escaped_energy_j'][a])/dt,
        r3_power_w=(run['process_totals']['gas_released_energy_j'][b]-run['process_totals']['gas_released_energy_j'][a])/dt)
    if not result['closure_accepted']:raise ValueError(f'ABS union accepted-heat closure failed: {residual:g} J')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--start',type=float,default=3300.);ap.add_argument('--end',type=float,default=3500.)
    ap.add_argument('--scene',type=Path);ap.add_argument('--gas-frames',type=Path)
    args=ap.parse_args()
    if args.run.suffix=='.gz':
        with gzip.open(args.run,'rt')as f:run=json.load(f)
    else:run=json.loads(args.run.read_text())
    value=audit(run,args.start,args.end)
    if args.gas_frames:
        if not args.scene:raise ValueError('Scene required for gas-frame topology')
        value['outside_surface']=outside_state(json.loads(args.scene.read_text()),run,args.gas_frames,args.end)
    value.update(run_sha256=sha(args.run),last_observed_s=run['time_s'][-1],
        trajectory_complete=not run.get('partial',True),calibrated=False,validation_passed=False)
    save(args.output,value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()

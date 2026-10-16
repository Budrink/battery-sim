#!/usr/bin/env python3
"""Sequential, warmed CPU/CUDA gas-only benchmarks; no physical calibration."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from calibrate_upv import save

PROBE=Path('/tmp/battery-sim-cuda-v1/battery_sim_gas_patch_benchmark')


def compare(run,reference):
    if not np.allclose(run['time_s'],reference['time_s'],rtol=0,atol=1e-13):
        raise ValueError('Different output times')
    ids=list(reference['gas'])
    def field(r,key):return np.array([r['gas'][i][key] for i in ids])
    p,q=field(run,'pressure_pa'),field(reference,'pressure_pa')
    t,u=field(run,'temperature_k'),field(reference,'temperature_k')
    m,n=field(run,'mass_kg'),field(reference,'mass_kg')
    return dict(max_pressure_difference_pa=float(abs(p-q).max()),
        max_relative_pressure_difference=float((abs(p-q)/np.maximum(abs(q),1.)).max()),
        max_temperature_difference_k=float(abs(t-u).max()),
        max_mass_l1_difference_kg=float(abs(m-n).sum(axis=0).max()),
        max_final_path_mass_difference_kg=max((abs(run['channels'][k]['transferred_mass_kg'][-1]-v['transferred_mass_kg'][-1])
            for k,v in reference['channels'].items()),default=0.))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('scene',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe',type=Path,default=PROBE);p.add_argument('--duration',type=float,default=3e-5)
    p.add_argument('--interval',type=float,default=3e-6);p.add_argument('--global-cap',type=float,default=2e-4)
    p.add_argument('--caps',type=float,nargs='+',default=[1e-6,2.5e-7,6.25e-8]);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--patch-size',type=int,default=9)
    p.add_argument('--backends',nargs='+',default=['cpu-global','cuda-global','production','cpu-local','cuda-local'])
    args=p.parse_args()
    if any(not math.isfinite(x) or x<=0 for x in [args.duration,args.interval,args.global_cap,*args.caps]):p.error('Invalid controls')
    if not 2<=args.patch_size<=9 or not 1<=args.repeats<=20:p.error('Invalid size/repetitions')
    source=json.loads(args.scene.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(args.probe.read_bytes()+Path(__file__).read_bytes()+args.scene.read_bytes()).hexdigest()
    manifest=dict(fingerprint=fingerprint,source=str(args.scene.resolve()),duration=args.duration,interval=args.interval,
        global_cap=args.global_cap,caps=args.caps,repeats=args.repeats,patch_size=args.patch_size,backends=args.backends,
        scope='Gas-only fixed network; single-thread C++ CPU, sequential benchmark runs, warmup excluded')
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:
        raise ValueError('Use a fresh output directory for changed inputs/binary')
    save(out/'manifest.json',manifest);save(out/'scenario.json',source)
    results=[]
    for backend in args.backends:
        for i,cap in enumerate(args.caps if backend.endswith('-local') else [args.global_cap]):
            path=out/f'{backend}-{i}.json'
            if path.exists():r=json.loads(path.read_text())
            else:
                print('starting',backend,'cap',cap,flush=True)
                completed=subprocess.run([str(args.probe),str(out/'scenario.json'),backend,str(args.duration),str(cap),
                    str(args.interval),str(args.repeats),str(args.patch_size)],stdout=subprocess.PIPE,text=True,check=True,timeout=1800)
                r=json.loads(completed.stdout);save(path,r)
            results.append((path.name,r))
    references=[r for _,r in results if r['backend']=='cpu-global']
    if not references:
        print('Individual benchmark files saved; no cpu-global reference requested');return
    reference=references[0];rows=[]
    for name,r in results:
        row=dict(file=name,backend=r['backend'],cap_s=r['max_dt_s'],solve_wall_median_s=r['solve_wall_median_s'],
            speedup_vs_cpu_global=reference['solve_wall_median_s']/r['solve_wall_median_s'],
            steps=r['trials'][-1]['steps'],patch_substeps=r['trials'][-1]['patch_substeps'],
            edge_evaluations=r['trials'][-1]['edge_evaluations'],kernel_launches=r['trials'][-1]['kernel_launches'],
            max_mass_residual_kg=max(v['max_mass_residual_kg'] for v in r['trials']),
            max_energy_residual_j=max(v['max_energy_residual_j'] for v in r['trials']),
            **compare(r,reference))
        paired=[other for _,other in results if other['backend']=='cpu-local' and other['max_dt_s']==r['max_dt_s']]
        if r['backend']=='cuda-local' and paired:row['cpu_cuda_local_difference']=compare(r,paired[0])
        rows.append(row)
    save(out/'summary.json',dict(nodes=reference['nodes'],edges=reference['edges'],patches=reference['patches'],
        colors=reference['colors'],rows=rows,reference='CPU global: verify against production and finer controls, not an exact solution',
        calibration_accepted=False,warning='Production includes placeholder thermal/transaction overhead; pure CUDA comparison is cuda-global. '
        'Samples are sparse and downloads timed separately. Wall timings are workstation measurements, not guaranteed hardware limits.'))
    for row in rows:print(json.dumps(row))


if __name__=='__main__':main()

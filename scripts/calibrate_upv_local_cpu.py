#!/usr/bin/env python3
"""CPU-local or CUDA-MC destruction replay. Frozen heat profile and numeric controls.

Each candidate has an immutable manifest and a wall-time budget; a timeout is
recorded as incomplete, never assigned a fitted score. Validation is separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import replay_upv_leading_aperture as leading
import calibrate_upv_mass_paths as fit
from calibrate_upv import save


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--duration', type=float, default=30.)
    p.add_argument('--backend',choices=['cpu_local','cuda_kawasaki','cuda_batched','cuda_deterministic','cuda_network'],default='cpu_local')
    p.add_argument('--sweeps',type=int,choices=[1,2,3,4,5,6,7,8],default=4,
                   help='fixed whole-graph deterministic mass sweeps per accepted thermal step')
    p.add_argument('--deterministic-pressure-fraction',type=float,default=.75,
                   help='per-node pressure budget for deterministic whole-graph sweeps (0, 1]')
    p.add_argument('--network-pressure-fraction',type=float,default=1.,
                   help='simultaneous network pressure-change limiter (0, .45]')
    p.add_argument('--batch-pressure-fraction',type=float,default=.2)
    p.add_argument('--batch-mean-threshold',type=float,default=32.)
    p.add_argument('--packet-fraction',type=float,default=.03125)
    p.add_argument('--tail-atol',type=float,default=100.)
    p.add_argument('--seed',type=int,default=20261015)
    p.add_argument('--max-dt', type=float, default=.001)
    p.add_argument('--exchange-interval', type=float, default=1e-5)
    p.add_argument('--integrator', choices=['explicit','implicit'],default='explicit')
    p.add_argument('--refinement', type=float, default=1.,
                   help='Scale numerical time/error controls only; 0 < value <= 1')
    p.add_argument('--pressure-atol', type=float, default=1.)
    p.add_argument('--pressure-rtol', type=float, default=1e-5)
    p.add_argument('--pressure', type=float, default=50e6)
    p.add_argument('--rate', type=float, default=20000.)
    p.add_argument('--timeout', type=float, default=600.)
    p.add_argument('--experiment', choices=['no-cooling', 'active-cooling'], default='no-cooling')
    args = p.parse_args()
    if args.backend in ('cuda_kawasaki','cuda_batched') and args.integrator!='explicit':
        raise ValueError('MC has no CPU implicit branch')
    if not (np.isfinite(args.packet_fraction) and 0<args.packet_fraction<=1 and
            np.isfinite(args.tail_atol) and args.tail_atol>=0 and args.seed>=0):
        raise ValueError('Invalid MC controls')
    if not all(np.isfinite(v) and v>0 for v in (args.pressure,args.rate,args.timeout,args.exchange_interval,args.pressure_atol,args.pressure_rtol)):
        raise ValueError('Invalid control')
    if not 0 < args.refinement <= 1:
        raise ValueError('Invalid refinement')
    if not np.isfinite(args.deterministic_pressure_fraction) or not 0 < args.deterministic_pressure_fraction <= 1:
        raise ValueError('Invalid deterministic pressure fraction')
    if not np.isfinite(args.network_pressure_fraction) or not 0 < args.network_pressure_fraction <= 1:
        raise ValueError('Invalid network pressure fraction')
    source, prior = leading.old.geometry.inputs(args.experiment)
    scene = leading.build(source,args.duration,args.max_dt)
    frozen = leading.old.physical_signature(scene)
    scene['gas_network'].update(mass_backend=args.backend,local_exchange_interval_s=args.exchange_interval,
        local_integrator=args.integrator,
        pressure_absolute_tolerance_pa=args.pressure_atol,pressure_relative_tolerance=args.pressure_rtol)
    if args.backend in ('cuda_kawasaki','cuda_batched'):
        scene['gas_network'].update(kawasaki_packet_fraction=args.packet_fraction,
            kawasaki_tail_atol_pa=args.tail_atol,kawasaki_seed=args.seed)
    if args.backend in ('cuda_batched','cuda_deterministic'):
        scene['gas_network'].update(batch_pressure_fraction=args.batch_pressure_fraction,
            batch_mean_threshold=args.batch_mean_threshold)
    if args.backend=='cuda_deterministic':
        scene['gas_network'].update(deterministic_sweeps_per_step=args.sweeps,
            deterministic_pressure_fraction=args.deterministic_pressure_fraction)
    if args.backend=='cuda_network':
        scene['gas_network'].update(network_pressure_fraction=args.network_pressure_fraction)
    for c in scene['gas_network']['channels']:
        if c['kind']=='rupture':
            c['opening_pressure_pa']=args.pressure
            c['damage_opening']['area_rate_per_s']=args.rate
    assert frozen==leading.old.physical_signature(scene)
    scene['metadata']['hybrid_trial']=dict(mass_backend=args.backend,heat_profile_frozen=True,
        parameters=dict(pressure_pa=args.pressure,rate_per_s=args.rate),calibration_accepted=False)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    manifest=dict(binary_sha256=hashlib.sha256(fit.PROBE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scene_sha256=leading.old.geometry.digest(scene),physical_thermal_sha256=frozen,
        experiment=args.experiment,refinement=args.refinement,calibration_accepted=False)
    if (out/'manifest.json').exists() and leading.old.geometry.read(out/'manifest.json')!=manifest:
        raise ValueError('Changed inputs; use a fresh output directory')
    save(out/'manifest.json',manifest);save(out/'scenario.json',scene)
    if (out/'run.json').exists():
        print('Replay already completed:',out,flush=True);return
    request=json.dumps(dict(scenario_path=str((out/'scenario.json').resolve()),refinement=args.refinement,
                           checkpoint_path=str((out/'partial.json').resolve())))+'\n'
    print('Starting',args.experiment,args.pressure,args.rate,'duration',args.duration,flush=True)
    # subprocess.run kills/waits for this exact child on timeout. No orphan worker.
    with (out/'progress.log').open('w') as log:
        try:
            result=subprocess.run([str(fit.PROBE),str(out/'scenario.json')],input=request,text=True,
                stdout=subprocess.PIPE,stderr=log,timeout=args.timeout,
                env={**os.environ,'BATTERY_SIM_PROGRESS':'1'},check=True)
        except subprocess.TimeoutExpired:
            partial=leading.old.geometry.read(out/'partial.json') if (out/'partial.json').exists() else None
            save(out/'status.json',dict(completed=False,reason='wall_time_budget',seconds=args.timeout,
                last_sample_s=partial['time_s'][-1] if partial else None,calibration_accepted=False))
            print('Incomplete: wall-time budget reached; no fit accepted',flush=True);return
    run=json.loads(result.stdout)
    if 'error' in run:
        save(out/'status.json',dict(completed=False,error=run['error'],calibration_accepted=False))
        raise RuntimeError(run['error'])
    fit.native.audit(run,scene);save(out/'run.json',run)
    sampled=leading.old.strength.observation_samples(run,prior)
    report=dict(completed=True,calibration_accepted=False,parameters=scene['metadata']['hybrid_trial']['parameters'],
        **leading.summary(run,scene),metrics=fit.metrics(sampled,prior) if len(sampled['time_s'])>1 else {})
    if args.duration>=30:
        report['score']=leading.old.protocol.pressure_score(run,prior)
        report['joint_objective']=report['score']['objective']+(report['metrics']['phases']['3-30']['rmse_k']/20)**2
    save(out/'summary.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()

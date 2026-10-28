#!/usr/bin/env python3
"""Fresh geometry control, then bounded gas-access continuations with explicit core.

No cross-binary checkpoint migration. A fresh 0..835 s r8 prefix is followed
by conservative, native-contract-checked geometry changes. Non-control trials
remain conditional on that prefix until independently restarted from ambient.
"""
import argparse
import copy
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import numpy as np
import calibrate_upv_r123 as cal
from check_upv_restart import native
from probe_upv_gas_access import plot
from upv_gas_access import candidate,shape_report,received_gas_heat
import upv_r123_data as dataio
from upv_observation_policy import CALIBRATION_CHANNELS


def geometry(source,outer,end=950.,jet=1.,edge=.1,iss3=None,protected=.001,lip=.001,pad=1.):
    return candidate(source,protected,end=end,lip_open_fraction=lip,
        backing_open_fraction=.01,witness_edge_open_fraction=edge,
        outer_insulation_open_fraction=outer,jet_wetted_fraction=jet,iss3_accessible_fraction=iss3,
        open_iss_pad_wetted_fraction=pad)


def same_prefix_physics(prefix,scene):
    """Full physical equality, excluding duration, output times and annotations."""
    def physical(value):
        value=copy.deepcopy(value);value.pop('metadata',None)
        value.pop('failure_sample_times_s',None)
        value['simulation'].pop('duration_s',None)
        return value
    return physical(prefix)==physical(scene)


def checked_seed_geometry(source,seed_prefix):
    """Reconstruct an existing geometry; never accept arbitrary seed physics."""
    g=seed_prefix['metadata']['gas_access_geometry']
    expected=geometry(source,g['outer_insulation_open_fraction'],end=835.,
        jet=g['jet_wetted_fraction'],edge=g['witness_edge_open_fraction'],
        iss3=g.get('iss3_accessible_fraction'),protected=g['protected_accessible_fraction'],
        lip=g['holder_lip_open_fraction'],pad=g.get('open_iss_pad_wetted_fraction',1.))
    if seed_prefix!=expected:
        raise ValueError('Seed geometry does not reconstruct from frozen physical source')
    return copy.deepcopy(seed_prefix)


def temperature_pruner(data):
    event=dataio.event(data)['time_s']
    def inspect(run):
        t=np.asarray(run['time_s']);obs=dataio.observed(data,t)
        for key,base_limit in [('101',100.),('102',100.),('107',100.),
                ('C2_TIM',300.),('108',1000.),('C2_ISS',1000.)]:
            if key not in CALIBRATION_CHANNELS:continue
            # Do not require peak-amplitude matching, but do not ignore it.
            limit=np.where((t>=event-2.)&(t<=event+12.),4.*base_limit,base_limit)
            bad=np.isfinite(obs[key])&(np.abs(np.asarray(run['temperature_k'][key])-obs[key])>limit)
            first=None
            for at,wrong in zip(t,bad):
                first=at if wrong and first is None else first if wrong else None
                if first is not None and at-first>=5.:
                    return f'{key}: sustained temperature divergence, {first:g}..{at:g} s'
        return None
    return inspect


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--outer-fractions',nargs='+',type=float,default=[1.,.5,.25])
    ap.add_argument('--jet-wetted-fraction',type=float,default=1.)
    ap.add_argument('--jet-wetted-fractions',nargs='+',type=float,
        help='Small explicit footprint grid; overrides the single footprint value')
    ap.add_argument('--open-iss-pad-fractions',nargs='+',type=float,default=[1.],
        help='Common pad-side footprint on ISS1/ISS2, retaining full facing-casing access')
    ap.add_argument('--witness-edge-fraction',type=float,default=.1)
    ap.add_argument('--iss3-access-fraction',type=float)
    ap.add_argument('--protected-fraction',type=float,default=.001)
    ap.add_argument('--lip-fraction',type=float,default=.001)
    ap.add_argument('--fresh-candidate',action='store_true',help='Verify one candidate from ambient with its own geometry, not the common r8 prefix')
    ap.add_argument('--selection-metric',choices=['shape_objective','shape_with_front_objective'],default='shape_objective')
    ap.add_argument('--seed',type=Path,help='Reuse this exact-core fresh r8 prefix; never migrate an old binary checkpoint')
    ap.add_argument('--reuse-seed-geometry',action='store_true',
        help='Use the explicitly verified seed geometry for the sealed prefix; trials remain conditional')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False)
    cal.PROBE=args.probe.resolve();contract=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    source=json.loads((args.source/'scenario.json').read_text())
    if 'gas_access_geometry'in source.get('metadata',{}):raise ValueError('Require explicit unmasked physical source')
    source['gas_network']['averaged_pressure_iterations']=2048
    cal.save(out/'physical-source.json',source)
    jets=args.jet_wetted_fractions or [args.jet_wetted_fraction]
    grid=list(itertools.product(args.outer_fractions,jets,args.open_iss_pad_fractions))
    # Validate the entire grid before spending GPU time.
    for fraction,jet,pad in grid:
        geometry(source,fraction,jet=jet,edge=args.witness_edge_fraction,
            iss3=args.iss3_access_fraction,protected=args.protected_fraction,lip=args.lip_fraction,pad=pad)
    if args.fresh_candidate and len(grid)!=1:
        raise ValueError('Fresh-candidate verification requires one explicit geometry')
    if args.reuse_seed_geometry and (not args.seed or args.fresh_candidate):
        raise ValueError('Explicit seed geometry requires --seed without --fresh-candidate')
    prefix=(geometry(source,args.outer_fractions[0],end=835.,jet=jets[0],
        edge=args.witness_edge_fraction,iss3=args.iss3_access_fraction,
        protected=args.protected_fraction,lip=args.lip_fraction,pad=args.open_iss_pad_fractions[0])
        if args.fresh_candidate else geometry(source,1.,end=835.))
    if args.reuse_seed_geometry:
        prefix=checked_seed_geometry(source,json.loads((args.seed/'prefix-scenario.json').read_text()))
    cal.save(out/'prefix-scenario.json',prefix)
    state=dict(status='running',phase='fresh_prefix',pid=os.getpid(),
        source=str(args.source.resolve()),native=str(cal.PROBE),native_sha256=contract,
        source_scenario_sha256=hashlib.sha256((args.source/'scenario.json').read_bytes()).hexdigest(),
        geometry_code_sha256=hashlib.sha256((Path(__file__).parent/'upv_gas_access.py').read_bytes()).hexdigest(),
        runner_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pressure_iterations_limit=2048,pressure_relative_acceptance=1e-6,
        minimum_regular_dt_s=prefix['simulation']['minimum_dt_s'],
        chemistry_frozen=True,materials_frozen=True,solid_contacts_frozen=True,
        accepted_calibration=False,conditional_geometry_search=True,trials=[])
    state['prefix_is_candidate_geometry']=args.fresh_candidate
    state['selection_metric']=args.selection_metric
    cal.save(out/'status.json',state)
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    data,_=dataio.load('no-cooling');prune=temperature_pruner(data);runs={}
    try:
        seed=args.seed.resolve()if args.seed else out
        if args.seed:
            previous=json.loads((seed/'status.json').read_text())
            if previous['native_sha256']!=contract:raise ValueError('Seed binary fingerprint mismatch')
            if json.loads((seed/'physical-source.json').read_text())!=source:
                raise ValueError('Seed physical source mismatch')
            if json.loads((seed/'prefix-scenario.json').read_text())!=prefix:
                raise ValueError('Seed prefix geometry or controls mismatch')
            with gzip.open(seed/'prefix/run.json.gz','rt')as stream:warm=json.load(stream)
            print('REUSE verified fresh prefix',seed,flush=True)
        else:
            print('FRESH PREFIX 0..835 s; new core, no old checkpoint',flush=True)
            warm=native(out/'prefix-scenario.json',out/'prefix',dict(restart_contract=contract,
                save_unvented_restart=str(out/'sealed-state.bin'),stop_after_first_rupture=False),1900.)
        if warm['advanced_to_s']!=835. or warm.get('first_rupture_s')is not None:
            raise ValueError('Expected sealed 835 s prefix')
        if warm.get('partial')or warm['started_at_s']!=0.:
            raise ValueError('Require complete fresh prefix from ambient')
        state['checkpoint_source']=str(seed)
        state['fresh_prefix_wall_s']=warm['wall_s']
        state['fresh_prefix_complete']=True
        mapping=[dict(to=i,fraction=1.,**{'from':i})for i in range(len(source['gas_network']['compartments']))]
        for i,(fraction,jet,pad) in enumerate(grid):
            folder=out/f'trial-{i:03d}';folder.mkdir()
            scene=geometry(source,fraction,jet=jet,edge=args.witness_edge_fraction,
                iss3=args.iss3_access_fraction,protected=args.protected_fraction,lip=args.lip_fraction,pad=pad)
            cal.save(folder/'scenario.json',scene)
            state.update(phase='conditional_suffix',active_trial=folder.name);cal.save(out/'status.json',state)
            print('SUFFIX',folder.name,'outer rim',fraction,'hot footprint',jet,'pad footprint',pad,flush=True)
            row=dict(trial=folder.name,outer_insulation_open_fraction=fraction,
                jet_wetted_fraction=jet,witness_edge_fraction=args.witness_edge_fraction,
                iss3_accessible_fraction=args.iss3_access_fraction,protected_fraction=args.protected_fraction,
                lip_fraction=args.lip_fraction,open_iss_pad_wetted_fraction=pad)
            try:
                run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=contract,
                    restart_from=str(seed/'sealed-state.bin'),restart_source_scenario=str(seed/'prefix-scenario.json'),
                    unvented_gas_remap=mapping,stop_after_first_rupture=False),700.,prune_fn=prune)
                report=shape_report(run,data);report['accepted_gas_heat_880_900_j']=received_gas_heat(run)
                cal.save(folder/'shape.json',report)
                row.update(status='complete',**report,realtime_ratio=115./run['wall_s'],
                    fresh_from_ambient=same_prefix_physics(prefix,scene))
                if row['fresh_from_ambient']:
                    row.update(full_trajectory_wall_s=warm['wall_s']+run['wall_s'],
                        full_trajectory_realtime_ratio=950./(warm['wall_s']+run['wall_s']))
                iss3=args.protected_fraction if args.iss3_access_fraction is None else args.iss3_access_fraction
                label=(f'Пятно {100*jet:g}%; ISS {100*pad:g}%; боковые {100*args.witness_edge_fraction:g}%; '
                    f'вход TIM {100*args.protected_fraction:g}%; вход ISS3 {100*iss3:g}%')
                runs[label]=run
                plot(out,runs,data,filename='shape-current.png')
            except (RuntimeError,TimeoutError)as exc:
                row.update(status='rejected',error=str(exc))
            state['trials'].append(row);cal.save(out/'status.json',state)
            print(json.dumps({k:row[k]for k in ['trial','status','shape_objective','error','realtime_ratio']if k in row}),flush=True)
        completed=[r for r in state['trials']if r['status']=='complete'and not r['no_heat_rejected']]
        if completed:state['selected_shape_candidate']=min(completed,key=lambda r:r[args.selection_metric])['trial']
        state.update(status='complete',active_trial=None)
    except KeyboardInterrupt:state.update(status='stopped')
    except BaseException as exc:state.update(status='failed',error=str(exc));raise
    finally:cal.save(out/'status.json',state)


if __name__=='__main__':main()

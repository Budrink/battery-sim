#!/usr/bin/env python3
"""Adaptive boundary geometry search using a verified sealed current-core state.

All TIM bays use the same fraction. Candidates are conditional continuations;
the selected geometry still requires a fresh full trajectory before acceptance.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as cal
from check_upv_restart import native, compare
from probe_upv_gas_access import plot
from refine_upv_gas_access import temperature_pruner
from upv_cooling_fit import rejection_reason
from upv_cooling_review import plot_cooling, phase_report
from upv_gas_response_refinement import front_prune_reason, CRITERIA_VERSION
from upv_gas_route_refinement import assessment
from upv_observation_policy import CALIBRATION_CHANNELS, metadata as observation_policy
from upv_tim_gas_access import candidate
from upv_gas_response_refinement import candidate as response_candidate
from upv_boundary_transport import candidate as boundary_candidate
import upv_r123_data as dataio


def candidate_rank(row):
    """Protect pressure/front, then improve the worst selected cooling trace.

    A better uncertain peak must not outrank a worse persistent cooling error.
    C2/ISS cannot participate even as a ranking tie-breaker.
    """
    pressure=row['preserved_front']['pressure_peak_relative_error']
    front_ok=row['arrival_accepted'] and pressure is not None and pressure<=.05
    worst=max(row['channels'][k]['rmse_k']/row['channels'][k]['tolerance_k']
              for k in CALIBRATION_CHANNELS)
    return (0 if front_ok else 1,worst,row['cooling_objective'],row['objective'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True,help='Fresh probe folder with sealed state and a complete or temperature-pruned trajectory')
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-trials',type=int,default=6)
    ap.add_argument('--max-slowdown',type=float,default=2.,help='Maximum wall/simulated ratio; user-approved 5 leaves all numerical guards intact')
    ap.add_argument('--reuse-neutral-from',type=Path,help='Reuse a checked suffix only after source/core/checkpoint hash agreement')
    ap.add_argument('--fractions',type=float,nargs='+',help='Explicit initial geometry bracket, each fraction in [1e-5,1]')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--damage-deltas',type=float,nargs='+',default=[0.],
                    help='Common reverse casing/gas damage-response increments, each in [-6,6]')
    ap.add_argument('--path-scales',type=float,nargs='+',default=[1.],help='Aperture counterexchange path relative to source, [0.5,10]')
    ap.add_argument('--jet-fractions',type=float,nargs='+',help='Common jet coverage of both open ISS bays, [0.1,1]')
    ap.add_argument('--entry-fractions',type=float,nargs='+',help='TIM under-fracture access, independent of rim exit, [1e-5,1]')
    ap.add_argument('--active-axes',nargs='+',choices=['fraction','damage_delta','path_scale','jet_fraction','entry_fraction'],
                    help='Coordinates eligible for local refinement; default: those varied by the initial bracket')
    args=ap.parse_args();source=args.source.resolve();out=args.output.resolve()
    if not 1.<=args.max_slowdown<=5.:raise ValueError('Max slowdown outside [1,5]')
    cal.PROBE=args.probe.resolve();core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    provenance=json.loads((source/'status.json').read_text())
    if provenance['native_sha256']!=core:raise ValueError('Source must use the current exact core')
    scene_path=source/'trial-000/scenario.json';scene=json.loads(scene_path.read_text())
    replay=source/'trial-000/replay';run_path=replay/'run.json.gz'
    complete_source=run_path.exists()
    if not complete_source:
        error=json.loads((source/'trial-000/result.json').read_text())['error']
        if not error.startswith('Pruned:'):raise ValueError('A numerical/speed failure cannot be used as a calibration reference')
        run_path=replay/'incomplete-run.json.gz'
    with gzip.open(run_path,'rt') as stream:baseline=json.load(stream)
    end=scene['simulation']['duration_s'];checkpoint=source/'sealed-state.bin'
    if baseline['started_at_s']!=0. or end<1100. or baseline['time_s'][-1]<925.:
        raise ValueError('Require a fresh trajectory covering the event and early relaxation')
    if complete_source and (baseline['advanced_to_s']!=end or baseline.get('partial')):
        raise ValueError('Completed source is inconsistent')
    if not checkpoint.exists() or baseline['unvented_restart']['time_s']!=835.:
        raise ValueError('Require the exact saved 835 s sealed state')
    for key,limit in [('max_thermal_residual_j',1e-4),('max_chemical_residual_j',1e-4),('max_mass_residual_kg',1e-10)]:
        if baseline[key]>limit:raise ValueError('Source conservation violation: '+key)
    out.mkdir(exist_ok=args.resume);data,record=dataio.load('no-cooling');event=dataio.event(data);strict=temperature_pruner(data)
    def prune(run):return strict(run) or rejection_reason(run,data,event) or front_prune_reason(run,data)
    base_fraction=scene['metadata']['gas_access_geometry']['protected_accessible_fraction']
    base_jet=scene['metadata']['gas_access_geometry']['jet_wetted_fraction']
    base_entry=scene['metadata']['gas_access_geometry'].get('protected_fracture_fraction',base_fraction)
    base_score=assessment(baseline,data,end)
    state=dict(status='running',phase='neutral_restart',pid=os.getpid(),source=str(source),native_sha256=core,
        criteria_version=CRITERIA_VERSION,observation_policy=observation_policy(),
        experimental_sha256=record['canonical_sha256'],source_scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),accepted_calibration=False,
        ranking_version='five-channel-minimax-cooling-v1',
        materials_frozen=True,chemistry_frozen=True,records=[],baseline_complete=complete_source,
        baseline_known_until_s=baseline['time_s'][-1],baseline_score=base_score,
        best=dict(fraction=base_fraction,source='fresh-reference',**base_score) if complete_source else None)
    state['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.resume:
        previous=json.loads((out/'status.json').read_text())
        if previous['status']=='running':raise ValueError('Stop the previous controller before resume')
        for key in ('native_sha256','source_scene_sha256','checkpoint_sha256','criteria_version','experimental_sha256'):
            if previous[key]!=state[key]:raise ValueError('Resume contract changed: '+key)
        previous.setdefault('resumes',[]).append(dict(previous_status=previous['status'],runner_sha256=state['runner_sha256']))
        previous.update(status='running',pid=os.getpid(),runner_sha256=state['runner_sha256'])
        state=previous
        recorded={r['trial'] for r in state['records']}
        for folder in sorted(out.glob('trial-*')):
            if folder.name in recorded:continue
            if (folder/'result.json').exists():
                state['records'].append(json.loads((folder/'result.json').read_text()));continue
            if (folder/'scenario.json').exists():
                fraction=json.loads((folder/'scenario.json').read_text())['metadata']['common_tim_gas_access']['fraction']
                params=json.loads((folder/'scenario.json').read_text())['metadata'].get('tim_cooling_parameters',{})
                row=dict(trial=folder.name,fraction=fraction,damage_delta=params.get('damage_delta',0.),
                    path_scale=params.get('path_scale',1.),jet_fraction=params.get('jet_fraction',base_jet),
                    entry_fraction=params.get('entry_fraction',fraction),status='interrupted',
                    error='Controller stopped before evaluation finished; partial preserved')
                state['records'].append(row);cal.save(folder/'result.json',row)
    fractions=args.fractions or [base_fraction*3.,base_fraction*10.,base_fraction/3.,base_fraction*30.]
    jets=args.jet_fractions or [base_jet]
    def proposal(fraction,damage,path_scale,jet,entry):
        proposed=candidate(scene,fraction,fracture_access_fraction=entry)
        if path_scale!=1. or jet!=base_jet:proposed=boundary_candidate(proposed,path_scale,jet,relative_to_current=True)
        if damage:proposed=response_candidate(proposed,damage_delta=damage,end=end)
        proposed['metadata']['tim_cooling_parameters']=dict(fraction=fraction,damage_delta=damage,path_scale=path_scale,jet_fraction=jet,entry_fraction=entry,
            common_for_all_cells=True,relative_to_source=True,materials_and_chemistry_unchanged=True)
        return proposed
    for fraction in fractions:
        for damage in args.damage_deltas:
            for path_scale in args.path_scales:
                for jet in jets:
                    for entry in args.entry_fractions or [base_entry*fraction/base_fraction]:proposal(fraction,damage,path_scale,jet,entry)
    state['initial_fractions']=fractions
    state['initial_damage_deltas']=args.damage_deltas
    state['initial_path_scales']=args.path_scales;state['initial_jet_fractions']=jets
    state['initial_entry_fractions']=args.entry_fractions
    names=('fraction','damage_delta','path_scale','jet_fraction','entry_fraction')
    groups=(fractions,args.damage_deltas,args.path_scales,jets,args.entry_fractions or fractions)
    active_axes={i for i,(name,values)in enumerate(zip(names,groups))
                 if name in args.active_axes} if args.active_axes else {i for i,values in enumerate(groups)if len(set(values))>1}
    state['active_axes']=[names[i]for i in sorted(active_axes)]
    eligible=[r for r in state['records']if r['status']=='complete' and r['preserved_front']['accepted']]
    if complete_source:eligible.append(dict(fraction=base_fraction,source='fresh-reference',**base_score))
    if eligible:state['best']=min(eligible,key=candidate_rank)
    cal.save(out/'status.json',state)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        control=out/'neutral';control_scene=json.loads(json.dumps(scene))
        control_end=925.;control_scene['simulation']['duration_s']=control_end
        control_scene['failure_sample_times_s']=[t for t in scene['failure_sample_times_s'] if t<=control_end]
        if control_scene['failure_sample_times_s'][-1]!=control_end:raise ValueError('Neutral end must already be an observation')
        if args.reuse_neutral_from:
            old=args.reuse_neutral_from.resolve();previous=json.loads((old/'status.json').read_text())
            for key in ('native_sha256','source_scene_sha256','checkpoint_sha256'):
                if previous[key]!=state[key]:raise ValueError('Borrowed neutral contract mismatch: '+key)
            checked=json.loads((old/'neutral-comparison.json').read_text())
            if not checked['bitwise_equal']or checked['compared_until_s']!=control_end:raise ValueError('Borrowed neutral is incomplete')
            cal.save(out/'neutral-comparison.json',checked)
            state['neutral_reused_from']=str(old);print('REUSE exact-core/source/checkpoint neutral suffix',flush=True)
        elif args.resume and (out/'neutral-comparison.json').exists():
            checked=json.loads((out/'neutral-comparison.json').read_text())
            if checked['compared_until_s']!=control_end:raise ValueError('Incomplete neutral restart test')
            print('REUSE exact neutral suffix 835..925 s',flush=True)
        else:
            control.mkdir();cal.save(control/'scenario.json',control_scene)
            print('START exact neutral suffix 835..925 s',flush=True)
            neutral=native(control/'scenario.json',control/'replay',dict(restart_contract=core,restart_from=str(checkpoint),
                stop_after_first_rupture=False),1.1*args.max_slowdown*(control_end-835.),prune_fn=prune,
                minimum_speed=1./args.max_slowdown,rolling_minimum=min(.25,1./args.max_slowdown))
            checked=compare(baseline,neutral,until_s=control_end);cal.save(out/'neutral-comparison.json',checked)
        if not checked['bitwise_equal']:raise ValueError('Restart changed physical trajectory')
        state['phase']='conditional_boundary_geometry';cal.save(out/'status.json',state)
        queue=[(f,d,p,j,r)for j in jets for p in args.path_scales for d in args.damage_deltas for f in fractions
               for r in args.entry_fractions or [base_entry*f/base_fraction]];local=[]
        def point_key(f,d,p,j,r):return (round(math.log(f),10),round(d,10),round(math.log(p),10),round(math.log(j),10),round(math.log(r),10))
        seen={point_key(base_fraction,0.,1.,base_jet,base_entry)};best_run=baseline
        for row in state['records']:
            if row['status'] in ('complete','rejected'):
                seen.add(point_key(row['fraction'],row.get('damage_delta',0.),row.get('path_scale',1.),row.get('jet_fraction',base_jet),row.get('entry_fraction',row['fraction'])))
        if state['best'] and state['best'].get('trial'):
            with gzip.open(out/state['best']['trial']/'replay/run.json.gz','rt') as stream:best_run=json.load(stream)
        first=max([int(p.name.split('-')[1])for p in out.glob('trial-*')if p.is_dir()]+[-1])+1
        for number in range(first,args.max_trials):
            fraction=None
            while queue or local:
                items=local if local and (number-first)%3!=2 else queue
                if not items:items=local if local else queue
                value,damage,path_scale,jet,entry=items.pop(0)
                value=min(1.,max(1e-5,value));damage=min(6.,max(-6.,damage))
                path_scale=min(10.,max(.5,path_scale));jet=min(1.,max(.1,jet))
                entry=min(1.,max(1e-5,entry));key=point_key(value,damage,path_scale,jet,entry)
                if key not in seen:seen.add(key);fraction=value;break
            if fraction is None:break
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise ValueError('Core changed during search')
            trial=out/f'trial-{number:03d}';trial.mkdir();proposed=proposal(fraction,damage,path_scale,jet,entry)
            cal.save(trial/'scenario.json',proposed);state['active_trial']=trial.name;cal.save(out/'status.json',state)
            print('START',trial.name,'TIM exit/entry',fraction,entry,'reverse damage',damage,'path scale',path_scale,'jet fraction',jet,flush=True)
            row=dict(trial=trial.name,fraction=fraction,damage_delta=damage,path_scale=path_scale,jet_fraction=jet,entry_fraction=entry,
                conditional_restart=True,requires_fresh_verification=True)
            try:
                os.environ['BATTERY_MIXING_FAILURE_SNAPSHOT']=str(trial/'mixing-failure.json')
                run=native(trial/'scenario.json',trial/'replay',dict(restart_contract=core,
                    restart_from=str(checkpoint),restart_source_scenario=str(scene_path),
                    unvented_gas_remap=[dict(to=i,fraction=1.,**{'from':i}) for i in range(len(scene['gas_network']['compartments']))],
                    stop_after_first_rupture=False),1.1*args.max_slowdown*(end-835.),prune_fn=prune,
                    minimum_speed=1./args.max_slowdown,rolling_minimum=min(.25,1./args.max_slowdown))
                result=assessment(run,data,end);cal.save(trial/'assessment.json',result)
                cal.save(trial/'phases.json',phase_report(run,data,end))
                row.update(status='complete',**result,realtime_ratio=(end-835.)/run['wall_s'])
                if result['preserved_front']['accepted'] and (state['best'] is None or candidate_rank(row)<candidate_rank(state['best'])):
                    old=(state['best']['fraction'],state['best'].get('damage_delta',0.),state['best'].get('path_scale',1.),
                         state['best'].get('jet_fraction',base_jet),state['best'].get('entry_fraction',state['best']['fraction'])) if state['best'] else (base_fraction,0.,1.,base_jet,base_entry)
                    state['best']=dict(row);best_run=run
                    # Continue the improving direction, then bracket it locally.
                    point=(fraction,damage,path_scale,jet,entry)
                    for axis,(before,after) in enumerate(zip(old,point)):
                        if before==after or axis not in active_axes:continue
                        values=(after+(2. if after>before else -2.),.5*(after+before)) if axis==1 else (
                            after*math.sqrt(after/before),math.sqrt(after*before))
                        for value in reversed(values):
                            next_point=list(point);next_point[axis]=value;local.insert(0,tuple(next_point))
                print('RESULT',trial.name,'objective',row['objective'],'speed',row['realtime_ratio'],flush=True)
            except (RuntimeError,TimeoutError) as exc:
                row.update(status='rejected',error=str(exc));print('REJECTED',trial.name,str(exc),flush=True)
            state['records'].append(row);cal.save(trial/'result.json',row);cal.save(out/'status.json',state)
            runs={'Свежий контроль'+(' (до остановки)' if not complete_source else ''):baseline}
            if best_run is not baseline:runs['Лучший условный вариант']=best_run
            plot_cooling(out,runs,data,{k:k for k in runs})
            plot(out,runs,data,filename='burst-comparison.png',title='Геометрия прорыва и струйного обогрева; C2/ISS справочно')
            if state['best'] and state['best'].get('accepted'):break
        state.update(status='conditional_search_complete',phase='requires_fresh_verification')
    except KeyboardInterrupt:state['status']='stopped'
    except BaseException as exc:state.update(status='failed',error=str(exc));raise
    finally:
        state.pop('active_trial',None);cal.save(out/'status.json',state)


if __name__=='__main__':main()

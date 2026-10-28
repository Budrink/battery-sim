#!/usr/bin/env python3
"""Audit a completed fresh 20/10 mm full-horizon pair; never fit physics.

The existing spatial driver owns the GPU jobs. This read-only monitor waits
for it, verifies provenance/pilot-prefix continuity and reports phase errors.
"""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from calibrate_fsri30_bayes import atomic_save
from check_fsri30_resolved_gas import compare_mesh,compact
from fsri30_data import sha


def load_run(folder):
    # Discard detailed per-channel histories before loading the other mesh.
    with gzip.open(folder/'replay/run.json.gz','rt') as f:return compact(json.load(f))


def physical_signature(scene):
    value=deepcopy(scene)
    for key in ('metadata','name','failure_sample_times_s'):value.pop(key,None)
    value['simulation'].pop('duration_s',None)
    return value


def verify_provenance(root,launch,contract):
    probe=Path(launch['command'][launch['command'].index('--probe')+1])
    if sha(probe)!=launch['native_sha256'] or sha(Path(contract['base_scenario']))!=launch['base_sha256']:
        raise ValueError('Pinned native binary or physical base changed')
    for name,digest in launch['scripts_sha256'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Pinned mesh code changed: '+name)
    if sha(Path(launch['pilot'])/'status.json')!=launch['pilot_status_sha256']:
        raise ValueError('Pilot evidence changed')


def aligned_times(a,b):
    ta,tb=np.asarray(a['time_s']),np.asarray(b['time_s'])
    if ta.shape!=tb.shape or not np.allclose(ta,tb,rtol=0.,atol=1e-7):
        raise ValueError('Observation times do not align')
    return ta


def prefix_check(pilot,full):
    tp=np.asarray(pilot['time_s']);tf=np.asarray(full['time_s']);n=len(tp)
    if len(tf)<n or not np.allclose(tp,tf[:n],rtol=0.,atol=1e-7):
        raise ValueError('Full replay does not contain the exact pilot observation schedule')
    differences={}
    def visit(a,b,path):
        if isinstance(a,dict):
            if a.keys()!=b.keys():raise ValueError('Pilot-prefix field keys changed: '+path)
            for key in a:visit(a[key],b[key],path+'/'+key)
        else:
            x,y=np.asarray(a,dtype=float),np.asarray(b,dtype=float)[:n]
            if x.shape!=y.shape:raise ValueError('Pilot-prefix shape changed: '+path)
            differences[path]=float(np.max(np.abs(x-y)/np.maximum(1.,np.maximum(np.abs(x),np.abs(y)))))
    for section in ('cell_mean_temperature_k','process_totals','gas_groups'):
        visit(pilot[section],full[section],section)
    worst=max(differences.values(),default=0.)
    return dict(accepted=worst<=1e-10,tolerance=1e-10,maximum_scaled_difference=worst,
        compared_fields=len(differences),until_s=float(tp[-1]),
        changed_fields={k:v for k,v in differences.items() if v>1e-10})


def phase_comparison(a,b,start,end,tolerance=.05):
    t=aligned_times(a,b);mask=(t>=start-1e-7)&(t<=end+1e-7)
    if mask.sum()<2:raise ValueError('Phase has fewer than two aligned observations')
    keys=sorted(a['cell_mean_temperature_k'])
    if keys!=sorted(b['cell_mean_temperature_k']):raise ValueError('Compared objects changed')
    va=np.array([a['cell_mean_temperature_k'][k] for k in keys])[:,mask]
    vb=np.array([b['cell_mean_temperature_k'][k] for k in keys])[:,mask]
    component=float(np.max(np.sqrt(np.mean(((va-vb)/np.maximum(va,vb))**2,axis=1))))
    ga=np.asarray(a['gas_groups']['outside']['maximum_temperature_k'])[mask]
    gb=np.asarray(b['gas_groups']['outside']['maximum_temperature_k'])[mask]
    gas=float(np.sqrt(np.mean(((ga-gb)/np.maximum(ga,gb))**2)))
    qa=np.asarray(a['process_totals']['gas_released_energy_j'])[mask]
    qb=np.asarray(b['process_totals']['gas_released_energy_j'])[mask]
    da,db=float(qa[-1]-qa[0]),float(qb[-1]-qb[0])
    energy=abs(da-db)/max(100.,abs(da),abs(db))
    shape=float(np.sqrt(np.mean(((qa-qa[0])-(qb-qb[0]))**2))/max(100.,abs(da),abs(db)))
    return dict(accepted=component<=tolerance and gas<=tolerance and energy<=tolerance and shape<=tolerance,
        start_s=float(t[mask][0]),end_s=float(t[mask][-1]),tolerance=tolerance,
        maximum_component_rms_relative_kelvin=component,hottest_gas_rms_relative_kelvin=gas,
        r3_increment_relative_difference=energy,r3_increment_shape_relative_difference=shape,
        r3_increment_j=[da,db])


def report_available_prefix(root,out):
    from fsri30_prefix_history import load_available,trim_observed,summary_at,plot_prefix
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    root=Path(root);launch=json.loads((root/'launch.json').read_text())
    contract=json.loads((root/'checks/contract.json').read_text());verify_provenance(root,launch,contract)
    runs={};events={};sources={};signatures={}
    for mm in (20,10):
        runs[mm],events[mm],sources[mm],scene=load_available(root/f'checks/h-{mm}mm')
        signatures[mm]=physical_signature(scene)
        pilot=json.loads((Path(launch['pilot'])/f'uniform-{mm}/h-{mm}mm/scenario.json').read_text())
        if physical_signature(pilot)!=signatures[mm]:raise ValueError('Full scene changed pilot physics')
    end=min(r['time_s'][-1] for r in runs.values())
    runs={mm:trim_observed(r,end) for mm,r in runs.items()}
    aligned_times(runs[20],runs[10])
    summaries={mm:summary_at(runs[mm],events[mm]) for mm in (20,10)}
    first=min(e['time_s'] for e in summaries[20]['model_conversion_events']+summaries[10]['model_conversion_events'])
    next_events=[e for mm in (20,10) for e in summaries[mm]['model_conversion_events'] if e['component']!='B10']
    second=min((e['time_s'] for e in next_events),default=end)
    begin=max(0.,5.*np.floor((first-30.)/5.));finish=min(end,5.*np.ceil((first+120.)/5.))
    onset=max(finish,5.*np.floor((second-30.)/5.))
    windows={}
    for name,start,stop in (('preheat',0.,begin),('first_burst',begin,finish),
                            ('between_bursts',finish,onset),('cascade_onset',onset,end),('through_before_cascade',0.,onset)):
        if stop-start>=5.:windows[name]=phase_comparison(runs[20],runs[10],start,stop)
    by10={e['component']:e['time_s'] for e in summaries[10]['model_conversion_events']}
    right_censored=[dict(component=e['component'],coarse_event_s=e['time_s'],fine_not_crossed_by_s=float(end),
                        fine_delay_lower_bound_s=float(end-e['time_s']))
        for e in summaries[20]['model_conversion_events'] if e['component'] not in by10]
    fine_live=sources[10].get('live_state') or {}
    result=dict(status='common-observed-prefix-compared',common_horizon_s=float(end),
        full_requested_horizon_s=5945.,full_horizon_spatial_check_passed=False,calibrated=False,validation_passed=False,
        native_sha256=launch['native_sha256'],base_sha256=launch['base_sha256'],sources=sources,
        global_comparison=compare_mesh(runs[20],runs[10],summaries[20],summaries[10]),phase_comparisons=windows,
        conversion_events={mm:summaries[mm]['model_conversion_events'] for mm in (20,10)},
        conversion_definition='10% R2 converted mass, linear interpolation between completed observations',
        right_censored_fine_events=right_censored,
        fine_live_state_beyond_observations=dict(time_s=fine_live.get('time_s'),
            B09=fine_live.get('objects',{}).get('B09'),used_as_observation=False),
        note='No extrapolation, no fitted time shift. A missing event at the cutoff is censored, not proof of a different cascade order.')
    plot=plot_prefix(out,runs,end,first,second);result['plot']=str(plot.resolve())
    atomic_save(out/'report.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True,type=Path,help='Root containing launch.json and checks/status.json')
    ap.add_argument('--available-prefix',action='store_true',help='Compare completed snapshots even if a full replay was stopped')
    ap.add_argument('--output',type=Path,help='New output directory for --available-prefix')
    args=ap.parse_args();root=args.run.resolve();status_path=root/'verification.json'
    if args.available_prefix:
        result=report_available_prefix(root,args.output or root/'available-prefix')
        print(json.dumps(result,indent=2),flush=True);return
    if args.output is not None:raise ValueError('--output is only supported with --available-prefix')
    if status_path.exists():raise ValueError('Verification output already exists')
    launch=json.loads((root/'launch.json').read_text());pilot=Path(launch['pilot'])
    state=dict(status='waiting-for-full-pair',pid=os.getpid(),native_sha256=launch['native_sha256'],
        duration_s=5945.,recommended_mesh_mm=None,calibrated=False,validation_passed=False,
        scope='Numerical spatial check only; sensor mapping and experimental agreement remain unresolved',
        controller_sha256=sha(Path(__file__)))
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);atomic_save(status_path,state)
    try:
        contract=json.loads((root/'checks/contract.json').read_text())
        if contract['native_sha256']!=launch['native_sha256'] or contract['base_sha256']!=launch['base_sha256']:
            raise ValueError('Full replay uses a different core or physical profile')
        verify_provenance(root,launch,contract)
        while True:
            full=json.loads((root/'checks/status.json').read_text())
            state['active_phase']=full.get('phase');atomic_save(status_path,state)
            if full['status']!='running':break
            os.kill(full['pid'],0);time.sleep(10.)
        if full['status'] not in ('complete','mesh-limit-not-converged'):
            try:
                report_available_prefix(root,root/'available-prefix-auto')
                state['available_prefix_report']=str(root/'available-prefix-auto/report.json')
            except Exception as exc:state['available_prefix_error']=str(exc)
            raise ValueError('Full numerical replay did not finish: '+full['status']+' '+full.get('error',''))
        before=json.loads((pilot/'status.json').read_text())
        verify_provenance(root,launch,contract)
        runs={};summaries={};prefix={}
        for mm in (20,10):
            label=f'h-{mm}mm';folder=root/'checks'/label
            result=full['results'][label]
            if result['status']!='complete' or not result['fresh_from_zero'] or result['advanced_to_s']<5945.-1e-7:
                raise ValueError('A full fresh trajectory is missing')
            runs[mm]=load_run(folder);summaries[mm]=result
            pilot_folder=Path(before['results'][f'uniform-{mm}']['folder'])
            pilot_scene=json.loads((pilot_folder/'scenario.json').read_text())
            full_scene=json.loads((folder/'scenario.json').read_text())
            if physical_signature(pilot_scene)!=physical_signature(full_scene):
                raise ValueError(f'{mm}mm full replay changed pilot physics or numerical controls')
            prefix[mm]=prefix_check(load_run(pilot_folder),runs[mm])
        state['pilot_prefix_continuity']=prefix
        global_check=compare_mesh(runs[20],runs[10],summaries[20],summaries[10]);state['global_comparison']=global_check
        windows={};events=summaries[20]['model_conversion_events']+summaries[10]['model_conversion_events']
        first=min((e['time_s'] for e in events),default=1800.)
        for name,start,end in (('pre_first_r2',0.,max(5.,first-30.)),
                               ('first_r2_impulse',max(0.,first-30.),min(5945.,first+120.)),
                               ('late_evolution',min(5940.,first+120.),5945.)):
            windows[name]=phase_comparison(runs[20],runs[10],start,end)
        # Fixed physical windows prevent a long cold tail from hiding a local
        # mismatch. They do not shift either trace or fit to thermocouple peaks.
        for start in np.arange(0.,5945.,300.):
            windows[f'window-{start:g}']=phase_comparison(runs[20],runs[10],start,min(5945.,start+300.))
        state['phase_comparisons']=windows
        accepted=global_check['accepted'] and all(x['accepted'] for x in windows.values()) and all(x['accepted'] for x in prefix.values())
        state.update(status='numerical-check-passed' if accepted else 'numerical-check-not-passed',
            full_horizon_spatial_check_passed=accepted,
            recommended_mesh_mm=20 if accepted and summaries[20]['realtime_ratio']>=.2 else None,
            realtime_ratios={mm:summaries[mm]['realtime_ratio'] for mm in (20,10)},
            conversion_events={mm:summaries[mm]['model_conversion_events'] for mm in (20,10)},
            conversion_definition='10% of cell mass reacted via R2, NOT a measured destruction flag',
            observed_cascade_count=None,
            report_note='Numerical agreement is not proof of a realistic cascade or experimental validation')
    except KeyboardInterrupt:state['status']='monitor-stopped-gpu-job-unaffected'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        atomic_save(status_path,state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

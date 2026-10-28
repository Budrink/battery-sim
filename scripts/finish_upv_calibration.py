#!/usr/bin/env python3
"""Bounded full-event training trials; never accept a prefix or fit validation."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as cal
from check_upv_restart import native
from refine_upv_gas_access import temperature_pruner
from upv_cooling_fit import rejection_reason
from upv_gas_response_refinement import front_prune_reason
from upv_gas_route_refinement import assessment
from upv_cooling_review import phase_report,plot_cooling
from upv_final_fit import candidate,DEFAULT,only_gas_contacts_changed
from upv_observation_policy import CALIBRATION_CHANNELS,metadata as observation_policy
import upv_r123_data as dataio


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--points',type=Path,required=True,help='Explicit bounded list of parameter dictionaries')
    ap.add_argument('--overlay-base',type=Path,help='Actual parameter pivot scene; conditional use requires exact equality of ALL physics except gas films with checkpoint scene')
    ap.add_argument('--fresh',action='store_true',help='Every trial starts from ambient, even gas-only changes')
    ap.add_argument('--zones',action='store_true',help='Record the existing native zone temperatures/inventories for physical diagnostics')
    ap.add_argument('--complete-window',action='store_true',help='Trajectory review: disable observation-error pruning; retain strict numerical, UI, conservation and speed checks. Old acceptance gates are still reported unchanged.')
    a=ap.parse_args();source=a.reference.resolve();out=a.output.resolve()
    prior=json.loads((source/'status.json').read_text());original=json.loads((source/'trial-000/scenario.json').read_text())
    cal.PROBE=a.probe.resolve();core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    if core!=prior['native_sha256'] and not a.fresh:
        raise ValueError('A core transition requires --fresh; checkpoints cannot cross native revisions')
    pivot=json.loads(a.overlay_base.read_text())if a.overlay_base else original
    points=json.loads(a.points.read_text());proposals=[candidate(pivot,p)for p in points]
    if a.zones:
        for scene,p in proposals:scene['metadata']['zone_diagnostics']=True
    if not a.fresh and a.overlay_base and any(not only_gas_contacts_changed(original,s)for s,p in proposals):
        raise ValueError('Overlay pivot changed physics beyond gas films; require --fresh')
    if not a.fresh and not a.overlay_base and any(s['metadata']['final_calibration_trial']['fresh_required']for s,p in proposals):
        raise ValueError('Solid/electrical/kinetic changes require --fresh')
    out.mkdir();end=original['simulation']['duration_s'];data,record=dataio.load('no-cooling');event=dataio.event(data)
    state=dict(status='running',pid=os.getpid(),native_sha256=core,reference=str(source),
        observation_policy=observation_policy(),current_input='measured',validation_used=False,
        max_slowdown=5.,records=[],best=None,accepted_calibration=False,points=points)
    state['observation_error_pruning']=not a.complete_window
    state['purpose']='complete trajectory review; strict acceptance reported separately'if a.complete_window else'bounded parameter calibration'
    if core!=prior['native_sha256']:
        state.update(previous_core_sha256=prior['native_sha256'],fresh_core_transition=True)
    if a.overlay_base:state.update(overlay_base=str(a.overlay_base.resolve()),overlay_base_sha256=hashlib.sha256(a.overlay_base.read_bytes()).hexdigest(),gas_contact_only_pivot_verified=not a.fresh)
    cal.save(out/'status.json',state);strict=temperature_pruner(data)
    def prune(run):return strict(run)or rejection_reason(run,data,event)or front_prune_reason(run,data)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        for i,(scene,p)in enumerate(proposals):
            trial=out/f'trial-{i:03d}';trial.mkdir();cal.save(trial/'scenario.json',scene)
            state['active_trial']=trial.name;cal.save(out/'status.json',state)
            request=dict(restart_contract=core,stop_after_first_rupture=False)
            if a.fresh:request.update(save_unvented_restart=str(trial/'sealed-state.bin'),unvented_restart_until_s=835.)
            else:
                request.update(restart_from=str(source/'sealed-state.bin'))
                response=p['damaged_span']!=1. or p['damaged_logit']!=0.
                if response:
                    if p['entry']!=original['metadata']['gas_access_geometry'].get('protected_fracture_fraction',.001)or p['jet_fraction']!=original['metadata']['gas_access_geometry']['jet_wetted_fraction']or p['witness_mouth']!=original['metadata']['gas_access_geometry']['iss3_accessible_fraction']or p['pad_footprint']!=original['metadata']['gas_access_geometry'].get('open_iss_pad_wetted_fraction',1.)or p['backing_fraction']!=original['metadata']['gas_access_geometry']['backing_open_fraction']or any(p[k]!=DEFAULT[k]for k in ('path','opening_rate','strength','wall_hot_scale','reverse_hot_logit','reverse_profile_power')):
                        raise ValueError('Mixed geometry and gas-response change requires a fresh trial')
                    request['restart_gas_response_scenario']=str(source/'trial-000/scenario.json')
                else:request.update(restart_source_scenario=str(source/'trial-000/scenario.json'),
                    unvented_gas_remap=[dict(to=j,fraction=1.,**{'from':j})for j in range(len(scene['gas_network']['compartments']))])
            row=dict(trial=trial.name,parameters=p,fresh=a.fresh,accepted=False)
            print('START',trial.name,'fresh'if a.fresh else'conditional suffix',json.dumps(p),flush=True)
            try:
                if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise RuntimeError('Core changed')
                run=native(trial/'scenario.json',trial/'replay',request,5.5*(end if a.fresh else end-835.),
                    prune_fn=None if a.complete_window else prune,minimum_speed=.2,rolling_minimum=.2)
                result=assessment(run,data,end);phases=phase_report(run,data,end)
                cal.save(trial/'assessment.json',result);cal.save(trial/'phases.json',phases)
                row.update(status='complete',**result,realtime_ratio=(run['advanced_to_s']-run['started_at_s'])/run['wall_s'])
                row['accepted']=bool(result['accepted']and a.fresh)
                row['warmup_rmse_k']={k:phases['phases'].get('pre_burst',{}).get('channels',{}).get(k,{}).get('time_weighted_rmse_k')for k in CALIBRATION_CHANNELS}if a.fresh else None
                if a.fresh:row['accepted']=bool(row['accepted']and all(v is not None and v<=5. for v in row['warmup_rmse_k'].values()))
                row['rank']=[0 if result['preserved_front']['accepted']and result['arrival_accepted']and result['preserved_front']['pressure_peak_relative_error']<=.05 else 1,
                    max(result['channels'][k]['rmse_k']/result['channels'][k]['tolerance_k']for k in CALIBRATION_CHANNELS),result['objective']]
                if a.fresh:
                    row['rank'][1]=max(row['rank'][1],max(v/5. for v in row['warmup_rmse_k'].values()))
                    row['rank'][2]+=sum((v/5.)**2 for v in row['warmup_rmse_k'].values())
                if state['best']is None or row['rank']<state['best']['rank']:
                    state['best']=row.copy();cal.save(out/'best-scenario.json',scene)
                    plot_cooling(out,{'best':run},data,{'best':'Лучший текущий кандидат; без сдвига времени'})
                print('RESULT',trial.name,'objective',row['objective'],'RMSE',json.dumps({k:round(row['channels'][k]['rmse_k'],2)for k in CALIBRATION_CHANNELS}),flush=True)
            except(RuntimeError,TimeoutError)as exc:
                row.update(status='rejected',error=str(exc));print('REJECTED',trial.name,str(exc),flush=True)
            state['records'].append(row);cal.save(trial/'result.json',row);cal.save(out/'status.json',state)
        state['status']='complete_trials_not_validated';state['accepted_calibration']=bool(state['best']and state['best']['accepted'])
    except KeyboardInterrupt:state['status']='stopped'
    finally:state.pop('active_trial',None);cal.save(out/'status.json',state)


if __name__=='__main__':main()

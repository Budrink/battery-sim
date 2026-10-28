#!/usr/bin/env python3
"""Check corrected phase conduction on training, then a full fixed holdout.

No held-out temperature/pressure selection, chemical refit, heat clamp, current
change or UI waiver. Hot electrical continuation is an explicit tested prior.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as cal
from check_upv_restart import native
from report_upv_frozen_comparison import plot
from upv_cooling_boundary import candidate as cooling_candidate
from upv_cooling_review import phase_report
from upv_frozen_validation import build, digest
from upv_gas_route_refinement import assessment
import upv_r123_data as dataio


def thermal_continuation(source, plateau_k):
    if not 375. <= plateau_k <= 450.:
        raise ValueError('Electrical plateau prior must be within375..450K')
    scene=copy.deepcopy(source)
    for cell in scene['cells']:
        if 'electrical'not in cell:continue
        law=cell['electrical'];a=law.pop('temperature_log_slope_per_k')
        ref=law['reference_temperature_k']
        law['temperature_log_response']=[[273.15,a*(273.15-ref)],[ref,0.],
                                        [plateau_k,a*(plateau_k-ref)],[2000.,a*(plateau_k-ref)]]
    scene['metadata']['electrical_closure_review']=dict(
        phase_conduction='Healthy polarized phase and nonpolarized products conduct in parallel',
        thermal_continuation='Existing cold log slope, explicit constant hot tail',
        plateau_k=plateau_k, high_temperature_material_law_measured=False,
        reference_scene_sha256=digest(source), chemistry_refitted=False,
        thermal_materials_contacts_or_gas_paths_changed=False,
        ui_guard='Strict unchanged; no heat/current clipping or diagnostic bypass',
        heldout_temperatures_used_for_selection=False)
    return scene


def read_run(folder):
    for name in ('run.json.gz','incomplete-run.json.gz'):
        path=folder/name
        if path.exists():
            with gzip.open(path,'rt')as stream:return json.load(stream)
    return None


def save_review(folder, state, experiment):
    run=read_run(folder/'replay')
    if run is None:return
    scene=json.loads((folder/'scenario.json').read_text());data,_=dataio.load(experiment)
    end=scene['simulation']['duration_s'];label=('Калибровочный опыт'if experiment=='no-cooling'else
        'Проверка с охлаждением по110–112; химия заморожена')
    if run.get('partial'):label+=' — НЕПОЛНЫЙ ПРОГОН'
    state.update(known_until_s=run['time_s'][-1],advanced_to_s=run['advanced_to_s'],wall_s=run['wall_s'],
        realtime_ratio=(run['advanced_to_s']-run['started_at_s'])/run['wall_s'],
        balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k})
    cal.save(folder/'phases.json',phase_report(run,data,end))
    for mode in ('overview','event','cooling'):
        plot(run,data,end,label,folder/(mode+'.png'),mode)
    if run.get('partial'):plot(run,data,run['time_s'][-1],label,folder/'known-prefix.png','overview')
    if not run.get('partial'):
        result=assessment(run,data,end);cal.save(folder/'assessment.json',result)
        state['strict_previous_shape_gates_passed']=bool(result['accepted'])
    cal.save(folder/'status.json',state)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--cooling-parameters',type=Path,required=True)
    ap.add_argument('--plateau-k',type=float,default=425.)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir()
    source=json.loads(args.source.read_text());source_hash=digest(source)
    cal.PROBE=args.probe.resolve();core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    training=thermal_continuation(source,args.plateau_k)
    cooled,data=build(training)
    cooling=json.loads(args.cooling_parameters.read_text())['parameters']
    cooled=cooling_candidate(cooled,data,cooling)
    state=dict(status='running',pid=os.getpid(),native_sha256=core,source_scene_sha256=source_hash,
        plateau_k=args.plateau_k,chemistry_refitted=False,ui_guard_strict=True,
        current_changed=False,heldout_temperature_pressure_selection=False,
        boundary_conditioned=True,fully_blind_validation=False,experiments={})
    cal.save(out/'status.json',state)
    for name,scene in [('no-cooling',training),('active-cooling',cooled)]:
        folder=out/name;folder.mkdir();cal.save(folder/'scenario.json',scene)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    active=None
    try:
        for name in ('no-cooling','active-cooling'):
            active=name;folder=out/name;scene=json.loads((folder/'scenario.json').read_text())
            row=dict(status='running',native_sha256=core,chemistry_refitted=False,
                current_changed=False,refitted=name=='active-cooling',accepted_validation=False,
                ui_guard_strict=True,fully_blind_validation=False)
            state['experiments'][name]=row;state['phase']=name;cal.save(out/'status.json',state)
            print('START',name,'strict UI, same frozen chemistry',flush=True)
            try:
                native(folder/'scenario.json',folder/'replay',dict(restart_contract=core,
                    stop_after_first_rupture=False,save_unvented_restart=str(folder/'sealed-state.bin')),
                    5.5*scene['simulation']['duration_s'],minimum_speed=.2,rolling_minimum=.2)
                row['status']='complete'
            except (RuntimeError,TimeoutError)as exc:
                row.update(status='rejected',error=str(exc));print('REJECTED',name,str(exc),flush=True)
            finally:
                save_review(folder,row,name);cal.save(out/'status.json',state)
            print('RESULT',name,row['status'],row.get('known_until_s'),flush=True)
            if row['status']!='complete':break
        state['status']='complete'if len(state['experiments'])==2 and all(r['status']=='complete'for r in state['experiments'].values())else'rejected'
    except KeyboardInterrupt:
        state['status']='stopped'
        if active:
            state['experiments'][active]['status']='stopped';save_review(out/active,state['experiments'][active],active)
    finally:
        if digest(json.loads(args.source.read_text()))!=source_hash:raise RuntimeError('Original source changed')
        cal.save(out/'status.json',state);print('FINAL',state['status'],flush=True)


if __name__=='__main__':main()

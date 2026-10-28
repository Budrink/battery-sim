#!/usr/bin/env python3
"""Full reference and restart-verified gas geometry on the current bounded solver.

Recalibration after a numerical fix must use a fresh current-core reference.
The captured sealed state is conditional for geometry trials, not a substitute
for final fresh-from-ambient verification of a selected geometry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal

import calibrate_upv_r123 as cal
from check_upv_restart import native, compare
from probe_upv_gas_access import plot
from refine_upv_gas_access import temperature_pruner
from upv_cooling_fit import rejection_reason
from upv_cooling_review import phase_report, plot_cooling
from upv_gas_access import received_gas_heat
from upv_gas_response_refinement import front_prune_reason, CRITERIA_VERSION
from upv_gas_route_refinement import assessment
from upv_tim_gas_access import candidate as tim_candidate
import upv_r123_data as dataio
from upv_observation_policy import metadata as observation_policy


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--single-full',action='store_true',
        help='Verify only the supplied geometry from ambient; no restart or further candidates')
    args=ap.parse_args();out=args.output.resolve();source=args.source.resolve()
    scene=json.loads(source.read_text());end=scene['simulation']['duration_s']
    if end<1100.:raise ValueError('Cooling tail required')
    if 835. not in scene['failure_sample_times_s']:raise ValueError('835 s must already be an observation')
    cal.PROBE=args.probe.resolve();core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    data,record=dataio.load('no-cooling');event=dataio.event(data);strict=temperature_pruner(data)
    def prune(run):return strict(run) or rejection_reason(run,data,event) or front_prune_reason(run,data)
    out.mkdir();checkpoint=out/'sealed-state.bin'
    state=dict(status='running',phase='fresh_current_core_reference',pid=os.getpid(),
        native_sha256=core,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        experimental_sha256=record['canonical_sha256'],criteria_version=CRITERIA_VERSION,
        minimum_regular_dt_s=scene['simulation']['minimum_dt_s'],end_s=end,
        chemistry_frozen=True,material_properties_frozen=True,accepted_calibration=False,records=[],
        observation_policy=observation_policy())
    state['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cal.save(out/'status.json',state)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    runs={}
    def execute(index,proposed,label,restart=False):
        folder=out/f'trial-{index:03d}';folder.mkdir();cal.save(folder/'scenario.json',proposed)
        state['active_trial']=folder.name;cal.save(out/'status.json',state)
        request=dict(restart_contract=core,stop_after_first_rupture=False)
        if restart:
            request.update(restart_from=str(checkpoint),restart_source_scenario=str(out/'trial-000/scenario.json'),
                unvented_gas_remap=[dict(to=i,fraction=1.,**{'from':i}) for i in range(len(scene['gas_network']['compartments']))])
        else:request.update(save_unvented_restart=str(checkpoint),unvented_restart_until_s=835.)
        print('START',folder.name,label,flush=True)
        row=dict(trial=folder.name,label=label,conditional_restart=restart)
        try:
            if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise ValueError('Native binary changed')
            os.environ['BATTERY_GAS_FAILURE_SNAPSHOT']=str(folder/'failure-snapshot.json')
            run=native(folder/'scenario.json',folder/'replay',request,2.2*(end-835.) if restart else 2.2*end,prune_fn=prune)
            if not restart and run['unvented_restart']['time_s']!=835.:raise ValueError('Wrong checkpoint time')
            if run['solver_profile']['minimum_regular_step_s']<scene['simulation']['minimum_dt_s']-1e-10:raise ValueError('Regular dt floor')
            result=assessment(run,data,end);cal.save(folder/'assessment.json',result)
            cal.save(folder/'phases.json',phase_report(run,data,end))
            cal.save(folder/'gas-heat.json',dict(burst=received_gas_heat(run),cooling=received_gas_heat(run,950.,end)))
            row.update(status='complete',**result,realtime_ratio=(end-run['started_at_s'])/run['wall_s'],
                balances={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k})
            runs[label]=run
            plot_cooling(out,runs,data,{k:k for k in runs})
            plot(out,runs,data,filename='burst-comparison.png',title='Текущий солвер с проверкой локальных запасов; неизменные материалы и химия')
            state['records'].append(row);cal.save(folder/'result.json',row);cal.save(out/'status.json',state)
            print('RESULT',folder.name,row['objective'],'speed',row['realtime_ratio'],flush=True)
            return run,row
        except (RuntimeError,TimeoutError) as exc:
            row.update(status='rejected',error=str(exc));state['records'].append(row)
            cal.save(folder/'result.json',row);cal.save(out/'status.json',state)
            print('REJECTED',folder.name,str(exc),flush=True)
            return None,row
    try:
        reference,base=execute(0,scene,'Текущая исходная геометрия')
        if reference is None:raise RuntimeError('Fresh reference rejected; do not calibrate geometry over a failed reference')
        if args.single_full:
            state['status']='single_full_complete_not_calibrated'
            return
        state['phase']='verify_exact_restart';cal.save(out/'status.json',state)
        neutral,_=execute(1,scene,'Контроль перезапуска',True)
        if neutral is None:raise RuntimeError('Neutral restart failed')
        check=compare(reference,neutral);cal.save(out/'neutral-restart-comparison.json',check)
        if not check['bitwise_equal']:raise RuntimeError('Neutral restart changed trajectory')
        runs.pop('Контроль перезапуска')
        state['phase']='conditional_common_tim_geometry';cal.save(out/'status.json',state)
        _,first=execute(2,tim_candidate(scene,.001),'Проходы TIM 0,1%',True)
        fraction=.003 if first.get('objective',float('inf'))<.99*base['objective'] else .0003
        execute(3,tim_candidate(scene,fraction),f'Проходы TIM {100*fraction:g}%',True)
        state['status']='geometry_contrasts_complete_not_calibrated'
    except KeyboardInterrupt:state['status']='stopped'
    except BaseException as exc:
        state.update(status='failed',error=str(exc));raise
    finally:
        state.pop('active_trial',None);cal.save(out/'status.json',state)


if __name__=='__main__':main()

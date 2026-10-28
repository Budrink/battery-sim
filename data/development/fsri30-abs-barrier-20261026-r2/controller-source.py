#!/usr/bin/env python3
"""Fresh FSRI diagnostic of pressure-opened ABS; retain the burst-qualified seed."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import numpy as np

import calibrate_upv_r123 as settings
from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import native
from check_fsri30_radiation_experiment import decision,read_run,REFERENCE
from check_fsri30_resolved_gas import compact
from fsri30_abs_barrier import damage_abs_barrier
from fsri30_burst_fit import assess
from fsri30_cooling_fit import converged
from fsri30_data import ROOT,load,save,sha
from fsri30_early_stop import rejection
from fsri30_objective import model_fields,cascade

PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-abs-barrier-20261026-r1'
CORE='57090f718325c766ceb23cfa639457bf7bbdabf9525c49ca68e09f1b478c1f33'
SOURCE=ROOT/'data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023'


def opening_summary(run):
    channels=[v for k,v in run.get('channels',{}).items()if k.startswith('abs-opening:')]
    def last(channel,key):return channel.get(key,[0.])[-1]
    return dict(observed_channels=len(channels),
        open_patches=sum(last(c,'opening_fraction')>0.for c in channels),
        effective_area_m2=sum(last(c,'effective_area_m2')for c in channels),
        fracture_work_j=sum(last(c,'fracture_work_j')for c in channels))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=3800.)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if sha(PROBE)!=CORE:raise ValueError('ABS diagnostic binary changed')
    base=json.loads((SOURCE/'scenario.json').read_text())
    scene=damage_abs_barrier(base,pressure_pa=1000.,onset_k=450.,rate_per_s=.01,
        temperature_scale_k=40.,area_rate_per_s=1.,fracture_j_m2=100.)
    scene['process_history_selection']['channel_prefixes'].append('abs-opening:')
    save(out/'scenario.json',scene);settings.PROBE=PROBE
    data,_=load();reference=compact(read_run(REFERENCE))
    state=dict(status='running',phase='cold-identity-check',pid=os.getpid(),native_sha256=CORE,
        source=str(SOURCE),source_sha256=sha(SOURCE/'scenario.json'),
        scenario_sha256=sha(out/'scenario.json'),fresh_from_zero=True,
        calibrated=False,validation_passed=False,abs_parameters_measured=False,
        geometry=scene['metadata']['abs_barrier_damage'])
    save(out/'contract.json',dict(state,controller_sha256=sha(Path(__file__)),
        adapter_sha256=sha(Path(__file__).with_name('fsri30_abs_barrier.py'))))
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        atomic_save(out/'status.json',state)
        cold=deepcopy(scene);cold['simulation']['duration_s']=10.
        cold['failure_sample_times_s']=[0.,5.,10.];save(out/'cold-scenario.json',cold)
        control=native(out/'cold-scenario.json',out/'cold-replay',dict(stop_after_first_rupture=False),90.,enforce_speed=False)
        archived=read_run(SOURCE/'replay/run.json.gz')
        ct,ids,cv=model_fields(control);rt,rids,rv=model_fields(archived)
        if ids!=rids:raise ValueError('Cold control changed component identities')
        expected=np.stack([np.interp(ct,rt,rv[:,j])for j in range(len(ids))],axis=1)
        error=float(np.max(np.abs(cv-expected)))
        identity=dict(maximum_component_difference_k=error,accepted=error<1e-8,
            same_observation_times=bool(np.all(np.isin(ct,rt))),**opening_summary(control))
        save(out/'cold-identity-check.json',identity)
        if not identity['accepted']or identity['open_patches']:
            raise ValueError('New closed ABS routes changed cold baseline')
        del archived,control
        state.update(phase='fresh-full-replay',cold_identity=identity)
        def prune(run):
            burst=assess(scene,run,data)
            state.update(last_completed_observation_s=run['time_s'][-1],
                live_advanced_to_s=run['advanced_to_s'],wall_s=run['wall_s'],
                minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
                burst=burst,abs_opening=opening_summary(run))
            atomic_save(out/'status.json',state)
            if burst['evaluated']and not burst['accepted']:
                reason=dict(reason=burst['reason'],burst=burst)
            elif burst['phase_window_complete']:
                reason=rejection(scene,run,data,absolute_limit_k=250.,relative_limit=.25)
            else:reason,_=decision(scene,run,reference,data)
            if reason:save(out/'early-stop.json',reason);return reason['reason']
            return None
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=CORE,
            stop_after_first_rupture=False,save_committed_restart=str(out/'post-cascade-3000.bin'),
            committed_restart_at_s=3000.),args.timeout,minimum_speed=.2,rolling_minimum=.2,prune_fn=prune)
        if run.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen ledger failed')
        if run['solver_profile']['minimum_regular_step_s']<.01-1e-9:raise ValueError('Step floor failed')
        burst=assess(scene,run,data);count=cascade(scene,run)['reacted_components']
        result=dict(burst=burst,cooling=converged(run,data,converted_components=count),
            abs_opening=opening_summary(run),last_observed_s=run['time_s'][-1],wall_s=run['wall_s'],
            run_sha256=sha(out/'replay/run.json.gz'),calibrated=False,validation_passed=False,
            balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k})
        save(out/'summary.json',result);state.update(status='complete',result=result)
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='pruned'if str(exc).startswith('Pruned: ')else'failed',error=str(exc))
    finally:
        atomic_save(out/'status.json',state)
        path=out/'replay/run.json.gz'
        if not path.exists():path=out/'replay/incomplete-run.json.gz'
        if path.exists()and read_run(path)['time_s'][-1]>=3600.:
            command=[sys.executable,str(ROOT/'scripts/report_fsri30_balance.py'),
                '--current',str(out),'--current-label','ABS','--previous',
                str(ROOT/'data/calibration/fsri30-balance-bracket-20261026-r2/trial-0026'),
                '--previous-label','26','--output',str(out/'graphs')]
            with(out/'graphs.log').open('w')as log:
                state['graphs_returncode']=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
            atomic_save(out/'status.json',state)
        print(json.dumps({k:v for k,v in state.items()if k!='geometry'},indent=2),flush=True)


if __name__=='__main__':main()

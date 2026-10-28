#!/usr/bin/env python3
"""One bounded fresh thermal-contact trial; strict physical ledgers and burst gate."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import calibrate_upv_r123 as settings
from calibrate_fsri30_bayes import atomic_save
from check_fsri30_radiation_experiment import BASE, CORE, PROBE, REFERENCE, decision, read_run
from check_fsri30_resolved_gas import compact
from check_upv_restart import native
from fsri30_abs_contact import hot_abs_contact
from fsri30_burst_fit import assess
from fsri30_cooling_fit import candidate, converged
from fsri30_data import ROOT, load, save, sha
from fsri30_early_stop import rejection
from fsri30_objective import cascade


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--emissivity',type=float,required=True)
    ap.add_argument('--stirring',type=float,required=True)
    ap.add_argument('--abs-strength',type=float,required=True)
    ap.add_argument('--timeout',type=float,default=5000.)
    ap.add_argument('--label',default='29')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if sha(PROBE)!=CORE:raise ValueError('Frozen physical core changed')
    base=json.loads(BASE.read_text())
    p=dict(effective_emissivity=args.emissivity,thermal_stirring_coefficient=args.stirring)
    scene=hot_abs_contact(candidate(base,p),strength=args.abs_strength)
    data,manifest=load();reference=compact(read_run(REFERENCE));settings.PROBE=PROBE
    save(out/'scenario.json',scene)
    state=dict(status='running',pid=os.getpid(),native_sha256=CORE,
        source_sha256=sha(BASE),scenario_sha256=sha(out/'scenario.json'),controller_sha256=sha(Path(__file__)),
        contact_model_sha256=sha(Path(__file__).with_name('fsri30_abs_contact.py')),
        csv_sha256=manifest['csv_sha256'],parameters=dict(p,abs_contact_strength=args.abs_strength),
        fresh_from_zero=True,calibrated=False,validation_passed=False,
        scope='Shared gas/ABS hot contact and exterior radiation; chemistry/residue/materials/topology frozen')
    save(out/'contract.json',state);atomic_save(out/'status.json',state)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        def prune(run):
            burst=assess(scene,run,data)
            state.update(last_completed_observation_s=run['time_s'][-1],live_advanced_to_s=run['advanced_to_s'],
                wall_s=run['wall_s'],minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],burst=burst)
            atomic_save(out/'status.json',state)
            if burst['evaluated']and not burst['accepted']:
                reason=dict(kind='burst',reason=burst['reason'],burst=burst)
            elif burst['phase_window_complete']:
                reason=rejection(scene,run,data,absolute_limit_k=250.,relative_limit=.25)
            else:
                reason,inherited=decision(scene,run,reference,data)
                if inherited:save(out/'inherited-pre-cooling-mismatch.json',inherited)
            if reason:
                save(out/'early-stop.json',reason);return reason['reason']
            return None
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=CORE,stop_after_first_rupture=False,
            save_committed_restart=str(out/'post-cascade-3000.bin'),committed_restart_at_s=3000.),
            args.timeout,minimum_speed=.2,rolling_minimum=.2,prune_fn=prune)
        if run.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen ledger failed')
        if run['solver_profile']['minimum_regular_step_s']<.01-1e-9:raise ValueError('Physical step floor violated')
        burst=assess(scene,run,data);converted=cascade(scene,run)['reacted_components']
        acceptance=converged(run,data,converted_components=converted)
        acceptance.update(burst_accepted=burst['accepted'],burst_timing_target_met=burst['timing_target_met'])
        acceptance['accepted']=bool(acceptance['accepted']and burst['accepted']and burst['timing_target_met'])
        result=dict(acceptance=acceptance,burst=burst,balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k},
            run_sha256=sha(out/'replay/run.json.gz'),wall_s=run['wall_s'],last_observed_s=run['time_s'][-1])
        save(out/'summary.json',result);state.update(status='complete',result=result,burst=burst)
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='pruned'if str(exc).startswith('Pruned: ')else'failed',error=str(exc))
    finally:
        atomic_save(out/'status.json',state)
        completed=out/'replay/run.json.gz';partial=out/'replay/incomplete-run.json.gz'
        path=completed if completed.exists()else partial
        if path.exists():
            observed=read_run(path)['time_s'][-1]
            if observed>=3600.:
                command=[sys.executable,str(ROOT/'scripts/report_fsri30_balance.py'),'--current',str(out),
                    '--current-label',args.label,'--previous',str(ROOT/'data/calibration/fsri30-balance-bracket-20261026-r2/trial-0026'),
                    '--previous-label','26','--output',str(out/'graphs')]
                with(out/'graphs.log').open('w')as log:
                    code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
                state['graphs_returncode']=code;atomic_save(out/'status.json',state)
        print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

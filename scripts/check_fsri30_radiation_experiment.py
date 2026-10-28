#!/usr/bin/env python3
"""Fresh FSRI30 radiation replay, with inherited warmup errors labelled.

Only radiation changes relative to the retained seam candidate. Old prefixes
are references, NEVER restart states or substitutes for an uncomputed tail.
"""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import signal

import numpy as np

import calibrate_upv_r123 as calibration
from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import native
from check_fsri30_resolved_gas import compact
from check_fsri30_source_reconstruction import metrics
from check_fsri30_open_boundary import cooling_metrics
from fsri30_data import ROOT, load, save, sha
from fsri30_early_stop import rejection
from fsri30_objective import model_fields, targets
from fsri30_radiation import add_exterior_radiation
from fsri30_cooling_fit import candidate as cooling_candidate
from fsri30_burst_fit import assess as assess_burst, POLICY as BURST_POLICY

BASE = ROOT/'data/development/fsri30-abs-seams-20261025-r2/scenario.json'
REFERENCE = BASE.parent/'replay/incomplete-run.json.gz'
PROFILE = ROOT/'data/development/fsri30-radiation-20261025-r1/scenario.json'
PROBE = ROOT/'build/calibration-probes/battery_sim_casing_probe-radiation-20261025-r1'
CORE = '3147360f49c85a456e290478ae0cc30c9d63a460e5c3c44109a02bea2015625f'


def read_run(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def prefix(run,end):
    """Never extend a reference beyond its last completed observation."""
    times=np.asarray(run['time_s'])
    if end>times[-1]+1e-7:raise ValueError('Reference does not cover requested prefix')
    n=int(np.searchsorted(times,end+1e-7,side='right'))
    return dict(time_s=times[:n].tolist(),cell_mean_temperature_k={
        key:values[:n] for key,values in run['cell_mean_temperature_k'].items()})


def recent_rmse(run,data,window=60.):
    t,_,v=model_fields(run);target=targets(data)
    mask=(target['time']>=t[-1]-window)&(target['time']<=t[-1]+1e-7)&~target['burst']
    if not np.any(mask):return None
    q=np.quantile(v,[0.,.5,.9,1.],axis=1).T
    pred=np.stack([np.interp(target['time'][mask],t,q[:,i])for i in range(4)],axis=1)
    return float(np.sqrt(np.mean((pred-target['q'][mask])**2)))


def decision(scene,run,reference,data):
    result=rejection(scene,run,data)
    if result is None:return None,None
    end=float(run['time_s'][-1])
    # A cooling-only sensitivity test must not be aborted by the IDENTICAL
    # already-known parent-pulse timing error of the unchanged baseline. Keep
    # that failure visible; tolerate it only if it is not made materially worse.
    # Absolute safety/ledger/speed guards and ALL cooling guards stay enabled.
    gross='persistent gross envelope divergence for 60 s outside uncertain peaks'
    if end<3600. and result['reason']==gross:
        old=prefix(reference,end)
        prior=rejection(scene,old,data)
        if prior and prior['reason']==gross:
            a,b=recent_rmse(run,data),recent_rmse(old,data)
            if a is not None and b is not None and a<=b+80.:
                return None,dict(result,candidate_rmse_k=a,reference_rmse_k=b,
                    note='Inherited pre-cooling mismatch, not accepted calibration; cooling test continues')
    return result,None


def plot(out,run,reference,data,label='Exterior radiation, effective epsilon 0.1'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    observed=np.array([data[f'T{i}_k']for i in range(1,10)]).T
    end=float(run['time_s'][-1])
    for ax in axes[0]:
        mask=data['time_s']<=end+1e-7
        ax.fill_between(data['time_s'][mask],np.min(observed[mask],axis=1)-273.15,
            np.max(observed[mask],axis=1)-273.15,color='.6',alpha=.25,label='Experiment: 9 TC range')
        ax.plot(data['time_s'][mask],np.median(observed[mask],axis=1)-273.15,
            color='black',lw=1.,label='Experiment: TC median')
    for label,r,color in (('No radiation (saved prefix)',reference,'tab:orange'),
                           (label,run,'tab:blue')):
        t,_,v=model_fields(r);mask=t<=end+1e-7;t=t[mask]
        q=np.quantile(v[mask],[0.,.5,.9,1.],axis=1).T-273.15
        for ax in axes[0]:
            ax.plot(t,q[:,1],color=color,label=label+' median')
            ax.plot(t,q[:,3],color=color,ls='--',alpha=.7,label=label+' maximum')
        axes[1,0].plot(t,np.array(r['process_totals']['gas_released_energy_j'])[mask]/1000.,color=color,label=label+' R3')
        if 'thermal_boundary_energy_out_j'in r['process_totals']:
            axes[1,1].plot(t,np.array(r['process_totals']['thermal_boundary_energy_out_j'])[mask]/1000.,color=color,label=label)
    for ax,title in zip(axes.flat,('Full temperature history, C','Cooling, C','Gas combustion energy, kJ','Radiative energy exported, kJ')):
        ax.set_title(title);ax.set_xlabel('Time, s');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0,0].set_xlim(0.,end)
    if end>3000.:axes[0,1].set_xlim(3000.,end)
    fig.suptitle('Fresh 20 mm / 30 cells; cooling-boundary candidate\nTC envelope vs component means, NOT matched sensors or independent validation; no time shift')
    fig.savefig(out/'comparison.png',dpi=155);plt.close(fig)


def report(out,scene,run,reference,data):
    end=min(float(run['time_s'][-1]),float(reference['time_s'][-1]))
    candidate=prefix(run,end);control=prefix(reference,end)
    result=dict(last_completed_observation_s=float(run['time_s'][-1]),
        common_comparison_until_s=end,common_control_cooling=cooling_metrics(control,data),
        common_candidate_cooling=cooling_metrics(candidate,data),
        candidate_cooling=cooling_metrics(run,data),calibrated=False,validation_passed=False,
        scope='Observed prefixes only; saved control is incomplete; unmapped TC envelope heuristic')
    if 'envelope_rmse_k'in result['common_candidate_cooling']:
        result['common_cooling_rmse_change_k']=result['common_candidate_cooling']['envelope_rmse_k']-result['common_control_cooling']['envelope_rmse_k']
    if 'process_totals'in run:
        result['radiative_energy_out_j']=run['process_totals']['thermal_boundary_energy_out_j'][-1]
    label=f"Radiation epsilon {scene['metadata']['exterior_radiation']['effective_emissivity']:.3g}, stirring {scene['gas_network']['thermal_stirring_coefficient']:.3g}"
    save(out/'comparison.json',result);plot(out,compact(run),reference,data,label)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--emissivity',type=float,default=.1)
    ap.add_argument('--stirring',type=float,default=.05)
    ap.add_argument('--explosion-priority',action='store_true')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    seed=json.loads(PROFILE.read_text());base=json.loads(BASE.read_text())
    if seed!=add_exterior_radiation(base,.1):raise ValueError('Radiation-only seed has changed')
    scene=cooling_candidate(base,dict(effective_emissivity=args.emissivity,thermal_stirring_coefficient=args.stirring))
    if sha(PROBE)!=CORE:raise ValueError('Frozen radiation core changed')
    calibration.PROBE=PROBE
    data,manifest=load();reference=compact(read_run(REFERENCE));save(out/'scenario.json',scene)
    state=dict(status='running',pid=os.getpid(),native_sha256=CORE,
        scenario_sha256=sha(out/'scenario.json'),source_profile_sha256=sha(PROFILE),
        reference_sha256=sha(REFERENCE),controller_sha256=sha(Path(__file__)),
        experimental_csv_sha256=manifest['csv_sha256'],fresh_from_zero=True,
        duration_s=5945.,mesh_mm=20.,parameters=scene['metadata']['cooling_search_parameters'],
        only_change='Shared exterior radiation and existing conservative free-gas mixing',
        explosion_priority=args.explosion_priority,
        condensed_residue_unchanged=True,calibrated=False,validation_passed=False)
    save(out/'contract.json',state);atomic_save(out/'status.json',state)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        def prune(partial):
            burst=assess_burst(scene,partial,data)if args.explosion_priority else None
            if burst is not None:state['burst']=burst
            if burst is not None and burst['evaluated']and not burst['accepted']:
                reason=dict(kind='burst',reason=burst['reason'],burst=burst);inherited=None
            elif burst is not None and burst['phase_window_complete']:
                # A moderately biased cooling tail cannot discard a good
                # explosion before its full tail is measured. Gross failures
                # and the original physical/numerical safety still stop it.
                reason=rejection(scene,partial,data,absolute_limit_k=250.,relative_limit=.25);inherited=None
            else:reason,inherited=decision(scene,partial,reference,data)
            state.update(last_completed_observation_s=partial['time_s'][-1],
                live_advanced_to_s=partial['advanced_to_s'],wall_s=partial['wall_s'],
                minimum_regular_step_s=partial['solver_profile']['minimum_regular_step_s'])
            if inherited:
                state['inherited_pre_cooling_mismatch']=inherited
                save(out/'inherited-pre-cooling-mismatch.json',inherited)
            atomic_save(out/'status.json',state)
            if reason:
                save(out/'early-stop.json',reason)
                return reason['reason']
            return None
        for key,name in (('BATTERY_AVERAGED_FAILURE_SNAPSHOT','pressure-failure.json'),
            ('BATTERY_MIXING_FAILURE_SNAPSHOT','mixing-failure.json'),('BATTERY_GAS_FAILURE_SNAPSHOT','constituent-failure.json')):
            os.environ[key]=str(out/name)
        run=native(out/'scenario.json',out/'replay',dict(restart_contract=CORE,
            stop_after_first_rupture=False,save_committed_restart=str(out/'post-cascade-3000.bin'),
            committed_restart_at_s=3000.,gas_frames_path=str(out/'gas-frames.bin')),
            5945.*5.2+60.,minimum_speed=.2,rolling_minimum=.2,prune_fn=prune)
        if run.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen ledger failed')
        if run['solver_profile']['minimum_regular_step_s']<.01-1e-9:raise ValueError('Physical step floor violated')
        result=metrics(scene,run,data);save(out/'summary.json',result)
        state.update(status='complete',result=report(out,scene,run,reference,data),
            run_sha256=sha(out/'replay/run.json.gz'),automatically_promoted=False)
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='pruned'if str(exc).startswith('Pruned: ')else'failed',error=str(exc))
    finally:
        partial=out/'replay/incomplete-run.json.gz'
        if partial.exists():
            try:
                run=read_run(partial);state['prefix_sha256']=sha(partial)
                state['result']=report(out,scene,run,reference,data)
            except Exception as exc:state['partial_report_error']=str(exc)
        atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

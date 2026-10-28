#!/usr/bin/env python3
"""Fresh 20 mm open-boundary trials; retain chemistry and internal geometry.

No old-core checkpoints or scores. Archived closed trajectory is a labelled
reference only; TC envelopes are NOT a fabricated per-cell validation.
"""
import argparse
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
from fsri30_data import ROOT,load,save,sha
from fsri30_objective import model_fields,targets
from fsri30_open_boundary import open_boundary
from fsri30_early_stop import rejection

BASE=ROOT/'data/development/fsri30-full20-10-20261025-r1/checks/h-20mm/scenario.json'
CLOSED=BASE.parent/'replay/run.json.gz'


def cooling_metrics(run,data):
    t,_,values=model_fields(run);target=targets(data)
    q=np.quantile(values,[0.,.5,.9,1.],axis=1).T
    mask=(target['time']>=3600.) & (target['time']<=t[-1]+1e-7)
    if not np.any(mask):return dict(complete=False)
    prediction=np.stack([np.interp(target['time'][mask],t,q[:,i]) for i in range(4)],axis=1)
    error=prediction-target['q'][mask]
    tail=t>=3600.;delta=np.diff(q[tail],axis=0)
    return dict(complete=bool(t[-1]>=5945.-1e-7),
        envelope_rmse_k=float(np.sqrt(np.mean(error**2))),
        quantile_rmse_k=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        tail_temperature_c=(q[-1]-273.15).tolist(),
        positive_rise_sum_k=np.maximum(0.,delta).sum(axis=0).tolist(),
        scope='9 TC quantiles vs 30 means, exploratory shape diagnostic; no sensor mapping')


def plot(out,runs,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    observed=np.array([data[f'T{i}_k'] for i in range(1,10)]).T
    for ax in axes[0]:
        ax.fill_between(data['time_s'],np.min(observed,axis=1)-273.15,
            np.max(observed,axis=1)-273.15,color='.7',alpha=.3,label='Experiment: 9 TC envelope')
        ax.plot(data['time_s'],np.median(observed,axis=1)-273.15,color='black',lw=1.,label='Experiment: TC median')
    for label,run in runs.items():
        t,_,v=model_fields(run);q=np.quantile(v,[.5,.9,1.],axis=1).T-273.15
        for ax in axes[0]:
            line,=ax.plot(t,q[:,0],label=label+' median')
            ax.plot(t,q[:,2],color=line.get_color(),ls='--',alpha=.7,label=label+' maximum')
        totals=run['process_totals']
        axes[1,0].plot(t,np.array(run['gas_groups']['outside']['maximum_temperature_k'])-273.15,label=label)
        axes[1,1].plot(t,np.array(totals['gas_released_energy_j'])/1000.,label=label)
        if 'escaped_energy_j' in totals:
            axes[2,0].plot(t,np.array(totals['escaped_energy_j'])/1000.,label=label+' sensible+latent')
            axes[2,0].plot(t,np.array(totals['escaped_chemical_j'])/1000.,ls='--',label=label+' unburned chemical')
            axes[2,1].plot(t,totals['ambient_gross_out_kg'],label=label+' gross out')
            axes[2,1].plot(t,totals['ambient_gross_in_kg'],ls='--',label=label+' gross in')
    titles=('Full temperature history, °C','Cooling tail, °C','Hottest external gas, °C',
        'Cumulative R3, kJ','Energy carried to ambient, kJ','Gas exchanged with ambient, kg')
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title);ax.set_xlabel('s');ax.grid(alpha=.2);ax.legend(fontsize=6)
    axes[0,1].set_xlim(3000.,5945.)
    fig.suptitle('Uniform 20 mm; chemistry/materials/internal contacts frozen\nOpen far-field diagnostic, NOT independent validation; no time shifts')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--base',type=Path,default=BASE)
    ap.add_argument('--flows-m3-s',type=float,nargs='+',default=[.01])
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    calibration.PROBE=args.probe.resolve();base=json.loads(args.base.read_text());data,_=load()
    if base['metadata']['air_mesh']['spacing_m']!=.02:raise ValueError('20 mm mesh required')
    if base['simulation']['duration_s']!=5945.:raise ValueError('Full published horizon required')
    core=sha(calibration.PROBE);base_hash=sha(args.base)
    state=dict(status='running',pid=os.getpid(),native_sha256=core,
        base_sha256=base_hash,base=str(args.base.resolve()),flows_m3_s=args.flows_m3_s,
        fresh_from_zero=True,mesh_mm=20.,chemistry_frozen=True,internal_geometry_frozen=True,
        calibrated=False,validation_passed=False,results={})
    save(out/'contract.json',dict(state,controller_sha256=sha(Path(__file__)),
        boundary_adapter_sha256=sha(Path(__file__).with_name('fsri30_open_boundary.py'))))
    with gzip.open(CLOSED,'rt') as f:reference=json.load(f)
    runs={'closed (archived reference)':compact(reference)}
    save(out/'closed-reference.json',dict(path=str(CLOSED),sha256=sha(CLOSED),
        cooling=cooling_metrics(reference,data),producer_core='e7386cc42effc14e1c8bf6a2ffa6c225b77935921458ca5aadfaa9df7cc59f56',
        recomputed_on_current_core=False))
    del reference
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        for flow in args.flows_m3_s:
            label=f'open-{flow:g}m3s';state['phase']=label;atomic_save(out/'status.json',state)
            if sha(calibration.PROBE)!=core or sha(args.base)!=base_hash:raise ValueError('Pinned source changed')
            folder=out/label;folder.mkdir();scene=open_boundary(base,flow)
            save(folder/'scenario.json',scene)
            def prune(prefix):
                decision=rejection(scene,prefix,data)
                if decision:
                    save(folder/'early-stop.json',decision)
                    return decision['reason']
                return None
            for key,name in (('BATTERY_AVERAGED_FAILURE_SNAPSHOT','pressure-failure.json'),
                ('BATTERY_MIXING_FAILURE_SNAPSHOT','mixing-failure.json'),('BATTERY_GAS_FAILURE_SNAPSHOT','constituent-failure.json')):
                os.environ[key]=str(folder/name)
            run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=core,
                stop_after_first_rupture=False,save_committed_restart=str(folder/'sealed-1100.bin'),
                committed_restart_at_s=1100.),5945.*5.2+60.,minimum_speed=.2,rolling_minimum=.2,prune_fn=prune)
            if run.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen balance failed')
            result=metrics(scene,run,data)
            result.update(cooling=cooling_metrics(run,data),renewal_m3_s=flow,
                escaped={k:v[-1] for k,v in run['process_totals'].items() if k.startswith(('escaped_','ambient_'))},
                scenario_sha256=sha(folder/'scenario.json'),run_sha256=sha(folder/'replay/run.json.gz'))
            if result['minimum_regular_step_s']<.01-1e-9:raise ValueError('Physical time floor violated')
            save(folder/'summary.json',result);state['results'][label]=result
            runs[label]=compact(run);del run;plot(out,runs,data);atomic_save(out/'status.json',state)
        state['status']='complete'
        state['selected_candidate']=min(state['results'],key=lambda k:state['results'][k]['cooling']['envelope_rmse_k'])
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='pruned'if str(exc).startswith('Pruned: ')else'failed',error=str(exc))
    finally:atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

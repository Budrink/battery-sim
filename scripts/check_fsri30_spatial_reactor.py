#!/usr/bin/env python3
"""Fresh coarse-grid quadrature trial against the preserved fine prefix.

Frozen physical laws/geometry/boundary; no fine-state restart or fitted clocks.
This is numerical-method qualification, NOT experimental validation.
"""
import argparse
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import signal

import numpy as np

import calibrate_upv_r123 as settings
from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import native,compare
from fsri30_data import ROOT,sha
from fsri30_prefix_history import conversion_events,trim_observed
from report_fsri30_full_mesh import phase_comparison


def load_run(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def trial_scene(base,enabled,end):
    scene=deepcopy(base)
    scene['simulation']['duration_s']=end
    times=[t for t in scene['failure_sample_times_s'] if t<=end+1e-9]
    if not times or times[-1]!=end:raise ValueError('Use an existing observation boundary')
    scene['failure_sample_times_s']=times
    scene['gas_network']['reactive_reconstruction']=enabled
    boxes=scene['metadata']['gas_volume_geometry']
    for node in scene['gas_network']['compartments']:
        if node['id'] in boxes:node['reconstruction_box']=deepcopy(boxes[node['id']])
    if scene['simulation']['minimum_dt_s']!=.01 or scene['gas_network'].get('pressure_pattern_predictor'):
        raise ValueError('Numerical floor or rejected pressure predictor changed')
    return scene


def compare_trial(scene,trial,coarse,fine):
    end=float(trial['time_s'][-1]);histories={name:trim_observed(run,end) for name,run in
        (('old20',coarse),('fine10',fine),('new20',trial))}
    windows={}
    for name,start,stop in (('first_burst',1100.,1230.),('cascade',2200.,2400.),('observed_prefix',0.,end)):
        if end>=stop:windows[name]={label:phase_comparison(histories[label],histories['fine10'],start,stop)
            for label in ('old20','new20')}
    events={name:conversion_events(scene,raw) for name,raw in (('old20',coarse),('fine10',fine),('new20',trial))}
    event_maps={name:{e['component']:e['time_s'] for e in rows if e['time_s']<=end} for name,rows in events.items()}
    timing={}
    for label in ('old20','new20'):
        shared=sorted(event_maps[label].keys()&event_maps['fine10'].keys())
        errors=[abs(event_maps[label][k]-event_maps['fine10'][k]) for k in shared]
        timing[label]=dict(shared_events=len(shared),mean_absolute_delay_s=float(np.mean(errors)) if errors else None,
            maximum_absolute_delay_s=max(errors,default=None),events=events[label])
    return dict(windows=windows,timing=timing,scope='Unshifted numerical comparison; event interpolation has 5 s sampling uncertainty',
        calibrated=False,experimental_validation=False,promoted=False)


def plot_comparison(out,trial,coarse,fine):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for label,run,style in (('20 mm baseline',coarse,':'),('10 mm reference',fine,'--'),('20 mm quadrature',trial,'-')):
        t=np.array(run['time_s']);mask=t<=trial['time_s'][-1]+1e-7;t=t[mask]
        for ax,key in zip(axes.flat[:3],('B09','B11','B29')):
            ax.plot(t,np.array(run['cell_mean_temperature_k'][key])[mask]-273.15,style,label=label)
            ax.set_title(key+' mean temperature, °C');ax.set_xlim(2180.,min(2525.,trial['time_s'][-1]))
        axes[1,1].plot(t,np.array(run['process_totals']['gas_released_energy_j'])[mask]/1e6,style,label=label)
    axes[1,1].set_title('Cumulative R3 heat, MJ');axes[1,1].set_xlim(2180.,trial['time_s'][-1])
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlabel('s');ax.legend(fontsize=8)
    fig.suptitle('Frozen chemistry and geometry; actual time, no curve shifts')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--coarse',type=Path,required=True);ap.add_argument('--fine',type=Path,required=True)
    ap.add_argument('--probe',type=Path,required=True);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=2525.)
    ap.add_argument('--baseline-until',type=float,default=1230.)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    probe=args.probe.resolve();core=sha(probe);settings.PROBE=probe
    if core!='6d67cc57da9e99e5ad3931569544bfb9ba920b8aab8c60963c82876e9b8be4dd':
        raise ValueError('Rejected prototype controller requires its ARCHIVED prototype binary, never the production core')
    base=json.loads((args.coarse/'scenario.json').read_text())
    reference=load_run(args.fine/'replay/incomplete-run.json.gz')
    coarse=load_run(args.coarse/'replay/run.json.gz')
    if reference['time_s'][-1]<args.duration-1e-7:raise ValueError('Cannot extrapolate the fine reference')
    pinned={str(p.resolve()):sha(p) for p in (probe,args.library,args.coarse/'scenario.json',args.coarse/'replay/run.json.gz',
        args.fine/'scenario.json',args.fine/'replay/incomplete-run.json.gz',Path(__file__))}
    shutil.copyfile(probe,out/'native-probe');shutil.copymode(probe,out/'native-probe')
    shutil.copyfile(args.library,out/'native-core.a');settings.PROBE=out/'native-probe'
    state=dict(status='running',pid=os.getpid(),native_sha256=core,results={},parameters_frozen=True,
        boundary='unchanged closed finite atmosphere',physical_minimum_step_s=.01,pinned_files=pinned,
        fresh_from_zero=True,promoted=False,experimental_validation=False)
    atomic_save(out/'contract.json',state);atomic_save(out/'status.json',state)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    def ensure_pinned():
        if any(sha(Path(p))!=value for p,value in pinned.items()):raise ValueError('Pinned input/core changed')
    try:
        # Candidate first, then a fresh same-core OFF check through first burst.
        for name,enabled,end in (('quadrature',True,args.duration),('baseline-off',False,args.baseline_until)):
            ensure_pinned();folder=out/name;folder.mkdir();scene=trial_scene(base,enabled,end)
            atomic_save(folder/'scenario.json',scene)
            state['phase']=name;atomic_save(out/'status.json',state)
            request=dict(restart_contract=core,stop_after_first_rupture=False,gas_frames_path=str(folder/'gas-fields.bin'),
                save_committed_restart=str(folder/'committed-1100s.bin'),committed_restart_at_s=1100.,stop_after_committed_restart=False)
            os.environ['BATTERY_AVERAGED_FAILURE_SNAPSHOT']=str(folder/'pressure-failure.json')
            os.environ['BATTERY_MIXING_FAILURE_SNAPSHOT']=str(folder/'mixing-failure.json')
            os.environ['BATTERY_GAS_FAILURE_SNAPSHOT']=str(folder/'constituent-failure.json')
            run=native(folder/'scenario.json',folder/'replay',request,math.inf,minimum_speed=.5,rolling_minimum=.25)
            ensure_pinned()
            if run.get('partial') or abs(run['advanced_to_s']-end)>1e-7:raise ValueError('Requested horizon incomplete')
            if run['solver_profile']['minimum_regular_step_s']<.01-1e-9:raise ValueError('Physical floor failed')
            if run.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen balance failed')
            result=dict(advanced_to_s=run['advanced_to_s'],wall_s=run['wall_s'],speed_x=end/run['wall_s'],
                run_sha256=sha(folder/'replay/run.json.gz'),balances={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k})
            if enabled:
                result['comparison']=compare_trial(scene,run,coarse,reference)
                atomic_save(out/'comparison.json',result['comparison']);plot_comparison(out,run,coarse,reference)
            else:
                result['off_vs_archived']=compare(coarse,run,until_s=end)
                # New optional code must not cause a material trajectory change when OFF.
                if result['off_vs_archived']['max_absolute_difference']>1e-7:raise ValueError('OFF branch differs from archived baseline')
            state['results'][name]=result;atomic_save(out/'status.json',state)
        state['status']='complete-unpromoted'
    except KeyboardInterrupt:state.update(status='stopped',error='User/controller interruption; preserved prefix only')
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)
    return 0 if state['status']=='complete-unpromoted' else 1


if __name__=='__main__':raise SystemExit(main())

#!/usr/bin/env python3
"""Refine aperture jets, not the entire room; freeze chemistry and conductance.

Fresh baseline OFF check first. A changed room topology is numerical spatial
refinement, not a fitted gas route. All physical volumes/areas/capacities close.
"""
import argparse
from copy import deepcopy
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
from check_fsri30_spatial_reactor import load_run,compare_trial
from fsri30_data import sha
from fsri30_resolved_gas import build_resolved


def local_scene(base,coarse,near,end):
    scene=build_resolved(base,spacing=.02,vent_spacing=near,wall_spacing=.01,
        stirring=coarse['gas_network']['thermal_stirring_coefficient'],
        stirring_length=coarse['gas_network']['thermal_stirring_length_m'])
    for section in ('cells','interfaces','boundaries','shells','primary_failures','heating_protocols'):
        if scene[section]!=coarse[section]:raise ValueError('Physical section changed: '+section)
    controls=lambda s:{k:v for k,v in s['gas_network'].items() if k not in ('compartments','channels','thermal_contacts')}
    if controls(scene)!=controls(coarse):raise ValueError('Gas constitutive/numerical law changed')
    internal=lambda s:[n for n in s['gas_network']['compartments'] if not n['id'].startswith('air-')]
    if internal(scene)!=internal(coarse):raise ValueError('Internal volumes/chemistry changed')
    def air_budget(s):
        nodes=[n for n in s['gas_network']['compartments'] if n['id'].startswith('air-')]
        return np.array([sum(n['volume_m3'] for n in nodes),sum(n['initial_mass_kg'] for n in nodes),
            sum(n['initial_mass_kg']*n['initial_oxidizer_mass_fraction'] for n in nodes)])
    if not np.allclose(air_budget(scene),air_budget(coarse),rtol=1e-12,atol=1e-14):raise ValueError('Air inventory changed')
    for key in ('failure_sample_times_s','thermal_path_trace','gas_film_trace','surface_probe_trace'):
        if key in coarse:scene[key]=deepcopy(coarse[key])
    scene['simulation']=deepcopy(coarse['simulation']);scene['simulation']['duration_s']=end
    scene['failure_sample_times_s']=[t for t in coarse['failure_sample_times_s'] if t<=end+1e-9]
    if scene['failure_sample_times_s'][-1]!=end:raise ValueError('New observation boundary')
    return scene


def plot(out,runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    end=runs['20 mm + fine jets']['time_s'][-1]
    for label,r in runs.items():
        t=np.array(r['time_s']);mask=t<=end+1e-7;t=t[mask]
        for ax,k in zip(axes.flat[:3],('B09','B11','B29')):
            ax.plot(t,np.array(r['cell_mean_temperature_k'][k])[mask]-273.15,label=label)
            ax.set_title(k+' mean, °C');ax.set_xlim(2180.,end)
        axes[1,1].plot(t,np.array(r['process_totals']['gas_released_energy_j'])[mask]/1e6,label=label)
    axes[1,1].set_title('Cumulative R3, MJ');axes[1,1].set_xlim(2180.,end)
    for ax in axes.flat:ax.set_xlabel('s');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same physical laws/boundary; fresh trajectories, actual time')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--coarse',type=Path,required=True)
    ap.add_argument('--fine',type=Path,required=True);ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--near-mm',type=float,default=10.)
    ap.add_argument('--duration',type=float,default=2525.);ap.add_argument('--baseline-until',type=float,default=1230.)
    ap.add_argument('--baseline-from',type=Path,help='Reuse a completed, identical, fresh baseline from this exact native core only')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=json.loads(args.base.read_text());coarse_scene=json.loads((args.coarse/'scenario.json').read_text())
    coarse=load_run(args.coarse/'replay/run.json.gz');fine=load_run(args.fine/'replay/incomplete-run.json.gz')
    if fine['time_s'][-1]<args.duration-1e-7:raise ValueError('Cannot extrapolate fine reference')
    probe=args.probe.resolve();core=sha(probe);shutil.copy2(probe,out/'native-probe');settings.PROBE=out/'native-probe'
    sources=[Path(__file__),Path(__file__).with_name('check_fsri30_spatial_reactor.py'),
        Path(__file__).with_name('free_gas_mesh.py'),Path(__file__).with_name('fsri30_resolved_gas.py')]
    pinned={str(p.resolve()):sha(p) for p in [args.base,args.coarse/'scenario.json',probe,*sources]}
    if args.baseline_from:
        for p in (args.baseline_from/'scenario.json',args.baseline_from/'replay/run.json.gz',args.baseline_from.parent/'contract.json'):
            pinned[str(p.resolve())]=sha(p)
    state=dict(status='running',phase='prepare',pid=os.getpid(),native_sha256=core,pinned_files=pinned,
        physical_parameters_frozen=True,reactive_reconstruction=False,physical_minimum_step_s=.01,
        fresh_from_zero=True,results={},promoted=False,experimental_validation=False)
    atomic_save(out/'status.json',state);atomic_save(out/'contract.json',state)
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    def check_frozen():
        if any(sha(Path(p))!=value for p,value in pinned.items()):raise ValueError('Pinned source/core changed')
    try:
        refined=local_scene(base,coarse_scene,args.near_mm/1000.,args.duration)
        state['total_gas_nodes']=len(refined['gas_network']['compartments'])
        baseline=deepcopy(coarse_scene);baseline['simulation']['duration_s']=args.baseline_until
        baseline['failure_sample_times_s']=[t for t in coarse_scene['failure_sample_times_s'] if t<=args.baseline_until+1e-9]
        for name,scene in (('baseline-off',baseline),('local-refinement',refined)):
            check_frozen();folder=out/name
            state['phase']=name;atomic_save(out/'status.json',state)
            for key,file in (('BATTERY_GAS_FAILURE_SNAPSHOT','constituents.json'),('BATTERY_AVERAGED_FAILURE_SNAPSHOT','pressure.json'),('BATTERY_MIXING_FAILURE_SNAPSHOT','mixing.json')):
                os.environ[key]=str(folder/file)
            request=dict(restart_contract=core,stop_after_first_rupture=False,gas_frames_path=str(folder/'gas-fields.bin'),
                save_committed_restart=str(folder/'committed-1100s.bin'),committed_restart_at_s=1100.,stop_after_committed_restart=False)
            if name=='baseline-off' and args.baseline_from:
                source=args.baseline_from.resolve()
                provenance=json.loads((source.parent/'contract.json').read_text())
                if provenance['native_sha256']!=core or json.loads((source/'scenario.json').read_text())!=scene:
                    raise ValueError('Reused baseline core/physical scene mismatch')
                run=load_run(source/'replay/run.json.gz')
                if run['started_at_s']!=0.:raise ValueError('Reused baseline did not start from zero')
                shutil.copytree(source,folder)
                state['baseline_reuse']=dict(source=str(source),native_sha256=core,run_sha256=sha(source/'replay/run.json.gz'),
                    scope='same-core identical-scene baseline only; refined trajectory still fresh from zero')
            else:
                folder.mkdir();atomic_save(folder/'scenario.json',scene)
                run=native(folder/'scenario.json',folder/'replay',request,math.inf,minimum_speed=.5,rolling_minimum=.25)
            check_frozen();end=scene['simulation']['duration_s']
            if run.get('partial') or abs(run['advanced_to_s']-end)>1e-7:raise ValueError('Incomplete horizon')
            if run['solver_profile']['minimum_regular_step_s']<.01-1e-9 or run.get('max_oxidizer_residual_kg',0.)>1e-10:
                raise ValueError('Physical floor/oxygen balance failed')
            result=dict(advanced_to_s=run['advanced_to_s'],wall_s=run['wall_s'],speed_x=end/run['wall_s'],
                balances={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k},
                run_sha256=sha(folder/'replay/run.json.gz'))
            if name=='baseline-off':
                result['off_vs_archived']=compare(coarse,run,until_s=end)
                if result['off_vs_archived']['max_absolute_difference']>1e-7:raise ValueError('OFF baseline changed')
            else:
                result['comparison']=compare_trial(scene,run,coarse,fine)
                atomic_save(out/'comparison.json',result['comparison'])
                plot(out,{'20 mm baseline':coarse,'10 mm reference':fine,'20 mm + fine jets':run})
            state['results'][name]=result;atomic_save(out/'status.json',state)
        state['status']='complete-unpromoted'
    except KeyboardInterrupt:state.update(status='stopped',error='Controller interrupted; preserved prefixes only')
    except Exception as e:state.update(status='failed',error=str(e))
    finally:atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)
    return 0 if state['status']=='complete-unpromoted' else 1


if __name__=='__main__':raise SystemExit(main())

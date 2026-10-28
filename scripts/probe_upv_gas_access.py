#!/usr/bin/env python3
"""Shape-first, frozen-physics gas-access geometry experiment."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from check_upv_restart import native
from calibrate_upv_r123 import PROBE,save
import upv_r123_data as dataio
from upv_observation_policy import label
from upv_gas_access import candidate,shape_report,received_gas_heat,CHANNELS


def plot(out,runs,data,filename='shape-comparison.png',title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'101':'C1 / ISS','102':'C1 / TIM','C2_ISS':'C2 / ISS','C2_TIM':'C2 / TIM','108':'C3 / ISS','107':'C3 / TIM'}
    fig,axes=plt.subplots(3,2,figsize=(14,10),constrained_layout=True)
    obs=dataio.observed(data,data['time_s']);m=(data['time_s']>=875)&(data['time_s']<=950)
    for ax,key in zip(axes.flat,CHANNELS):
        ax.plot(data['time_s'][m],obs[key][m]-273.15,color='black',lw=2,label='Эксперимент')
        for name,run in runs.items():
            t=np.asarray(run['time_s']);keep=(t>=875)&(t<=950)
            ax.plot(t[keep],np.asarray(run['temperature_k'][key])[keep]-273.15,
                lw=1.5,ls='--'if name=='Исходный r5'else'-',label=name)
        ax.set(title=label(key,labels[key]),xlabel='Время, с',ylabel='°C',xlim=(875,950));ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(title or 'Сначала форма и распределение: меняется только геометрия доступа газа\nБез сдвига времени и масштабирования температур; пики и охлаждение не подгонялись')
    fig.savefig(out/filename,dpi=150);plt.close(fig)


def capture(source,prior,out,contract):
    """Zero-step contract-checked capture of the same committed 835 s state.

    Native gas-response checker validates the changed hot response laws first.
    No physics-identity bypass and no time advancement: this is a conditional
    diagnostic state transfer, not a claimed fresh preheating verification.
    """
    scene=json.loads((source/'scenario.json').read_text());scene['simulation']['duration_s']=835.
    scene['failure_sample_times_s']=[0.,835.];save(out/'capture-scenario.json',scene)
    request=dict(restart_contract=contract,restart_from=str(prior/'sealed-state.bin'),
        restart_gas_response_scenario=str(prior/'scenario.json'),
        save_unvented_restart=str(out/'sealed-state.bin'),stop_after_first_rupture=False)
    with (out/'capture.json').open('w')as dst,(out/'capture.log').open('w')as err:
        result=subprocess.run([str(PROBE),str(out/'capture-scenario.json')],input=json.dumps(request),
            text=True,stdout=dst,stderr=err,timeout=60.)
    if result.returncode:raise RuntimeError((out/'capture.log').read_text())
    r=json.loads((out/'capture.json').read_text())
    if 'error'in r:raise RuntimeError(r['error'])
    if r['started_at_s']!=835. or r['advanced_to_s']!=835.:raise ValueError('Unexpected checkpoint time')
    return scene,r


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--fractions',type=float,nargs='+',default=[.001])
    ap.add_argument('--lip-fraction',type=float,default=1.)
    ap.add_argument('--plume-exit-fraction',type=float,default=1.)
    ap.add_argument('--backing-fraction',type=float,default=1.)
    ap.add_argument('--witness-edge-fraction',type=float,default=1.)
    ap.add_argument('--jet-wetted-fraction',type=float,default=1.)
    ap.add_argument('--outer-insulation-fraction',type=float,default=1.)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False)
    source=args.source.resolve();prior=args.prior.resolve();contract=hashlib.sha256(PROBE.read_bytes()).hexdigest()
    state=dict(status='running',pid=os.getpid(),source=str(source),native_sha256=contract,
        source_scenario_sha256=hashlib.sha256((source/'scenario.json').read_bytes()).hexdigest(),
        geometry_code_sha256=hashlib.sha256((Path(__file__).parent/'upv_gas_access.py').read_bytes()).hexdigest(),
        runner_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        chemistry_frozen=True,material_properties_frozen=True,shape_first=True,
        accepted_calibration=False,geometry_is_unmeasured_hypothesis=True,trials=[])
    save(out/'status.json',state)
    data,_=dataio.load('no-cooling')
    with gzip.open(source/'replay/run.json.gz','rt')as f:baseline=json.load(f)
    runs={'Исходный r5':baseline}
    try:
        original,captured=capture(source,prior,out,contract)
        if any(captured['temperature_k'][k][0]!=baseline['temperature_k'][k][0]for k in baseline['temperature_k']):
            raise ValueError('Checkpoint capture does not match the baseline state')
        state['baseline_shape']=shape_report(baseline,data)
        for i,fraction in enumerate(args.fractions):
            folder=out/f'trial-{i:03d}';folder.mkdir();scene=candidate(json.loads((source/'scenario.json').read_text()),fraction,
                lip_open_fraction=args.lip_fraction,plume_exit_fraction=args.plume_exit_fraction,
                backing_open_fraction=args.backing_fraction,witness_edge_open_fraction=args.witness_edge_fraction,
                jet_wetted_fraction=args.jet_wetted_fraction,outer_insulation_open_fraction=args.outer_insulation_fraction)
            save(folder/'scenario.json',scene)
            mapping=[dict(to=j,fraction=1.,**{'from':j})for j in range(len(scene['gas_network']['compartments']))]
            save(folder/'remap.json',mapping);state.update(active_trial=folder.name);save(out/'status.json',state)
            print('START',folder.name,'fraction',fraction,'lip',args.lip_fraction,flush=True)
            row=dict(label=folder.name,protected_fraction=fraction,lip_fraction=args.lip_fraction,
                plume_exit_fraction=args.plume_exit_fraction,backing_fraction=args.backing_fraction,
                witness_edge_fraction=args.witness_edge_fraction,jet_wetted_fraction=args.jet_wetted_fraction,
                outer_insulation_open_fraction=args.outer_insulation_fraction)
            try:
                run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=contract,
                    restart_from=str(out/'sealed-state.bin'),restart_source_scenario=str(out/'capture-scenario.json'),
                    unvented_gas_remap=mapping,stop_after_first_rupture=False),700.)
                report=shape_report(run,data)
                report['accepted_gas_heat_880_900_j']=received_gas_heat(run)
                save(folder/'shape.json',report)
                row.update(status='complete',**report,wall_s=run['wall_s'],speed=115./run['wall_s'],
                    initial_conservation=run['rebase_initial_residuals'])
                runs[f'Доступ {100*fraction:g}%']=run;plot(out,runs,data)
            except (RuntimeError,TimeoutError)as exc:row.update(status='rejected',error=str(exc))
            state['trials'].append(row);save(out/'status.json',state)
            print(json.dumps({k:row[k]for k in ['label','status','shape_objective','error','speed']if k in row}),flush=True)
        state.update(status='complete',active_trial=None)
    except BaseException as exc:state.update(status='failed',error=str(exc));raise
    finally:save(out/'status.json',state)


if __name__=='__main__':main()

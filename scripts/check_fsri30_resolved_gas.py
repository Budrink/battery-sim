#!/usr/bin/env python3
"""Fresh spatial convergence ladder; frozen kinetics, reservoirs and can laws.

A finite resolved atmosphere is not open-room CFD. A 5% mesh check is NOT
experimental validation. All levels start from the same fresh physical state.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import signal

import numpy as np

import calibrate_upv_r123 as settings
from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import native
from check_fsri30_source_reconstruction import DEFAULT_BASE,metrics
from fsri30_data import ROOT,load,save,sha
from fsri30_resolved_gas import build_resolved


def compact(run):
    return {k:run[k] for k in ('time_s','cell_mean_temperature_k','process_totals','gas_groups','heating_protocols') if k in run}


def compare_mesh(a,b,ma,mb):
    ta,tb=np.asarray(a['time_s']),np.asarray(b['time_s'])
    # Different accepted substeps reach the SAME requested observation with
    # different last-bit clock roundoff. This is not an allowed time shift.
    if ta.shape!=tb.shape or not np.allclose(ta,tb,rtol=0.,atol=1e-7):
        raise ValueError('Spatial observations do not align')
    va=np.array([a['cell_mean_temperature_k'][k] for k in sorted(a['cell_mean_temperature_k'])]).T
    vb=np.array([b['cell_mean_temperature_k'][k] for k in sorted(b['cell_mean_temperature_k'])]).T
    relative=np.sqrt(np.mean(((va-vb)/np.maximum(va,vb))**2,axis=0))
    energy_relative=abs(ma['all_gas_r3_j']-mb['all_gas_r3_j'])/max(100.,ma['all_gas_r3_j'],mb['all_gas_r3_j'])
    same_count=ma['r2_converted_components']==mb['r2_converted_components']
    ea,eb=ma['model_conversion_events'],mb['model_conversion_events']
    same_order=[e['component'] for e in ea]==[e['component'] for e in eb]
    timing=max((abs(x['time_s']-y['time_s'])/max(1.,x['time_s'],y['time_s']) for x,y in zip(ea,eb)),default=0.)
    exercised=bool(ea and eb and max(ma['all_gas_r3_j'],mb['all_gas_r3_j'])>10.)
    qa=np.array(a['process_totals']['gas_released_energy_j']);qb=np.array(b['process_totals']['gas_released_energy_j'])
    r3_shape=float(np.sqrt(np.mean((qa-qb)**2))/max(100.,float(max(qa[-1],qb[-1]))))
    ta=np.array(a['gas_groups']['outside']['maximum_temperature_k']);tb=np.array(b['gas_groups']['outside']['maximum_temperature_k'])
    external_shape=float(np.sqrt(np.mean(((ta-tb)/np.maximum(ta,tb))**2)))
    return dict(accepted=bool(exercised and max(relative)<=.05 and energy_relative<=.05 and r3_shape<=.05 and external_shape<=.05 and same_count and same_order and timing<=.05),
        maximum_component_rms_relative_kelvin=float(max(relative)),r3_energy_relative_difference=energy_relative,
        reactive_window_exercised=exercised,r3_cumulative_rms_relative_difference=r3_shape,
        hottest_external_gas_rms_relative_kelvin=external_shape,
        same_conversion_count=same_count,same_conversion_order=same_order,maximum_conversion_time_relative_difference=timing,
        tolerance=.05,scope='spatial convergence on computed horizon only; NOT experimental validation')


def plot(out,runs,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for label,r in runs.items():
        t=np.array(r['time_s']);v=np.array(list(r['cell_mean_temperature_k'].values())).T
        axes[0,0].plot(t,np.max(v,axis=1)-273.15,label=label)
        axes[0,1].plot(t,np.median(v,axis=1)-273.15,label=label)
        axes[1,0].plot(t,np.array(r['process_totals']['gas_released_energy_j'])/1000,label=label)
        axes[1,1].plot(t,np.array(r['gas_groups']['outside']['maximum_temperature_k'])-273.15,label=label)
    obs=np.array([data[f'T{i}_k'] for i in range(1,10)]).T-273.15
    for ax,series in ((axes[0,0],np.max(obs,axis=1)),(axes[0,1],np.median(obs,axis=1))):
        ax.plot(data['time_s'],series,c='black',alpha=.4,lw=1.,label='9 TC envelope (not matched locations)')
        if runs:ax.set_xlim(0,max(r['time_s'][-1] for r in runs.values()))
    for ax,title in zip(axes.flat,('Maximum battery mean, °C','Median battery mean, °C','Cumulative gas R3, kJ','Hottest external air voxel, °C')):
        ax.set_title(title);ax.set_xlabel('s');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Fixed chemistry; spatial gas refinement — diagnostic, NOT fitted/validated')
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,default=DEFAULT_BASE)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=1800.)
    ap.add_argument('--levels-mm',type=float,nargs='+',default=[40.,20.,10.])
    ap.add_argument('--first-level-mm',type=float,help='Verify a previously failing mesh first; comparison order stays coarse-to-fine')
    ap.add_argument('--stirring',type=float,default=.05)
    ap.add_argument('--stirring-length-mm',type=float,default=20.)
    ap.add_argument('--pressure-patterns',action='store_true')
    ap.add_argument('--vent-spacing-mm',type=float,help='Fine aperture jet spacing; other gas regions follow --levels-mm')
    ap.add_argument('--wall-spacing-mm',type=float,help='Fixed ABS spacing, to isolate gas-grid changes')
    ap.add_argument('--all-levels',action='store_true')
    ap.add_argument('--no-heater-control',action='store_true')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not 1.<=args.duration<=5945.:raise ValueError('Outside published horizon')
    if not 0.<args.stirring_length_mm<=50.:raise ValueError('Physical stirring length must be in (0,50] mm')
    if any(x<=y for x,y in zip(args.levels_mm,args.levels_mm[1:])):raise ValueError('Levels must decrease')
    execution=list(args.levels_mm)
    if args.first_level_mm is not None:
        if args.first_level_mm not in execution:raise ValueError('First mesh is not in the ladder')
        execution.remove(args.first_level_mm);execution.insert(0,args.first_level_mm)
    base=json.loads(args.base.read_text());settings.PROBE=args.probe.resolve();data,_=load()
    state=dict(status='running',pid=os.getpid(),results={},comparisons={},duration_s=args.duration,
        levels_mm=args.levels_mm,execution_order_mm=execution,stirring_prior=args.stirring,
        stirring_length_mm=args.stirring_length_mm,pressure_patterns=args.pressure_patterns,native_sha256=sha(settings.PROBE),
        vent_spacing_mm=args.vent_spacing_mm,wall_spacing_mm=args.wall_spacing_mm,
        base_scenario=str(args.base.resolve()),base_sha256=sha(args.base),fresh_from_zero=True,
        calibrated=False,validation_passed=False,spatially_converged=False)
    contract=dict(state,controller_sha256=sha(Path(__file__)),geometry_sha256=sha(Path(__file__).with_name('fsri30_resolved_gas.py')),
        mesh_sha256=sha(Path(__file__).with_name('free_gas_mesh.py')))
    save(out/'contract.json',contract);runs={}
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        for mm in execution:
            label=f'h-{mm:g}mm';state['phase']=label;atomic_save(out/'status.json',state)
            scene=build_resolved(base,mm/1000,stirring=args.stirring,controlled_heater=not args.no_heater_control,
                stirring_length=args.stirring_length_mm/1000,pressure_patterns=args.pressure_patterns,
                vent_spacing=None if args.vent_spacing_mm is None else args.vent_spacing_mm/1000,
                wall_spacing=None if args.wall_spacing_mm is None else args.wall_spacing_mm/1000)
            scene['simulation']['duration_s']=args.duration
            scene['failure_sample_times_s']=sorted(set([0.,args.duration]+list(np.arange(5.,args.duration,5.))))
            scene['gas_film_trace']=False
            scene['metadata'].update(fresh_from_zero=True,conditioned_transport_diagnostic=False)
            folder=out/label;folder.mkdir();save(folder/'scenario.json',scene)
            # Failure-only observers: preserve a rejected algebraic window for
            # an isolated GPU regression, without repeating the warmup again.
            for variable,filename in (
                    ('BATTERY_AVERAGED_FAILURE_SNAPSHOT','pressure-failure.json'),
                    ('BATTERY_MIXING_FAILURE_SNAPSHOT','mixing-failure.json'),
                    ('BATTERY_GAS_FAILURE_SNAPSHOT','constituent-failure.json')):
                os.environ[variable]=str(folder/filename)
            try:
                run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=state['native_sha256'],
                    stop_after_first_rupture=False,gas_frames_path=str(folder/'gas-fields.bin')),
                    args.duration*5.2+60.,minimum_speed=.2,rolling_minimum=.2)
                result=metrics(scene,run,data)
                if result['minimum_regular_step_s']<scene['simulation']['minimum_dt_s']-1e-9:raise ValueError('Physical timestep floor violated')
                if run.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen conservation failed')
                result.update(external_nodes=scene['metadata']['air_mesh']['external_nodes'],
                    scenario_sha256=sha(folder/'scenario.json'),run_sha256=sha(folder/'replay/run.json.gz'),
                    heater_cutoff=[e for e in run['events'] if e['kind']=='heater_cutoff'])
                save(folder/'summary.json',result);state['results'][label]=result;runs[label]=compact(run);del run
                for coarse,fine in zip(args.levels_mm,args.levels_mm[1:]):
                    a,b=f'h-{coarse:g}mm',f'h-{fine:g}mm';key=a+' / '+b
                    if a not in runs or b not in runs or key in state['comparisons']:continue
                    comparison=compare_mesh(runs[a],runs[b],state['results'][a],state['results'][b])
                    state['comparisons'][key]=comparison
                    state['spatially_converged']=comparison['accepted']
                    if comparison['accepted']:
                        state.update(spatially_converged=True,converged_on_horizon_s=args.duration,selected_mesh_mm=fine)
                    else:
                        state.pop('converged_on_horizon_s',None);state.pop('selected_mesh_mm',None)
                plot(out,runs,data);atomic_save(out/'status.json',state)
                if state['spatially_converged'] and not args.all_levels:break
            except Exception as exc:
                state['results'][label]=dict(status='failed',error=str(exc));atomic_save(out/'status.json',state)
                # Numerical failure is not a coarse mesh score. Do not blindly
                # spend the remaining ladder on an unverified failing solver.
                raise
        state['status']='complete' if state['spatially_converged'] else 'mesh-limit-not-converged'
    except KeyboardInterrupt:state['status']='stopped'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

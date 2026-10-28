#!/usr/bin/env python3
"""Fresh 2x2 geometry / UPV-kinetics diagnostic, no fitting or forced ignition."""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import signal

import numpy as np

import calibrate_upv_r123 as settings
from calibrate_fsri30_bayes import atomic_save
from check_upv_restart import native
from fsri30_data import ROOT, load, save, sha
from fsri30_objective import loss
from fsri30_reconstruction import source_geometry, transfer_r2, R2_KEYS
from model_evidence import record

DEFAULT_BASE = ROOT/'data/calibration/ul-fsri-30-reactive-refinement-20261024-r8/trial-0003/scenario.json'
DEFAULT_PROBE = ROOT/'build/calibration-probes/battery_sim_casing_probe-closed-exhaust-20261024-r1'
CASES = ('combined', 'r2-only', 'geometry-only', 'reference')


def cases(base):
    # Function calls are separate: a geometry-only run cannot silently acquire
    # UPV chemistry or a newly created reserve, and input is never mutated.
    values = {'reference':deepcopy(base), 'r2-only':transfer_r2(base),
              'geometry-only':source_geometry(base)}
    values['combined'] = transfer_r2(values['geometry-only'])
    for name, scene in values.items():
        scene['name'] = 'FSRI source reconstruction / '+name
        scene['gas_film_trace'] = True
        scene['process_history_selection']['gas_prefixes'] = ['body-', 'top-', 'base-', 'plume-', 'hood']
        scene['process_history_selection']['channel_prefixes'] = ['rupture-', 'vent-', 'outside-', 'exhaust-']
        scene['metadata'].update(fresh_from_zero=True, conditioned_transport_diagnostic=False,
            validation_passed=False, calibration_status='UNVALIDATED geometry/kinetics transfer diagnostic')
    return values


def metrics(scene, run, data):
    t = np.array(run['time_s']); source = scene['metadata']['heated_component']
    events = []; peaks = []
    for cell in scene['cells']:
        if cell['role'] != 'battery': continue
        ident = cell['id']; law = cell['conversion']
        # Unlike residue+gas, this does NOT mistake inert R1 products for R2.
        # The present FSRI law has constant .33 gas yield. Reject other laws
        # rather than silently reusing this exact inference for variable yield.
        if law.get('gas_yield_temperature_per_k', 0.) or law.get('gas_yield_density_exponent', 0.) or law['gas_yield'] <= 0.:
            raise ValueError('R2 reporting needs a measured direct ledger or constant gas yield')
        converted = np.array(run['chemistry'][ident]['r2_generated_gas_kg'])/law['gas_yield']
        hit = np.flatnonzero(converted >= .1*law['mass_kg'])
        if len(hit):
            i = int(hit[0]); when = float(t[i]) if i == 0 else float(np.interp(
                .1*law['mass_kg'], converted[i-1:i+1], t[i-1:i+1]))
            events.append(dict(component=ident, time_s=when))
        peaks.append(dict(component=ident, mean_c=float(max(run['cell_mean_temperature_k'][ident]))-273.15,
                          r2_converted_mass_fraction=float(converted[-1]/law['mass_kg'])))
    events.sort(key=lambda e:(e['time_s'],e['component']))
    peaks.sort(key=lambda e:e['mean_c'],reverse=True)
    envelope=loss(scene,run,data,include_features=False)
    envelope['complete']=bool(envelope['complete'])
    return dict(status='partial' if run.get('partial') else 'complete', advanced_to_s=run['advanced_to_s'],
        fresh_from_zero=run['started_at_s']==0., calibrated=False, validation_passed=False,
        r2_parameters={k:next(c for c in scene['cells'] if c['role']=='battery')['conversion'][k] for k in R2_KEYS},
        r2_converted_components=len(events), model_conversion_events=events,
        observed_failure_count=None, conversion_reporting='10% R2 mass, inferred from constant-yield gas ledger; not R1 residue',
        neighbour_peaks=[p for p in peaks if p['component']!=source][:6],
        source_peak=next(p for p in peaks if p['component']==source),
        all_gas_r3_j=run['process_totals']['gas_released_energy_j'][-1],
        final_all_gas_unburned_j=run['process_totals']['gas_chemical_j'][-1],
        final_all_gas_oxygen_kg=run['process_totals']['gas_oxidizer_kg'][-1],
        maximum_recorded_cavity_pressure_pa=max(max(v['pressure_pa']) for k,v in run['gas'].items() if k.startswith(('body-','top-','base-'))),
        wall_s=run['wall_s'], realtime_ratio=(run['advanced_to_s']-run['started_at_s'])/run['wall_s'],
        minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
        balances={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k},
        experimental_envelope_diagnostic=envelope)


def geometry_plot(out, base, reconstructed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    from generate_fsri30 import BOX, DIAMETER
    geom=reconstructed['metadata']['geometry_reconstruction']
    points=np.array(list(reconstructed['metadata']['approximate_positions_m'].values()))
    fig,axes=plt.subplots(1,2,figsize=(10,8),constrained_layout=True)
    ax=axes[0]
    for poly in geom['polygon_xy_m']:
        ax.add_patch(Polygon(np.array(poly)*1000,fill=False,ec='.65',lw=.7))
    for j,(x,y,_) in enumerate(points):
        ax.add_patch(Circle((x*1000,y*1000),DIAMETER*500,fc='#da8360' if j==10 else '#cddce6',ec='.3',lw=.6))
        ax.text(x*1000,y*1000,f'{j:02}',ha='center',va='center',fontsize=7)
    for face in geom['shared_faces']:
        a,b=points[face['a']],points[face['b']]
        ax.plot([a[0]*1000,b[0]*1000],[a[1]*1000,b[1]*1000],c='tab:blue',alpha=.25,lw=.6)
    ax.add_patch(Rectangle((-BOX[0]*500,-BOX[1]*500),BOX[0]*1000,BOX[1]*1000,fill=False,ec='black'))
    for y,label in [(-BOX[1]*500,'Back vent'),(BOX[1]*500,'Front vent')]:
        ax.annotate(label,xy=(0,y),xytext=(0,y+(-12 if y<0 else 12)),ha='center',
                    arrowprops=dict(arrowstyle='->',color='crimson'),color='crimson')
    ax.set_aspect('equal');ax.set_xlim(-55,55);ax.set_ylim(110,-110)
    ax.set_xlabel('mm');ax.set_ylabel('mm');ax.set_title('Source layout + bounded gas-control regions\nVent sizes and mid-height remain priors')
    temperature=np.linspace(298.15,650.,300)
    for scene,label in [(base,'Previous FSRI hypothesis'),(transfer_r2(base),'Accepted UPV training R2')]:
        law=next(c for c in scene['cells'] if c['role']=='battery')['conversion']
        k=law['reaction_rate_per_s']*np.exp(law['reaction_activation_j_per_mol']/8.314462618*(1/law['reaction_reference_k']-1/temperature))
        axes[1].semilogy(temperature-273.15,k,label=label)
    axes[1].set_xlabel('Temperature, °C');axes[1].set_ylabel('R2 rate, 1/s');axes[1].grid(alpha=.2);axes[1].legend()
    axes[1].set_title('Only the R2 Arrhenius law is transferred\nNo imposed ignition or changed energy reserve')
    fig.savefig(out/'geometry-and-r2.png',dpi=160);plt.close(fig)


def comparison(out, runs, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True,constrained_layout=True)
    observed=np.array([data[f'T{i}_k'] for i in range(1,10)]).T-273.15
    for ax,name in zip(axes.flat,CASES):
        ax.fill_between(data['time_s'],np.nanmin(observed,axis=1),np.nanmax(observed,axis=1),color='black',alpha=.12,label='9 measured TC range')
        if name in runs:
            run=runs[name]
            for ident,values in run['cell_mean_temperature_k'].items():
                ax.plot(run['time_s'],np.array(values)-273.15,lw=.7,alpha=.7)
            ax.plot([],[],c='tab:blue',label='30 model battery means')
        ax.set_title(name+(' (not run)' if name not in runs else ''));ax.grid(alpha=.2);ax.legend(fontsize=7)
        ax.set_xlabel('Time from experiment start, s');ax.set_ylabel('Temperature, °C')
    fig.suptitle('Geometry / transferred R2: diagnostic only; TC-to-cell mapping unknown; no time shifts')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--base',type=Path,default=DEFAULT_BASE)
    ap.add_argument('--probe',type=Path,default=DEFAULT_PROBE)
    ap.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    ap.add_argument('--duration',type=float,default=5945.)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if not 1 <= args.duration <= 5945:raise ValueError('Outside published duration')
    base=json.loads(args.base.read_text());base['simulation']['duration_s']=args.duration
    base['failure_sample_times_s']=[t for t in base['failure_sample_times_s'] if t<=args.duration]
    base['failure_sample_times_s']=sorted(set(base['failure_sample_times_s']+[args.duration]))
    data,_=load();scenes=cases(base);settings.PROBE=args.probe.resolve()
    state=dict(status='running',pid=os.getpid(),native_sha256=sha(settings.PROBE),
        source_scenario=str(args.base.resolve()),source_scenario_sha256=sha(args.base),
        cases=args.cases,results={},fresh_from_zero=True,calibrated=False,validation_passed=False)
    save(out/'contract.json',dict(state,controller_sha256=sha(Path(__file__)),
        geometry_module_sha256=sha(Path(__file__).with_name('fsri30_reconstruction.py'))))
    geometry_plot(out,base,scenes['combined']);runs={}
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    atomic_save(out/'status.json',state)
    try:
        for name in args.cases:
            state['phase']=name;atomic_save(out/'status.json',state)
            folder=out/name;folder.mkdir();scene=scenes[name];save(folder/'scenario.json',scene)
            try:
                run=native(folder/'scenario.json',folder/'replay',dict(restart_contract=state['native_sha256'],
                    stop_after_first_rupture=False),args.duration*5.2+60.,minimum_speed=.2,rolling_minimum=.2)
                result=metrics(scene,run,data);result.update(scenario_sha256=sha(folder/'scenario.json'),
                    run_sha256=sha(folder/'replay/run.json.gz'))
                if result['minimum_regular_step_s'] < scene['simulation']['minimum_dt_s']-1e-9:
                    raise ValueError('Physical step below preserved floor')
                save(folder/'summary.json',result);runs[name]=run;state['results'][name]=result
            except Exception as exc:
                state['results'][name]=dict(status='failed',error=str(exc))
                partial=folder/'replay/incomplete-run.json.gz'
                if partial.exists():
                    with gzip.open(partial,'rt') as stream:runs[name]=json.load(stream)
            comparison(out,runs,data);atomic_save(out/'status.json',state)
        state['status']='complete' if all(r['status']=='complete' for r in state['results'].values()) else 'completed-with-failures'
        if state['status']=='complete' and args.duration==5945.:
            evidence=dict(result_path=str(out/'status.json'),fresh_from_zero=True,independent_validation=dict(accepted=False),
                source_training_only=True,chemistry_equivalence_confirmed=False)
            law=scenes['combined']['metadata']['r2_transfer']['parameters']
            state['r2_transfer_record']=record('constitutive_law','upv-to-fsri30-r2-hypothesis',law,evidence,
                compatibility=dict(model='r1-r2-r3',target_material='unidentified-18650',source_scenario_sha256=scenes['combined']['metadata']['r2_transfer']['source_scenario_sha256']),
                producer_core=state['native_sha256'],activate_development=False)
    except KeyboardInterrupt:state['status']='stopped'
    finally:
        atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

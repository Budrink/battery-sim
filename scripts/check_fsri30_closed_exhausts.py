#!/usr/bin/env python3
"""Closed-module A/B diagnostic, never a calibrated geometry replacement."""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import signal

import numpy as np

import calibrate_upv_r123 as settings
from calibrate_fsri30_bayes import atomic_save,read_run
from check_upv_restart import native
from fsri30_conditioned_start import prepare,same_series
from fsri30_data import save,sha
from fsri30_heat_path_audit import audit
from fsri30_objective import cascade
from model_evidence import record


def exits(scene):
    nodes={n['id'] for n in scene['gas_network']['compartments']}
    outer={n for n in nodes if n=='hood' or n.startswith('plume-')}
    cut=[]
    for e in scene['gas_network']['channels']:
        if (e['from'] in outer)==(e['to'] in outer):continue
        if e['kind']!='passage' or not e['from'].startswith('top-') or e['to'] not in outer:
            raise ValueError('Unresolved exterior crossing; cannot certify a closed module: '+e['id'])
        cut.append(e['id'])
    if len(cut)!=30:raise ValueError('Expected all thirty explicit module exits')
    return cut


def close_exhausts(scene):
    result=deepcopy(scene);cut=set(exits(scene))
    for e in result['gas_network']['channels']:
        if e['id'] in cut:e['area_m2']=0.
    result.setdefault('metadata',{}).update(closed_exhaust_diagnostic=True,
        sealed_exhaust_ids=sorted(cut),full_warmup_validation=False,
        closure_scope='All module-to-exterior mass and mixing paths; solid heat rejection unchanged')
    return result


def observations(scene):
    result=deepcopy(scene);result['gas_film_trace']=True
    select=result['process_history_selection']
    select['channel_prefixes']=sorted(set(select['channel_prefixes']+['exhaust-']))
    select['gas_prefixes']=sorted(set(select['gas_prefixes']+['base-','inside-B10-']))
    return result


def same_observations(reference,control,until):
    # New traces add base cavities/exhaust ledgers; NEVER drop an old field
    # when qualifying the unchanged open control against the archived run.
    control=deepcopy(control)
    def projection(a,b):
        return {k:projection(v,b[k]) for k,v in a.items()} if isinstance(a,dict) else b
    for section in ('temperature_k','gas','gas_observers','channels','chemistry','chemistry_zones','electrical','joule_energy_j'):
        if section in reference:control[section]=projection(reference[section],control[section])
    return same_series(reference,control,until)


def metrics(scene,run):
    ids=sorted(run['cell_mean_temperature_k']);t=np.array(run['time_s'])
    maxima=np.array(run['source_max_temperature_k']).reshape(len(t),len(ids))
    source=scene['metadata']['heated_component']
    peaks=sorted((max(run['cell_mean_temperature_k'][k]),k) for k in ids if k!=source)[::-1]
    leakage=[]
    for key in exits(scene):
        v=run['channels'][key]
        for field in ('transferred_mass_kg','transferred_chemical_j','transferred_oxidizer_kg',
                      'transferred_vapor_kg','transferred_latent_j','mixed_mass_kg','delivered_gas_energy_j'):
            deviation=max(abs(x-v[field][0]) for x in v[field])
            if deviation!=0.:leakage.append(dict(channel=key,field=field,change=deviation))
        if any(x!=0. for x in v['effective_area_m2'][1:]):leakage.append(dict(channel=key,error='nonzero open area'))
    heat=audit(scene,run)
    return dict(complete=not run.get('partial',False),advanced_to_s=run['advanced_to_s'],
        cascade=cascade(scene,run),neighbour_peaks=[dict(id=k,mean_c=temp-273.15,
            local_c=float(max(maxima[:,ids.index(k)]))-273.15) for temp,k in peaks[:6]],
        maximum_recorded_pressure_pa=max(max(v['pressure_pa']) for v in run['gas'].values()),
        pressure_scope='Recorded cavities including source pores, not a global all-voxel maximum',
        thermal_audit=heat,closed_cut_exact=not leakage,leakage=leakage,
        minimum_regular_step_s=run['solver_profile']['minimum_regular_step_s'],
        balances={k:v for k,v in run.items() if k.startswith('max_') and 'residual' in k},
        wall_s=run['wall_s'],realtime_ratio=(run['advanced_to_s']-run['started_at_s'])/run['wall_s'],
        conditioned_transport_diagnostic=True,full_warmup_validation=False,
        calibrated=False,validation_passed=False)


def plot(out,baseline,closed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ax,ident in zip(axes.flat,('B10','B11','B09','B03')):
        for run,label in [(baseline,'Open reference'),(closed,'Closed module')]:
            t=np.array(run['time_s']);ax.plot(t,np.array(run['cell_mean_temperature_k'][ident])-273.15,label=label)
        ax.set_title(ident);ax.set_ylabel('Mean temperature, °C');ax.grid(alpha=.2);ax.legend()
    for ax in axes[-1]:ax.set_xlabel('Time since experiment start, s')
    fig.suptitle('Conditioned boundary diagnostic: closure at 1755 s; NOT independent validation')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for run,label in [(baseline,'Open reference'),(closed,'Closed module')]:
        t=np.array(run['time_s'])
        cavities=[v for k,v in run['gas'].items() if k.startswith(('body-','top-'))]
        pressure=np.max(np.array([v['pressure_pa'] for v in cavities]),axis=0)
        axes[0].plot(t,pressure/1e5,label=label)
        q=run['process_totals']['gas_released_energy_j']
        axes[1].plot(t,(np.array(q)-q[0])/1000.,label=label)
    axes[0].set_ylabel('Max body/top pressure, bar absolute')
    axes[1].set_ylabel('All-gas R3 release since restart, kJ');axes[1].set_xlabel('Time since experiment start, s')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Boundary diagnostic: unchanged chemistry, no exterior mass/mixing paths in closed case')
    fig.tight_layout();fig.savefig(out/'gas-pressure.png',dpi=150);plt.close(fig)


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    source=args.reference.resolve();probe=args.probe.resolve();base=json.loads((source/'scenario.json').read_text())
    identity=dict(native_sha256=sha(probe),reference=str(source),reference_scenario_sha256=sha(source/'scenario.json'),
        diagnostic_only=True,closure_time_s=1755.,duration_s=base['simulation']['duration_s'])
    path=out/'contract.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Diagnostic contract changed; use new output')
    save(path,identity);state=dict(status='running',phase='prepare-same-core-start',pid=os.getpid(),**identity)
    atomic_save(out/'status.json',state);settings.PROBE=probe
    try:
        _,checkpoint,prefix=prepare(args.start,probe=probe)
        baseline=read_run(source)
        opened=observations(base);control=deepcopy(opened);control['simulation']['duration_s']=1800.
        control['failure_sample_times_s']=[t for t in control['failure_sample_times_s'] if t<=1800.]
        def replay(scene,folder,sealed=False):
            folder.mkdir(exist_ok=True);save(folder/'scenario.json',scene)
            if (folder/'replay/run.json.gz').exists():return read_run(folder)
            return native(folder/'scenario.json',folder/'replay',dict(restart_from=str(checkpoint),
                restart_contract=sha(probe),restart_transport_scenario=str(prefix/'scenario.json'),
                diagnostic_reactive_transit=True,diagnostic_closed_exhausts=sealed,
                stop_after_first_rupture=False),args.timeout,minimum_speed=.2,rolling_minimum=.2)
        state.update(phase='open-reference-identity');atomic_save(out/'status.json',state)
        check=same_observations(baseline,replay(control,out/'open-control'),1800.)
        save(out/'open-control-equivalence.json',check)
        if not check['accepted']:raise ValueError('Open control does not match archived best during main burst')
        closed=close_exhausts(opened);save(out/'closed/scenario.json',closed)
        state.update(phase='closed-module-full-suffix',active_folder=str(out/'closed'))
        atomic_save(out/'status.json',state)
        result=replay(closed,out/'closed',True);summary=metrics(closed,result)
        if not summary['closed_cut_exact']:raise ValueError('A supposedly closed exit leaked')
        if summary['minimum_regular_step_s']<.005-1e-10:raise ValueError('Physical minimum step violated')
        if result.get('max_oxidizer_residual_kg',0.)>1e-10:raise ValueError('Oxygen conservation failed')
        save(out/'summary.json',summary);plot(out,baseline,result)
        evidence=dict(result_path=str((out/'summary.json').resolve()),result_sha256=sha(out/'summary.json'),
            conditioned_transport_diagnostic=True,full_warmup_validation=False,
            note='Counterfactual closed boundary, not measured geometry; preserve all old vent history')
        ident=record('geometry_prior','fsri30-closed-exhaust-ablation',{'module_exits_closed':True},evidence,
            compatibility={'fixture':'fsri30-reconstruction-v1','stage':'conditioned-at-1755s'},
            producer_core=sha(probe),activate_development=False)
        state.update(status='complete',phase='closed-boundary-diagnostic-finished',summary=str(out/'summary.json'),
            evidence_record=ident,reacted_components=summary['cascade']['reacted_components'],
            hottest_neighbour_c=summary['neighbour_peaks'][0]['mean_c'])
    except KeyboardInterrupt:state.update(status='stopped',phase='interrupted')
    except Exception as exc:state.update(status='needs-attention',error=str(exc))
    finally:atomic_save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)
    return 0 if state['status']=='complete' else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('output','start','reference','probe'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--timeout',type=float,default=21600.);args=ap.parse_args()
    def stop(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    raise SystemExit(run(args))


if __name__=='__main__':main()

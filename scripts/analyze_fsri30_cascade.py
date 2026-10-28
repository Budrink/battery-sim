#!/usr/bin/env python3
"""Diagnose the completed cascade in an explicitly stopped fine-grid prefix.

No solver, chemistry or boundary changes; aligned shapes are diagnostic only.
Threshold intervals expose the 5 s observation uncertainty instead of treating
linear interpolation as an exact physical event clock.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from calibrate_fsri30_bayes import atomic_save
from check_fsri30_resolved_gas import compact,compare_mesh
from fsri30_data import sha
from fsri30_prefix_history import conversion_events,trim_observed,summary_at
from fsri30_spatial_patterns import event_patterns,compare_spatial_fields
from report_fsri30_full_mesh import prefix_check,phase_comparison,verify_provenance


def threshold_intervals(scene,raw):
    times=np.asarray(raw['time_s']);result={}
    for cell in scene['cells']:
        if cell['role']!='battery':continue
        law=cell['conversion'];ident=cell['id']
        if law.get('gas_yield_temperature_per_k',0.) or law.get('gas_yield_density_exponent',0.) or law['gas_yield']<=0.:
            raise ValueError('Threshold intervals require constant gas yield')
        reacted=np.asarray(raw['chemistry'][ident]['r2_generated_gas_kg'])/law['gas_yield']
        hit=np.flatnonzero(reacted>=.1*law['mass_kg'])
        if len(hit):
            i=int(hit[0]);result[ident]=[float(times[max(0,i-1)]),float(times[i])]
    return result


def event_shape_details(runs,events,skip_jump_s=10.):
    by={m:{e['component']:e['time_s'] for e in events[m]} for m in (20,10)}
    shared=sorted(by[20].keys()&by[10].keys())
    tau=np.arange(-60.,120.+1e-8,5.);far=np.abs(tau)>skip_jump_s;after=tau>skip_jump_s
    result=[]
    for ident in shared:
        curves={};peaks={}
        for mm in (20,10):
            times=np.asarray(runs[mm]['time_s']);values=np.asarray(runs[mm]['cell_mean_temperature_k'][ident])
            requested=by[mm][ident]+tau
            if requested[0]<times[0] or requested[-1]>times[-1]:
                raise ValueError('Shape comparison would extrapolate an unobserved tail')
            curves[mm]=np.interp(requested,times,values)
            observed=(times>=requested[0])&(times<=requested[-1])
            peaks[mm]=float(values[observed].max())
        a,b=curves[20],curves[10];relative=np.abs(a-b)/np.maximum(a,b)
        result.append(dict(component=ident,raw_arrival_delay_s=by[10][ident]-by[20][ident],
            aligned_all_rms_relative_kelvin=float(np.sqrt(np.mean(relative**2))),
            aligned_outside_jump_rms_relative_kelvin=float(np.sqrt(np.mean(relative[far]**2))),
            aligned_after_jump_rms_relative_kelvin=float(np.sqrt(np.mean(relative[after]**2))),
            recorded_peak_k=peaks,recorded_peak_relative_difference=abs(peaks[20]-peaks[10])/max(peaks.values())))
    return dict(rows=result,excluded_jump_half_width_s=skip_jump_s,
        scope='Own-event diagnostic, not an unshifted convergence acceptance or a new physical timing law')


def snapshot(raw,when):
    t=np.asarray(raw['time_s']);i=int(np.argmin(abs(t-when)))
    if abs(t[i]-when)>1e-7:raise ValueError('Snapshot time was not observed')
    result=dict(time_s=float(t[i]),battery_mean_k={k:v[i] for k,v in raw['cell_mean_temperature_k'].items()},
        gas_groups={name:{k:v[i] for k,v in group.items()} for name,group in raw['gas_groups'].items()},
        global_r3_j=raw['process_totals']['gas_released_energy_j'][i])
    keys=('transferred_mass_kg','mixed_mass_kg','transferred_chemical_j','delivered_gas_energy_j')
    result['signed_module_to_air_vent_ledgers']={k:sum(v[k][i] for name,v in raw['channels'].items() if name.startswith('vent-')) for k in keys}
    result['r2_generated_gas_kg']={k:v['r2_generated_gas_kg'][i] for k,v in raw['chemistry'].items()}
    return result


def plot_diagnosis(out,runs,events,details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    event_times={mm:{e['component']:e['time_s'] for e in events[mm]} for mm in (20,10)}
    for ident,color in (('B09','tab:blue'),('B11','tab:orange')):
        for mm,style in ((20,'-'),(10,'--')):
            t=np.asarray(runs[mm]['time_s']);v=np.asarray(runs[mm]['cell_mean_temperature_k'][ident])-273.15
            axes[0,0].plot(t,v,style,color=color,label=f'{ident}, {mm} mm')
            axes[0,1].plot(t-event_times[mm][ident],v,style,color=color,label=f'{ident}, {mm} mm')
    axes[0,0].set_xlim(2180.,2320.);axes[0,0].set_title('Actual time: first propagated events')
    axes[0,1].set_xlim(-25.,120.);axes[0,1].axvspan(-5.,5.,color='gray',alpha=.13,label='5 s recording scale')
    axes[0,1].set_title('Own-event aligned: diagnostic only')
    rows=details['rows'];x=np.arange(len(rows))
    axes[1,0].plot(x,[100*r['recorded_peak_relative_difference'] for r in rows],'o-',label='Recorded peak difference')
    axes[1,0].plot(x,[100*r['aligned_after_jump_rms_relative_kelvin'] for r in rows],'x--',label='After-jump aligned RMS')
    axes[1,0].set_xticks(x,[r['component'] for r in rows],rotation=90,fontsize=7)
    axes[1,0].set_ylabel('Relative to Kelvin, %');axes[1,0].set_title('Magnitude and post-event shape, all 30 cells')
    for mm,style in ((20,'-'),(10,'--')):
        axes[1,1].plot(runs[mm]['time_s'],np.asarray(runs[mm]['process_totals']['gas_released_energy_j'])/1e6,style,label=f'{mm} mm')
    axes[1,1].set_xlim(2180.,runs[10]['time_s'][-1]);axes[1,1].set_ylabel('R3 cumulative heat, MJ')
    axes[1,1].set_title('Delayed heat release catches up')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0,0].set_xlabel('s');axes[0,1].set_xlabel('s relative to own 10% R2 event')
    axes[0,0].set_ylabel('Battery mean, °C');axes[0,1].set_ylabel('Battery mean, °C');axes[1,1].set_xlabel('s')
    fig.suptitle('Frozen 20/10 mm cascade diagnosis; no solver correction or experimental validation')
    path=out/'cascade-diagnosis.png';fig.savefig(path,dpi=160);plt.close(fig)
    return str(path.resolve())


def analyze(source,fine,out):
    source,fine,out=Path(source).resolve(),Path(fine).resolve(),Path(out).resolve()
    launch=json.loads((source/'launch.json').read_text());contract=json.loads((source/'checks/contract.json').read_text())
    verify_provenance(source,launch,contract)
    state=json.loads((fine/'status.json').read_text())
    if state['status']!='stopped':raise ValueError('Expected the explicitly stopped reference')
    if state['native_sha256']!=launch['native_sha256'] or sha(fine/'scenario.json')!=sha(source/'checks/h-10mm/scenario.json'):
        raise ValueError('Native core or pinned physical scene differs')
    if any(sha(Path(p))!=v for p,v in state['pinned_files'].items()):raise ValueError('Pinned reference inputs changed')
    paths={20:source/'checks/h-20mm/replay/run.json.gz',10:fine/'replay/incomplete-run.json.gz'}
    raw={};scenes={};events={};runs={};intervals={}
    for mm,folder in ((20,source/'checks/h-20mm'),(10,fine)):
        with gzip.open(paths[mm],'rt') as f:raw[mm]=json.load(f)
        scenes[mm]=json.loads((folder/'scenario.json').read_text())
        if raw[mm]['started_at_s']!=0.:raise ValueError('Not a fresh trajectory')
        events[mm]=conversion_events(scenes[mm],raw[mm]);intervals[mm]=threshold_intervals(scenes[mm],raw[mm])
    end=float(raw[10]['time_s'][-1])
    for mm in (20,10):runs[mm]=trim_observed(compact(raw[mm]),end)
    source_id=scenes[10]['metadata']['heated_component']
    if any(len(events[m])!=30 for m in (20,10)):raise ValueError('The full 30-cell conversion cascade was not observed')
    full=event_patterns(runs,events,source=source_id);details=event_shape_details(runs,events)
    inversions=[]
    ordered=[e['component'] for e in events[20]];d10={e['component']:e['time_s'] for e in events[10]}
    for i,a in enumerate(ordered):
        for b in ordered[i+1:]:
            if d10[a]>d10[b]:
                certain=intervals[20][a][1]<intervals[20][b][0] and intervals[10][b][1]<intervals[10][a][0]
                inversions.append(dict(a=a,b=b,order_reversal_resolved_by_observations=bool(certain)))
    for row in full['events']:
        a,b=intervals[20][row['component']],intervals[10][row['component']]
        row['observed_event_brackets_s']={20:a,10:b};row['delay_bracket_s']=[b[0]-a[1],b[1]-a[0]]
    phases={name:phase_comparison(runs[20],runs[10],a,b) for name,a,b in
        (('before_cascade',0.,2200.),('cascade',2200.,2400.),('after_cascade',2400.,end))}
    with gzip.open(source/'checks/h-10mm/replay/incomplete-run.json.gz','rt') as f:old=compact(json.load(f))
    result=dict(status='stopped-cascade-compared',observed_through_s=end,
        fine_live_time_s=raw[10].get('live_state',{}).get('time_s'),fine_wall_s=raw[10]['wall_s'],
        full_requested_horizon_s=5945.,full_horizon_complete=False,native_sha256=launch['native_sha256'],
        previous_fine_prefix_repeat=prefix_check(old,runs[10]),
        global_comparison=compare_mesh(runs[20],runs[10],summary_at(runs[20],events[20]),summary_at(runs[10],events[10])),
        phases=phases,events=full,shape_details=details,observed_order_inversions=inversions,
        reaction_cascade_span_s={mm:events[mm][-1]['time_s']-next(e['time_s'] for e in events[mm] if e['component']!=source_id) for mm in (20,10)},
        snapshots={str(t):{mm:snapshot(raw[mm],t) for mm in (20,10)} for t in (1115.,1200.,1800.,2100.,2200.,2230.,2400.,end)},
        balances={mm:{k:v for k,v in raw[mm].items() if k.startswith('max_') and 'residual' in k} for mm in (20,10)},
        observation_spacing_s=float(np.median(np.diff(runs[10]['time_s']))),
        calibrated=False,validation_passed=False,patterns_promoted=False,
        interpretation='Timing, amplitude and local redistribution are separate. Aligned curves never replace unshifted acceptance. 10 mm is a comparison reference, not a proven continuum limit.',
        provenance=dict(controller_sha256=sha(Path(__file__)),raw_results={mm:dict(path=str(p),sha256=sha(p)) for mm,p in paths.items()}))
    out.mkdir(parents=True,exist_ok=False)
    spatial=compare_spatial_fields(scenes[20],scenes[10],source/'checks/h-20mm/gas-fields.bin',fine/'gas-fields.bin',end)
    atomic_save(out/'spatial-fields.json',spatial)
    result['spatial_fields']={k:v for k,v in spatial.items() if k!='history'}
    result['plot']=plot_diagnosis(out,runs,events,details)
    atomic_save(out/'report.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path);ap.add_argument('--fine',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();r=analyze(a.source,a.fine,a.output)
    print(json.dumps(dict(observed_through_s=r['observed_through_s'],reaction_cascade_span_s=r['reaction_cascade_span_s'],
        global_comparison=r['global_comparison'],shape_details=r['shape_details'],plot=r['plot']),indent=2))


if __name__=='__main__':main()

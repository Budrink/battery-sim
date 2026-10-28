"""Full frozen-pair comparison; propagation diagnostics cannot time-fit a pass."""
import argparse
import json
from pathlib import Path

from calibrate_fsri30_bayes import atomic_save
from check_fsri30_resolved_gas import compare_mesh,plot
from fsri30_data import load,sha
from fsri30_prefix_history import load_available,summary_at
from fsri30_spatial_patterns import compare_spatial_fields,event_patterns
from report_fsri30_full_mesh import aligned_times,phase_comparison,prefix_check,verify_provenance


def verify_fine_reference(source,fine):
    launch=json.loads((source/'launch.json').read_text())
    contract=json.loads((source/'checks/contract.json').read_text())
    verify_provenance(source,launch,contract)
    if sha(fine/'scenario.json') != sha(source/'checks/h-10mm/scenario.json'):
        raise ValueError('Fine reference changed the pinned 10 mm scenario')
    state=json.loads((fine/'status.json').read_text())
    if state['native_sha256'] != launch['native_sha256']:
        raise ValueError('Fine reference uses a different native core')
    if state.get('run_sha256') != sha(fine/'replay/run.json.gz'):
        raise ValueError('Fine reference raw result differs from its completion record')


def plot_patterns(out,runs,events,scene,patterns,spatial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    rows=patterns['events'];locations=scene['metadata']['approximate_positions_m']
    if rows:
        rank=np.arange(len(rows))
        for key,label,style in (('coarse_event_s','20 mm','o-'),('fine_event_s','10 mm','x--')):
            axes[0,0].plot(rank,[r[key] for r in rows],style,label=label,ms=4)
        axes[0,0].set_xticks(rank,[r['component'] for r in rows],rotation=90,fontsize=7)
        axes[0,0].set_ylabel('R2 10% conversion time, s');axes[0,0].legend()
        points=np.array([locations[r['component']] for r in rows])*1000.
        art=axes[0,1].scatter(points[:,0],points[:,1],c=[r['raw_delay_s'] for r in rows],cmap='coolwarm',s=190)
        for point,row in zip(points,rows):axes[0,1].text(point[0],point[1],row['component'],ha='center',va='center',fontsize=7)
        fig.colorbar(art,ax=axes[0,1],label='10 mm event delay relative to 20 mm, s')
    axes[0,0].set_title('Raw event arrival times — no time shifts')
    axes[0,1].set_title('Where the propagation front differs');axes[0,1].set_aspect('equal')
    axes[0,1].set_xlabel('mm');axes[0,1].set_ylabel('mm')
    tau=np.arange(-60.,120.+1e-9,5.)
    for mm,style in ((20,'-'),(10,'--')):
        by={e['component']:e['time_s'] for e in events[mm]};r=runs[mm];t=np.array(r['time_s'])
        shapes=[]
        for row in rows:
            event=by[row['component']]
            if event+tau[0]>=t[0] and event+tau[-1]<=t[-1]:
                shapes.append(np.interp(event+tau,t,r['cell_mean_temperature_k'][row['component']])-273.15)
        if shapes:
            q=np.percentile(shapes,[25.,50.,75.],axis=0)
            line,=axes[1,0].plot(tau,q[1],style,label=f'{mm} mm median, IQR shade')
            axes[1,0].fill_between(tau,q[0],q[2],color=line.get_color(),alpha=.12)
    axes[1,0].set_title('Own-event aligned shapes — diagnostic only')
    axes[1,0].set_xlabel('s relative to own R2 10% event');axes[1,0].set_ylabel('Battery mean, °C');axes[1,0].legend(fontsize=8)
    history=spatial['history'];t=[r['time_s'] for r in history]
    for key,label in (('air_fuel_l1_relative','Fuel field L1 / larger inventory'),
                      ('normalized_fuel_distribution_total_variation','Normalized fuel distribution TV'),
                      ('air_volume_rms_relative_kelvin','Air volume RMS temperature / K')):
        axes[1,1].plot(t,[100.*r[key] if r[key] is not None else np.nan for r in history],label=label)
    axes[1,1].set_xlabel('Unshifted time, s');axes[1,1].set_ylabel('%');axes[1,1].legend(fontsize=7)
    axes[1,1].set_title('Spatial redistribution on common physical boxes')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Frozen 20/10 mm propagation patterns; CLOSED-air numerical diagnostic, not experimental validation')
    path=out/'propagation-patterns.png';fig.savefig(path,dpi=160);plt.close(fig)
    return str(path.resolve())


def write_report(source,fine,out):
    source,fine,out=Path(source),Path(fine),Path(out)
    verify_fine_reference(source,fine)
    out.mkdir(parents=True,exist_ok=False)
    runs={};events={};metadata={};scenes={}
    for mm,folder in ((20,source/'checks/h-20mm'),(10,fine)):
        runs[mm],events[mm],metadata[mm],scenes[mm]=load_available(folder)
        if metadata[mm]['partial'] or abs(runs[mm]['time_s'][-1]-5945.)>1e-7:
            raise ValueError('Cannot claim a full pair from an incomplete reference')
    aligned_times(runs[20],runs[10])
    old,_,_,_=load_available(source/'checks/h-10mm')
    continuity=prefix_check(old,runs[10])
    summaries={mm:summary_at(runs[mm],events[mm]) for mm in (20,10)}
    comparison=compare_mesh(runs[20],runs[10],summaries[20],summaries[10])
    windows={}
    for name,start,end in (('preheat',0.,1075.),('first_burst',1075.,1230.),
            ('between_bursts',1230.,2200.),('cascade',2200.,2500.),('late_evolution',2500.,5945.)):
        windows[name]=phase_comparison(runs[20],runs[10],start,end)
    for start in range(0,5945,300):
        windows[f'fixed-{start}s']=phase_comparison(runs[20],runs[10],float(start),min(5945.,float(start+300)))
    result=dict(status='full-pair-compared',duration_s=5945.,sources=metadata,
        analysis_sha256=sha(Path(__file__)),prefix_repeat=continuity,global_comparison=comparison,
        phase_comparisons=windows,conversion_events=events,
        conversion_definition='10% R2 converted mass; interpolated observed ledger, not a measured explosion',
        spatially_converged=bool(continuity['accepted'] and comparison['accepted'] and all(w['accepted'] for w in windows.values())),
        calibrated=False,validation_passed=False,patterns_promoted=False,
        note='Frozen CLOSED-air numerical check, not experimental validation. No time shift in acceptance metrics.')
    result['propagation_patterns']=event_patterns(runs,events)
    spatial=compare_spatial_fields(scenes[20],scenes[10],source/'checks/h-20mm/gas-fields.bin',fine/'gas-fields.bin',5945.)
    atomic_save(out/'spatial-fields.json',spatial)
    result['spatial_fields']={k:v for k,v in spatial.items() if k!='history'}
    result['spatial_analysis_sha256']=sha(Path(__file__).with_name('fsri30_spatial_patterns.py'))
    result['macroscopic_comparison_accepted']=result['spatially_converged']
    # Do not call a visible gas redistribution mismatch converged merely because
    # a large cold outer volume dilutes the RMS error. Keep this extra field
    # qualification separate from the historical macro acceptance definition.
    result['field_pattern_qualification']=dict(tolerance=.05,
        accepted=bool(spatial['maximum_air_fuel_l1_relative']<=.05 and spatial['maximum_air_oxygen_l1_relative']<=.05
            and spatial['air_volume_rms_relative_kelvin']<=.05 and (spatial['same_node_maximum_local_relative_kelvin'] or 0.)<=.05),
        scope='Conservative fuel/O2 restriction and same-volume gas response; volume-averaged temperature is not an energy balance')
    result['spatially_converged']=bool(result['macroscopic_comparison_accepted'] and result['field_pattern_qualification']['accepted'])
    result['pattern_plot']=plot_patterns(out,runs,events,scenes[20],result['propagation_patterns'],spatial)
    data,_=load();plot(out,{f'{mm} mm':r for mm,r in runs.items()},data)
    result['plot']=str((out/'comparison.png').resolve())
    atomic_save(out/'report.json',result)
    return dict(path=str((out/'report.json').resolve()),plot=result['plot'],
        spatially_converged=result['spatially_converged'],global_comparison=comparison)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--fine',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    print(json.dumps(write_report(args.source,args.fine,args.output),indent=2))


if __name__=='__main__':main()

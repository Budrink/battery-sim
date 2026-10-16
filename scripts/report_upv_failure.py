#!/usr/bin/env python3
"""Plot the failed/successful post-event trial without editing fit parameters."""
import argparse
import json
import numpy as np
from calibrate_upv import save
from calibrate_upv_failure import OUTPUT, CHANNELS, load_record, event_marker, digest, fingerprint

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=__import__('pathlib').Path,default=OUTPUT);a=p.parse_args();out=a.output
    profile=json.loads((out/'profile.json').read_text())
    if profile['model_sha256']!=fingerprint():raise ValueError('Changed model; refit before reporting')
    reports=[json.loads((out/(k+'.json')).read_text()) for k in ('fit','validate')]
    for r in reports:
        if r['frozen_profile_sha256']!=digest(out/'profile.json') or r['parameters']!=profile['parameters'] or r['validation_refitted']:raise ValueError('Mixed fit/validation provenance')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,6,figsize=(21,7),constrained_layout=True)
    for row,r in enumerate(reports):
        t=np.asarray(r['result']['time_s'])-r['meta']['event']['bracket_width_s']
        for col,k in enumerate(CHANNELS):
            ax=axes[row,col];y=np.array([np.nan if v is None else v for v in r['observed_k'][k]])-273.15
            ax.plot(t,y,color='#273c75',lw=1.4,label='Experiment');ax.plot(t,np.array(r['result']['temperature_k'][k])-273.15,color='#d35400',ls='--',label='CUDA trial')
            ax.set_title(f'{k}: RMSE {r["metrics"]["per_channel"][k]["rmse_k"]:.1f} K');ax.grid(alpha=.2);ax.set_xlabel('Seconds after primary event')
            if col==0:ax.set_ylabel(r['experiment']+'\nTemperature, °C');ax.legend(fontsize=8)
    fig.suptitle('First post-event trial: uniform finite-energy C2 pulse + two switched contacts\nAll capacities and intact links frozen. Ambient/cooling measured. Model NOT accepted.',fontsize=13)
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True);events={}
    for ax,name in zip(axes,['single-cell','no-cooling','active-cooling']):
        d,record=load_record(name);t=d['time_s']
        if name=='single-cell':
            v=d['U115_v'];ambient=d['T109_k']
            candidates=np.flatnonzero((v[1:]<4.2)&(v[:-1]>4.2)&(np.diff(ambient)>50))+1
            if len(candidates)!=1:raise ValueError('Ambiguous single-cell voltage/thermal event')
            i=int(candidates[0]);event=dict(last_pre_event_s=float(t[i-1]),first_event_s=float(t[i]),bracket_width_s=float(t[i]-t[i-1]),method='Voltage falls below 4.2 V while ambient rises >50 K in one scan; event bracket, not an ignition criterion')
            keys=('101','102','103','104','109')
        else:event=event_marker(d);i=event['index'];keys=('103','104','105','106','108','109')
        selected=(t>=t[i]-5)&(t<=t[i]+60)
        for k in keys:ax.plot(t[selected]-t[i],d[f'T{k}_k'][selected]-273.15,label=k+(' ambient' if k=='109' else ''))
        ax.set_title(name);ax.set_xlabel('Seconds relative to first event scan');ax.set_ylabel('Raw sensor temperature, °C');ax.grid(alpha=.2);ax.legend(fontsize=8)
        event['source_csv_sha256']=record['canonical_sha256'];event['peaks_first_60s_c']={k:float(np.nanmax(d[f'T{k}_k'][(t>=t[i])&(t<=t[i]+60)])-273.15) for k in keys}
        events[name]=event
    fig.suptitle('Raw event observations: surface spikes and ambient spikes are not separately identified bulk temperatures')
    fig.savefig(out/'event-observations.png',dpi=150);plt.close(fig)
    assessment=dict(status='rejected_as_complete_destruction_calibration',default_scene_replaced=False,
        all_capacities_fixed=True,validation_refitted=False,profile_sha256=digest(out/'profile.json'),
        fit_metrics=reports[0]['metrics'],validation_metrics=reports[1]['metrics'],events=events,
        numerical_checks={r['experiment']:dict(energy_residual_j=r['result']['energy_residual_j'],
            half_dt_max_change_k=r['time_check_max_change_k'],export_identical=r['export_identical']) for r in reports},
        conclusion='Simple solid-only pulse/contact-switch trial fails; fitted E/duration/h are diagnostic compromise, NOT identified material properties.',
        next_unmodeled_blocks=['Localized vent/gas heat transfer and its energy partition',
            'Post-event sensor contact/measurement operator', 'Spatial source and irreversible internal structure changes'],
        limits='Measured event timing and ambient/cooling boundaries; no autonomous overcharge prediction or secondary-runaway validation; isolated witness spike is not proof of cascade.')
    save(out/'assessment.json',assessment)
    print(json.dumps({k:assessment[k] for k in ('status','numerical_checks')},indent=2))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Compare two links-only trials without selecting or refitting parameters."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from calibrate_upv import model_hash, save
from check_upv_core_shell import compare
from generate_upv import ROOT, TARGETS, load_case

def main():
    plain=ROOT/'data/calibration/upv-links-3x3x3'
    contrast=ROOT/'data/calibration/upv-links-contrast-3x3x3'
    profiles={};reports={}
    for name,directory in (('references',plain),('contrast',contrast)):
        profile=json.loads((directory/'profile.json').read_text());profiles[name]=profile
        digest=hashlib.sha256((directory/'profile.json').read_bytes()).hexdigest()
        reports[name]={}
        for mode in ('fit','validate'):
            report=json.loads((directory/f'{mode}.json').read_text())
            if report['frozen_profile_sha256']!=digest or report['model_sha256']!=profile['model_sha256']:
                raise ValueError('Report/profile mismatch')
            if not report['all_capacities_fixed'] or report['validation_refitted']:raise ValueError('Fit protocol mismatch')
            for obj in report['result']['applied_model']['objects']:
                if 'capacity_table' in obj:raise ValueError('Variable capacity in links-only report')
            reports[name][mode]=report
    if profiles['contrast']['model_sha256']!=model_hash():raise ValueError('Latest native model changed')
    # The five-parameter trial is a historical profile from before adding the
    # opt-in contrast knob; do not rewrite its original binary/source hashes.
    summary=dict(status='exploratory_tradeoff',all_capacities_fixed=True,capacity_trial_active=False,
        default_scene_replaced=False,validation_refitted=False,experiments={},
        parameters={k:p['parameters'] for k,p in profiles.items()},
        near_bounds={k:p['near_bounds'] for k,p in profiles.items()},
        interpretation='Lower calibration RMSE and worst validation error; overall validation RMSE essentially unchanged, three channels worse. Not a universal improvement or identified material law.',
        caveat=profiles['contrast']['protocol']['caveat'],
        development_note=profiles['contrast']['protocol']['development_note'])
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    errors,eaxes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    labels={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS'}
    for row,mode in enumerate(('fit','validate')):
        r=reports['contrast'][mode];old=reports['references'][mode];experiment=r['experiment']
        if compare(r['baseline_result'],old['baseline_result'])['max_change_k']>1e-10:raise ValueError('Inconsistent baselines')
        observed,record,_=load_case(ROOT/'data/experiments/upv',experiment)
        if r['source_csv_sha256']!=record['canonical_sha256']:raise ValueError('Observed data changed')
        summary['experiments'][experiment]=dict(before=r['baseline_metrics'],references=old['metrics'],contrast=r['metrics'],
            reference_time_check=old['time_check'],contrast_time_check=r['time_check'])
        tracks=[('Before',r['baseline_result'],'#187eae','--'),('Contact refit',old['result'],'#939393',':'),
                ('With internal ΔT',r['result'],'#c95828','-')]
        tm=np.asarray(observed['time_s']);mask=tm<=r['result']['time_s'][-1]
        for ax,ea,ch in zip(axes[row],eaxes[row],TARGETS):
            ax.plot(tm[mask],np.asarray(observed[f'T{ch}_k'])[mask]-273.15,color='#20242a',label='Measured',lw=1.7)
            for label,run,color,style in tracks:
                t=np.asarray(run['time_s']);temperature=np.asarray(run['temperature_k'][ch])
                ax.plot(t,temperature-273.15,label=label,color=color,ls=style,lw=1.3)
                error=temperature-np.interp(t,tm,observed[f'T{ch}_k']);ea.plot(t,error,label=label,color=color,ls=style,lw=1.3)
            ax.set(title=f'{experiment}: {labels[ch]}',xlabel='Time, s',ylabel='Temperature, °C')
            ea.set(title=f'{experiment}: {labels[ch]}',xlabel='Time, s',ylabel='Model − measurement, K')
            ea.axhline(0,color='#20242a',lw=.7)
            for a in (ax,ea):a.grid(alpha=.2);a.legend(fontsize=8)
    fig.suptitle('Connectivity-only calibration, 3×3×3 — ALL capacities fixed\nFit: no cooling; frozen check: active cooling. Conditional pre-event model; numbering unconfirmed.',fontsize=12)
    errors.suptitle('Residuals: smaller worst error, not a uniform improvement across sensors\nNo capacity, boundary, geometry or sensor fitting.',fontsize=12)
    fig.savefig(contrast/'comparison.png',dpi=160);errors.savefig(contrast/'residuals.png',dpi=160)
    plt.close(fig);plt.close(errors)
    save(contrast/'assessment.json',summary)
    print(json.dumps({k:{stage:v[stage]['rmse_k'] for stage in ('before','references','contrast')} for k,v in summary['experiments'].items()},indent=2))

if __name__=='__main__':main()

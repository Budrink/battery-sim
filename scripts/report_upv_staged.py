#!/usr/bin/env python3
"""Report both frozen stages; no fitting or choice based on validation data."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from calibrate_upv import model_hash, save
from generate_upv import ROOT, TARGETS, load_case

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'data/calibration/upv-staged-3x3x3')
    directory=parser.parse_args().directory
    profile=json.loads((directory/'profile.json').read_text())
    digest=hashlib.sha256((directory/'profile.json').read_bytes()).hexdigest()
    if profile['model_sha256']!=model_hash():raise ValueError('Model changed after fit')
    reports=[json.loads((directory/f'{kind}.json').read_text()) for kind in ('fit','validate')]
    for r in reports:
        if r['frozen_profile_sha256']!=digest or r['model_sha256']!=profile['model_sha256']:raise ValueError('Unfrozen comparison')
    summary=dict(status='exploratory_bound_limited',frozen_profile_sha256=digest,default_scene_replaced=False,
        validation_refitted=False,conductivity_parameters=profile['conductivity']['parameters'],
        capacity_parameters=profile['capacity']['parameters'],
        near_bounds={k:profile[k]['near_bounds'] for k in ('conductivity','capacity')},experiments={},
        caveat=profile['protocol']['caveat'],capacity_identified=False,
        interpretation='Transport gives most improvement; capacity correction barely improves the second case, hits both bounds and worsens three validation channels. Do not infer measured C(T) or defect dependence.')
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    labels={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS'}
    for row,r in enumerate(reports):
        experiment=r['experiment'];observed,*_=load_case(ROOT/'data/experiments/upv',experiment)
        tm=np.asarray(observed['time_s']);mask=tm<=r['capacity']['result']['time_s'][-1]
        summary['experiments'][experiment]={k:r[k]['metrics'] for k in ('baseline','conductivity','capacity')}
        summary['experiments'][experiment]['time_step_checks']={k:r[k]['time_check'] for k in ('conductivity','capacity')}
        regions=r['capacity']['result']['battery_regions']
        summary['experiments'][experiment]['battery_node_temperature_range_c']=[
            min(t for cell in regions.values() for region in cell.values() for t in region['min_temperature_k'] if t is not None)-273.15,
            max(t for cell in regions.values() for region in cell.values() for t in region['max_temperature_k'] if t is not None)-273.15]
        for ax,ch in zip(axes[row],TARGETS):
            ax.plot(tm[mask],np.asarray(observed[f'T{ch}_k'])[mask]-273.15,label='Measured',color='#20242a',lw=1.8)
            for kind,label,color,style in (('baseline','Before','#999999',':'),('conductivity','G fit, fixed C','#187eae','--'),('capacity','G frozen, C fit','#c95828','-')):
                run=r[kind]['result'];ax.plot(run['time_s'],np.asarray(run['temperature_k'][ch])-273.15,label=label,color=color,ls=style,lw=1.3)
            ax.set(title=f'{experiment}: {labels[ch]}\nFinal RMSE {r["capacity"]["metrics"]["per_channel"][ch]["rmse_k"]:.2f} K',xlabel='Time, s',ylabel='Temperature, °C')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('3×3×3: transport first, storage second — CUDA in both stages\nSecond experiment excluded from fitting; topology numbering remains unconfirmed.',fontsize=12)
    fig.savefig(directory/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    p=profile['conductivity']['parameters'];tc=np.linspace(25,75,300)
    gknots=[0.,p['internal_log_35'],p['internal_log_50'],p['internal_log_70']]
    def factor(x):return .2+2.8/(1.+np.exp(-(np.log(.8/2.)+np.asarray(x))))
    axes[0].plot(tc,factor(np.interp(tc,[25,35,50,70],gknots)),color='#187eae')
    axes[0].scatter([25,35,50,70],factor(gknots),color='#187eae')
    axes[0].set(xlabel='Donor temperature, °C',ylabel='Internal G / Gref',title='Internal connectivity, not measured bulk k(T)')
    c=profile['capacity']['parameters'];cknots=[1.,c['capacity_45'],c['capacity_70']]
    axes[1].plot(tc,np.interp(tc,[25,45,70],cknots),color='#c95828')
    axes[1].scatter([25,45,70],cknots,color='#c95828')
    axes[1].axhline(.8,color='gray',ls=':');axes[1].axhline(1.2,color='gray',ls=':')
    axes[1].set(xlabel='Node temperature, °C',ylabel='C / Cref',title='Exploratory capacity: both limits reached',ylim=(.75,1.25))
    for ax in axes:ax.axhline(1,color='gray',lw=.7);ax.axvspan(70,75,color='gray',alpha=.1);ax.grid(alpha=.2)
    fig.savefig(directory/'constitutive-laws.png',dpi=160);plt.close(fig)
    save(directory/'assessment.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

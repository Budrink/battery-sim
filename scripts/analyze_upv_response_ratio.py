#!/usr/bin/env python3
"""Curve-ratio diagnosis on the 3^3 constant-C, constant-internal-G baseline.

Ratios of temperature rises are empirical OUTPUT corrections, not recovered
heat capacities or conductances. A face-gradient ratio is a conductance proxy
only under an additional equal-through-flux assumption, not verified here.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from calibrate_upv import Worker, save, verify_fixed_properties
from calibrate_upv_internal import sha
from check_upv_core_shell import compare
from generate_upv import ROOT, TARGETS, load_case

LABELS={'101':'C1 / ISS (hot face)','102':'C1 / TIM (cold face)',
        '107':'C3 / TIM (hot face)','108':'C3 / ISS (cold face)'}


def rise_ratio(model,observed,min_rise=1.):
    model=np.asarray(model);observed=np.asarray(observed)
    dm=model-model[0];de=observed-observed[0]
    mask=(dm>=min_rise)&(de>=min_rise)&np.isfinite(dm)&np.isfinite(de)
    ratio=np.full(dm.shape,np.nan);ratio[mask]=de[mask]/dm[mask]
    return ratio,mask


def sample_function(x,y,knots):
    x=np.asarray(x);y=np.asarray(y);valid=np.isfinite(x)&np.isfinite(y)
    x=x[valid];y=y[valid]
    # Do not silently sort a cooling/heating loop into a single-valued law.
    if len(x)<2 or np.any(np.diff(x)<=0):return None
    return [[float(k),float(np.interp(k,x,y))] for k in knots if x[0]<=k<=x[-1]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=('no-cooling','active-cooling'),default='no-cooling')
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/calibration/upv-response-ratio')
    a=p.parse_args();ex=a.experiment
    scenario_path=ROOT/f'data/calibration/upv-core-shell-check/{ex}-3x3x3-scenario.json'
    scene=json.loads(scenario_path.read_text())
    for c in scene['cells']:
        if c['role']=='battery' and (c['thermal']['divisions']!=[3,3,3] or 'bulk_connectivity' in c['thermal']):
            raise ValueError('Requires the 3^3 baseline without an internal temperature-response law')
    worker=Worker(a.probe,scenario_path)
    try:run=worker.run({});verify_fixed_properties(run,scene)
    finally:worker.close()
    recorded=json.loads((ROOT/f'data/calibration/upv-core-shell-check/{ex}-3x3x3.json').read_text())['result']
    if compare(run,recorded)['max_change_k']>1e-7:raise ValueError('Baseline changed')
    observed,record,_=load_case(ROOT/'data/experiments/upv',ex)
    if scene['metadata']['canonical_sha256']!=record['canonical_sha256']:raise ValueError('Data mismatch')
    t=np.asarray(run['time_s']);obs={c:np.interp(t,observed['time_s'],observed[f'T{c}_k']) for c in TARGETS}
    mod={c:np.asarray(run['temperature_k'][c]) for c in TARGETS}
    out=dict(kind='empirical-output-ratio-diagnostic',experiment=ex,source_csv_sha256=record['canonical_sha256'],
             scenario_sha256=sha(scenario_path),probe_sha256=sha(a.probe),script_sha256=sha(__file__),
             heat_capacity_fixed=True,internal_connectivity_fixed=True,external_contacts_refitted=False,
             ratio='(T_exp(t)-T_exp(0))/(T_model(t)-T_model(0))',
             min_rise_k=1.,time_s=list(t),channels={},face_pairs={},
             caveat='No unique C(T) or G(T) identification by dividing temperature curves. Ratios below are diagnostics.',
             result=run)
    fig,axes=plt.subplots(3,4,figsize=(15,10),constrained_layout=True)
    for col,c in enumerate(TARGETS):
        ratio,mask=rise_ratio(mod[c],obs[c]);mask &= t>=60
        axes[0,col].plot(t,obs[c]-273.15,c='#254f9e',label='Experiment')
        axes[0,col].plot(t,mod[c]-273.15,c='#dd6632',ls='--',label='Constant C, internal G')
        axes[0,col].set(title=LABELS[c],ylabel='Temperature, °C',xlabel='Time, s');axes[0,col].legend(fontsize=8)
        axes[1,col].plot(t[mask],ratio[mask],c='#754c9b')
        axes[1,col].axhline(1,c='#555555',ls=':',lw=1)
        axes[1,col].set(xlabel='Time, s',ylabel='Experimental rise / modeled rise')
        axes[2,col].plot(mod[c][mask]-273.15,ratio[mask],c='#208875')
        axes[2,col].axhline(1,c='#555555',ls=':',lw=1)
        axes[2,col].set(xlabel='Modeled face temperature, °C',ylabel='Empirical correction R(T_model)')
        out['channels'][c]=dict(label=LABELS[c],final_model_temperature_k=float(mod[c][-1]),
            final_experimental_temperature_k=float(obs[c][-1]),
            final_rise_ratio=float(ratio[-1]) if mask[-1] else None,
            ratio=[float(v) if valid else None for v,valid in zip(ratio,mask)],
            empirical_function=dict(input='Modeled surface temperature (not donor temperature)',
                temperature_k=list(mod[c][mask]),factor=list(ratio[mask]),
                warning='Sampled parametric curve; may be multivalued during cooling, do not extrapolate as a material law.'))
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle(f'{ex}: divide temperature RISES, not absolute temperatures\nC = 412.87 J/K per battery; no internal quadratic correction. Ratios are not identified C(T) or G(T).',fontsize=12)
    a.output_dir.mkdir(parents=True,exist_ok=True);fig.savefig(a.output_dir/f'{ex}-ratios.png',dpi=150)
    pairs,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for col,(cell,hot,cold) in enumerate((('C1','101','102'),('C3','107','108'))):
        dm=mod[hot]-mod[cold];de=obs[hot]-obs[cold]
        mask=(t>=60)&(dm>=1)&(de>=1);proxy=np.full(t.shape,np.nan);proxy[mask]=dm[mask]/de[mask]
        axs[0,col].plot(t,de,label='Experimental face contrast',c='#254f9e')
        axs[0,col].plot(t,dm,label='Modeled face contrast',c='#dd6632',ls='--')
        axs[0,col].set(title=cell,xlabel='Time, s',ylabel='Hot face − cold face, K');axs[0,col].legend(fontsize=8)
        axs[1,col].plot((mod[hot][mask]+mod[cold][mask])/2-273.15,proxy[mask],c='#208875')
        axs[1,col].axhline(1,c='#555555',ls=':',lw=1)
        axs[1,col].set(xlabel='Mean modeled face temperature, °C',ylabel='Modeled contrast / experimental contrast')
        out['face_pairs'][cell]=dict(hot_channel=hot,cold_channel=cold,
            final_contrast_ratio=float(proxy[-1]) if mask[-1] else None,
            proxy=[float(v) if valid else None for v,valid in zip(proxy,mask)],
            provisional_function=dict(argument='Mean MODELED face temperature, degrees C',
                knots=sample_function((mod[hot]+mod[cold])/2-273.15,proxy,(30,35,40,45,50)),
                interpolation='piecewise linear; no extrapolation; null means no single-valued curve',
                identified_material_law=False),
            interpretation='G_needed/G_model ONLY if through-flux were unchanged; internal storage/contact feedback not removed.')
    for ax in axs.flat:ax.grid(alpha=.2)
    pairs.suptitle('Face-contrast ratio: a starting diagnostic for internal transfer\nNot an exact reconstruction: transient heat storage and changing boundary flux remain.',fontsize=12)
    pairs.savefig(a.output_dir/f'{ex}-face-contrast.png',dpi=150)
    save(a.output_dir/f'{ex}.json',out)
    print(json.dumps(dict(experiment=ex,final_rise_ratios={c:v['final_rise_ratio'] for c,v in out['channels'].items()},
                          final_contrast_ratios={c:v['final_contrast_ratio'] for c,v in out['face_pairs'].items()}),indent=2))


if __name__=='__main__':main()

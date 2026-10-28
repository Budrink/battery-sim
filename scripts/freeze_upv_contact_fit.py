"""Freeze a completed contact fit, check dt/2, then replay held-out data once.

No optimization or validation-dependent parameter selection takes place here.
All errors use the same uniformly sampled 600-second charging window.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if ROOT==Path('/') or not (ROOT/'scripts').exists():
    ROOT=Path('/home/konstantin/work/battery-sim')
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_upv_r123 as cal
import upv_r123_data as dataio
import upv_r123_model as model

LABELS={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM',
        '108':'C3 / ISS','C2_TIM':'C2 / TIM','C2_ISS':'C2 / ISS'}
CONTACTS={'iss_h','tim_h','case_h','battery_iss_factor','battery_tim_factor','surface_iss'}


def read(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def save_run(path,run):
    with gzip.open(path,'wt') as stream:json.dump(run,stream,allow_nan=False)


def metrics(run,meta,onset):
    at=onset+np.arange(0.,600.001,5.)
    if run['time_s'][-1]<at[-1]-1e-5:raise ValueError('Incomplete charging window')
    obs=dataio.observed(meta['data'],at);result={};errors=[]
    for key in LABELS:
        err=np.interp(at,run['time_s'],run['temperature_k'][key])-obs[key]
        result[key]=dict(rmse_k=float(np.sqrt(np.nanmean(err**2))),end_bias_k=float(err[-1]))
        errors.extend(err[np.isfinite(err)].tolist())
    dp=np.interp(at,run['time_s'],dataio.chamber_pressure_pa(run))/1e5-obs['pressure_bar']
    return dict(channels=result,rmse_all_six_k=float(np.sqrt(np.mean(np.array(errors)**2))),
                pressure_rmse_bar=float(np.sqrt(np.nanmean(dp**2))),wall_s=run['wall_s'],
                balances={k:run[k]for k in ('max_thermal_residual_j','max_chemical_residual_j','max_mass_residual_kg')})


def compare_plot(run,old,meta,onset,name,path):
    at=onset+np.arange(0.,600.001,1.);obs=dataio.observed(meta['data'],at)
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
    for ax,key in zip(axes.flat,LABELS):
        ax.plot(at-onset,obs[key]-273.15,color='#d95f02',label='Эксперимент')
        ax.plot(np.array(old['time_s'])-onset,np.array(old['temperature_k'][key])-273.15,
                color='.55',ls='--',label='До подбора контактов')
        ax.plot(np.array(run['time_s'])-onset,np.array(run['temperature_k'][key])-273.15,
                color='#0072b2',label='После подбора контактов')
        ax.set_title(LABELS[key]);ax.set_xlim(0,600);ax.grid(alpha=.2)
        ax.set_xlabel('После включения тока, с');ax.set_ylabel('Температура, °C')
    axes[0,0].legend()
    title='Калибровка, без охлаждения' if name=='no-cooling' else 'Валидация, жидкостное охлаждение — без подгонки'
    fig.suptitle(title+'\n600 с нагрева; химия, электрические параметры и свойства материалов фиксированы')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(path,dpi=140);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--activate',action='store_true')
    args=ap.parse_args();args.fit=args.fit.resolve();args.output=args.output.resolve()
    state=json.loads((args.fit/'status.json').read_text())
    fingerprint=cal.fingerprint()
    if state['status']!='complete' or not state['diagnostics']['full']:raise ValueError('Completed fit required')
    if state['fingerprint']!=fingerprint:raise ValueError('Fit fingerprint is stale')
    pointer_path=ROOT/'data/calibration/upv-charge-degradation-v5/current-profile.json'
    previous=json.loads(pointer_path.read_text());prior=json.loads((ROOT/previous['parameters_file']).read_text())['parameters']
    p=state['parameters']
    for key,value in prior.items():
        if key not in CONTACTS and p[key]!=value:raise ValueError('Frozen parameter changed: '+key)
    if p['surface_iss']!=0:raise ValueError('Healthy ISS thermal suppression must remain disabled')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Refusing to overwrite an existing verification')
    args.output.mkdir(parents=True,exist_ok=True)
    phash=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    frozen=args.output/'frozen-parameters.json'
    cal.save(frozen,dict(parameters=p,parameters_sha256=phash,model_fingerprint=fingerprint,
        calibration_source=str(args.fit.relative_to(ROOT)),previous_profile=previous['parameters_file'],
        charge_window_s=600.,full_explosion_calibrated=False,validation_used_for_selection=False,
        scope='Shared material-pair contacts only; R1/R2/R3, electrical, bulk C/k and numerical solver frozen'))
    cal.save(args.output/'previous-profile.json',previous)
    run=read(args.fit/'run.json.gz');scene,meta=model.build('no-cooling',p,warmup=True,duration=626.956)
    onset=dataio.charge_onset_s('no-cooling',meta['data'],meta['event'])
    old=read(ROOT/previous['calibration_result']/'run.json.gz')
    summary=dict(parameters_sha256=phash,model_fingerprint=fingerprint,charge_window_s=600.,
                 full_explosion_calibrated=False,validation_used_for_selection=False,
                 calibration=metrics(run,meta,onset),previous_calibration=metrics(old,meta,onset),
                 parameter_changes={k:[prior.get(k,1.),v]for k,v in p.items()if v!=prior.get(k,1.)})
    compare_plot(run,old,meta,onset,'no-cooling',args.output/'calibration.png')
    fine_dir=args.output/'half-step';fine_dir.mkdir()
    print('START HALF STEP',flush=True)
    fine=cal.replay(scene,fine_dir,'cuda',200.,refinement=.5);save_run(fine_dir/'run.json.gz',fine)
    t=np.array(run['time_s'])
    delta={k:float(np.max(abs(np.interp(t,fine['time_s'],fine['temperature_k'][k])-run['temperature_k'][k])))for k in LABELS}
    pressure_delta=float(np.max(abs(np.interp(t,fine['time_s'],dataio.chamber_pressure_pa(fine))-dataio.chamber_pressure_pa(run))))
    summary['half_step_check']=dict(max_temperature_delta_k=delta,chamber_pressure_max_delta_pa=pressure_delta,wall_s=fine['wall_s'])
    cal.save(args.output/'summary.json',summary);print('HALF STEP',summary['half_step_check'],flush=True)
    if max(delta.values())>.25:raise ValueError('dt/2 difference exceeds 0.25 K')
    folder=args.output/'active-cooling';folder.mkdir()
    data,_=dataio.load('active-cooling');mark=dataio.event(data)
    onset=dataio.charge_onset_s('active-cooling',data,mark)
    scene,meta=model.build('active-cooling',p,warmup=True,duration=onset+600.)
    cal.save(folder/'scenario.json',scene)
    print('START FROZEN VALIDATION',flush=True)
    val=cal.replay(scene,folder,'cuda',200.);save_run(folder/'run.json.gz',val)
    summary['validation']=metrics(val,meta,onset)
    old=read(ROOT/previous['validation_result']/'run.json.gz')
    summary['previous_validation']=metrics(old,meta,onset)
    compare_plot(val,old,meta,onset,'active-cooling',args.output/'validation.png')
    if cal.fingerprint()!=fingerprint:raise RuntimeError('Core changed during frozen check')
    cal.save(folder/'status.json',dict(status='complete',parameters_sha256=phash,model_fingerprint=fingerprint,
        metrics=summary['validation'],validation_used_for_selection=False))
    cal.save(args.output/'summary.json',summary)
    if args.activate:
        cal.save(pointer_path,dict(status='600 s contact heating fit; held-out validation completed; full explosion not calibrated',
            parameters_file=str(frozen.relative_to(ROOT)),previous_working_profile=previous['parameters_file'],
            model_fingerprint=fingerprint,parameters_sha256=phash,defaults_reset=False,
            kernel_changed=False,fixed_materials=True,fixed_chemistry_and_electrical=True,
            calibration_result=str(args.fit.relative_to(ROOT)),validation_result=str(folder.relative_to(ROOT)),
            numerical_check=str((args.output/'summary.json').relative_to(ROOT)),charge_window_s=600.,
            both_replays_complete=True,validation_used_for_selection=False,
            first_300s_joule_share=state['diagnostics']['first300_joule_share'],
            calibration_rmse_all_six_channels_k=summary['calibration']['rmse_all_six_k'],
            validation_rmse_all_six_channels_k=summary['validation']['rmse_all_six_k'],
            half_step_max_temperature_delta_k=max(delta.values()),
            half_step_max_chamber_pressure_delta_pa=pressure_delta,full_explosion_calibrated=False))
    print('DONE',json.dumps(summary),flush=True)


if __name__=='__main__':main()

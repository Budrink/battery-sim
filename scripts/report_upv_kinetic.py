#!/usr/bin/env python3
"""Report the frozen kinetic fit and held-out experiment, with identical v3 scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
import calibrate_upv_kinetic as kin
from calibrate_upv import save
from report_upv_failure_transition import intact_regression

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=kin.OUTPUT)
    p.add_argument('--intact-probe',required=True,type=Path);a=p.parse_args();out=a.output
    profile=json.loads((out/'profile.json').read_text());checksum=kin.base.digest(out/'profile.json')
    if profile['model_sha256']!=kin.fingerprint():raise ValueError('Core or calibration protocol changed after fit')
    reports=[json.loads((out/(mode+'.json')).read_text()) for mode in ('fit','validate')]
    old=[json.loads((kin.prior.OUTPUT/(mode+'.json')).read_text()) for mode in ('fit','validate')]
    for r,o in zip(reports,old):
        if r['parameters']!=profile['parameters'] or r['frozen_profile_sha256']!=checksum or r['validation_refitted']:
            raise ValueError('Fit and validation do not use the same frozen profile')
        np.testing.assert_allclose(r['result']['time_s'],o['result']['time_s'],rtol=0,atol=1e-9)
        if r['observed_k']!=o['observed_k']:raise ValueError('Different experimental observations')
    regression=intact_regression(a.intact_probe)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    labels={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS'}
    for row,(r,o) in enumerate(zip(reports,old)):
        t=np.asarray(r['result']['time_s'])-r['meta']['event']['bracket_width_s'];mask=(t>=300)&(t<=1800)
        for col,k in enumerate(kin.prior.WITNESSES):
            ax=axes[row,col];observed=np.array([np.nan if x is None else x for x in r['observed_k'][k]])
            ax.plot(t[mask],observed[mask]-273.15,color='#154c79',label='Experiment')
            ax.plot(t[mask],np.asarray(r['result']['temperature_k'][k])[mask]-273.15,color='#dc652e',label='Kinetic v4')
            ax.plot(t[mask],np.asarray(o['result']['temperature_k'][k])[mask]-273.15,color='#999999',ls='--',label='Previous v3')
            ax.set_title(f'{labels[k]} | RMSE {r["metrics"]["per_channel"][k]["rmse_k"]:.2f} K')
            ax.set_xlim(300,1800);ax.grid(alpha=.18);ax.set_xlabel('Seconds after measured event')
            if col==0:ax.set_ylabel(('FIT: no cooling' if row==0 else 'VALIDATION: cooling')+'\nTemperature, °C');ax.legend(fontsize=8)
    fig.suptitle('Process-based CUDA model: fixed Cp, reference k and intact links\nSame late-witness objective; validation parameters frozen; no invented vent')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for row,r in enumerate(reports):
        run=r['result'];t=np.asarray(run['time_s'])-r['meta']['event']['bracket_width_s']
        axes[row,0].plot(t,np.asarray(run['released_energy_j_history'])/1000,label='Released',color='#dc652e')
        axes[row,0].plot(t,np.asarray(run['remaining_energy_j'])/1000,label='Remaining',color='#154c79')
        axes[row,0].set_ylabel(r['experiment']+'\nEnergy, kJ');axes[row,0].set_xlim(0,120);axes[row,0].legend()
        axes[row,1].plot(t,run['source_max_temperature_k'],label='Maximum',color='#dc652e')
        axes[row,1].plot(t,run['source_min_temperature_k'],label='Minimum',color='#154c79')
        axes[row,1].set_xlim(0,300);axes[row,1].set_ylabel('Modeled C2 temperature, K');axes[row,1].legend()
        lo=np.asarray(run['structure_min_xyz']);hi=np.asarray(run['structure_max_xyz'])
        for axis,label,color in [(0,'Through-plane','#652980'),(1,'In-plane','#128080')]:
            axes[row,2].fill_between(t,lo[:,axis],hi[:,axis],color=color,alpha=.25)
            axes[row,2].plot(t,hi[:,axis],color=color,label=label)
        axes[row,2].set_xlim(0,120);axes[row,2].set_yscale('log');axes[row,2].set_ylabel('Local structural factor');axes[row,2].legend()
        for ax in axes[row]:ax.set_xlabel('Seconds after measured event');ax.grid(alpha=.18)
    fig.suptitle('Inferred source history: diagnostic, NOT measured chemistry\nSampled at experimental scan times; rapid power peaks are under-resolved in this export')
    fig.savefig(out/'process.png',dpi=150);plt.close(fig)
    rate_checks=profile['sensitivity']['rate_400_per_s']
    identifiability=dict(
        multi_start_solutions=[dict(parameters=x['parameters'],training_rmse_k=x['rmse_k']) for x in profile['starts']],
        rate_range_per_s=[min(x['value'] for x in rate_checks),max(x['value'] for x in rate_checks)],
        rate_sweep_rmse_span_k=max(x['metrics']['rmse_k'] for x in rate_checks)-min(x['metrics']['rmse_k'] for x in rate_checks),
        note='This is demonstrated weak sensitivity/parameter tradeoff, not a confidence interval or a proof of global non-identifiability')
    assessment=dict(status='exploratory_fit_kinetics_not_identified',kinetics_identified=False,default_scene_replaced=False,
        profile_sha256=checksum,model_sha256=kin.fingerprint(),validation_refitted=False,validation_used_for_selection=False,
        parameters=profile['parameters'],near_bounds=profile['near_bounds'],all_capacities_fixed=True,
        results={r['experiment']:dict(metrics=r['metrics'],previous_v3_metrics=o['metrics'],time_check_max_change_k=r['time_check_max_change_k'],
            process=r['process'],thermal_residual_j=r['result']['max_thermal_residual_j'],chemical_residual_j=r['result']['max_chemical_residual_j'],
            export_identical=r['export_identical']) for r,o in zip(reports,old)},
        training_sensitivity=profile['sensitivity'],identifiability_evidence=identifiability,intact_regression=regression,
        limitations=['Measured primary event time, ambient and cooling histories are prescribed; no overcharge onset prediction',
            'Only late witness channels scored; source thermocouples and first 300 seconds diagnostic only',
            'Ea, damage law and activity peak factors are frozen illustrative priors, not identified material constants',
            'E is effective accessible solid-heat budget with no outlet, not total reaction/ejecta energy',
            'No gas jet transport, no outlet invented for unspecified pouch vent geometry',
            'C1/C3 acquisition numbering remains an unconfirmed hypothesis; post-event probe attachment not established',
            'No secondary ignition kinetics calibrated; held-out experiment was seen during earlier development'])
    save(out/'assessment.json',assessment)
    print(json.dumps({k:assessment[k] for k in ('status','near_bounds','results','intact_regression')},indent=2))

if __name__=='__main__':main()

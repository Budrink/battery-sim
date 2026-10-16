#!/usr/bin/env python3
"""Report frozen late-witness fit, transient priors and intact-model regression."""
import argparse
import json
from pathlib import Path
import numpy as np
import calibrate_upv_failure_transition as tr
from calibrate_upv import Worker, save

def intact_regression(probe):
    result={}
    for mode,name in [('fit','no-cooling'),('validate','active-cooling')]:
        path=tr.v1.PRIOR/(name+'-fitted-scenario.json')
        old_path=tr.v1.PRIOR/(mode+'.json');old=json.loads(old_path.read_text())
        w=Worker(probe,path)
        try:r=w.run({})
        finally:w.close()
        np.testing.assert_allclose(r['time_s'],old['result']['time_s'],rtol=0,atol=1e-9)
        delta=max(float(np.max(np.abs(np.asarray(r['temperature_k'][k])-old['result']['temperature_k'][k]))) for k in tr.WITNESSES)
        if delta>1e-8 or abs(r['energy_residual_j'])>1e-6:raise ValueError('Intact transport regression: '+name)
        result[name]=dict(max_change_k=delta,energy_residual_j=r['energy_residual_j'],
            reference_rmse_k=old['metrics']['rmse_k'],scenario_sha256=tr.v1.digest(path),reference_sha256=tr.v1.digest(old_path))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=tr.OUTPUT)
    parser.add_argument('--intact-probe',type=Path,required=True)
    a=parser.parse_args();out=a.output;profile=json.loads((out/'profile.json').read_text())
    if profile['model_sha256']!=tr.fingerprint():raise ValueError('Model changed after fit')
    reports=[json.loads((out/(mode+'.json')).read_text()) for mode in ('fit','validate')]
    for r in reports:
        if r['parameters']!=profile['parameters'] or r['validation_refitted'] or r['frozen_profile_sha256']!=tr.v1.digest(out/'profile.json'):
            raise ValueError('Mixed fit/validation profiles')
    regression=intact_regression(a.intact_probe)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    labels={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS'}
    for row,r in enumerate(reports):
        t=np.asarray(r['result']['time_s'])-r['meta']['event']['bracket_width_s']
        selected=(t>=300)&(t<=1800)
        for col,k in enumerate(tr.WITNESSES):
            ax=axes[row,col];y=np.array([np.nan if x is None else x for x in r['observed_k'][k]])-273.15
            ax.plot(t[selected],y[selected],color='#24487b',lw=1.5,label='Experiment')
            ax.plot(t[selected],np.asarray(r['result']['temperature_k'][k])[selected]-273.15,color='#d45a26',lw=1.5,label='Transient medium')
            ax.plot(t[selected],np.asarray(r['v1_replay']['result']['temperature_k'][k])[selected]-273.15,color='#929292',ls='--',label='Old v1, unchanged')
            ax.set_xlim(300,1800);ax.grid(alpha=.2);ax.set_xlabel('Seconds after primary event')
            ax.set_title(f'{labels[k]}: RMSE {r["metrics"]["per_channel"][k]["rmse_k"]:.2f} K')
            if col==0:ax.set_ylabel(r['experiment']+'\nTemperature, °C');ax.legend(fontsize=8)
    fig.suptitle('Late witnesses only: source thermocouples and first 300 s NOT fitted\nFixed Cp and intact links; source timing/environment measured; rapid pulse and structural transient are priors')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    p=profile['parameters'];fig,axes=plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
    def fraction(t,tau):return np.where(t>=6*tau,1.,-np.expm1(-np.maximum(t,0)/tau)/(-np.expm1(-6.)))
    t=np.linspace(0,6*p['bulk_relaxation_time_s']*1.15,600)
    factor=p['destroyed_bulk_factor']+(p['bulk_peak_factor']-p['destroyed_bulk_factor'])*(1-fraction(t,p['bulk_relaxation_time_s']))
    axes[0].plot(t,factor,color='#652980');axes[0].axhline(1,color='#999999',ls='--',label='Reference bulk k')
    axes[0].set_title('Structural transient: peak/time NOT fitted');axes[0].set_ylabel('Effective bulk factor');axes[0].legend(fontsize=8)
    t=np.linspace(0,6*p['duration_s'],600)
    power=sum(weight*np.exp(-t/tau)/tau/(-np.expm1(-6.))*(t<6*tau) for weight,tau in [(p['fast_fraction'],p['duration_s']),(1-p['fast_fraction'],p['tail_time_s'])])
    axes[1].plot(t,p['energy_j']*power/1000,color='#d45a26');axes[1].set_ylabel('Deposited power, kW');axes[1].set_title('Chemical pulse: shape NOT fitted')
    t=np.linspace(0,6*p['tail_time_s']*1.05,600)
    released=p['energy_j']*(p['fast_fraction']*fraction(t,p['duration_s'])+(1-p['fast_fraction'])*fraction(t,p['tail_time_s']))
    axes[2].plot(t,released/1000,color='#24487b');axes[2].axhline(p['energy_j']/1000,color='#999999',ls='--')
    axes[2].set_ylabel('Released heat, kJ');axes[2].set_title('One finite chemical reserve')
    for ax in axes:ax.set_xlabel('Seconds after local rupture');ax.grid(alpha=.2)
    fig.suptitle('Two distinct processes: changing connectivity redistributes heat; the chemical reserve supplies it')
    fig.savefig(out/'rupture-law.png',dpi=150);plt.close(fig)
    assessment=dict(status='exploratory_not_accepted_as_complete_destruction_calibration',default_scene_replaced=False,
        model_sha256=tr.fingerprint(),profile_sha256=tr.v1.digest(out/'profile.json'),validation_refitted=False,
        all_capacities_fixed=True,parameters=profile['parameters'],fitted_parameters=list(tr.FIT_NAMES),
        fixed_unidentified_priors=dict(tr.PULSE_PRIOR,**tr.STRUCTURE_PRIOR),near_bounds=profile['near_bounds'],
        results={r['experiment']:dict(metrics=r['metrics'],old_v1_same_window_metrics=r['v1_replay']['metrics'],
            energy_residual_j=r['result']['energy_residual_j'],half_dt_max_change_k=r['time_check_max_change_k'],
            export_identical=r['export_identical'],legacy_v1_max_change_k=r['v1_replay']['max_change_k']) for r in reports},
        intact_regression=regression,intact_probe_sha256=tr.v1.digest(a.intact_probe),
        limits=['Transient peak and relaxation time are hypotheses, not measured constants',
            'Fitted E is heat deposited in modeled solid, not total chemical energy or calorimetry',
            'No explicit gas/mass field or new remote heat-transfer path',
            'Primary time, environment and cooling temperatures are prescribed; secondary kinetics not calibrated',
            'No claim that a post-event thermocouple spike measures whole-cell temperature',
            'Historical v1 peak-window RMSE is not comparable to this late-only objective'])
    save(out/'assessment.json',assessment)
    print(json.dumps({k:assessment[k] for k in ('status','near_bounds','intact_regression','results')},indent=2))

if __name__=='__main__':main()

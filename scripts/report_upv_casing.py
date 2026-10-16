#!/usr/bin/env python3
"""Compare frozen v5 casing trials to prior profiles on unchanged objectives."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import calibrate_upv_casing as trial
from calibrate_upv import save


def main():
    cli=argparse.ArgumentParser(__doc__);cli.add_argument('--output',type=Path,default=trial.OUT);args=cli.parse_args()
    assessment={'status':'conditional proxy trial, not identified packaging or validated destruction',
                'validation_refitted':False,'default_scene_replaced':False,'experiments':{}}
    labels={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS'}
    gas_fig,gas_axes=plt.subplots(2,2,figsize=(12,7),sharex='col');have_gas=False
    for stage,prior in [('heat','upv-links-contrast-3x3x3'),('failure','upv-kinetic-v4')]:
        fig,axes=plt.subplots(4,2,figsize=(13,11),sharex='col');available=False
        for column,(mode,name) in enumerate([('fit','no-cooling'),('validate','active-cooling')]):
            directory='failure-fast' if stage=='failure' else stage
            path=args.output/directory/(mode+'.json')
            if not path.exists():continue
            available=True;r=json.loads(path.read_text());before=json.loads((trial.ROOT/'data/calibration'/prior/(mode+'.json')).read_text())
            data,_=trial.old.load_record(name);start=0 if stage=='heat' else r['meta']['event']['last_pre_event_s']
            model=r['result'];t=np.asarray(model['time_s']);raw_t=data['time_s']-start
            select=(raw_t>=0)&(raw_t<=t[-1]);row=dict(previous_rmse_k=before['metrics']['rmse_k'],rmse_k=r['metrics']['rmse_k'],
                per_channel=r['metrics']['per_channel'],max_refinement_difference_k=r.get('max_refinement_difference_k'),
                max_thermal_residual_j=model['max_thermal_residual_j'],max_chemical_residual_j=model['max_chemical_residual_j'],max_mass_residual_kg=model['max_mass_residual_kg'])
            if 'max_scored_refinement_difference_k' in r:row['max_scored_refinement_difference_k']=r['max_scored_refinement_difference_k']
            extra=args.output/'numerics'/(stage+'-'+name+'.json')
            if extra.exists():row['full_cfl_refinement']=json.loads(extra.read_text())
            assessment['experiments'][stage+'/'+name]=row
            for index,(key,label) in enumerate(labels.items()):
                ax=axes[index,column];ax.plot(raw_t[select],data['T'+key+'_k'][select]-273.15,color='#444444',lw=1,label='Experiment')
                ax.plot(t,np.asarray(model['temperature_k'][key])-273.15,color='#d1495b',lw=1.5,label='Physical casing')
                oldrun=before['result'];ax.plot(oldrun['time_s'],np.asarray(oldrun['temperature_k'][key])-273.15,color='#287c9b',ls='--',lw=1,label='Previous model')
                if stage=='failure':ax.axvspan(0,300,color='grey',alpha=.12)
                ax.set_title(name+' — '+label);ax.set_ylabel('°C');ax.grid(alpha=.2)
            axes[0,column].legend(fontsize=8);axes[-1,column].set_xlabel('time from '+('test start' if stage=='heat' else 'last pre-event scan')+' [s]')
            if stage=='failure':
                have_gas=True
                pressure=2.5*data['U114_v'];chamber=model['gas']['chamber']
                p=np.interp(t,raw_t,pressure)*1e5;late=t>=300
                row['chamber_pressure_diagnostic']={'rmse_bar_after_300s':float(np.sqrt(np.mean((np.asarray(chamber['pressure_pa'])[late]-p[late])**2))/1e5),
                    'predicted_peak_bar':max(chamber['pressure_pa'])/1e5,'observed_peak_bar':float(np.max(pressure[select])),
                    'note':'Not a fitted target; chamber average temperature/combustion/gas composition are unmeasured'}
                row['rupture_events']=[e for e in model['events'] if e['kind']=='shell_rupture']
                row['event_bracket_width_s']=r['meta']['event']['bracket_width_s']
                row['source_diagnostics']={'max_volume_temperature_k':max(model['source_max_temperature_k']),
                    'final_damage':model['source_damage'][-1],
                    'remaining_chemical_energy_j':model['remaining_energy_j'][-1],
                    'net_released_to_thermal_history_j':model['released_energy_j_history'][-1],
                    'final_chamber_gas_mass_kg':chamber['mass_kg'][-1],
                    'note':'Effective event-conditioned heat fit, not identified total cell calorimetry'}
                ax=gas_axes[0,column]
                ax.plot(raw_t[select],pressure[select],color='#444444',label='Measured chamber pressure')
                ax.plot(t,np.asarray(chamber['pressure_pa'])/1e5,color='#d1495b',label='Modeled chamber pressure')
                ax.set_title(name);ax.set_ylabel('absolute pressure [bar]');ax.legend(fontsize=8);ax.grid(alpha=.2)
                ax=gas_axes[1,column]
                ax.plot(t,np.asarray(chamber['temperature_k'])-273.15,color='#d1495b',label='Modeled chamber gas')
                ax.plot(raw_t[select],data['T109_k'][select]-273.15,color='#444444',label='Measured ambient forcing (ch109)')
                ax.set_ylabel('temperature [°C]');ax.set_xlabel('time from last pre-event scan [s]');ax.legend(fontsize=8);ax.grid(alpha=.2)
        if available:
            fig.suptitle(stage+' — identical scoring windows; packaging is an unmeasured proxy',fontsize=12)
            fig.tight_layout();fig.savefig(args.output/(stage+'-comparison.png'),dpi=160)
            if stage=='failure':
                for ax in axes.flat:
                    values=[]
                    for line in ax.lines:
                        x=np.asarray(line.get_xdata());y=np.asarray(line.get_ydata())
                        values.extend(y[(x>=300)&(x<=1800)&np.isfinite(y)])
                    if values:
                        lo,hi=min(values),max(values);pad=max(1.,.05*(hi-lo))
                        ax.set_xlim(300,1800);ax.set_ylim(lo-pad,hi+pad)
                fig.suptitle('Failure — scored late window only (300–1800 s); early burst NOT shown',fontsize=12)
                fig.savefig(args.output/'failure-scored-comparison.png',dpi=160)
        plt.close(fig)
    if have_gas:
        gas_fig.suptitle('Unfitted gas diagnostics — no calibrated wall loss / external combustion',fontsize=12)
        gas_fig.tight_layout();gas_fig.savefig(args.output/'gas-diagnostics.png',dpi=160)
    plt.close(gas_fig)
    for stage in ('heat','failure'):
        path=args.output/('failure-fast' if stage=='failure' else stage)/'profile.json'
        if path.exists():
            p=json.loads(path.read_text());assessment[stage+'_search']=p['search'];assessment[stage+'_parameters']=p['parameters']
    save(args.output/'assessment.json',assessment);print(json.dumps(assessment,indent=2))


if __name__=='__main__':main()

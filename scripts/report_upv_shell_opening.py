#!/usr/bin/env python3
"""Plots and optional dense replay for a frozen shell-only calibration."""
import argparse
import copy
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import calibrate_upv_shell_opening as fit
from calibrate_upv import save

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=fit.Path,default=fit.OUT)
    p.add_argument('--probe',type=fit.Path,default=fit.Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe'))
    p.add_argument('--dense',action='store_true');a=p.parse_args()
    # The second stage shares the report format but has its own freeze hash.
    energy=a.output/'profile.json'
    stored=json.loads(energy.read_text())
    if stored['protocol']['revision'].startswith('progressive-shell-frozen-energy-refit-'):
        import calibrate_upv_shell_energy as joint
        expected=joint.fingerprint(a.probe)
    else:expected=fit.fingerprint(a.probe)
    profile=json.loads((a.output/'profile.json').read_text())
    if profile['fingerprint']!=expected:raise ValueError('Frozen model changed')
    pressure,axes=plt.subplots(2,2,figsize=(12,7));thermal,taxes=plt.subplots(4,2,figsize=(12,11))
    opening,oaxes=plt.subplots(3,2,figsize=(12,9));summary=dict(profile=profile,experiments={})
    runner=fit.gas.t.Runner(a.probe,a.output)
    try:
        for col,(mode,name) in enumerate([('fit','no-cooling'),('validate','active-cooling')]):
            r=json.loads((a.output/(mode+'.json')).read_text());v=r['result'];at=np.asarray(v['time_s'])
            old=json.loads((fit.PRIOR/(mode+'.json')).read_text());previous=old['result']
            dense_path=a.output/(name+'-dense.json')
            if a.dense:
                s=json.loads((a.output/(name+'-'+mode+'-scenario.json')).read_text())
                s['simulation']['duration_s']=30.
                s['failure_sample_times_s']=np.unique(np.r_[np.arange(0,5,.01),np.arange(5,30,.1),30.,[t for t in s['failure_sample_times_s'] if t<30]]).tolist()
                dense=runner.run(s);save(dense_path,dense);print(name,'dense steps',dense['steps'],'wall',dense['wall_s'],flush=True)
            dense=json.loads(dense_path.read_text()) if dense_path.exists() else v
            old_dense_path=fit.PRIOR/'dense-check'/(name+'.json')
            old_dense=json.loads(old_dense_path.read_text())['coarse'] if old_dense_path.exists() else previous
            for row in range(2):
                ax=axes[row,col];ax.plot(at,r['observed_pressure_bar'],color='#414141',label='Experiment')
                vv=dense if row==0 else v;ov=old_dense if row==0 else previous
                ax.plot(vv['time_s'],np.asarray(vv['gas']['chamber']['pressure_pa'])/1e5,color='#c43c39',label='Progressive shell')
                ax.plot(ov['time_s'],np.asarray(ov['gas']['chamber']['pressure_pa'])/1e5,'--',color='#277da8',label='Frozen v7')
                ax.set(xlim=(0,10 if row==0 else 1800),ylabel='Chamber pressure [bar abs]',xlabel='Time after primary event [s]',title=name);ax.grid(alpha=.2)
            axes[0,col].legend(fontsize=8)
            for row,(key,label) in enumerate([('101','C1 / ISS'),('102','C1 / TIM'),('107','C3 / TIM'),('108','C3 / ISS')]):
                ax=taxes[row,col];mask=at>=300;old_at=np.asarray(previous['time_s']);old_mask=old_at>=300
                ax.plot(at[mask],np.asarray(r['observed_temperature_k'][key],dtype=float)[mask]-273.15,color='#414141',label='Experiment')
                ax.plot(at[mask],np.asarray(v['temperature_k'][key])[mask]-273.15,color='#c43c39',label='Progressive shell')
                ax.plot(old_at[old_mask],np.asarray(previous['temperature_k'][key])[old_mask]-273.15,'--',color='#277da8',label='Frozen v7')
                ax.set(title=name+' — '+label,ylabel='Temperature [C]',xlabel='Time [s]');ax.grid(alpha=.2)
            taxes[0,col].legend(fontsize=8)
            for label,values in dense['channels'].items():
                oaxes[0,col].plot(dense['time_s'],1e6*np.asarray(values['effective_area_m2']),label=label.replace('C2-progressive-',''))
                oaxes[1,col].plot(dense['time_s'],values['max_bulk_damage'],label=label.replace('C2-progressive-',''))
                oaxes[2,col].plot(dense['time_s'],values['max_seam_damage'],label=label.replace('C2-progressive-',''))
            for row,label in enumerate(['Effective area [mm2]','Bulk damage','Seam damage']):
                oaxes[row,col].set(xlim=(0,5),title=name,ylabel=label,xlabel='Time [s]');oaxes[row,col].grid(alpha=.2)
            oaxes[0,col].legend(fontsize=8,ncol=3)
            pressure_values=np.asarray(dense['gas']['chamber']['pressure_pa']);peak=int(np.argmax(pressure_values))
            summary['experiments'][name]=dict(metrics=r['metrics'],previous_metrics=old['metrics'],
                dense_pressure_peak_bar=float(pressure_values[peak]/1e5),dense_peak_time_s=dense['time_s'][peak],
                internal_pressure_peak_bar=float(np.max(dense['gas']['inside-C2']['pressure_pa'])/1e5),
                internal_pressure_note='Unobserved internal pseudo-gas pressure; not a calibrated pressure rating of the pouch',
                events=v['events'],ledgers={k:x for k,x in v.items() if 'residual' in k},
                source_scene_sha256=fit.digest(a.output/(name+'-'+mode+'-scenario.json')))
        for fig,file,title in [(pressure,'pressure-comparison.png','Chamber pressure: frozen calibration / held-out replay'),
                               (thermal,'witness-comparison.png','Late witness temperatures; battery properties unchanged'),
                               (opening,'opening-history.png','Progressive casing opening: effective area and local damage')]:
            fig.suptitle(title);fig.tight_layout();fig.savefig(a.output/file,dpi=150);plt.close(fig)
        refinement=a.output/'refinement.json'
        if refinement.exists():summary['refinement']=json.loads(refinement.read_text())
        save(a.output/'assessment.json',summary)
    finally:runner.close()

if __name__=='__main__':main()

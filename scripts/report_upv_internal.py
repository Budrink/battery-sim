#!/usr/bin/env python3
"""Compare the frozen internal trial with its unchanged 3^3 baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from calibrate_upv import save
from generate_upv import ROOT, TARGETS, load_case

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory',type=Path,default=ROOT/'data/calibration/upv-internal-3x3x3')
a=p.parse_args();directory=a.directory
profile_path=directory/'profile.json';profile=json.loads(profile_path.read_text())
digest=hashlib.sha256(profile_path.read_bytes()).hexdigest()
reports=[json.loads((directory/f'{stage}.json').read_text()) for stage in ('fit','validate')]
for r in reports:
    if r['frozen_profile_sha256']!=digest or r['parameters']!=profile['parameters']:raise ValueError('Frozen profile mismatch')
summary=dict(status='exploratory',parameters=profile['parameters'],near_bounds=profile['near_bounds'],
             frozen_profile_sha256=digest,validation_refitted=False,default_scene_replaced=False,experiments={})
fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
labels={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS'}
for row,r in enumerate(reports):
    ex=r['experiment'];run=r['result'];observed,*_=load_case(ROOT/'data/experiments/upv',ex)
    baseline=json.loads((ROOT/f'data/calibration/upv-core-shell-check/{ex}-3x3x3.json').read_text())['result']
    checks=r['numerical_checks']
    summary['experiments'][ex]=dict(baseline_rmse_k=r['baseline_metrics']['rmse_k'],rmse_k=r['metrics']['rmse_k'],
        baseline_max_error_k=r['baseline_metrics']['max_abs_error_k'],max_error_k=r['metrics']['max_abs_error_k'],
        max_grid_3_vs_12_k=checks['battery_grid']['12']['max_change_k'],
        max_time_step_change_k=checks['time_step']['max_change_k'],grid_3_passed=checks['grid_3_passed'],
        reference_8_vs_12_passed=checks['reference_8_vs_12']['max_change_k']<=.1,
        energy_residual_j=r['metrics']['energy_residual_j'])
    t=np.asarray(run['time_s']);tm=np.asarray(observed['time_s']);mask=tm<=t[-1]
    for ax,ch in zip(axes[row],TARGETS):
        ax.plot(tm[mask],np.asarray(observed[f'T{ch}_k'])[mask]-273.15,label='Measured',c='#263f9e',lw=1.5)
        ax.plot(baseline['time_s'],np.asarray(baseline['temperature_k'][ch])-273.15,label='Before',c='#909090',ls=':')
        ax.plot(t,np.asarray(run['temperature_k'][ch])-273.15,label='Internal fit',c='#d25624',ls='--',lw=1.5)
        ax.set(title=f'{ex}: {labels[ch]}\nRMSE {r["baseline_metrics"]["per_channel"][ch]["rmse_k"]:.2f} → {r["metrics"]["per_channel"][ch]["rmse_k"]:.2f} K',
               xlabel='Time, s',ylabel='Temperature, °C')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
passed=all(v['grid_3_passed'] and v['reference_8_vs_12_passed'] and v['max_time_step_change_k']<=.05 for v in summary['experiments'].values())
summary['status']='exploratory_numerics_passed' if passed else 'exploratory_grid_sensitive'
summary['accepted_as_mesh_independent_calibration']=False
summary['caveat']='Two-parameter graph-edge fit, not identified k(T). Quadratic bound reached. Pre-event and unconfirmed numbering only.'
save(directory/'assessment.json',summary)
fig.suptitle('3×3×3 internal-edge trial: external contacts and material reference properties fixed\nFit: no cooling; frozen check: active cooling. Exploratory, not mesh-independent calibration.',fontsize=12)
fig.savefig(directory/'comparison.png',dpi=150)
print(json.dumps(summary,indent=2));print(directory/'comparison.png')

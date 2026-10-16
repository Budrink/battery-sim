#!/usr/bin/env python3
"""Compare independent 3^3/4^3/8^3 fits with the same frozen CUDA physics."""
import argparse
import copy
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from calibrate_upv import model_hash, residuals, save
from calibrate_upv_internal import sha
from generate_upv import ROOT, TARGETS, load_case


def physical_scene(scene):
    out=copy.deepcopy(scene);out.pop('metadata',None)
    for c in out['cells']:
        if c['role']=='battery':c['thermal'].pop('divisions')
    return out


def comparable_protocol(protocol):
    out=copy.deepcopy(protocol);out.pop('grid');out.pop('revision');return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/calibration/upv-internal-mesh-study')
    a=p.parse_args();runs={};profiles={};native_hash=model_hash();binary_hash=sha(a.probe)
    summary=dict(kind='independent-grid-refits',grids=[3,4,8],model_sha256=native_hash,probe_sha256=binary_hash,
                 contacts_refitted=False,validation_refitted=False,default_scene_replaced=False,
                 caveats=['This measures agreement with observations, not physical accuracy of continuum k(T).',
                          'No change to the half-volume treatment: boundary half-resistances still use fixed k_ref.',
                          'C1/C3 numbering is an unconfirmed hypothesis; second experiment previously inspected.',
                          'All fits are pre-event conditional heating; no runaway/degradation validation.'],results={})
    for n in (3,4,8):
        folder=ROOT/f'data/calibration/upv-internal-{n}x{n}x{n}'
        profile=json.loads((folder/'profile.json').read_text());profiles[n]=profile;runs[n]={}
        if not profile['optimizer_success'] or profile['validation_used']:raise ValueError('Converged no-cooling-only fit required')
        if profile['model_sha256']!=native_hash or profile['probe_sha256']!=binary_hash:raise ValueError('Native model/binary differs')
        if comparable_protocol(profile['protocol'])!=comparable_protocol(profiles[3]['protocol']):raise ValueError('Fit protocol differs')
        if profile['frozen_external_parameters']!=profiles[3]['frozen_external_parameters']:raise ValueError('External contacts differ')
        if n!=3 and profile['search']!=profiles[4]['search']:raise ValueError('Search settings differ')
        result=dict(volumes_per_battery=n**3,core_volumes=(n-2)**3,shell_volumes=n**3-(n-2)**3,
                    parameters=profile['parameters'],near_bounds=profile['near_bounds'],starts=profile['starts'],
                    local_jacobian_condition_number=profile['local_jacobian_condition_number'],
                    evaluation_count=len(profile['evaluation_history']),profile_sha256=sha(folder/'profile.json'),experiments={})
        for stage in ('fit','validate'):
            report=json.loads((folder/f'{stage}.json').read_text());runs[n][stage]=report
            if report['parameters']!=profile['parameters'] or report['frozen_profile_sha256']!=sha(folder/'profile.json'):
                raise ValueError('Report not tied to its frozen profile')
            if report['source_csv_sha256']!=runs[3][stage]['source_csv_sha256']:raise ValueError('Observation data differs')
            if report['metrics']['evaluated_values']!=runs[3][stage]['metrics']['evaluated_values']:raise ValueError('Scoring window differs')
            scene=json.loads((folder/f'{report["experiment"]}-scenario.json').read_text())
            base=json.loads((ROOT/f'data/calibration/upv-internal-3x3x3/{report["experiment"]}-scenario.json').read_text())
            if physical_scene(scene)!=physical_scene(base):raise ValueError('Something other than battery divisions changed')
            if any(c['thermal']['divisions']!=[n]*3 for c in scene['cells'] if c['role']=='battery'):
                raise ValueError('Wrong mesh in scenario')
            checks=report['numerical_checks'];m=report['metrics'];old=runs[3][stage]['metrics']
            result['experiments'][report['experiment']]=dict(
                metrics=m,rmse_change_from_3_k=m['rmse_k']-old['rmse_k'],
                rmse_change_from_3_percent=100*(m['rmse_k']/old['rmse_k']-1),
                baseline_rmse_k=report['baseline_metrics']['rmse_k'],
                numerical_checks=checks,export_equivalence=report['export_equivalence'],
                report_sha256=sha(folder/f'{stage}.json'))
        summary['results'][str(n)]=result
    fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    labels={'101':'C1 / ISS','102':'C1 / TIM','107':'C3 / TIM','108':'C3 / ISS'}
    colors={3:'#285da8',4:'#d36328',8:'#209170'}
    for row,stage in enumerate(('fit','validate')):
        experiment=runs[3][stage]['experiment'];observed,*_=load_case(ROOT/'data/experiments/upv',experiment)
        for n in (3,4,8):
            r=runs[n][stage];run=r['result'];t=np.asarray(run['time_s']);e=residuals(run,observed)
            for ax,ch in zip(axes[row],TARGETS):
                ax.plot(t[t>=60],e[ch],label=f'{n}³: RMSE {r["metrics"]["per_channel"][ch]["rmse_k"]:.3f} K',
                        c=colors[n],ls={3:'-',4:'--',8:'-.'}[n],lw=1.6)
        for ax,ch in zip(axes[row],TARGETS):
            ax.axhline(0,c='#333333',lw=.7);ax.grid(alpha=.2)
            ax.set(title=f'{experiment}: {labels[ch]}',xlabel='Time, s',ylabel='Simulation − measurement, K')
            ax.legend(fontsize=8)
    fig.suptitle('Independent internal-link refits: only the battery grid changes\nSame two parameters, bounds, starts and frozen external contacts; no active-cooling refit.',fontsize=12)
    a.output_dir.mkdir(parents=True,exist_ok=True);fig.savefig(a.output_dir/'residuals.png',dpi=150)
    save(a.output_dir/'summary.json',summary)
    for n,result in summary['results'].items():
        print(n,result['parameters'],{ex:dict(rmse=v['metrics']['rmse_k'],max_error=v['metrics']['max_abs_error_k'],
                                            change_percent=v['rmse_change_from_3_percent']) for ex,v in result['experiments'].items()})
    print(a.output_dir/'residuals.png')


if __name__=='__main__':main()

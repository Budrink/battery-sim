#!/usr/bin/env python3
"""Choose a fresh probe from measured common-window responses, never a final score."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from check_fsri30_radiation_experiment import read_run
from fsri30_data import ROOT,load,save,sha
from fsri30_objective import model_fields,targets
from fsri30_balance import timing_compensation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    root=ROOT/'data/development/fsri30-deadline-20261026-r1'
    folders=dict(base=ROOT/'data/calibration/fsri30-balance-bracket-20261026-r2/trial-0026',
        lower_mix=ROOT/'data/calibration/fsri30-balance-bracket-20261026-r1/trial-0025',
        higher_epsilon=ROOT/'data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023',
        hot_contact=root/'abs-hot-suffix/vent-suffix')
    data,manifest=load();truth=targets(data);runs={};sources={}
    for label,folder in folders.items():
        path=folder/'replay/run.json.gz'
        if not path.exists():path=folder/'replay/incomplete-run.json.gz'
        raw=read_run(path);t,_,v=model_fields(raw);runs[label]=(t,v)
        sources[label]=dict(path=str(path),sha256=sha(path),start_s=float(t[0]),end_s=float(t[-1]))
    end=min(float(t[-1])for t,v in runs.values())
    if end<3900.:raise ValueError('Need actual cooling through at least3900s')
    mask=(truth['time']>=3600.)&(truth['time']<=end+1e-7);grid=truth['time'][mask]
    levels=np.linspace(0.,1.,9)
    observed=np.array([data[f'T{i}_k']for i in range(1,10)]).T
    target=np.quantile(observed[mask],levels,axis=1).T
    curves={}
    for label,(t,v)in runs.items():
        q=np.quantile(v,levels,axis=1).T
        curves[label]=np.stack([np.interp(grid,t,q[:,j])for j in range(9)],axis=1)
    params=dict(base=dict(effective_emissivity=.3932387554878369,thermal_stirring_coefficient=.0786871284805109),
        lower_mix=dict(thermal_stirring_coefficient=.07452091483809048),
        higher_epsilon=dict(effective_emissivity=.7664775109756737,thermal_stirring_coefficient=.06722339346183823))
    b=params['base'];delta_mix=np.log(params['lower_mix']['thermal_stirring_coefficient']/b['thermal_stirring_coefficient'])
    dm=(curves['lower_mix']-curves['base'])/delta_mix
    de=((curves['higher_epsilon']-curves['base']-dm*np.log(params['higher_epsilon']['thermal_stirring_coefficient']/b['thermal_stirring_coefficient']))/
        (params['higher_epsilon']['effective_emissivity']-b['effective_emissivity']))
    da=curves['hot_contact']-curves['base']
    # All nine ranks have equal weight. No claim of a known TC-to-cell assignment.
    # Later observed samples have modestly higher weight; unknown future is absent.
    weight=np.sqrt(np.linspace(1.,2.,len(grid)))[:,None]
    residual=curves['base']-target
    matrix=np.stack([(de*weight).ravel(),(da*weight).ravel()],axis=1)
    fit=lsq_linear(matrix,-(residual*weight).ravel(),bounds=([.02-b['effective_emissivity'],0.],[.95-b['effective_emissivity'],1.]))
    epsilon=b['effective_emissivity']+float(fit.x[0]);strength=float(fit.x[1])
    rows=json.loads((ROOT/'data/calibration/fsri30-balance-bracket-20261026-r2/observations.json').read_text())
    latest=json.loads((root/'burst-corrected/status.json').read_text())
    center=dict(parameters=latest['parameters'],burst=latest['burst'],trial=str(root/'burst-corrected'))
    if not center['burst'].get('evaluated'):
        # A speed-censored prefix did not measure a burst. Retain actual older
        # phase evidence; do not treat the new proposal as a measured event.
        center=next(r for r in rows if r['trial'].endswith('/trial-0026'))
    next_parameters,phase_forecast=timing_compensation(rows,epsilon,center)
    next_parameters['abs_contact_strength']=strength
    predicted=curves['base']+fit.x[0]*de+strength*da
    # Forecast is not used as an observed trajectory and is never sent to plotting.
    result=dict(parameters=next_parameters,observed_cooling_window_s=[float(grid[0]),float(grid[-1])],
        common_window_only=True,full_score_available=False,proposal_only=True,actual_fresh_replay_required=True,
        base_rank_rmse_k=float(np.sqrt(np.mean(residual**2))),
        conditioned_hot_contact_rank_rmse_k=float(np.sqrt(np.mean((curves['hot_contact']-target)**2))),
        predicted_rank_rmse_k=float(np.sqrt(np.mean((predicted-target)**2))),
        predicted_rank_bias_k=np.mean(predicted-target,axis=0).tolist(),
        phase_forecast=phase_forecast,sources=sources,csv_sha256=manifest['csv_sha256'],
        phase_center_trial=center['trial'],new_hot_contact_phase_response_measured=False,
        scope='Measured local response proposal, nine rank curves versus nine TC order statistics; positions unresolved; NOT validation')
    save(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

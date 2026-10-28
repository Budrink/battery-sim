"""Early rejection for the cooling-target diagnostic, not a validation score."""
import numpy as np
from scipy.ndimage import maximum_filter1d
from fsri30_objective import model_fields,targets,reject_reason


def rejection(scene,run,data,*,window_s=120.,absolute_limit_k=80.,relative_limit=.10):
    """Use observed prefix only; soften uncertain peaks but never ignore them.

    Nine TC envelopes and thirty means are different observations. This is a
    deliberately coarse SEARCH guard. Absolute error is K; its relative scale
    is absolute Kelvin, never Celsius. The complete fit still has to improve.
    """
    gross=reject_reason(scene,run,data)
    if gross:return dict(reason=gross,last_observed_s=float(run['time_s'][-1]))
    t,_,values=model_fields(run);end=float(t[-1])
    # This controller tests cooling, not a new warmup chemical fit. The
    # existing gross-divergence guard above remains active in all phases.
    if end<3600.+window_s:return None
    target=targets(data)
    mask=(target['time']>=end-window_s)&(target['time']<=end+1e-7)
    if np.count_nonzero(mask)<window_s*.9:return None
    grid=target['time'][mask];truth=target['q'][mask]
    q=np.quantile(values,[0.,.5,.9,1.],axis=1).T
    prediction=np.stack([np.interp(grid,t,q[:,i])for i in range(4)],axis=1)
    # A noisy TC must not soften every branch of the cooling envelope.
    sharp=(abs(np.gradient(prediction,grid,axis=0))>5.)|(abs(np.gradient(truth,grid,axis=0))>5.)
    peaks=maximum_filter1d(sharp.astype(int),size=61,axis=0)>0
    stable=~peaks;error=prediction-truth
    info=dict(last_observed_s=end,window_start_s=float(grid[0]),window_end_s=float(grid[-1]),
        scope='Cooling search rejection only; unmapped TC-envelope heuristic',
        absolute_limit_k=absolute_limit_k,relative_kelvin_limit=relative_limit,
        stable_samples=int(np.any(stable,axis=1).sum()),peak_samples=int(np.any(peaks,axis=1).sum()),
        stable_quantile_entries=int(stable.sum()),peak_quantile_entries=int(peaks.sum()))
    if np.any(stable,axis=1).sum()>=60:
        rmse=float(np.sqrt(np.mean(error[stable]**2)))
        limit=max(absolute_limit_k,relative_limit*float(np.sqrt(np.mean(truth[stable]**2))))
        if rmse>limit:return dict(info,reason=f'Cooling prefix RMSE {rmse:.1f} K exceeds {limit:.1f} K for stable window',rmse_k=rmse,limit_k=limit)
    if peaks.sum()>=15:
        rmse=float(np.sqrt(np.mean(error[peaks]**2)))
        limit=max(250.,.25*float(np.sqrt(np.mean(truth[peaks]**2))))
        if rmse>limit:return dict(info,reason=f'Peak prefix RMSE {rmse:.1f} K exceeds softened {limit:.1f} K',rmse_k=rmse,limit_k=limit)
    return None

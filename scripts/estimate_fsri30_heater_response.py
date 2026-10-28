#!/usr/bin/env python3
"""Two-state empirical heater-response proposal, not a replacement solver."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from fsri30_data import load, save, sha


def fit(paths):
    design=[];targets=[]
    for path in paths:
        with gzip.open(path,'rt') as stream:r=json.load(stream)
        t=np.asarray(r['time_s']);h=r['heating_protocols']['fsri-film-ramp']
        y=np.asarray(r['cell_mean_temperature_k']['B10'])-298.15
        sensor=np.asarray(h['sensor_temperature_k'])-298.15;power=np.asarray(h['commanded_power_w'])
        dt=np.diff(t)
        design.append(np.column_stack([y[:-1],sensor[:-1],power[:-1],np.ones(len(dt))]))
        targets.append(np.column_stack([np.diff(y)/dt,np.diff(sensor)/dt]))
    return np.linalg.lstsq(np.concatenate(design),np.concatenate(targets),rcond=None)[0]


def predict(matrix,gain,feed,times):
    def rhs(t,state):
        power=np.clip(feed+gain*(.1*t-state[1]),0.,30.)
        return np.array([*state,power,1.])@matrix
    return solve_ivp(rhs,[0.,float(times[-1])],[0.,0.],t_eval=times,rtol=1e-7,atol=1e-8).y


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--training',type=Path,nargs='+',required=True)
    ap.add_argument('--holdout',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    matrix=fit(a.training);d,_=load();times=np.arange(0.,1501.,5.);truth=np.interp(times,d['time_s'],d['T5_k'])-298.15
    def objective(p):
        e=predict(matrix,p[0],p[1],times)[0,20:]-truth[20:]
        return float(np.sqrt(np.mean(e**2)))
    result=differential_evolution(objective,[(.125,2.),(0.,8.)],seed=7,popsize=6,maxiter=30,tol=.01)
    with gzip.open(a.holdout,'rt') as stream:hold=json.load(stream)
    t=np.asarray(hold['time_s']);actual=np.asarray(hold['cell_mean_temperature_k']['B10'])-298.15
    observed=predict(matrix,.125,2.,t)[0];err=observed-actual
    save(a.output,dict(training=[dict(path=str(p.resolve()),sha256=sha(p)) for p in a.training],
        holdout=dict(path=str(a.holdout.resolve()),sha256=sha(a.holdout),until_s=float(t[-1]),
            rmse_k=float(np.sqrt(np.mean(err**2))),final_error_k=float(err[-1])),
        derivative_coefficients=matrix.tolist(),proposal=dict(gain=float(result.x[0]),feedforward_w=float(result.x[1]),
            estimated_rmse_k=float(result.fun)),scope='Short-prefix empirical proposal; long extrapolation not validated, CUDA confirmation required'))


if __name__=='__main__':main()

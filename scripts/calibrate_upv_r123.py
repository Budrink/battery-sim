#!/usr/bin/env python3
"""Fresh GPU-only UPV calibration: R123 chemistry + I²R, from ambient to failure.

No old scenarios/profiles are read. Every candidate keeps measured material
properties fixed. The held-out experiment is never used to select parameters.
"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np
from scipy.optimize import lsq_linear
import upv_r123_model as model
import upv_r123_data as dataio

OUT=dataio.ROOT/'data/calibration/upv-r123-overcharge-v2'
PROBE=Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe')
HEAT=('resistance_ohm','resistance_temperature','overcharge_gain','r2_rate','r2_activation','iss_h','tim_h','case_h',
      'environment_h','surface_iss','surface_tim','bulk_slope')
FAILURE=('energy_j','gas_fraction','gas_yield','gas_rate','shell_strength_pa','opening_rate')
LINEAR={'resistance_temperature','resistance_damage','r2_activation','surface_iss','surface_tim','bulk_slope','gas_fraction','gas_yield'}


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def fingerprint():
    files=[Path(__file__),Path(model.__file__),Path(dataio.__file__),PROBE]
    files+=sorted((dataio.ROOT/'native').rglob('*.hpp'))+sorted((dataio.ROOT/'native').rglob('*.cuh'))
    files+=sorted((dataio.ROOT/'native').rglob('*.cpp'))+sorted((dataio.ROOT/'native').rglob('*.cu'))
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


def coordinate(p,key):
    return p[key] if key in LINEAR else np.log(p[key])


def value(x,key):
    return float(x if key in LINEAR else np.exp(x))


def range_for(key):
    lo,hi=model.BOUNDS[key]
    return (lo,hi) if key in LINEAR else (np.log(lo),np.log(hi))


def residual(run,meta,stage):
    at=np.asarray(run['time_s']);obs=dataio.observed(meta['data'],at);event=meta['event']['time_s'];blocks=[]
    windows=[(0.,min(300.,at[-1]),2.),(300.,min(event-20.,at[-1]),2.)] if stage=='heat' else [
        (0.,event-5.,3.),(event-5.,event+3.,100.),(event+3.,event+30.,30.),
        (event+30.,event+300.,10.),(event+300.,event+1800.,5.)]
    for lo,hi,scale in windows:
        mask_time=(at>=lo)&(at<=hi)
        for name in dataio.CHANNELS:
            # Source thermocouple attachment after ejection is not established;
            # keep its curves diagnostic rather than identifying bulk chemistry
            # from a jet-contaminated broken sensor.
            if stage!='heat' and name.startswith('C2_') and hi>event:continue
            error=np.asarray(run['temperature_k'][name])-obs[name]
            mask=mask_time&np.isfinite(error)
            if mask.any():blocks.append(error[mask]/(scale*np.sqrt(mask.sum())))
    if stage=='heat':
        dp=np.asarray(run['gas']['chamber']['pressure_pa'])/1e5-obs['pressure_bar']
        mask=np.isfinite(dp)
        if mask.any():blocks.append(dp[mask]/(.02*np.sqrt(mask.sum())))
    else:
        dp=np.asarray(run['gas']['chamber']['pressure_pa'])/1e5-obs['pressure_bar']
        for lo,hi,scale in [(0.,event,.03),(event,event+30.,.25),(event+30.,event+1800.,.1)]:
            mask=(at>=lo)&(at<=hi)&np.isfinite(dp)
            if mask.any():blocks.append(dp[mask]/(scale*np.sqrt(mask.sum())))
    return np.concatenate(blocks)


def metrics(run,meta):
    at=np.asarray(run['time_s']);obs=dataio.observed(meta['data'],at);event=meta['event']['time_s'];result={}
    for phase,lo,hi in [('warmup',0.,event-20.),('peak',event-3.,event+3.),
                        ('early',event+3.,event+30.),('propagation',event+30.,event+300.),('cooldown',event+300.,event+1800.)]:
        rows={}
        for channel in dataio.CHANNELS:
            error=np.asarray(run['temperature_k'][channel])-obs[channel]
            selected=error[(at>=lo)&(at<=hi)&np.isfinite(error)]
            if len(selected):rows[channel]=dict(rmse_k=float(np.sqrt(np.mean(selected**2))),bias_k=float(np.mean(selected)))
        if rows:result[phase]=rows
    pressure=np.asarray(run['gas']['chamber']['pressure_pa'])/1e5
    result['pressure']=dict(peak_bar=float(np.max(pressure)),observed_peak_bar=float(np.nanmax(obs['pressure_bar'])),
        peak_time_s=float(at[np.argmax(pressure)]),rmse_bar=float(np.sqrt(np.mean((pressure-obs['pressure_bar'])**2))))
    result['joule_j']={key:values[-1] for key,values in run['joule_energy_j'].items()}
    result['wall_s']=run['wall_s'];result['steps']=run['steps']
    result['balance']={key:run[key] for key in ('max_thermal_residual_j','max_chemical_residual_j','max_mass_residual_kg')}
    return result


def replay(scene,directory,label,timeout,refinement=1.):
    check_initial_voltage(scene)
    # Native output is transient. Retain only the best compressed replay and
    # small metrics; do not regenerate gigabytes of discarded trial fields.
    with tempfile.TemporaryDirectory(prefix='battery-r123-trial-') as tmp:
        root=Path(tmp);path=root/'scene.json';save(path,scene)
        env=dict(os.environ,BATTERY_SIM_PROGRESS='1')
        with (root/'run.json').open('w') as output,(directory/(label+'.log')).open('w') as progress:
            child=subprocess.Popen([str(PROBE),str(path)],stdin=subprocess.PIPE,stdout=output,stderr=progress,text=True,env=env)
            try:child.communicate(json.dumps(dict(refinement=refinement,checkpoint_path=str(root/'partial.json')))+'\n',timeout=timeout)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        if child.returncode:raise RuntimeError((directory/(label+'.log')).read_text()[-3000:])
        run=json.loads((root/'run.json').read_text())
        if 'error' in run:raise RuntimeError(run['error'])
    if run['max_thermal_residual_j']>1e-4 or run['max_chemical_residual_j']>1e-4 or run['max_mass_residual_kg']>1e-10:
        raise ValueError('Candidate failed energy/mass conservation')
    actual={c['id']:c for c in run['applied_objects']}
    for c in scene['cells']:
        a=actual[c['id']];t=c['thermal']
        if a['capacity_j_k']!=t['heat_capacity_j_per_k'] or a['conductivity_w_m_k']!=t['conductivity_w_per_m_k']:
            raise ValueError('Measured material properties changed')
    if scene['metadata']['experiment']=='no-cooling':
        electrical,_=dataio.electrical_record();offset=scene['metadata']['electrical']['offset_s']
        electrical_t=electrical[:,0]+offset;power=np.maximum(0.,electrical[:,1]*electrical[:,2])
        # Bound cumulative dissipated Joule heat by actually supplied electrical
        # work. The remaining work is not silently treated as another heat source.
        accumulated=np.r_[0.,np.cumsum(.5*(power[:-1]+power[1:])*np.diff(electrical_t))]
        available=np.interp(run['time_s'],electrical_t,accumulated,left=0.,right=accumulated[-1])
        joule=np.asarray(run['joule_energy_j']['C2'])
        if np.any(joule>available*1.01+1.):raise ValueError('I²R dissipates more than supplied U*I work')
    return run


def check_initial_voltage(scene):
    if scene['metadata']['experiment']!='no-cooling':return
    electrical,_=dataio.electrical_record()
    source=next(c for c in scene['cells'] if c['id']=='C2');law=source['electrical']
    temperature=source['thermal']['initial_temperature_k']
    resistance=np.clip(law['resistance_ohm']*np.exp(law['temperature_log_slope_per_k']*
        (temperature-law['reference_temperature_k'])),law['min_resistance_ohm'],law['max_resistance_ohm'])
    # Conservative engineering lower prior, NOT an inferred OCV(t). It prevents
    # replacing missing chemistry with an initial I*R drop incompatible with the
    # voltage of a fully charged 3.65-V-nominal NMC cell.
    if 3.65+electrical[0,2]*resistance>electrical[0,1]+.05:
        raise ValueError('Initial I*R contradicts measured terminal voltage and 3.65-V lower reversible-voltage prior')


def write_best(directory,stage,scene,run,p,meta,score):
    save(directory/(stage+'-parameters.json'),dict(parameters=p,objective=score,metrics=metrics(run,meta),
        fingerprint=fingerprint(),validation_used_for_selection=False,status='candidate, not independently validated'))
    save(directory/(stage+'-scenario.json'),scene)
    with gzip.open(directory/(stage+'-run.json.gz'),'wt') as stream:json.dump(run,stream,allow_nan=False)


def fit(args):
    stage=args.stage;directory=args.output;directory.mkdir(parents=True,exist_ok=True)
    names=HEAT if stage=='heat' else FAILURE
    p=dict(model.DEFAULT)
    if args.resume:
        p.update(json.loads((directory/(stage+'-parameters.json')).read_text())['parameters'])
    if args.start:
        # Analytic stack-balance seed. It fixes nothing: the objective is still
        # measured by an actual CUDA trajectory, on the same data and windows.
        seed=json.loads(args.start.read_text());p.update((seed.get('fit') or seed)['parameters'])
    if stage=='failure':p.update(json.loads((directory/'heat-parameters.json').read_text())['parameters'])
    history=[];best=float('inf');count=0;started=time.monotonic()
    def evaluate(candidate,label):
        nonlocal best,count
        count+=1;scene,meta=model.build('no-cooling',candidate,warmup=stage=='heat',duration=args.duration)
        save(directory/'status.json',dict(stage=stage,candidate=label,parameters=candidate,evaluations=count))
        print('START',label,flush=True)
        try:run=replay(scene,directory,label,args.timeout)
        except (subprocess.TimeoutExpired,ValueError,RuntimeError) as error:
            history.append(dict(label=label,parameters=candidate,status='failed',error=str(error)))
            save(directory/(stage+'-history.json'),history);print('FAILED',label,str(error),flush=True);return None
        r=residual(run,meta,stage);score=float(r@r);improved=score<best
        summary=dict(label=label,parameters=candidate,status='complete',objective=score,metrics=metrics(run,meta))
        history.append(summary);save(directory/(stage+'-history.json'),history)
        if improved:
            best=score;write_best(directory,stage,scene,run,candidate,meta,score)
        print('DONE',label,'objective',round(score,5),'best',round(best,5),'native_s',round(run['wall_s'],2),
              'elapsed_s',round(time.monotonic()-started),flush=True)
        return r,score
    base=evaluate(p,'baseline-'+stage)
    if base is None:raise RuntimeError('Baseline could not run; diagnose before optimizing')
    # Broad electrical/kinetic regimes first, not a long local optimizer trapped
    # on a flat objective. Every proposal is actually evaluated on CUDA.
    regimes=[('resistance-rising',dict(resistance_ohm=.015,resistance_temperature=.01)),
             ('resistance-low',dict(resistance_ohm=.003)),
             ('chemistry-early',dict(overcharge_gain=6.,r2_rate=1000.)),
             ('chemistry-late',dict(overcharge_gain=.5,r2_rate=30.))] if stage=='heat' else [
             ('energy-high',dict(energy_j=500000.)),('gas-fast',dict(gas_rate=100.)),
             ('strong-shell',dict(shell_strength_pa=1e7)),('gas-rich',dict(gas_fraction=.85))]
    if not args.no_regimes:
        for label,changes in regimes:
            result=evaluate(dict(p,**changes),label)
            if result and result[1]<base[1]:p=dict(p,**changes);base=result
    lo=np.array([range_for(k)[0] for k in names]);hi=np.array([range_for(k)[1] for k in names])
    for iteration in range(args.rounds):
        x=np.array([coordinate(p,k) for k in names]);columns=[];steps=[]
        for j,key in enumerate(names):
            direction=-1. if x[j]>.5*(lo[j]+hi[j]) else 1.
            step=direction*min(.08*(hi[j]-lo[j]),(hi[j]-x[j]) if direction>0 else (x[j]-lo[j]))
            if abs(step)<.001*(hi[j]-lo[j]):step=-direction*.08*(hi[j]-lo[j])
            q=dict(p);q[key]=value(x[j]+step,key);result=evaluate(q,f'{stage}-r{iteration}-{key}')
            columns.append((result[0]-base[0])/step if result else np.zeros_like(base[0]));steps.append(step)
        jac=np.column_stack(columns);span=hi-lo
        # One bounded log-secant proposal per round, with a fixed trust region.
        solution=lsq_linear(jac*span,-base[0],bounds=(np.maximum((lo-x)/span,-.25),np.minimum((hi-x)/span,.25)))
        proposed=np.clip(x+solution.x*span,lo,hi)
        q=dict(p,**{key:value(proposed[j],key) for j,key in enumerate(names)})
        result=evaluate(q,f'{stage}-r{iteration}-joint')
        best_row=min((v for v in history if v['status']=='complete'),key=lambda v:v['objective'])
        if best_row['objective']<base[1]:
            p=best_row['parameters'];base=evaluate(p,f'{stage}-r{iteration}-accepted')
        else:break
    final=json.loads((directory/(stage+'-parameters.json')).read_text())
    save(directory/'status.json',dict(stage=stage,status='finished',best_objective=final['objective'],evaluations=count,
         wall_s=time.monotonic()-started,validation_used_for_selection=False))
    print(json.dumps(final['metrics'],indent=2),flush=True)


def main():
    cli=argparse.ArgumentParser(__doc__);cli.add_argument('stage',choices=['heat','failure','trial','validate'])
    cli.add_argument('--output',type=Path,default=OUT);cli.add_argument('--timeout',type=float,default=600.)
    cli.add_argument('--rounds',type=int,default=1);cli.add_argument('--no-regimes',action='store_true')
    cli.add_argument('--resume',action='store_true',help='Start from the best current R123/Joule candidate, never a legacy profile')
    cli.add_argument('--start',type=Path,help='Seed from scripts/analyze_upv_heat_balance.py output, evaluated like any other candidate')
    cli.add_argument('--duration',type=float);args=cli.parse_args()
    if args.stage in ('heat','failure'):fit(args);return
    args.output.mkdir(parents=True,exist_ok=True);p=dict(model.DEFAULT)
    name='active-cooling' if args.stage=='validate' else 'no-cooling'
    if args.stage=='validate':
        profile=json.loads((args.output/'failure-parameters.json').read_text())
        if profile['fingerprint']!=fingerprint():raise ValueError('Model changed after freeze')
        if (args.output/'validate.json').exists():raise ValueError('Held-out replay already exists')
        p=profile['parameters']
    scene,meta=model.build(name,p,duration=args.duration);run=replay(scene,args.output,args.stage,args.timeout)
    save(args.output/(args.stage+'.json'),dict(parameters=p,metrics=metrics(run,meta),validation_used_for_selection=False))
    write_best(args.output,args.stage,scene,run,p,meta,float(residual(run,meta,'failure')@residual(run,meta,'failure')))
    print(json.dumps(metrics(run,meta),indent=2))


if __name__=='__main__':main()

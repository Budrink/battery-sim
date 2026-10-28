#!/usr/bin/env python3
"""Small, frozen-parameter gas-physics ablation. NOT a calibration search."""
import argparse
import gzip
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
import calibrate_upv_r123 as cal
import upv_r123_model as model
import upv_r123_data as dataio

# Engineering pseudo-fluid, NOT measured electrolyte properties. Its mass is
# allocated within the measured inert heat capacity, never added to that C.
PHASE=dict(initial_mass_kg=.03,liquid_cp=1800.,density_kg_m3=1200.,
    latent_j_per_kg=500000.,boiling_temperature_k=400.,saturation_reference_pa=101325.,
    # First comparison must NOT improve by inventing an additional fuel stock.
    # Vapor carries latent/sensible energy; R3 retains the original R2 budget.
    # The supported combustible-vapor option needs a separate identified or
    # explicitly repartitioned chemical-energy budget before calibration.
    vapor_specific_r=287.,relaxation_per_s=10.,chemical_j_per_kg=0.)
CASES=dict(control={},mixing=dict(mixing_diffusivity_m2_s=2e-5,mixing_flow_fraction=.25),
    phase=dict(volatile_phase=PHASE),
    combined=dict(mixing_diffusivity_m2_s=2e-5,mixing_flow_fraction=.25,volatile_phase=PHASE))


def summarize(run,meta):
    t=np.asarray(run['time_s']);gas=run['gas'];event=meta['event']['time_s'];c=run['chemistry']['C2']
    obs=dataio.observed(meta['data'],t);rows=[]
    for target in (600.,event-5.,event,event+4.,event+15.,t[-1]):
        if target>t[-1]:continue
        j=int(np.argmin(abs(t-target)))
        groups={}
        for label,predicate in [('chamber',lambda n:n=='chamber'),('near',lambda n:n.startswith('near-')),
                                ('outer_void',lambda n:n.startswith('void-')),
                                ('gaps',lambda n:n.startswith('gap-')),('inside',lambda n:n.startswith('inside-'))]:
            groups[label]={k:float(sum(v[k][j] for n,v in gas.items()if predicate(n)))
                for k in ('mass_kg','chemical_j','oxidizer_kg','released_energy_j','vapor_kg','latent_j')}
        row=dict(time_s=float(t[j]),c2_mean_c=run['cell_mean_temperature_k']['C2'][j]-273.15,
            precursor_g=c['r2_precursor_kg'][j]*1000,r2_power_w=c['r2_power_w'][j],
            r2_gas_g=c['r2_generated_gas_kg'][j]*1000,evaporated_g=c['evaporated_kg'][j]*1000,
            condensed_g=c['condensed_kg'][j]*1000,
            c3_iss_c=run['temperature_k']['108'][j]-273.15,c3_iss_observed_c=float(obs['108'][j]-273.15),groups=groups)
        zones=run.get('chemistry_zones',{}).get('C2',{})
        row['zones']=[dict(id=int(n),temperature_c=z['temperature_k'][j]-273.15,
            precursor_g=z['precursor_kg'][j]*1000,r2_power_w=z['r2_power_w'][j],
            r2_rate_per_s=z['r2_rate_per_s'][j],r2_gas_g=z['r2_generated_gas_kg'][j]*1000)
            for n,z in zones.items()]
        rows.append(row)
    pressure=np.asarray(dataio.chamber_pressure_pa(run))/1e5
    return dict(end_s=float(t[-1]),wall_s=run['wall_s'],peak_chamber_bar=float(pressure.max()),
        peak_time_s=float(t[pressure.argmax()]),observed_sampled_peak_bar=float(np.nanmax(obs['pressure_bar'])),
        residuals={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k},snapshots=rows,
        calibrated=False,validation_used=False)


def plot(out,cases,reference=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8));observed=False
    inputs=[(name,out/name/'trial-run.json.gz')for name in cases]
    if reference is not None:inputs.append(('previous geometry',reference))
    for name,p in inputs:
        if not p.exists():continue
        with gzip.open(p,'rt')as f:r=json.load(f)
        t=np.asarray(r['time_s']);d,_=dataio.load('no-cooling');obs=dataio.observed(d,t);event=dataio.event(d)['time_s']
        for ax,key in zip(axes.flat,('pressure','108','C2_ISS','107')):
            y=np.asarray(dataio.chamber_pressure_pa(r))/1e5 if key=='pressure'else np.asarray(r['temperature_k'][key])-273.15
            ax.plot(t,y,label=name)
            if not observed:ax.plot(t,obs['pressure_bar']if key=='pressure'else obs[key]-273.15,'k--',lw=1,label='experiment')
            ax.set_title({'pressure':'Chamber pressure [bar]','108':'C3 / ISS [C]','C2_ISS':'C2 / ISS [C]','107':'C3 / TIM [C]'}[key])
            ax.set_xlim(event-30,t[-1]);ax.grid(alpha=.2);ax.legend(fontsize=8)
        observed=True
    fig.suptitle('Frozen solid/electrical/R1/R2 parameters; new gas priors, NOT calibrated')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cases',nargs='+',choices=CASES,default=['control','combined'])
    ap.add_argument('--duration',type=float,default=950.);ap.add_argument('--timeout',type=float,default=1800.)
    ap.add_argument('--refinement',type=float,default=1.)
    ap.add_argument('--reference',type=Path,help='Saved no-cooling replay to overlay, never rerun or fit')
    ap.add_argument('--coupled-films',action='store_true',help='Local exponential R3/cooling integration; same physical laws')
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    base=json.loads(a.base.read_text());base=base.get('parameters',base)
    fp=cal.fingerprint();state=dict(status='running',fingerprint=fp,cases={},parameters=base,diagnostic=True,
        reference=str(a.reference)if a.reference else None)
    cal.save(out/'status.json',state)
    def stop(_sig,_frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        for label in a.cases:
            if fp!=cal.fingerprint():raise RuntimeError('Model changed during physics comparison')
            folder=out/label;folder.mkdir(exist_ok=False)
            scene,meta=model.build('no-cooling',base,duration=a.duration,gas_extensions=CASES[label])
            scene['gas_network']['coupled_gas_films']=a.coupled_films
            if a.coupled_films:scene['simulation']['minimum_dt_s']=.005
            scene['metadata']['zone_diagnostics']=True
            cal.save(folder/'scenario.json',scene)
            cal.save(folder/'parameters.json',dict(parameters=base,gas_extensions=CASES[label],fingerprint=fp,
                calibrated=False,solid_and_electrical_parameters_frozen=True))
            print('START',label,flush=True);started=time.monotonic()
            def reject(run):
                t=run['time_s'];obs=dataio.observed(meta['data'],t)
                if t[-1]<meta['event']['time_s']-20 and t[-1]>120:
                    mask=np.asarray(t)>t[-1]-30
                    for key in ('C2_TIM','C2_ISS'):
                        err=np.asarray(run['temperature_k'][key])-obs[key]
                        if mask.sum()>3 and (np.all(err[mask]>40)or np.all(err[mask]<-40)):
                            return 'Gross persistent pre-event mismatch '+key
                return None
            try:
                run=cal.replay(scene,folder,'trial',a.timeout,refinement=a.refinement,reject_partial=reject)
                with gzip.open(folder/'trial-run.json.gz','wt')as f:json.dump(run,f,allow_nan=False)
                state['cases'][label]=summarize(run,meta)
            except KeyboardInterrupt:raise
            except Exception as exc:
                state['cases'][label]=dict(error=str(exc),wall_s=time.monotonic()-started)
            cal.save(folder/'summary.json',state['cases'][label]);cal.save(out/'status.json',state)
            plot(out,a.cases,a.reference);print('DONE',label,state['cases'][label].get('peak_chamber_bar',state['cases'][label].get('error')),flush=True)
        state['status']='complete'if all('error'not in v for v in state['cases'].values())else'failed'
    except KeyboardInterrupt:state['status']='stopped'
    finally:cal.save(out/'status.json',state)

if __name__=='__main__':main()

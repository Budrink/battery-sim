#!/usr/bin/env python3
"""Frozen-physics averaged CUDA transport acceptance and coupled replay."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import calibrate_upv_r123 as cal
import upv_r123_data as dataio
from check_upv_restart import native
from check_upv_local_max import summary, local_plot
from check_upv_monotone_gas import plot

AUDIT = Path('/tmp/battery-sim-cuda-v1/battery_sim_gas_limiter_snapshot')
LABELS = dict(baseline='Исходный парный перенос', averaged='Усреднённый сквозной перенос')
TITLE = 'Усреднённый газоперенос; химия, материалы и контакты не подбираются'


def averaged_scene(original,duration):
    if not np.isfinite(duration) or duration<=0.:
        raise ValueError('Duration must be positive and finite')
    scene=copy.deepcopy(original)
    scene['gas_network']['network_integrator']='averaged'
    scene['simulation']['duration_s']=duration
    scene['failure_sample_times_s']=sorted({float(t)for t in scene.get('failure_sample_times_s',[])
        if 0.<=t<=duration}|{0.,duration})
    return scene


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=950.)
    ap.add_argument('--plot-only',action='store_true')
    args=ap.parse_args();out=args.output.resolve()
    if args.plot_only:
        local_plot(out,labels=LABELS,title=TITLE);return 0
    out.mkdir(exist_ok=False)
    source=dataio.ROOT/'data/calibration/upv-separated-chemistry-v6/thermal-gas-mixing-20261020-r1/temperature'
    original=json.loads((source/'scenario.json').read_text());scene=averaged_scene(original,args.duration)
    core=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()
    state=dict(status='running',source=str(source),native_sha256=core,diagnostic_only=True,
        calibrated=False,validation_run=False,cases={},acceptance={},duration_s=args.duration)
    cal.save(out/'status.json',state)
    try:
        # Characterization of the old bug, and regression requirements for the
        # replacement. The observed orifice input is not used as a ground-truth
        # physical flux at arbitrarily large area/long time.
        fixture=json.loads(subprocess.check_output([str(AUDIT),'--throughflow','averaged'],text=True))
        cal.save(out/'throughflow.json',fixture)
        def entry(volume,dt,area):
            return next(r for r in fixture['rows']if r['volume_m3']==volume and r['dt_s']==dt and r['area_factor']==area)
        first=entry(1e-7,.005,1.);larger=entry(1e-7,.005,10.);longer=entry(1e-7,.01,1.)
        assert first['accepted_inlet_kg_s']>.99*first['raw_flow_kg_s'],'Transit still inventory capped'
        assert larger['accepted_inlet_kg_s']>9.*first['accepted_inlet_kg_s'],'Area does not increase throughput'
        assert abs(longer['accepted_inlet_kg_s']/first['accepted_inlet_kg_s']-1)<.005,'Window changes throughput dramatically'
        assert abs(entry(1e-5,.005,1.)['accepted_inlet_kg_s']/first['accepted_inlet_kg_s']-1)<1e-6,'Artificial transit-volume dependence'
        state['acceptance']['throughflow']=dict(flow_kg_s=first['accepted_inlet_kg_s'],
            area_x10_flow_kg_s=larger['accepted_inlet_kg_s'],window_x2_flow_kg_s=longer['accepted_inlet_kg_s'])
        inputs=dataio.ROOT/'data/calibration/upv-separated-chemistry-v6/limiter-snapshots-20261020-r1'
        for at in ('882.877000','930.000000','950.000000'):
            snapshot=inputs/f'baseline-{at}-0.005-input.json'
            result=json.loads(subprocess.check_output([str(AUDIT),str(snapshot),'averaged'],text=True))
            cal.save(out/f'snapshot-{at}.json',result)
            if result['net_mass_scale']<.99:raise ValueError('Real snapshot still globally throttled: '+at)
            state['acceptance'][at]=dict(net_mass_scale=result['net_mass_scale'],entropy_backtracks=result['entropy_backtracks'],
                mass_residual_kg=result['mass_residual_kg'],energy_residual_j=result['energy_residual_j'])
        cal.save(out/'status.json',state)
        folder=out/'averaged';folder.mkdir();cal.save(folder/'scenario.json',scene)
        print('START averaged coupled replay',args.duration,flush=True)
        run=native(folder/'scenario.json',folder/'replay',dict(stop_after_first_rupture=False),max(1800.,args.duration*2.2))
        report=summary(scene,run);report['averaged_profile']={k:v for k,v in run['solver_profile'].items()if 'averaged'in k}
        cal.save(folder/'report.json',report);state['cases']['averaged']=report;cal.save(out/'status.json',state)
        with gzip.open(source/'replay/run.json.gz','rt')as stream:baseline=json.load(stream)
        state['cases']={'baseline':summary(original,baseline),'averaged':report}
        data,_=dataio.load('no-cooling');plot(out,dict(baseline=baseline,averaged=run),data,labels=LABELS,title=TITLE)
        if hashlib.sha256(cal.PROBE.read_bytes()).hexdigest()!=core:raise ValueError('Binary changed during replay')
        state['status']='complete'
    except Exception as exc:state.update(status='failed',error=str(exc))
    finally:
        cal.save(out/'status.json',state);print('AVERAGED_STATUS',state['status'],state.get('error',''),flush=True)
    return 0 if state['status']=='complete'else 1


if __name__=='__main__':raise SystemExit(main())

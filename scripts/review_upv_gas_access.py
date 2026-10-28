#!/usr/bin/env python3
"""Review explicitly selected completed geometry trials without rerunning."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from calibrate_upv_r123 import save
from probe_upv_gas_access import plot
from upv_gas_access import shape_report,received_gas_heat,CHANNELS
from upv_gas_heat_budget import gas_region
import upv_r123_data as dataio


def load(path):
    with gzip.open(path,'rt')as f:return json.load(f)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--trials',type=Path,nargs='+',required=True)
    ap.add_argument('--source-label',default='Исходный r5')
    ap.add_argument('--selection-metric',choices=['shape_objective','shape_with_front_objective'],default='shape_objective')
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False);data,record=dataio.load('no-cooling')
    baseline=load(args.source/'replay/run.json.gz')
    review=dict(source=str(args.source.resolve()),source_label=args.source_label,selection_metric=args.selection_metric,experimental_sha256=record['canonical_sha256'],
        shape_code_sha256=hashlib.sha256((Path(__file__).parent/'upv_gas_access.py').read_bytes()).hexdigest(),
        baseline=shape_report(baseline,data),trials=[],accepted_calibration=False,
        stage='Heating shape and gas distribution only; peak amplitudes and cooling not fitted')
    best=None;best_run=None
    for path in args.trials:
        run=load(path/'replay/run.json.gz')
        if run.get('partial')or run['advanced_to_s']<950.:raise ValueError('Incomplete diagnostic trial')
        scene=json.loads((path/'scenario.json').read_text());geometry=scene['metadata']['gas_access_geometry']
        report=shape_report(run,data)
        report.update(trial=str(path.resolve()),geometry={k:geometry[k]for k in
            ['protected_accessible_fraction','holder_lip_open_fraction','plume_exit_fraction','backing_open_fraction','witness_edge_open_fraction','jet_wetted_fraction','outer_insulation_open_fraction','iss3_accessible_fraction','open_iss_pad_wetted_fraction','cover_outer_insulation']if k in geometry},
            accepted_gas_heat_880_900_j=received_gas_heat(run),
            realtime_ratio=(run['advanced_to_s']-run['started_at_s'])/run['wall_s'],
            balances={k:v for k,v in run.items()if k.startswith('max_')and'residual'in k})
        report['peak_times_s']={k:float(np.asarray(run['time_s'])[np.argmax(
            np.where((np.array(run['time_s'])>=881.)&(np.array(run['time_s'])<=930.),run['temperature_k'][k],-np.inf))])for k in CHANNELS}
        balances={}
        for pad in ['ISS1','TIM1','ISS2','TIM2','ISS3','TIM3']:
            names={n for n in run['gas']if n.startswith('gap-'+pad+'-')}
            before,after=(gas_region(scene,run,names,at)for at in [880.,900.])
            balances[pad]={k:after[k]-before[k]for k in
                ['net_transport_j','r3_j','actual_heat_to_surfaces_j','gas_energy_change_j','energy_closure_j']}
        report['slit_energy_balances_880_900']=balances
        if max(abs(v['energy_closure_j'])for v in balances.values())>1e-4:
            raise ValueError('Local gas/surface heat budget does not close')
        review['trials'].append(report)
        if not report['no_heat_rejected']and(best is None or report[args.selection_metric]<best[args.selection_metric]):
            best=report;best_run=run
    if best is None:raise ValueError('No adequately excited shape candidate')
    review['selected_shape_candidate']=best['trial']
    plot(out,{args.source_label:baseline,'Новая геометрия':best_run},data)
    plot(out,{'Новая геометрия':best_run},data,filename='shape-current.png')
    save(out/'review.json',review)
    print(json.dumps(dict(output=str(out),baseline_shape_objective=review['baseline']['shape_objective'],
        selection_metric=args.selection_metric,
        candidates=[dict(trial=r['trial'],score=r[args.selection_metric],
            curve_shape_objective=r['shape_objective'],no_heat_rejected=r['no_heat_rejected'])for r in review['trials']],
        selected=best['trial'])),flush=True)


if __name__=='__main__':main()

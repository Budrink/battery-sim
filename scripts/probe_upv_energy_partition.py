#!/usr/bin/env python3
"""Fresh causal probe: fixed chemical energy, different local/mobile partition."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import calibrate_upv_r123 as cal
import upv_r123_data as dataio
import upv_cooling_fit as fit
from check_upv_restart import native
from resolve_upv_rupture_impulse import compact,plot_impulse
from upv_cooling_review import plot_cooling


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fraction',type=float,default=.84)
    ap.add_argument('--rate-scale',type=float,default=1.)
    ap.add_argument('--end',type=float,default=1500.)
    args=ap.parse_args()
    if not .25<=args.rate_scale<=4.:raise ValueError('Rate multiplier outside diagnostic bounds')
    out=args.output.resolve();out.mkdir(exist_ok=False)
    original=json.loads((args.reference/'scenario.json').read_text())
    scene=fit.partition_candidate(original,args.fraction)
    for cell in scene['cells']:
        if cell.get('conversion'):cell['conversion']['reaction_rate_per_s']*=args.rate_scale
    scene['metadata']['energy_partition_probe']['additional_rate_multiplier']=args.rate_scale
    scene['simulation']['duration_s']=args.end
    scene['process_history_selection']=dict(gas_prefixes=['inside-C2-','gap-'],channel_prefixes=['rupture-'])
    data,_=dataio.load('no-cooling');marker=dataio.event(data)
    scene['failure_sample_times_s']=dataio.observation_times(data,marker,args.end)
    cal.save(out/'scenario.json',scene)
    cal.save(out/'contract.json',dict(source=str(args.reference.resolve()),
        source_sha256=hashlib.sha256((args.reference/'scenario.json').read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest(),
        fresh_start=True,validation_used=False,energy_partition_probe=scene['metadata']['energy_partition_probe']))
    run=None
    try:
        run=native(out/'scenario.json',out/'replay',dict(stop_after_first_rupture=False,
            restart_contract=hashlib.sha256(cal.PROBE.read_bytes()).hexdigest(),
            save_unvented_restart=str(out/'sealed-state.bin')),max(2000.,args.end*2.2),
            prune_fn=lambda r:fit.rejection_reason(r,data,marker))
        status=dict(status='complete',accepted_calibration=False)
    except (RuntimeError,TimeoutError)as exc:
        status=dict(status='rejected',reason=str(exc),accepted_calibration=False)
        partial=out/'replay/incomplete-run.json.gz'
        if partial.exists():run=json.load(gzip.open(partial,'rt'))
    if run is not None:
        status['fit']=fit.score(run,data,args.end)
        status['residuals']={k:run[k]for k in ('max_thermal_residual_j','max_chemical_residual_j','max_mass_residual_kg')}
        curves=dict(candidate=compact(scene,run));labels=dict(candidate='Доля химпотенциала газа '+str(args.fraction))
        path=args.reference/'curves.json'
        if path.exists():curves['reference']=json.loads(path.read_text());labels['reference']='Исходный профиль'
        cal.save(out/'curves.json',curves['candidate']);plot_cooling(out,curves,data,labels);plot_impulse(out,curves,data,labels)
    cal.save(out/'report.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()

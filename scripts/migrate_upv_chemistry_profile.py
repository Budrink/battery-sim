#!/usr/bin/env python3
"""Explicit v5 -> separated charge/R1 seed migration; never a fitted profile."""
import argparse
import hashlib
import json
from pathlib import Path
import calibrate_upv_r123 as cal
import upv_r123_model as model
import upv_r123_data as dataio

REVISION='separated-charge-r1-v6'

def migrate(parameters):
    p=dict(parameters)
    if 'overcharge_gain' in p:
        if 'charge_formation_gain' in p:raise ValueError('Conflicting charge coefficients')
        p['charge_formation_gain']=p.pop('overcharge_gain')
    if p.pop('precursor_voltage',0.)!=0.:
        raise ValueError('Nonzero retired voltage yield needs manual review, not silent migration')
    unknown=set(p)-set(model.DEFAULT)
    if unknown:raise ValueError('Unsupported seed fields: '+', '.join(sorted(unknown)))
    return p

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();pointer_path=dataio.ROOT/'data/calibration/upv-charge-degradation-v5/current-profile.json'
    raw=pointer_path.read_bytes();pointer=json.loads(raw);source=dataio.ROOT/pointer['parameters_file']
    original=json.loads(source.read_text())['parameters'];p=migrate(original)
    if set(p)!=set(model.DEFAULT):raise ValueError('Full seed required; defaults must not fill missing fitted fields')
    if args.output.exists():raise ValueError('Refusing to overwrite a seed')
    old=json.loads((dataio.ROOT/pointer['calibration_result']/'scenario.json').read_text())
    new,_=model.build('no-cooling',p,warmup=True,duration=626.956)
    for key in ('ports','interfaces','shells','boundaries','gas_network','measurement_probes'):
        if new[key]!=old[key]:raise ValueError('Frozen transfer geometry changed: '+key)
    old_cells=[{k:v for k,v in c.items()if k not in ('conversion','electrical')}for c in old['cells']]
    new_cells=[{k:v for k,v in c.items()if k not in ('conversion','electrical')}for c in new['cells']]
    if old_cells!=new_cells:raise ValueError('Solid geometry/materials changed')
    cal.save(args.output,dict(revision=REVISION,status='uncalibrated migration seed, NOT a calibration',
        parameters=p,model_fingerprint=cal.fingerprint(),defaults_reset=False,
        source=pointer['parameters_file'],source_pointer_sha256=hashlib.sha256(raw).hexdigest(),
        source_parameters_sha256=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest(),
        parameters_sha256=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),
        migration={'overcharge_gain':'charge_formation_gain','precursor_voltage':'removed; verified zero'},
        frozen_transfer_geometry_verified=True,calibration_metrics_inherited=False))
    print(args.output)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""One fresh sensitivity to UNMEASURED module vent area, open far field fixed.

Wait for the current control; never compete for the GPU. Effective subgrid
aperture area is a fixture prior, not a new material/chemical calibration.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

from fsri30_data import ROOT,save,sha
from check_fsri30_open_boundary import BASE


def aperture_trial(base,scale):
    if not 1. <= scale <= 4.:raise ValueError('Outside declared area prior [1,4]')
    scene=deepcopy(base);changed=[]
    for channel in scene['gas_network']['channels']:
        if not channel['id'].startswith('vent-'):continue
        if channel['kind']!='passage' or channel['from'] not in ('body-07','body-22') or not channel['to'].startswith('air-'):
            raise ValueError('Unexpected module exit; do not change can rupture')
        before=channel['area_m2'];channel['area_m2']*=scale
        if channel['area_m2']>.02**2:raise ValueError('Aperture exceeds receiving 20 mm face')
        changed.append(dict(channel=channel['id'],before_m2=before,after_m2=channel['area_m2']))
    if len(changed)!=2:raise ValueError('Expected exactly two published front/back module vents')
    scene['name']+=f' / module effective vent area x{scale:g}'
    scene['metadata']['exhaust_aperture_diagnostic']=dict(scale=scale,changed=changed,
        measured_dimensions=False,subgrid_aperture=True,
        scope='Effective opening-area sensitivity only; solid meshes/capacities/contact exposure retained',
        geometry_error_note='Added opening is <1% of each complete ABS face; no claim of exact CAD reconstruction')
    return scene


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--wait-for',type=Path,required=True);ap.add_argument('--area-scale',type=float,default=4.)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    base=json.loads(BASE.read_text());scene=aperture_trial(base,args.area_scale)
    save(out/'scenario-base.json',scene)
    state=dict(status='waiting-for-control',area_scale=args.area_scale,renewal_m3_s=.01,
        native_sha256=sha(args.probe),source_sha256=sha(BASE),fresh_from_zero=True,
        chemistry_frozen=True,mesh_mm=20.,calibrated=False,validation_passed=False)
    save(out/'status.json',state)
    while True:
        control=json.loads((args.wait_for/'status.json').read_text())
        if control['status']!='running':break
        time.sleep(5.)
    if control['status']!='complete':
        state.update(status='not-started',error='Control did not complete; inspect it before further fitting')
        save(out/'status.json',state);return
    state['status']='running';save(out/'status.json',state)
    cmd=[sys.executable,str(ROOT/'scripts/check_fsri30_open_boundary.py'),
        '--probe',str(args.probe.resolve()),'--base',str(out/'scenario-base.json'),
        '--output',str(out/'check'),'--flows-m3-s','.01']
    with (out/'controller.log').open('w') as log:
        child=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    result=json.loads((out/'check/status.json').read_text())
    state.update(status=result['status'],controller_returncode=child.returncode,results=result['results'])
    save(out/'status.json',state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

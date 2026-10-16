#!/usr/bin/env python3
"""Export the frozen contact law to an ordinary native viewer/headless scenario.

The export is not an autonomous runaway model. C2 remains measured forcing.
Calibration probe snapshots instead keep baseline parameters for repeated fits.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from calibrate_upv import CONTACTS,PARAMETER_SETS,model_hash,save,validate_profile
from generate_upv import ROOT


def contact_group(a,b):
    pair=frozenset((a,b))
    names={frozenset(('battery','ISS')):'battery_iss_scale',
           frozenset(('battery','TIM')):'battery_tim_scale',
           frozenset(('ISS','aluminium')):'iss_aluminium_scale',
           frozenset(('ISS','steel')):'iss_steel_scale',
           frozenset(('TIM','aluminium')):'tim_aluminium_scale'}
    return names[pair]


def calibrated_scene(scene,profile):
    validate_profile(profile)
    family=profile['parameter_set']
    if family not in ('state_geometry','directed_geometry') or not profile['optimizer_success']:raise ValueError('Converged geometry profile required')
    if scene['metadata']['connectivity_model']!=family:raise ValueError('Matching geometry scenario required')
    out=copy.deepcopy(scene);params=profile['parameters']
    if set(params)!=set(PARAMETER_SETS[family]):raise ValueError('Unexpected parameter set')
    def response(r,material):
        r['linear']=[params['surface_linear']]*3;r['quadratic']=[params['surface_quadratic']]*3
        if family=='directed_geometry':r['temperature_contrast']=[params[material.lower()+'_temperature_contrast']]*3
    def reference(law,group):
        h=100*params[group]
        if not 1<h<10000:raise ValueError('Reference contact outside model bounds')
        law['geometry_reference_logit']=math.log((h-1)/(10000-h))
    objects={c['id']:c for c in out['cells']};ports={p['id']:p for p in out['ports']}
    for c in out['cells']:response(c['thermal']['surface_connectivity'],c['material_id'])
    for e in out['interfaces']:
        a=objects[ports[e['port_a']]['cell_id']]['material_id'];b=objects[ports[e['port_b']]['cell_id']]['material_id']
        reference(e['law'],contact_group(a,b))
    for b in out['boundaries']:
        if b['id'].startswith('source-'):
            material=objects[ports[b['port_id']]['cell_id']]['material_id'];group=contact_group('battery',material)
            reference(b['connectivity'],group);response(b['remote_surface_connectivity'],'battery')
            b['conductance_w_per_k']*=params[group]
        elif b['id'].startswith('ambient-'):b['conductance_w_per_k']*=params['ambient_scale']
    out['metadata'].update(calibration_status='Frozen pre-event contact fit; topology hypothesis and unmeasured fixture priors remain',
                           applied_parameters=params,model_sha256=profile['model_sha256'],
                           execution='Native viewer/headless directly. Do not reapply calibration probe scales to this exported scene.')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    profile=json.loads(a.profile.read_text());report=json.loads(a.report.read_text())
    if profile['model_sha256']!=model_hash() or report['model_sha256']!=model_hash():raise ValueError('Model changed')
    digest=hashlib.sha256(a.profile.read_bytes()).hexdigest()
    if report['frozen_profile_sha256']!=digest:raise ValueError('Profile changed')
    path=Path(report['scenario_path']);path=path if path.is_absolute() else ROOT/path
    result=calibrated_scene(json.loads(path.read_text()),profile)
    result['metadata']['frozen_profile_sha256']=digest
    save(a.output,result);print(a.output)


if __name__=='__main__':main()

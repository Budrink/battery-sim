"""Prepare a wall-contact hypothesis; never starts a simulation."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

# Contact-only h(T_donor) before unchanged material half-resistances.
# Same table for both directions, selected donor response, never averaged.
TARGET_H = [(25.,75.),(100.,100.),(200.,175.),(300.,350.),(500.,700.),(800.,950.)]


def candidate(source):
    scene=deepcopy(source)
    batteries={c['id'] for c in scene['cells'] if c.get('role')=='battery'}
    shells=[s for s in scene['shells'] if s['contents_id'] in batteries]
    assert len(shells)==len(batteries)==30
    for shell in shells:
        reference=shell['contents_contact_h']
        table=[]
        for celsius,h in TARGET_H:
            opening=(h/reference-.01)/1.98
            assert 0.<opening<1.
            table.append([celsius+273.15,math.log(opening/(1.-opening))])
        shell['contents_surface_connectivity']={'temperature_log_response':table}
    restored=deepcopy(scene)
    for a,b in zip(restored['shells'],source['shells']):
        a.pop('contents_surface_connectivity',None)
        if 'contents_surface_connectivity' in b:
            a['contents_surface_connectivity']=deepcopy(b['contents_surface_connectivity'])
    assert restored==source
    scene['metadata']['wall_contact_hypothesis']={
        'scope':'all 180 contents-to-shell contacts; no bulk, seams, gas films or heater-contact changes',
        'mechanism':'existing directed connectivity; reversible temperature response, damage unchanged',
        'temperature_variable':'hotter donor temperature, K; no endpoint averaging',
        'contact_h_table_c_w_m2_k':TARGET_H,
        'serial_resistances_unchanged':True,
        'material_identity_not_measured':True,
        'chemistry_and_mass_transfer_unchanged':True,
        'calibrated':False,'validation_passed':False}
    return scene


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    scene=candidate(json.loads(args.source.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(scene,f,indent=2)
    print(json.dumps(scene['metadata']['wall_contact_hypothesis'],indent=2))

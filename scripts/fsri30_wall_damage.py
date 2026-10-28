"""Irreversible inner-insulation candidate. Preparation only, no solver launch."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

ONSET_K=423.15  # 150 C; negligible rate here, steep acceleration near 200 C.
RATE=1e-6
SCALE_K=5.
INTACT_H=75.
DESTROYED_H=950.


def contact_law():
    floor,span=5.,990.
    logit=lambda h:math.log((h-floor)/(floor+span-h))
    return dict(kind='directed_geometry', background_conductance_w_per_k=floor,
        active_conductance_w_per_k=span, geometry_reference_logit=logit(INTACT_H),
        interface_damage_log_response=logit(DESTROYED_H)-logit(INTACT_H),
        damage_onset_k=ONSET_K, damage_rate_at_onset_per_s=RATE,
        damage_temperature_scale_k=SCALE_K, endpoint_damage_weight=0.,
        destruction_damage=1., destroyed_background_conductance_w_per_k=DESTROYED_H,
        destroyed_active_conductance_w_per_k=0.)


def candidate(source):
    scene=deepcopy(source)
    ids={c['id'] for c in scene['cells'] if c.get('role')=='battery'}
    shells=[s for s in scene['shells'] if s['contents_id'] in ids]
    assert len(shells)==len(ids)==30
    for shell in shells:
        shell.pop('contents_surface_connectivity',None)
        shell['contents_connectivity_per_m2']=contact_law()
    # Only the inner-contact law replaces the reversible temperature profile.
    restored=deepcopy(scene)
    for a,b in zip(restored['shells'],source['shells']):
        for key in ['contents_surface_connectivity','contents_connectivity_per_m2']:
            a.pop(key,None)
            if key in b:a[key]=deepcopy(b[key])
    assert restored==source
    scene['metadata']['wall_contact_hypothesis']=dict(
        scope='all 180 contents/casing contacts; insulation damage is separate from gas apertures',
        mechanism='existing accumulated interface damage feeds directed log-connectivity',
        kinetic_temperature='existing arithmetic mean of the two adjacent solid temperatures; not a transport average',
        rate_per_s='0 below150 C, otherwise 1e-6*exp((T_C-150)/5)',
        intact_h_w_m2_k=INTACT_H,destroyed_h_w_m2_k=DESTROYED_H,
        reversible_temperature_profile_removed=True,
        complete_damage_at_200c_hold_s=1/(RATE*math.exp((473.15-ONSET_K)/SCALE_K)),
        full_damage_temperature_at_point1_k_s=ONSET_K-273.15+SCALE_K*math.log1p(.1/(RATE*SCALE_K)),
        mechanical_shell_and_valves_unchanged=True,chemistry_and_mass_transfer_unchanged=True,
        material_identity_not_measured=True,calibrated=False,validation_passed=False)
    return scene


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();s=candidate(json.loads(a.source.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(s,f,indent=2)
    print(json.dumps(s['metadata']['wall_contact_hypothesis'],indent=2))

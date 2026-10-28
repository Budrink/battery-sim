#!/usr/bin/env python3
"""Read-only missing-path inventory; no artificial losses added to the solver."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from fsri30_data import save,sha
from report_fsri30_abs_cooling import audit,outside_state

SIGMA=5.670374419e-8


def radiative_potential(scene,run,time):
    t=np.asarray(run['time_s'])
    if time<t[0]-1e-7 or time>t[-1]+1e-7:raise ValueError('Radiation snapshot not observed')
    j=int(np.argmin(abs(t-time)));rows=[]
    areas=scene['metadata']['air_mesh']['actual_abs_exposure']
    ambient=298.15
    for ident,obj in run['thermal_paths']['objects'].items():
        if not ident.startswith('ABS:'):continue
        area=areas[ident.split(':')[1]]['outside_m2']
        mean=obj['reference_capacity_weighted_temperature_k'][j]
        lo=obj['minimum_temperature_k'][j];hi=obj['maximum_temperature_k'][j]
        rows.append(dict(solid=ident,area_m2=area,center_mean_temperature_k=mean,
            center_min_temperature_k=lo,center_max_temperature_k=hi,
            blackbody_mean_center_estimate_w=SIGMA*area*(mean**4-ambient**4),
            blackbody_min_center_estimate_w=SIGMA*area*(lo**4-ambient**4),
            blackbody_max_center_estimate_w=SIGMA*area*(hi**4-ambient**4)))
    if not rows:raise ValueError('No observed ABS panels')
    total=sum(r['blackbody_mean_center_estimate_w']for r in rows)
    return dict(time_s=float(t[j]),panels=rows,
        blackbody_center_based_potential_per_unit_emissivity_viewfactor_w=total,
        sensitivity_not_assumed_physics={str(e):e*total for e in (.05,.1,.3)},
        emissivity_measured=False,view_factors_measured=False,added_to_solver=False,
        limitations='Potential using voxel-center temperatures, not solved surface radiation or a predicted cooling trajectory; material survival uncertain')


def material_inventory(scene,run):
    residue=capacity=energy=0.
    for c in scene['cells']:
        if c['role']!='battery':continue
        m=run['chemistry'][c['id']]['residue_kg'][-1]
        cp=c['conversion']['residue_cp'];temperature=run['cell_mean_temperature_k'][c['id']][-1]
        residue+=m;capacity+=m*cp;energy+=m*cp*(temperature-298.15)
    return dict(last_observed_s=run['time_s'][-1],retained_r1_r2_residue_kg=residue,
        retained_residue_capacity_j_k=capacity,
        retained_residue_sensible_energy_uniform_mean_estimate_j=energy,
        condensed_ejecta_transport_implemented=False,
        note='NOT all residue should be ejected; requires cell-specific post-test/ejecta mass evidence. Existing gas/vapor/latent energy already exported and must not be counted twice.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder',type=Path,required=True);args=ap.parse_args();folder=args.folder
    path=folder/'replay/incomplete-run.json.gz'
    with gzip.open(path,'rt')as f:run=json.load(f)
    scene=json.loads((folder/'scenario.json').read_text())
    result=dict(run_sha256=sha(path),scene_sha256=sha(folder/'scenario.json'),
        last_observed_s=run['time_s'][-1],advanced_live_s_not_observation=run['advanced_to_s'],
        partial=True,calibrated=False,validation_passed=False,
        heat_paths=audit(run,3300.,3500.),
        outside_surface=outside_state(scene,run,folder/'gas-frames.bin',3500.),
        radiation=radiative_potential(scene,run,3500.),residue=material_inventory(scene,run),
        abs=dict(structural_damage_implemented=False,pyrolysis_implemented=False,
            temperature_dependent_material_removal_implemented=False),
        support=dict(explicit_solid_contact_present=False,experiment_support_contact_confirmed=False),
        sources=dict(ejecta='https://www.nasa.gov/centers-and-facilities/nesc/calorimetry-of-lithium-ion-cells-during-thermal-runaway/',
            fractional_mass_data='https://www.nlr.gov/transportation/battery-failure'))
    save(folder/'missing-losses-audit.json',result)
    print(json.dumps({k:v for k,v in result.items()if k not in ('heat_paths','radiation')},indent=2))
    print('Radiative potential per unit emissivity/viewfactor W:',result['radiation']['blackbody_center_based_potential_per_unit_emissivity_viewfactor_w'])


if __name__=='__main__':main()

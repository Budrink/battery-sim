"""Uniform slower insulation damage + timing correction; full recorded candidate."""
from copy import deepcopy
import math
from fsri30_r2_gas_shape_probe import main, ROOT

SOURCE=ROOT/'data/development/fsri30-wall-damage-replay-20261027-r1/scenario.json'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-wall-delay-record-20261027-r1'
CORE='812722f565a339d09a19f2c935f0ef9cb43f31f56f442861589d59e7a7424aea'


def candidate(source):
    scene=deepcopy(source)
    # Empirical local timing sensitivity, not a uniquely identified R2 law.
    r2_factor=math.exp((1737.-1763.)/219.3857)
    cells=[c for c in scene['cells'] if c.get('role')=='battery']
    assert len(cells)==30
    for c in cells:
        c['conversion']['reaction_rate_per_s']*=r2_factor
    shells=[s for s in scene['shells'] if s['contents_id'] in {c['id'] for c in cells}]
    assert len(shells)==30
    for shell in shells:
        shell['contents_connectivity_per_m2']['damage_rate_at_onset_per_s']*=.1
    # Demonstrate that no other physics/geometry/electrical settings changed.
    restored=deepcopy(scene)
    for a,b in zip(restored['cells'],source['cells']):
        if a.get('role')=='battery':a['conversion']['reaction_rate_per_s']=b['conversion']['reaction_rate_per_s']
    for a,b in zip(restored['shells'],source['shells']):
        if 'contents_connectivity_per_m2' in b:
            a['contents_connectivity_per_m2']=deepcopy(b['contents_connectivity_per_m2'])
    assert restored==source
    scene['simulation']['duration_s']=3000.
    scene['failure_sample_times_s']=list(map(float,range(0,3001,5)))
    controls=dict(r2_rate_multiplier=r2_factor,r2_rate_per_s=cells[0]['conversion']['reaction_rate_per_s'],
        insulation_damage_rate_multiplier=.1,insulation_rate_per_s=1e-7,
        insulation_onset_c=150.,insulation_temperature_scale_k=5.,
        complete_damage_hold_200c_s=1/(1e-7*math.exp(10)),
        complete_damage_hold_220c_s=1/(1e-7*math.exp(14)),
        full_damage_ramp_point1_k_s_c=150+5*math.log1p(.1/(1e-7*5)),
        same_law_all_batteries=True,irreversible=True,heater_r3_gas_valves_unchanged=True,
        cold_and_destroyed_conductances_unchanged=True,
        method='Tentative thermal-dose delay and archived local R2 timing sensitivity',
        objective='Later primary; delayed secondary via irreversible inner insulation',
        fresh_initial_state=True,calibrated=False,validation_passed=False)
    scene['metadata']['r2_gas_shape_probe']=controls
    wall=scene['metadata']['wall_contact_hypothesis']
    wall.update(rate_per_s='0 below150 C, otherwise 1e-7*exp((T_C-150)/5)',
        complete_damage_at_200c_hold_s=controls['complete_damage_hold_200c_s'],
        full_damage_temperature_at_point1_k_s=controls['full_damage_ramp_point1_k_s_c'],
        chemistry_and_mass_transfer_unchanged=False,
        chemistry_note='Only R2 prefactor adjusted; gas transfer unchanged')
    return scene


if __name__=='__main__':
    main(source=SOURCE,probe=PROBE,expected_core=CORE,candidate_fn=candidate)

"""Uniform cap-material hypothesis with less gas chemical energy; same laws."""
from copy import deepcopy
import math
from pathlib import Path
from fsri30_data import sha
from fsri30_r2_gas_shape_probe import main, ROOT

SOURCE = ROOT/'data/development/fsri30-r2x128-vent25-diff3-20261027-r1/scenario.json'
PROBE = ROOT/'build/calibration-probes/battery_sim_casing_probe-cap-material-20261027-r1'
ENERGY_FACTOR = .7
CAP_K = .05
RATE_FACTOR = math.exp(-2./219.3857)


def candidate(source):
    scene = deepcopy(source)
    batteries = [c for c in scene['cells'] if c.get('role') == 'battery']
    assert len(batteries) == 30
    for cell in batteries:
        law = cell['conversion']
        # The existing inventory is linear in q, including the initial charged
        # energy. Preserve q*(1-f) for R2, scale q*f for transported R3 fuel.
        q, f = law['net_heat_j_per_kg'], law['gas_chemical_fraction']
        scale = (1.-f) + ENERGY_FACTOR*f
        law['net_heat_j_per_kg'] = q*scale
        law['gas_chemical_fraction'] = ENERGY_FACTOR*f/scale
        assert math.isclose(law['net_heat_j_per_kg']*(1-law['gas_chemical_fraction']), q*(1-f), rel_tol=1e-14)
        assert math.isclose(law['net_heat_j_per_kg']*law['gas_chemical_fraction'], ENERGY_FACTOR*q*f, rel_tol=1e-14)
        law['reaction_rate_per_s'] *= RATE_FACTOR
        law['volatile_phase']['chemical_j_per_kg'] *= ENERGY_FACTOR
        assert not cell.get('electrical'), 'This energy identity assumes the externally heated fixture'
    scene['gas_network']['vapor_fuel_heat_j_kg'] *= ENERGY_FACTOR
    ids={c['id'] for c in batteries}
    shells=[s for s in scene['shells'] if s['contents_id'] in ids]
    assert len(shells)==30
    for shell in shells:
        assert 'face_materials' not in shell
        shell['face_materials']={'+z':dict(material_id='cap-low-conductivity-PROXY', conductivity_w_per_m_k=[CAP_K]*3)}
    # Prove all unrelated material, geometry, numerical, heater and valve
    # parameters remain identical to the stopped run.
    restored=deepcopy(scene)
    for a,b in zip(restored['cells'],source['cells']):
        if a.get('role')=='battery': a['conversion']=deepcopy(b['conversion'])
    for shell in restored['shells']:shell.pop('face_materials',None)
    restored['gas_network']['vapor_fuel_heat_j_kg']=source['gas_network']['vapor_fuel_heat_j_kg']
    assert restored==source
    scene['metadata']['r2_gas_shape_probe']=dict(
        cap_conductivity_w_m_k=CAP_K, cap_faces='+z only, all 30 batteries',
        cap_material_is_unmeasured_proxy=True, gas_fuel_energy_multiplier=ENERGY_FACTOR,
        direct_r2_heat_per_mass_unchanged=True, r2_rate_relative_to_last_run=RATE_FACTOR,
        r2_rate_per_s=batteries[0]['conversion']['reaction_rate_per_s'], r3_rate_multiplier=1.,
        outlet_hydraulic_resistance=25., mixing_diffusivity_m2_s=scene['gas_network']['mixing_diffusivity_m2_s'],
        fresh_initial_state=True, calibrated=False, validation_passed=False,
        objective='Primary onset and cooling shape; secondary timing must also qualify',
        timing_correction='Offline previous onset 1761 -> target1763 using -219.3857 s/log(rate); cap feedback is not fitted')
    scene['metadata']['secondary_timing_guard']=True
    return scene


if __name__=='__main__':
    # Binary fingerprint is persisted by the shared runner before execution.
    manifest=PROBE.with_suffix('.sha256').read_text().split()[0]
    main(source=SOURCE, probe=PROBE, expected_core=manifest, candidate_fn=candidate)

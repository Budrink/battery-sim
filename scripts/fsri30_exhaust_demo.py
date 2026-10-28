"""DEMO apparatus: independent captured vent risers, not an experimental fit."""
from copy import deepcopy
import math

from fsri30_r2_gas_shape_probe import main, ROOT

SOURCE=ROOT/'data/development/fsri30-wall-delay-full-20261027-r1/scenario.json'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-wall-delay-record-20261027-r1'
CORE='812722f565a339d09a19f2c935f0ef9cb43f31f56f442861589d59e7a7424aea'
FLOW_PER_DUCT_M3_S=.002
DUCT_DIAMETER_M=.018
SEGMENT_LENGTH_M=.03
SUCTION_PA=500.


def candidate(source):
    scene=deepcopy(source)
    batteries=[c for c in scene['cells'] if c.get('role')=='battery']
    ids={c['id'] for c in batteries}
    assert len(ids)==30
    net=scene['gas_network'];meta=scene['metadata']
    template=next(n for n in net['compartments'] if n['id'].startswith('aligned-inner-'))
    area=math.pi*(DUCT_DIAMETER_M/2)**2
    volume=area*SEGMENT_LENGTH_M
    temperature=net['reference_temperature_k']
    density=101325./(net['specific_r_j_per_kg_k']*temperature)
    base_node_count=len(net['compartments']);base_channel_count=len(net['channels'])
    owners={};contact_count=0
    for channel in net['channels']:
        if channel['kind']!='rupture':continue
        owner=channel['port_id'].split(':')[0]
        if owner not in ids:continue
        # Keep aperture mechanics/pressure law and area EXACTLY as recorded.
        if ':case:+z:' not in channel['port_id']:raise ValueError('Unaccounted side rupture')
        owners[channel['id']]=owner
        channel['to']=f'demo-exhaust-{owner}-capture'
    assert len(owners)==90
    for contact in net['thermal_contacts']:
        owner=contact['port_id'].split(':')[0]
        if owner in ids and ':case:+z:' in contact['port_id']:
            # Same top-surface heat law; the adjacent gas is now inside the hood.
            contact['compartment']=f'demo-exhaust-{owner}-capture';contact_count+=1
    for cell in batteries:
        name=cell['id'];x,y,z=cell['pose']['position_m']
        top=z+meta['packed_cylinder_geometry']['height_m']/2
        for k,part in enumerate(['capture','outlet']):
            node={key:deepcopy(value) for key,value in template.items() if key in
                ['reaction_rate_per_s','reaction_reference_k','reaction_activation_j_per_mol','oxidizer_heat_j_per_kg','ignition_temperature_k']}
            node.update(id=f'demo-exhaust-{name}-{part}',volume_m3=volume,
                initial_mass_kg=density*volume,initial_temperature_k=temperature,
                initial_oxidizer_mass_fraction=.232)
            if part=='outlet':
                node['ambient_boundary']=dict(pressure_pa=101325.-SUCTION_PA,
                    temperature_k=temperature,oxidizer_mass_fraction=.232,area_m2=area,
                    discharge_coefficient=.7,renewal_flow_m3_s=FLOW_PER_DUCT_M3_S)
            net['compartments'].append(node)
            # Schematic riser display: same volume as its finite gas inventory.
            width=math.sqrt(area)
            meta['gas_volume_geometry'][node['id']]=dict(
                low_m=[x-width/2,y-width/2,top+k*SEGMENT_LENGTH_M],
                high_m=[x+width/2,y+width/2,top+(k+1)*SEGMENT_LENGTH_M])
        net['channels'].append(dict(id=f'demo-exhaust-{name}-up',kind='passage',
            **{'from':f'demo-exhaust-{name}-capture','to':f'demo-exhaust-{name}-outlet'},
            area_m2=area,mixing_length_m=SEGMENT_LENGTH_M,mixing_axis=2,
            discharge_coefficient=.7,opening_pressure_pa=0.,hydraulic_resistance_factor=1.))
    # Prove that the difference is apparatus topology only, not hidden chemistry,
    # primary heating or a special immunity flag for secondary batteries.
    restored=deepcopy(scene)
    restored['gas_network']['compartments']=restored['gas_network']['compartments'][:base_node_count]
    restored['gas_network']['channels']=restored['gas_network']['channels'][:base_channel_count]
    for a,b in zip(restored['gas_network']['channels'],source['gas_network']['channels']):
        if a['id'] in owners:a['to']=b['to']
    for a,b in zip(restored['gas_network']['thermal_contacts'],source['gas_network']['thermal_contacts']):
        a['compartment']=b['compartment']
    restored['metadata']=deepcopy(source['metadata'])
    assert restored==source
    meta['exhaust_demo']=dict(
        kind='DEMO / ideal sealed capture risers; not an experimentally validated design',
        reference_run=SOURCE.parent.name,capture_fraction_assumed=1.,ducts=30,gas_nodes_added=60,
        rerouted_ruptures=len(owners),rerouted_top_thermal_contacts=contact_count,
        duct_equivalent_diameter_m=DUCT_DIAMETER_M,duct_length_m=2*SEGMENT_LENGTH_M,
        added_apparatus_gas_volume_m3=60*volume,added_initial_gas_mass_kg=60*density*volume,
        outlet_suction_pa=SUCTION_PA,renewal_flow_per_duct_m3_s=FLOW_PER_DUCT_M3_S,
        total_nominal_renewal_m3_h=30*FLOW_PER_DUCT_M3_S*3600,
        boundary_model='Existing compressible pressure opening plus analytic well-mixed throughflow; not a fan curve',
        energy_accounting='Existing ambient mass, enthalpy, chemical and oxygen ledgers',
        duct_walls='Ideal insulated and sealed; no cross-duct gas or thermal contact',
        display_geometry='Schematic risers added to apparatus, not a remesh of existing free air',
        direct_metal_contacts_already_disabled=True,
        heater_chemistry_battery_shells_and_numerics_unchanged=True,
        limits='No duct leakage, duct wall heating, fan power/failure or downstream external flame model',
        success_not_assumed=True,calibrated=False,validation_passed=False)
    assert not meta['can_contact_hypothesis']['direct_contacts_enabled']
    meta['r2_gas_shape_probe']=dict(meta['exhaust_demo'],fresh_initial_state=True,
        objective='Retain B10 primary runaway; test whether sealed extraction prevents secondary runaway through 3000 s')
    meta['secondary_timing_guard']=False
    scene['name']='DEMO — isolated exhaust risers / same baseline chemistry'
    selection=scene.setdefault('process_history_selection',{})
    for key in ['gas_prefixes','channel_prefixes']:
        selection.setdefault(key,[]).append('demo-exhaust-')
    return scene


if __name__=='__main__':
    main(source=SOURCE,probe=PROBE,expected_core=CORE,candidate_fn=candidate)

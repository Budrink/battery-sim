"""Fresh check of the analytically estimated irreversible-insulation seed."""
from copy import deepcopy
from fsri30_r2_gas_shape_probe import main, ROOT

SOURCE=ROOT/'data/development/fsri30-wall-damage-seed-20261027-r1/scenario.json'
PROBE=ROOT/'build/calibration-probes/battery_sim_casing_probe-wall-damage-20261027-r1'
CORE='cbb53739f5d0e02d4d5920c209c1aa591e3a16a39dc8e294b8d5c35371e96edc'


def checked_seed(source):
    count=0
    for shell in source['shells']:
        if not shell['contents_id'].startswith('B'):continue
        assert 'contents_surface_connectivity' not in shell
        law=shell['contents_connectivity_per_m2']
        assert law['damage_rate_at_onset_per_s']>0 and law['interface_damage_log_response']>0
        assert law['endpoint_damage_weight']==0
        count+=1
    assert count==30
    assert source['metadata']['secondary_timing_guard']
    return deepcopy(source)


if __name__=='__main__':
    main(source=SOURCE,probe=PROBE,expected_core=CORE,candidate_fn=checked_seed)

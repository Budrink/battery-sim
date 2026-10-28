import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_upv_electrical_closure import thermal_continuation


class ElectricalContinuation(unittest.TestCase):
    def test_changes_only_the_declared_electrical_temperature_law(self):
        source=dict(cells=[dict(id='C2',electrical=dict(resistance_ohm=.016,
            reference_temperature_k=298.15,temperature_log_slope_per_k=.0068,
            charge_log_slope=8.,current_profile_a=[[0.,35.],[1500.,35.]]),
            conversion=dict(reaction_rate_per_s=.03),thermal=dict(heat_capacity_j_per_k=400.)),
            dict(id='wall',thermal=dict(heat_capacity_j_per_k=20000.))],
            metadata={},shells=[dict(strength=1.)],gas_network=dict(channels=[]))
        before=copy.deepcopy(source);trial=thermal_continuation(source,425.)
        self.assertEqual(source,before)
        self.assertEqual(trial['shells'],source['shells']);self.assertEqual(trial['gas_network'],source['gas_network'])
        self.assertEqual(trial['cells'][1],source['cells'][1])
        old=copy.deepcopy(source['cells'][0]);new=copy.deepcopy(trial['cells'][0])
        a=old['electrical'].pop('temperature_log_slope_per_k');table=new['electrical'].pop('temperature_log_response')
        self.assertEqual(old,new)
        self.assertEqual(table[1],[298.15,0.])
        self.assertAlmostEqual(table[2][1],a*(425.-298.15))
        self.assertEqual(table[2][1],table[3][1])
        self.assertFalse(trial['metadata']['electrical_closure_review']['heldout_temperatures_used_for_selection'])

    def test_no_arbitrary_hot_range(self):
        for value in (200.,1000.):
            with self.assertRaises(ValueError):thermal_continuation(dict(cells=[],metadata={}),value)

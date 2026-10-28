import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_multilevel import remesh, inventory
from fsri30_sequence_model import initial, candidate

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023/scenario.json'


@unittest.skipUnless(SOURCE.exists(),'Retained FSRI fixture not installed')
class MultilevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=json.loads(SOURCE.read_text())
        cls.levels={h:remesh(cls.base,h)for h in (80,40,20)}

    def test_target_is_exact_identity_and_input_untouched(self):
        self.assertEqual(self.levels[20],self.base)
        self.assertEqual(json.loads(SOURCE.read_text()),self.base)

    def test_both_thermal_and_gas_mesh_are_coarsened(self):
        counts=[inventory(self.levels[h])for h in (80,40,20)]
        for key in ('gas_nodes','thermal_zones'):
            self.assertLess(counts[0][key],counts[1][key])
            self.assertLess(counts[1][key],counts[2][key])
        for h,zones in [(80,1),(40,3),(20,27)]:
            batteries=[c for c in self.levels[h]['cells']if c.get('role')=='battery']
            self.assertEqual(len(batteries),30)
            self.assertTrue(all(math.prod(c['thermal']['divisions'])==zones for c in batteries))

    def test_every_battery_keeps_one_valve_and_full_physical_inventory(self):
        for h in (80,40,20):
            s=self.levels[h]
            self.assertEqual(sum('valve'in e.get('damage_opening',{})for e in s['gas_network']['channels']),30)
            for key in ('gas_mass_kg','gas_volume_m3','gas_oxygen_kg','solid_capacity_j_k',
                        'module_exit_area_m2','renewal_m3_s'):
                self.assertTrue(math.isclose(inventory(s)[key],inventory(self.base)[key],rel_tol=2e-11))
            self.assertEqual(s['heating_protocols'],self.base['heating_protocols'])
            self.assertEqual(s['shells'],self.base['shells'])

    def test_parameters_change_shared_laws_not_inventories_or_time_step(self):
        p=initial(self.base);p.update(r2_rate_scale=.02,gas_can_contact_scale=2.,effective_emissivity=.4)
        for h in (80,40,20):
            s=self.levels[h];changed=candidate(s,p)
            self.assertEqual(changed['simulation']['minimum_dt_s'],s['simulation']['minimum_dt_s'])
            self.assertEqual(changed['heating_protocols'],s['heating_protocols'])
            for a,b in zip(changed['cells'],s['cells']):
                self.assertEqual(a['thermal'],b['thermal'])
                if a.get('role')=='battery':
                    aa=dict(a['conversion']);bb=dict(b['conversion'])
                    self.assertAlmostEqual(aa.pop('reaction_rate_per_s'),bb.pop('reaction_rate_per_s')*.02)
                    self.assertEqual(aa,bb)
            radiation=[b['radiation']for b in changed['boundaries']if 'radiation'in b]
            self.assertTrue(radiation)
            self.assertTrue(all(r['effective_emissivity']==.4 for r in radiation))


if __name__=='__main__':unittest.main()

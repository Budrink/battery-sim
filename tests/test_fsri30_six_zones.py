import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fsri30_six_zones import six_zone_batteries


class SixZoneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT/'data/development/fsri30-packed-fine-2406-20261027-r1/cascade-0000/scenario.json'
        if not source.exists(): raise unittest.SkipTest('Local FSRI diagnostic scenario not present')
        cls.old = json.loads(source.read_text()); cls.new = six_zone_batteries(cls.old)

    def test_inventories_and_topology(self):
        old, new = self.old['gas_network'], self.new['gas_network']
        for key in ('volume_m3','initial_mass_kg'):
            self.assertTrue(math.isclose(sum(x[key] for x in old['compartments']),
                                         sum(x[key] for x in new['compartments']), rel_tol=1e-12))
        ids = {x['id'] for x in new['compartments']}
        self.assertEqual(len(ids), len(new['compartments']))
        self.assertTrue(all(e['from'] in ids and e['to'] in ids for e in new['channels']))
        for cell in (x for x in self.new['cells'] if x.get('role')=='battery'):
            self.assertEqual(cell['thermal']['divisions'], [3,1,2])
            nodes = [n for n in new['compartments'] if n['id'].startswith('inside-'+cell['id']+'-')]
            self.assertEqual(len(nodes),6)
            self.assertEqual({n['local_equilibrium']['thermal_volume'] for n in nodes},set(range(6)))
            openings = [e for e in new['channels'] if e['id'].startswith('rupture-'+cell['id']+'-')]
            self.assertEqual(len(openings),3)
            self.assertEqual(sum('valve' in e.get('damage_opening',{}) for e in openings),1)


if __name__ == '__main__': unittest.main()

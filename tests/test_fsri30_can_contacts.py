import copy
import json
import sys
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fsri30_can_contacts import add_can_contacts


class CanContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT/'data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023/scenario.json'
        cls.base = json.loads(path.read_text())

    def test_contact_strips_do_not_bypass_heater(self):
        base = self.base
        original = copy.deepcopy(base)
        result = add_can_contacts(base)
        self.assertEqual(base,original)
        records = result['metadata']['can_contact_hypothesis']['contact_pairs']
        self.assertEqual(len(records),67)
        self.assertEqual({r['b'] for r in records if r['a'] == 'B10'}, {'B11','B17','B18'})
        self.assertFalse(any((r['a'],r['b']) in {('B02','B10'),('B03','B10')} for r in records))
        for record in records: self.assertAlmostEqual(record['area_m2'],.001*.065)
        self.assertEqual(result['cells'],base['cells'])
        self.assertEqual(result['shells'],base['shells'])
        self.assertEqual(result['gas_network']['channels'],base['gas_network']['channels'])

    def test_strip_width_reduces_exposed_gas_area(self):
        base = self.base
        a = add_can_contacts(base,.001,1000.)
        b = add_can_contacts(base,.002,1000.)
        ra = a['metadata']['can_contact_hypothesis']['contact_pairs'][0]
        rb = b['metadata']['can_contact_hypothesis']['contact_pairs'][0]
        self.assertAlmostEqual(rb['area_m2'],2*ra['area_m2'])
        pid = f"{ra['a']}:case:{ra['face_a']}:outer"
        def film(s):
            return next(x for x in s['gas_network']['thermal_contacts'] if x['port_id'] == pid)
        g0 = film(base)['connectivity']['background_conductance_w_per_k']
        g1 = film(a)['connectivity']['background_conductance_w_per_k']
        g2 = film(b)['connectivity']['background_conductance_w_per_k']
        self.assertTrue(0 < g2 < g1 < g0)
        self.assertAlmostEqual(g0-g2,2*(g0-g1))


if __name__ == '__main__': unittest.main()

import copy
import json
import math
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fsri30_packed_geometry import close_packed
from fsri30_can_contacts import add_can_contacts


class PackedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=json.loads((ROOT/'data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023/scenario.json').read_text())
        cls.original=copy.deepcopy(cls.base);cls.packed=close_packed(cls.base)

    def test_tangencies_and_conservation(self):
        self.assertEqual(self.base,self.original)
        p=self.packed;pos=p['metadata']['approximate_positions_m'];pairs=p['metadata']['neighbour_pairs']
        self.assertEqual(len(pairs),69)
        for a,b in pairs:self.assertAlmostEqual(math.dist(pos[f'B{a:02d}'],pos[f'B{b:02d}']),.018,places=10)
        ids=list(pos)
        for i,a in enumerate(ids):
            for b in ids[i+1:]:self.assertGreaterEqual(math.dist(pos[a],pos[b]),.018-1e-10)
        geom=p['metadata']['packed_cylinder_geometry']
        self.assertAlmostEqual(geom['volume_before_m3'],geom['volume_after_m3'],places=11)
        self.assertAlmostEqual(geom['mass_before_kg'],geom['mass_after_kg'],places=11)
        self.assertEqual([c['thermal']['divisions'] for c in p['cells'] if c['id']=='B10'],[[3,3,3]])

    def test_contacts_and_vent_area(self):
        p=add_can_contacts(self.packed,.0005,1000.)
        contacts=p['metadata']['can_contact_hypothesis']['contact_pairs']
        self.assertEqual(len(contacts),67)
        for c in contacts:self.assertAlmostEqual(c['area_m2'],.0005*.065)
        self.assertFalse(any((c['a'],c['b']) in {('B02','B10'),('B03','B10')} for c in contacts))
        old=sum(e['area_m2'] for e in self.base['gas_network']['channels'] if e['id'].startswith('vent-'))
        new=sum(e['area_m2'] for e in p['gas_network']['channels'] if e['id'].startswith('vent-'))
        self.assertAlmostEqual(old,new,places=14)
        self.assertEqual(len({e['id'] for e in p['gas_network']['channels']}),len(p['gas_network']['channels']))


if __name__=='__main__':unittest.main()

import json
import math
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fsri30_packed_geometry import close_packed
from fsri30_packed_pockets import split_tangent_pockets
from fsri30_can_contacts import add_can_contacts
from fsri30_external_screen import exterior_screen


class PocketTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023/scenario.json'
        cls.base=json.loads(path.read_text());cls.packed=close_packed(cls.base)
        cls.scene=add_can_contacts(split_tangent_pockets(cls.packed),.0005,1000.)

    def test_conserved_gas_and_cylinder_exposure(self):
        s=self.scene;geom=s['metadata']['packed_pocket_geometry']
        self.assertEqual(geom['pocket_count'],40)
        self.assertAlmostEqual(geom['original_body_volume_m3'],geom['total_body_volume_m3'],places=14)
        for fractions in geom['film_fractions'].values():self.assertAlmostEqual(sum(fractions.values()),1.)
        before=sum(n['initial_mass_kg'] for n in self.packed['gas_network']['compartments'])
        after=sum(n['initial_mass_kg'] for n in s['gas_network']['compartments'])
        self.assertAlmostEqual(before,after,places=12)
        self.assertEqual(len(s['metadata']['can_contact_hypothesis']['contact_pairs']),67)

    def test_pockets_have_only_top_and_bottom_paths(self):
        s=self.scene;channels=s['gas_network']['channels'];pockets=s['metadata']['packed_pocket_geometry']['pockets']
        for i,vertices in enumerate(pockets):
            name=f'body-pocket-{i:02d}'
            edges=[e for e in channels if e['from']==name or e['to']==name]
            self.assertEqual(len(edges),6)
            self.assertEqual({e['to'] for e in edges}, {f'{band}-{v:02d}' for v in vertices for band in ('top','base')})
            self.assertTrue(all(e['mixing_axis']==2 for e in edges))
        self.assertFalse(any(e['id'].startswith('free-body-') for e in channels))
        self.assertTrue(all(e['from']=='body-exterior' for e in channels if e['id'].startswith('vent-')))
        self.assertTrue(all(not e['from'].startswith('body-pocket-') or not e['to'].startswith('body-pocket-') for e in channels))

    def test_coarse_exterior_preserves_internal_physics(self):
        screen=add_can_contacts(split_tangent_pockets(close_packed(exterior_screen(self.base,80))),.0005,1000.)
        fine=self.scene
        def inventory(s):
            nodes=s['gas_network']['compartments']
            return (sum(n['volume_m3'] for n in nodes),sum(n['initial_mass_kg'] for n in nodes),
                    sum(c['thermal']['heat_capacity_j_per_k'] for c in s['cells']))
        for a,b in zip(inventory(fine),inventory(screen)):self.assertAlmostEqual(a,b,places=10)
        for c in screen['cells']:
            if c.get('role')=='battery':self.assertEqual(c['thermal']['divisions'],[3,3,3])
        self.assertEqual(screen['metadata']['packed_pocket_geometry']['pockets'],fine['metadata']['packed_pocket_geometry']['pockets'])
        self.assertEqual(len(screen['metadata']['can_contact_hypothesis']['contact_pairs']),67)


if __name__=='__main__':unittest.main()

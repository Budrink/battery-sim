from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_abs_seams import join_abs_panels
from fsri30_data import ROOT

BASE=ROOT/'data/development/fsri30-open-boundary-20261025-r2/open-0.01m3s/scenario.json'


class AbsSeamTests(unittest.TestCase):
    def test_twelve_geometric_seams_and_frozen_physics(self):
        base=json.loads(BASE.read_text());old=deepcopy(base);new=join_abs_panels(base)
        self.assertEqual(base,old)
        self.assertEqual(len(new['interfaces'])-len(base['interfaces']),12)
        self.assertEqual(len(new['ports'])-len(base['ports']),24)
        for key in ('cells','shells','gas_network','boundaries','simulation','heating_protocols'):
            self.assertEqual(base[key],new[key],key)
        ports={p['id']:p for p in new['ports']}
        for edge in new['interfaces'][len(base['interfaces']):]:
            a,b=(ports[edge[k]]for k in ('port_a','port_b'))
            self.assertEqual(a['centroid_m'],b['centroid_m'])
            self.assertEqual(a['rectangle_m'],b['rectangle_m'])
            self.assertEqual(a['normal'],[-v for v in b['normal']])
            self.assertTrue(a['cell_id'].startswith('ABS:'))
            self.assertAlmostEqual(a['area_m2'],a['rectangle_m'][0]*a['rectangle_m'][1])
        with self.assertRaisesRegex(ValueError,'already'):join_abs_panels(new)

    def test_no_contact_across_a_gap_or_at_line_only(self):
        def cell(name,position):
            return dict(id=name,material_id='ABS',geometry=dict(dimensions_m=[1.,1.,1.]),
                pose=dict(position_m=position),thermal=dict(divisions=[1,1,1],conductivity_w_per_m_k=[.18]*3))
        base=dict(cells=[cell('ABS:a',[0,0,0]),cell('ABS:b',[1.01,0,0])],ports=[],interfaces=[])
        self.assertEqual(join_abs_panels(base,expected_pairs=0)['interfaces'],[])
        base['cells'][1]['pose']['position_m']=[1,1,0]
        self.assertEqual(join_abs_panels(base,expected_pairs=0)['interfaces'],[])
        base['cells'][1]['pose']['position_m']=[1,0,0]
        trial=join_abs_panels(base,expected_pairs=1)
        reference=trial['metadata']['abs_seam_continuity']['added'][0]
        self.assertAlmostEqual(reference['reference_material_conductance_w_k'],.18)
        self.assertEqual(reference['contact_to_material_resistance_ratio'],1e-6)
        base['cells'][1]['pose']['position_m']=[.99,0,0]
        with self.assertRaisesRegex(ValueError,'Overlapping'):join_abs_panels(base,expected_pairs=0)


if __name__=='__main__':unittest.main()

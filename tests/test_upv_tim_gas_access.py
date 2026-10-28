import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from upv_tim_gas_access import candidate


class CommonTimAccess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/damaged-contact-20261022-r1/trial-002/scenario.json'
        if not p.exists():raise unittest.SkipTest('Optional current calibration geometry')
        cls.source=json.loads(p.read_text())

    def test_neutral_exact_and_no_mutation(self):
        old=copy.deepcopy(self.source)
        new=candidate(old, old['metadata']['gas_access_geometry']['protected_accessible_fraction'])
        for k in old:
            if k!='metadata':self.assertEqual(old[k],new[k])
        self.assertEqual(old,self.source)

    def test_same_rule_and_only_mass_paths_change(self):
        new=candidate(self.source,.001)
        for k in self.source:
            if k not in ('metadata','gas_network'):self.assertEqual(new[k],self.source[k])
        for k in self.source['gas_network']:
            if k!='channels':self.assertEqual(new['gas_network'][k],self.source['gas_network'][k])
        for a,b in zip(self.source['gas_network']['channels'],new['gas_network']['channels']):
            expected=copy.deepcopy(a)
            if a['id'].startswith('escape-TIM'):
                expected['area_m2']*=10.
            if a['kind']=='rupture' and a['to'].startswith('gap-TIM'):
                expected['discharge_coefficient']*=10.
                expected['mixing_length_m']/=10.
            self.assertEqual(b,expected)
        counts=new['metadata']['common_tim_gas_access']['counts']
        self.assertEqual(counts['TIM1'],counts['TIM2'])
        self.assertEqual(counts['TIM2'],counts['TIM3'])

    def test_invalid_fraction(self):
        for fraction in (0.,1.01,float('nan')):
            with self.assertRaises(ValueError):candidate(self.source,fraction)

    def test_fully_open_stays_within_nominal_geometry(self):
        scene=candidate(self.source,1.)
        for edge in scene['gas_network']['channels']:
            if edge['kind']=='rupture' and edge['to'].startswith('gap-TIM'):
                self.assertAlmostEqual(edge['discharge_coefficient'],.7)
                self.assertAlmostEqual(edge['mixing_length_m'],.001)

    def test_inlet_access_does_not_open_protected_rim_passages(self):
        fraction=self.source['metadata']['gas_access_geometry']['protected_accessible_fraction']
        scene=candidate(self.source,fraction,fracture_access_fraction=.01)
        changed=0
        for a,b in zip(self.source['gas_network']['channels'],scene['gas_network']['channels']):
            if a['kind']=='rupture' and a['to'].startswith('gap-TIM'):
                self.assertAlmostEqual(b['discharge_coefficient']/a['discharge_coefficient'],.01/fraction)
                self.assertAlmostEqual(a['mixing_length_m']/b['mixing_length_m'],.01/fraction)
                changed+=1
            else:self.assertEqual(a,b)
        self.assertGreater(changed,0)
        self.assertEqual(self.source['gas_network']['thermal_contacts'],scene['gas_network']['thermal_contacts'])
        neutral=candidate(scene,fraction)
        self.assertEqual(neutral['gas_network'],scene['gas_network'])

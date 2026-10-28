import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_wall_response import candidate, TARGET_H


class WallResponseTest(unittest.TestCase):
    def setUp(self):
        self.source={'cells':[{'id':f'B{i:02}', 'role':'battery', 'conversion':{'sentinel':i}} for i in range(30)],
                     'shells':[{'contents_id':f'B{i:02}', 'contents_contact_h':500.,
                                'conductivity_w_per_m_k':[15.]*3} for i in range(30)],
                     'metadata':{},'gas_network':{'sentinel':'unchanged'},'heating_protocols':['unchanged']}

    def test_source_and_other_physics_unchanged(self):
        original=copy.deepcopy(self.source)
        result=candidate(self.source)
        self.assertEqual(self.source,original)
        for shell in result['shells']:shell.pop('contents_surface_connectivity')
        result['metadata'].pop('wall_contact_hypothesis')
        self.assertEqual(result,original)

    def test_exact_monotone_contact_targets(self):
        result=candidate(self.source)
        profiles=[s['contents_surface_connectivity'] for s in result['shells']]
        self.assertTrue(all(p==profiles[0] for p in profiles))
        values=[]
        for (t,z),(c,h) in zip(profiles[0]['temperature_log_response'],TARGET_H):
            got=500.*(.01+1.98/(1+math.exp(-z)))
            self.assertAlmostEqual(t,c+273.15)
            self.assertAlmostEqual(got,h)
            values.append(got)
        self.assertEqual(values,sorted(values))
        self.assertLess(values[0],500.)
        self.assertGreater(values[-1],500.)


if __name__=='__main__':unittest.main()

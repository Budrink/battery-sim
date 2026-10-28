from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_data import ROOT
from fsri30_radiation import add_exterior_radiation


class RadiationGeometryTests(unittest.TestCase):
    def test_only_radiative_edges_added_and_vent_area_removed(self):
        base=json.loads((ROOT/'data/development/fsri30-abs-seams-20261025-r2/scenario.json').read_text())
        trial=add_exterior_radiation(base,.1)
        for key in ('cells','shells','interfaces','gas_network','ports','simulation','heating_protocols'):
            self.assertEqual(base[key],trial[key],key)
        self.assertEqual(base['boundaries'],trial['boundaries'][:len(base['boundaries'])])
        meta=trial['metadata']['exterior_radiation']
        self.assertAlmostEqual(sum(meta['visible_area_m2'].values()),.0739384056218045,places=8)
        self.assertTrue(any(b['radiation']['area_fraction']<1 for b in trial['boundaries'] if 'radiation'in b))
        with self.assertRaisesRegex(ValueError,'already'):add_exterior_radiation(trial,.1)

    def test_parameter_domain(self):
        for value in (-1.,1.1,float('nan')):
            with self.assertRaisesRegex(ValueError,'emissivity'):add_exterior_radiation({},value)


if __name__=='__main__':unittest.main()

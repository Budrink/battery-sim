import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from check_fsri30_open_boundary import BASE
from check_fsri30_exhaust_apertures import aperture_trial


class AperturePrior(unittest.TestCase):
    def test_only_two_exit_areas_change(self):
        base=json.loads(BASE.read_text());scene=aperture_trial(base,4.)
        for section in ('cells','shells','ports','interfaces','boundaries','heating_protocols'):
            self.assertEqual(base[section],scene[section])
        self.assertEqual(base['gas_network']['compartments'],scene['gas_network']['compartments'])
        changed=0
        for a,b in zip(base['gas_network']['channels'],scene['gas_network']['channels']):
            if a!=b:
                self.assertTrue(a['id'].startswith('vent-'));self.assertEqual(b['area_m2'],4.*a['area_m2'])
                changed+=1;b['area_m2']=a['area_m2'];self.assertEqual(a,b)
        self.assertEqual(changed,2)

    def test_no_unbounded_geometry_search(self):
        for scale in (.9,4.1,float('nan')):
            with self.assertRaises(ValueError):aperture_trial(json.loads(BASE.read_text()),scale)


if __name__=='__main__':unittest.main()

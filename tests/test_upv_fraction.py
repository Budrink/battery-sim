import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import calibrate_upv_fraction as model


class FixedFractionProtocol(unittest.TestCase):
    def test_frozen_thermal_chemical_and_mass_geometry(self):
        scene,_=model.area.old.geometry.inputs('no-cooling')
        first=model.build(scene,model.BASE)
        second=model.build(scene,dict(model.BASE,fraction=.05,pressure_pa=5e6,rate_per_s=.2))
        self.assertEqual(first['cells'],second['cells'])
        self.assertEqual(first['shells'],second['shells'])
        self.assertEqual(first['gas_network']['compartments'],second['gas_network']['compartments'])
        self.assertEqual(first['gas_network']['thermal_contacts'],second['gas_network']['thermal_contacts'])
        for a,b in zip(first['gas_network']['channels'],second['gas_network']['channels']):
            a,b=copy.deepcopy(a),copy.deepcopy(b)
            if a['kind']=='rupture':
                self.assertEqual(a['damage_opening']['mode'],'stochastic_fraction')
                self.assertNotIn('fragment_mean_area_m2',a['damage_opening'])
                for c in (a,b):
                    c.pop('opening_pressure_pa')
                    for k in ('pixel_rate_per_s','fixed_area_fraction'):c['damage_opening'].pop(k)
            self.assertEqual(a,b)

    def test_reject_undeclared_parameters(self):
        scene,_=model.area.old.geometry.inputs('no-cooling')
        for p in (dict(model.BASE,conductivity=1),dict(model.BASE,fraction=0),dict(model.BASE,fraction=.5)):
            with self.assertRaises(ValueError):model.build(scene,p)


if __name__=='__main__':unittest.main()

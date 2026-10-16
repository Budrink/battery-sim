import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import calibrate_upv_area as model

class AreaProtocol(unittest.TestCase):
    def test_only_barrier_inputs_change(self):
        scene,_=model.old.geometry.inputs('no-cooling');original=copy.deepcopy(scene)
        a=model.build(scene,model.BASE,17)
        self.assertEqual(scene,original)
        for _,p in model.CASES:
            b=model.build(scene,p,29)
            for key in ('cells','shells','interfaces','ports','boundaries','measurement_probes'):
                self.assertEqual(a[key],b[key])
            for key in ('compartments','thermal_contacts'):
                self.assertEqual(a['gas_network'][key],b['gas_network'][key])
            for before,after in zip(a['gas_network']['channels'],b['gas_network']['channels']):
                expected=copy.deepcopy(before)
                if before['kind']=='rupture':
                    expected['opening_pressure_pa']=p['pressure_pa']
                    expected['damage_opening'].update(pixel_rate_per_s=p['rate_per_s'],fragment_mean_area_m2=p['fragment_area_m2'],seed=29)
                self.assertEqual(expected,after)

    def test_fragment_scale_not_mesh_dependent(self):
        scene,_=model.old.geometry.inputs('no-cooling')
        a=model.build(scene,model.BASE,17,64);b=model.build(scene,model.BASE,17,32)
        self.assertEqual(a['gas_network']['channels'],b['gas_network']['channels'])
        self.assertNotEqual(a['shells'][0]['surface_divisions'],b['shells'][0]['surface_divisions'])

    def test_reject_extra_or_bad_fit_parameters(self):
        scene,_=model.old.geometry.inputs('no-cooling')
        for p in [dict(model.BASE,heat_capacity=1),dict(model.BASE,fragment_area_m2=0),dict(model.BASE,fragment_area_m2=float('nan'))]:
            with self.assertRaises(ValueError):model.build(scene,p,17)

if __name__=='__main__':unittest.main()

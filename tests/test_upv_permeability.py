import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import calibrate_upv_permeability as model


class PermeabilityProtocol(unittest.TestCase):
    def test_only_mass_barrier_parameters_change(self):
        scene,_=model.geometry.inputs('no-cooling');original=copy.deepcopy(scene)
        baseline=model.build(scene,model.BASE)
        for _,changes in model.CASES:
            parameters=dict(model.BASE,**changes);trial=model.build(scene,parameters)
            for key in ('cells','shells','interfaces','ports','boundaries','measurement_probes'):
                self.assertEqual(baseline[key],trial[key])
            for key in ('compartments','thermal_contacts'):
                self.assertEqual(baseline['gas_network'][key],trial['gas_network'][key])
            for a,b in zip(baseline['gas_network']['channels'],trial['gas_network']['channels']):
                if a['kind']!='rupture':
                    self.assertEqual(a,b)
                else:
                    expected=copy.deepcopy(a)
                    expected['opening_pressure_pa']*=parameters['strength_factor']
                    expected['discharge_coefficient']=parameters['discharge_coefficient']
                    expected['damage_opening']['pressure_damage_rate_per_s']*=parameters['rate_factor']
                    expected['damage_opening']['exponent']=parameters['exponent']
                    self.assertEqual(expected,b)
        self.assertEqual(scene,original)

    def test_forbidden_or_nonfinite_parameters(self):
        scene,_=model.geometry.inputs('no-cooling')
        for p in [dict(model.BASE,energy_j=1),dict(model.BASE,exponent=float('nan')),
                  dict(model.BASE,discharge_coefficient=2)]:
            with self.assertRaises(ValueError):
                model.build(scene,p)

    def test_dense_output_is_not_extra_pressure_evidence(self):
        prior=dict(result=dict(time_s=[0.,1.,4.]),observed_pressure_bar=[1.,3.,2.])
        sparse=dict(time_s=[0.,1.,4.],gas={'chamber':dict(pressure_pa=[1e5,4e5,3e5])})
        dense=dict(time_s=[0.,.02,1.,2.,4.],gas={'chamber':dict(pressure_pa=[1e5,1e8,4e5,1e8,3e5])})
        self.assertEqual(model.pressure_score(sparse,prior),model.pressure_score(dense,prior))

    def test_pressure_and_temperature_are_not_enough_for_refinement(self):
        delta=dict(max_witness_change_k=.04,max_pressure_change_bar=.007,
                   max_open_fraction_change=.001,max_final_aperture_relative_change=.01,
                   fracture_work_change_j=.0001,fine_fracture_work_j=.02,max_net_export_change_g=.01)
        self.assertTrue(model.refinement_passes(delta))
        for key,value in [('max_open_fraction_change',.1),('max_final_aperture_relative_change',.5),
                          ('fracture_work_change_j',.02),('max_net_export_change_g',1.)]:
            self.assertFalse(model.refinement_passes(dict(delta,**{key:value})))


if __name__=='__main__':
    unittest.main()

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_upv_shell_strength as trial


class ShellStrengthProtocol(unittest.TestCase):
    def test_only_strength_changes(self):
        scene,_=trial.geometry.inputs('no-cooling')
        original=copy.deepcopy(scene);base=trial.build(scene,1.,30.)
        for factor in (2.,4.,8.):
            stronger=trial.build(scene,factor,30.)
            self.assertEqual(base['cells'],stronger['cells'])
            self.assertEqual(base['shells'],stronger['shells'])
            self.assertEqual(base['gas_network']['compartments'],stronger['gas_network']['compartments'])
            self.assertEqual(base['gas_network']['thermal_contacts'],stronger['gas_network']['thermal_contacts'])
            self.assertEqual(trial.geometry.thermal_signature(stronger),trial.geometry.thermal_signature(scene))
            for a,b in zip(base['gas_network']['channels'],stronger['gas_network']['channels']):
                expected=copy.deepcopy(a)
                if expected['kind']=='rupture':
                    expected['opening_pressure_pa']*=factor
                self.assertEqual(b,expected)
        self.assertEqual(scene,original)

    def test_dense_diagnostics_do_not_add_observations(self):
        run=dict(time_s=[0.,.02,.04,.47],temperature_k={'101':[1.,2.,3.,4.]})
        prior=dict(result=dict(time_s=[0.,.47,1.]))
        selected=trial.observation_samples(run,prior)
        self.assertEqual(selected['time_s'],[0.,.47])
        self.assertEqual(selected['temperature_k']['101'],[1.,4.])

    def test_first_actual_rupture_not_initial_open_internal_channels(self):
        scene=dict(gas_network=dict(channels=[dict(id='burst',kind='rupture')]))
        run=dict(time_s=[0.,.1,.2,.3],
                 gas={'chamber':dict(pressure_pa=[101000,301000,201000,151000]),
                      'inside-C2':dict(pressure_pa=[101000,400000,250000,150000])},
                 channels={'burst':dict(transferred_mass_kg=[0,.001,.002,.003])},
                 events=[dict(id='internal',kind='vent_open',time_s=0),
                         dict(id='burst',kind='shell_rupture',time_s=.05)])
        result=trial.features(run,scene)
        self.assertEqual(result['first_open_s'],.05)
        self.assertEqual(result['first_net_gram_s'],.1)
        self.assertEqual(result['sampled_max_source_pressure_bar'],4.)


if __name__=='__main__':
    unittest.main()

import copy
import unittest

import numpy as np

from seed_upv_stored_charge import seed


class StoredChargeSeed(unittest.TestCase):
    def fixture(self):
        time=np.linspace(0.,850.,30);u=.25*(1.-np.exp(-time/400.));temperature=298.15+time**2/10000.
        law=dict(resistance_ohm=.016,temperature_log_slope_per_k=-.012,charge_log_slope=14.6,
                 product_conductivity_ratio=.6,reference_temperature_k=298.15)
        scene=dict(cells=[dict(id='C2',electrical=law,conversion=dict(mass_kg=.2))],
                   shells=[dict(material='unchanged')],metadata=dict(actual_profile='preserve'))
        run=dict(time_s=time,chemistry={'C2':dict(virgin_kg=.2*(1.-u),healthy_charged_kg=.2*u,
                    r2_precursor_kg=.2*u,residue_kg=np.zeros_like(u))},
                 electrical={'C2':dict(resistance_ohm=.016*np.exp(.006*(temperature-298.15)+8.*u))},
                 cell_mean_temperature_k={'C2':temperature})
        return scene,run

    def test_projects_only_electrical_law_from_training_stock(self):
        source,run=self.fixture();snapshot=copy.deepcopy(source);trial=seed(source,run)
        self.assertEqual(source,snapshot)
        actual=copy.deepcopy(trial);actual.pop('metadata');expected=copy.deepcopy(source);expected.pop('metadata')
        actual['cells'][0].pop('electrical');expected['cells'][0].pop('electrical')
        self.assertEqual(actual,expected)
        law=trial['cells'][0]['electrical']
        self.assertAlmostEqual(law['resistance_ohm'],.016,places=10)
        self.assertAlmostEqual(law['temperature_log_slope_per_k'],.006,places=10)
        self.assertAlmostEqual(law['charge_log_slope'],8.,places=8)
        self.assertEqual(law['charge_state_input'],'stored_precursor')
        self.assertFalse(trial['metadata']['stored_charge_revision']['calibrated'])
        self.assertFalse(trial['metadata']['stored_charge_revision']['validation_used'])

    def test_does_not_silently_migrate_twice(self):
        scene,run=self.fixture()
        with self.assertRaisesRegex(ValueError,'already'):
            seed(seed(scene,run),run)


if __name__=='__main__':unittest.main()

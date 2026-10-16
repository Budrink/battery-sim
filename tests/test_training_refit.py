import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import refit_upv_training as refit


class TrainingRefitTests(unittest.TestCase):
    def test_only_declared_intact_links_change(self):
        heat = refit.read(refit.model.shell.HEAT/'no-cooling-fitted-scenario.json')
        original, _ = refit.model.inputs('no-cooling')
        before = copy.deepcopy(original)
        heat['shells'][0]['contents_contact_h'] = 1234.
        next(c for c in heat['cells'] if c['id']=='C1')['thermal']['bulk_response']['temperature_contrast'] = [-7.]*3
        result = refit.update_post_links(original, heat)
        self.assertEqual(original, before)
        self.assertEqual(result['gas_network'], original['gas_network'])
        self.assertEqual(result['boundaries'], original['boundaries'])
        for a, b in zip(result['cells'], original['cells']):
            expected = copy.deepcopy(b)
            if a['role']=='battery':
                expected['thermal']['bulk_response'] = next(c for c in heat['cells'] if c['id']=='C1')['thermal']['bulk_response']
            self.assertEqual(a, expected)
        for a, b in zip(result['interfaces'], original['interfaces']):
            expected = copy.deepcopy(b)
            expected['law']['geometry_reference_logit'] = a['law']['geometry_reference_logit']
            self.assertEqual(a, expected)
        self.assertTrue(all(s['contents_contact_h']==1234. for s in result['shells']))

    def test_trial_inventory_and_design_rank(self):
        seed = refit.read(refit.SEED/'profile.json')['parameters']
        total = seed['solid_energy_j']+seed['gas_energy_j']; f0 = seed['gas_energy_j']/total
        design = []
        for case in refit.CASES:
            p = refit.parameters(seed, case)
            self.assertAlmostEqual(p['solid_energy_j']+p['gas_energy_j'], total)
            for k in ('gas_yield', 'wall_h', 'surface_h', 'strength_pa', 'damage_rate_per_s'):
                self.assertEqual(p[k], seed[k])
            design.append(refit.features([p['gas_energy_j']/total-f0,
                np.log(p['exposure_scale']/seed['exposure_scale']), np.log(p['solid_rate']/seed['solid_rate'])]))
        self.assertEqual(np.linalg.matrix_rank(design), 5)

    def test_training_preparation_never_loads_validation(self):
        profile = refit.read(refit.model.shell.HEAT/'profile.json')
        original_loader = refit.model.native.old.load_record
        original_inputs = refit.model.inputs
        calls = []
        def load(name):
            self.assertEqual(name, 'no-cooling'); calls.append(name)
            return original_loader(name)
        def inputs(name):
            self.assertEqual(name, 'no-cooling')
            return original_inputs(name)
        class StopBeforeCUDA:
            def run(self, scene):
                if any(b['id'].startswith('cooling-') for b in scene['boundaries']):
                    raise AssertionError('Cooling in training')
                raise RuntimeError('stopped before native integration')
        with patch.object(refit.model.native.old, 'load_record', side_effect=load), \
             patch.object(refit.model, 'inputs', side_effect=inputs):
            with self.assertRaisesRegex(RuntimeError, 'stopped before native'):
                refit.prepare('no-cooling', profile, StopBeforeCUDA())
        self.assertEqual(calls, ['no-cooling'])

    def test_cooling_range_is_declared_not_selected(self):
        self.assertEqual([c['sink_conductance_w_per_k'] for c in refit.COOLING_CASES], [3., 10., 30.])
        self.assertTrue(all(c['contact_h_w_per_m2_k']==10000. for c in refit.COOLING_CASES))


if __name__ == '__main__':
    unittest.main()

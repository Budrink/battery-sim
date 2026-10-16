import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mass_paths', ROOT/'scripts/upv_mass_paths.py')
model = importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


class MassPaths(unittest.TestCase):
    def setUp(self):
        self.scene, _ = model.inputs('no-cooling')

    def test_frozen_thermal_model(self):
        before = copy.deepcopy(self.scene)
        for gap in (.00005, .0002, .001):
            result = model.attach(self.scene, dict(model.initial(), gap_m=gap))
            self.assertEqual(model.thermal_signature(result), model.thermal_signature(before))
            self.assertEqual(self.scene, before)
            self.assertEqual([x['h_w_per_m2_k'] for x in result['gas_network']['thermal_contacts']],
                             [x['h_w_per_m2_k'] for x in before['gas_network']['thermal_contacts']])

    def test_local_paths_and_conservation(self):
        s = model.attach(self.scene, model.initial());gas = s['gas_network'];nodes = {n['id']: n for n in gas['compartments']}
        self.assertEqual(sum(n.get('local_equilibrium', {}).get('cell_id') == 'C2' for n in nodes.values()), 27)
        external = [n for n in nodes.values() if not n['id'].startswith('inside-C2')]
        old = next(n for n in self.scene['gas_network']['compartments'] if n['id']=='chamber')
        self.assertAlmostEqual(sum(n['volume_m3'] for n in external), .12, places=12)
        self.assertAlmostEqual(sum(n['initial_mass_kg'] for n in external), old['initial_mass_kg'], places=12)
        cv, tref = gas['cv_j_per_kg_k'], gas['reference_temperature_k']
        energy = lambda n:n['initial_mass_kg']*cv*(n['initial_temperature_k']-tref)
        self.assertAlmostEqual(sum(map(energy, external)), energy(old), places=8)
        for ch in gas['channels']:
            self.assertNotEqual(ch['to'], 'outside_domain')
            self.assertNotIn('surface_link', ch)
            if ch['kind']=='rupture':
                self.assertNotEqual(ch['to'], 'chamber')
                self.assertIn('local_equilibrium', nodes[ch['from']])
        # No through-pad path to neighbouring cells; local patches escape only
        # along their own interface or into the declared perimeter sectors.
        for ch in gas['channels']:
            if ch['from'].startswith('gap-'):
                self.assertTrue(ch['to'].startswith(('gap-', 'near-')))
        for pad in ('ISS1', 'TIM1', 'ISS2', 'TIM2', 'ISS3', 'TIM3'):
            self.assertEqual(sum(n.get('local_equilibrium', {}).get('cell_id')==pad for n in nodes.values()), 8)

    def test_forbidden_fit_knobs(self):
        for key in ('heat_capacity', 'conductivity', 'contact_h', 'surface_h', 'cooling_g'):
            with self.assertRaises(ValueError):
                model.attach(self.scene, dict(model.initial(), **{key: 1}))

    def test_cooling_plate_blocks_direct_bottom_exit(self):
        scene, _ = model.inputs('active-cooling')
        result = model.attach(scene, model.initial())
        paths = [p for p in result['gas_network']['channels'] if p['from']=='near--z']
        self.assertEqual({p['to'] for p in paths}, {'near--y', 'near-+y'})
        self.assertEqual(model.thermal_signature(scene), model.thermal_signature(result))

    def test_protocol_hash_survives_json_round_trip(self):
        protocol = dict(bounds=model.BOUNDS, parameters=model.PARAMETERS)
        self.assertEqual(model.digest(protocol), model.digest(json.loads(json.dumps(protocol))))


if __name__ == '__main__':
    unittest.main()

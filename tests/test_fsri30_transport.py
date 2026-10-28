import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import fsri30_transport_model as model
import search_fsri30_transport as worker
from fsri30_data import load
from fsri30_search_model import candidate as legacy_candidate, initial_point as legacy_initial
from generate_fsri30 import build


def fixture():
    return legacy_candidate(build(5945.), legacy_initial(), accuracy_level=1)


def synthetic(scene, neighbour_k=313.15):
    t = np.arange(0., 5946., 5.); ids = [f'B{i:02d}' for i in range(30)]
    temperature = np.full((len(t), 30), 298.15)
    temperature[t >= 1700., 10] = 1500.
    for i in (2, 3, 17):
        temperature[t >= 1800., i] = neighbour_k
    chemistry = {}
    for i, ident in enumerate(ids):
        reacted = np.where(t >= 1700., .025 if i == 10 else 0., 0.)
        chemistry[ident] = dict(residue_kg=(reacted*.67).tolist(), r2_generated_gas_kg=(reacted*.33).tolist())
    return dict(time_s=t.tolist(), partial=False,
        cell_mean_temperature_k={k: temperature[:, j].tolist() for j, k in enumerate(ids)},
        chemistry=chemistry)


class GeometryTests(unittest.TestCase):
    def test_all_extremes_preserve_fixed_physics_and_gas_inventory(self):
        base = fixture(); untouched = copy.deepcopy(base)
        points = [model.initial_point(), np.zeros(len(model.SPACE)), np.ones(len(model.SPACE))]
        points += [x for _, x in model.experiments()]
        for x in points:
            candidate = model.candidate(base, x)
            self.assertEqual(model.frozen(candidate), model.frozen(base))
            np.testing.assert_allclose(model.inventory(candidate), model.inventory(base), rtol=5e-14)
            self.assertEqual(candidate['primary_failures'], [])
            self.assertEqual(candidate['simulation']['minimum_dt_s'], .01)
        self.assertEqual(base, untouched)

    def test_no_chemistry_drive_timing_or_material_search_dimensions(self):
        names = {p.name for p in model.SPACE}
        self.assertFalse(any('r2' in n or 'r3' in n or 'heater' in n or 'casing' in n for n in names))
        self.assertEqual(len(model.experiments()), 5)

    def test_face_area_conductance_not_duplicated_by_shared_exposure(self):
        base = fixture()
        candidate = model.candidate(base, model.point(shared_gap_fraction=.8, upper_sleeve_fraction=.4))
        def sums(s):
            result = {}
            for c in s['gas_network']['thermal_contacts']:
                if not c['port_id'].startswith('B'):
                    continue
                ident = ':'.join(c['port_id'].split(':')[:3])
                g = c['connectivity']
                result[ident] = result.get(ident, 0.)+g['background_conductance_w_per_k']+.5*g['active_conductance_w_per_k']
            return result
        before, after = sums(base), sums(candidate)
        self.assertEqual(before.keys(), after.keys())
        for k in before:
            self.assertAlmostEqual(before[k], after[k], places=12)
        # Shared gas uses facing graph neighbours only, never a core or an
        # external flame linked through a solid ABS barrier.
        self.assertFalse(any(c['compartment'].startswith('plume') for c in candidate['gas_network']['thermal_contacts']))
        for c in candidate['gas_network']['thermal_contacts']:
            if c['port_id'].startswith('B'):
                self.assertIn(':case:', c['port_id'])

    def test_upper_rectangles_tile_the_actual_shell_face(self):
        base = fixture(); fraction = .4
        candidate = model.candidate(base, model.point(upper_sleeve_fraction=fraction))
        ports = {p['id']: p for p in candidate['ports']}
        for ident in ('B00', 'B10', 'B29'):
            core = next(c for c in base['cells'] if c['id'] == ident)
            for face in ('+x', '-x', '+y', '-y'):
                upper = ports[f'{ident}:case:{face}:gas-upper']
                lower = ports[f'{ident}:case:{face}:gas-body']
                contact = next(c for c in base['gas_network']['thermal_contacts'] if c['port_id'] == f'{ident}:case:{face}:outer')
                full_area = contact['connectivity']['background_conductance_w_per_k']/(.01*20.)
                self.assertAlmostEqual(upper['area_m2']+lower['area_m2'], full_area)
                axis = 'xyz'.index(face[1]); ri = [(axis+1)%3, (axis+2)%3].index(2)
                self.assertAlmostEqual(upper['centroid_m'][2]-upper['rectangle_m'][ri]/2,
                                       lower['centroid_m'][2]+lower['rectangle_m'][ri]/2)
                self.assertIn(':case:', upper['cell_id'])

    def test_hot_range_keeps_cold_both_direction_conductance(self):
        original = next(c for c in fixture()['gas_network']['thermal_contacts'] if c['port_id'].startswith('B'))
        def g(contact, log_response):
            l = contact['connectivity']; z = l['geometry_reference_logit']+log_response
            return l['background_conductance_w_per_k']+l['active_conductance_w_per_k']/(1+math.exp(-z))
        hot = model.hot_contact(original, 30.)
        self.assertAlmostEqual(g(hot, 0.), g(original, 0.), places=14)
        self.assertGreater(g(hot, math.log(10.)), g(original, math.log(10.)))

    def test_shared_weights_are_local_and_normalized(self):
        scene = fixture()
        for cell in ('B00', 'B10', 'B29'):
            for face in ('+x', '-x', '+y', '-y'):
                weights = model.facing_neighbours(scene, cell, face)
                if weights:
                    self.assertAlmostEqual(sum(w for _, w in weights), 1.)
                    self.assertTrue(all(w > 0 and j != int(cell[1:]) for j, w in weights))


class DirectionTests(unittest.TestCase):
    def test_real_neighbour_heat_beats_source_clock_feature(self):
        scene = fixture(); data, _ = load()
        cold = model.transport_loss(scene, synthetic(scene), data)
        warm = model.transport_loss(scene, synthetic(scene, 448.15), data)
        self.assertLess(warm['objective'], cold['objective']-10.)
        self.assertTrue(model.useful_response(cold, warm))
        self.assertEqual(warm['cascade']['reacted_components'], 1)
        self.assertFalse(warm['cascade']['autonomous_cascade'])
        self.assertFalse(warm['calibrated'])
        self.assertFalse(warm['validation_passed'])

    def test_sensitivity_gate_does_not_reward_only_hotter_source(self):
        scene = fixture(); data, _ = load(); run = synthetic(scene)
        baseline = model.transport_loss(scene, run, data)
        run['cell_mean_temperature_k']['B10'] = [3000.]*len(run['time_s'])
        hotter = model.transport_loss(scene, run, data)
        self.assertFalse(model.useful_response(baseline, hotter))
        self.assertEqual(baseline['transport_loss_parts']['neighbour_preheat'], hotter['transport_loss_parts']['neighbour_preheat'])

    def test_worker_does_not_launch_gp_after_insensitive_controls(self):
        scene = fixture(); data, _ = load(); run = synthetic(scene)
        score = dict(model.loss(scene, run, data), complete=True, constraint=-1., wall_s=10.,
            scene_hash=worker.canonical_hash(scene), native_sha256=worker.sha(worker.DEFAULT_PROBE))
        with tempfile.TemporaryDirectory(prefix='fsri30-direction-') as temp:
            root = Path(temp); reference = root/'reference'; reference.mkdir()
            worker.save(reference/'scenario.json', scene); worker.save(reference/'evaluation.json', score)
            args = SimpleNamespace(reference=reference, output=root/'search', probe=worker.DEFAULT_PROBE,
                trial_timeout=20., budget_hours=1., trials=10, seed=123)
            def evaluate(scene, folder, data, args):
                return dict(score, **model.transport_loss(scene, run, data))
            with patch.object(worker, 'read_run', return_value=run), \
                 patch.object(worker, 'evaluate_transport', side_effect=evaluate) as evaluate_mock, \
                 patch.object(worker, 'diagnostic_report'), patch.object(worker, 'propose') as gp, redirect_stdout(StringIO()):
                self.assertEqual(worker.run(args), 2)
            self.assertEqual(evaluate_mock.call_count, 5)
            gp.assert_not_called()
            status = json.loads((args.output/'status.json').read_text())
            self.assertEqual(status['phase'], 'no-neighbour-energy-response')
            self.assertEqual(status['completed_trials'], 6)
            self.assertFalse(status['calibrated'])


if __name__ == '__main__':
    unittest.main()

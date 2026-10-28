import copy
import gzip
import json
import re
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
from bounded_bayes import GaussianProcess, expected_improvement, propose, trust_region
from fsri30_data import load
from fsri30_objective import cascade, data_audit, loss, numerical_comparison, reject_reason, targets
from fsri30_search_model import (SPACE, admissible, candidate, conserved_fingerprint, decode,
                                initial_point, iteration_bounds, split_hood, validate_controls)
from fsri30_numerical_policy import exploration_policy, select as select_numerics
from fsri30_uncertainty import throughflow
from generate_fsri30 import build
import calibrate_fsri30_bayes as controller


def synthetic_run(scene, temperature=None, converted=None):
    t = np.arange(0., 5946., 5.)
    if t[-1] != 5945.:
        t = np.r_[t, 5945.]
    ids = [c['id'] for c in scene['cells'] if c['role'] == 'battery']
    y = np.full((len(t), 30), 298.15) if temperature is None else temperature
    chemistry = {}
    for j, key in enumerate(ids):
        reacted = np.zeros(len(t)) if converted is None else converted[:, j]
        chemistry[key] = dict(residue_kg=reacted*.67, r2_generated_gas_kg=reacted*.33)
    return dict(time_s=t.tolist(), cell_mean_temperature_k={k: y[:, j].tolist() for j, k in enumerate(ids)},
        chemistry=chemistry, released_energy_j_history=np.zeros(len(t)*30).tolist(),
        gas={'hood': {'released_energy_j': np.zeros(len(t)).tolist()}}, partial=False)


class SearchGeometryTests(unittest.TestCase):
    def test_roundtrip_bounds_and_parameter_immutability(self):
        x = initial_point()
        self.assertTrue(np.all((x >= 0.)&(x <= 1.)))
        self.assertEqual(set(decode(x)), {p.name for p in SPACE})
        for p, z in zip(SPACE, x):
            self.assertAlmostEqual(p.decode(z)/p.initial, 1., places=12)
        with self.assertRaises(ValueError):
            decode([-.1]*len(SPACE))
        base = build(5945.)
        old = copy.deepcopy(base)
        for point in (x, np.zeros(len(SPACE)), np.ones(len(SPACE))):
            s = candidate(base, point, layers=2, fine=True)
            self.assertEqual(conserved_fingerprint(s), conserved_fingerprint(base))
            self.assertEqual(s['primary_failures'], [])
            self.assertEqual(s['simulation']['minimum_dt_s'], .01)
            self.assertLess(s['simulation']['max_dt_s'], base['simulation']['max_dt_s'])
            self.assertEqual(base, old)

    def test_spatial_refinement_conserves_air_volume_mass_oxygen(self):
        base = build(5945.)
        def inventory(scene):
            nodes = scene['gas_network']['compartments']
            return np.array([sum(n['volume_m3'] for n in nodes), sum(n['initial_mass_kg'] for n in nodes),
                sum(n['initial_mass_kg']*n['initial_oxidizer_mass_fraction'] for n in nodes)])
        old = inventory(base)
        for layers in (1, 2, 4):
            s = copy.deepcopy(base)
            split_hood(s, layers)
            np.testing.assert_allclose(inventory(s), old, rtol=3e-15, atol=2e-15)
            nodes = {n['id']: n for n in s['gas_network']['compartments']}
            self.assertEqual(len(nodes), 901+30*layers)
            self.assertTrue(all(e['from'] in nodes and e['to'] in nodes for e in s['gas_network']['channels']))
            self.assertEqual(s['gas_network']['thermal_contacts'], base['gas_network']['thermal_contacts'])
            path = [e for e in s['gas_network']['channels'] if e['id'].startswith('exhaust-path-0-')]
            old_edge = next(e for e in base['gas_network']['channels'] if e['id'] == 'exhaust-0')
            self.assertAlmostEqual(sum(1/(e['area_m2']*e['discharge_coefficient'])**2 for e in path)*
                                   (old_edge['area_m2']*old_edge['discharge_coefficient'])**2, 1.)
            self.assertAlmostEqual(sum(e['mixing_length_m']/e['area_m2'] for e in path)*
                                   old_edge['area_m2']/old_edge['mixing_length_m'], 1.)
            with self.assertRaises(ValueError):
                split_hood(s)

    def test_shared_laws_and_no_unfunded_heater_work(self):
        s = candidate(build(5945.), np.ones(len(SPACE)))
        gas = s['gas_network']
        self.assertEqual(len({n['reaction_rate_per_s'] for n in gas['compartments']}), 1)
        self.assertEqual(len({c['conversion']['reaction_rate_per_s'] for c in s['cells'] if c['role'] == 'battery'}), 1)
        self.assertTrue(all('connectivity' in c for c in gas['thermal_contacts']))
        heater = next(c for c in s['cells'] if c['id'] == 'heater')['electrical']
        i = heater['current_profile_a'][0][1]
        u = heater['voltage_profile_v'][0][1]
        self.assertAlmostEqual(i*u, 9.)
        self.assertAlmostEqual(i*i*heater['resistance_ohm'], i*u)

    def test_cold_gas_stability_is_a_known_search_prior(self):
        self.assertTrue(admissible(initial_point()))
        invalid = initial_point()
        invalid[7], invalid[8] = 1., 0.  # high k600 + low activation burns cold gas
        self.assertFalse(admissible(invalid))
        rows = [dict(x=initial_point().tolist(), objective=10., complete=True, constraint=-1., wall_s=10.)]
        x, _ = propose(rows, len(SPACE), initial_point(), admissible=admissible, pool_size=128)
        self.assertTrue(admissible(x))
        for level in range(4):
            s = candidate(build(5945.), initial_point(), accuracy_level=level)
            self.assertGreaterEqual(s['gas_network']['coupled_transport_interval_s'], .005)
            self.assertEqual(s['simulation']['minimum_dt_s'], .01)

    def test_entire_ladder_respects_real_native_limits(self):
        header = (controller.ROOT/'native/include/battery_native/model.hpp').read_text()
        native_max = int(re.search(r'maximum_pressure_iterations\s*=\s*(\d+)', header)[1])
        self.assertEqual(iteration_bounds()['averaged_pressure_iterations'][1], native_max)
        base = build(5945.)
        for level in range(4):
            for layers in (1, 2, 4):
                s = candidate(base, initial_point(), accuracy_level=level, layers=layers)
                validate_controls(s)
                self.assertLessEqual(s['gas_network']['averaged_pressure_iterations'], native_max)
                self.assertLessEqual(s['gas_network']['averaged_payload_iterations'], 128)
                self.assertGreaterEqual(s['gas_network']['coupled_transport_interval_s'], s['simulation']['minimum_dt_s'])
                self.assertEqual(conserved_fingerprint(s), conserved_fingerprint(base))
        final = candidate(base, initial_point(), accuracy_level=3)
        self.assertEqual(final['gas_network']['averaged_pressure_iterations'], 2048)
        self.assertEqual(final['gas_network']['coupled_transport_interval_s'], .01)

    def test_invalid_controls_rejected_before_launch_or_output_creation(self):
        scene = candidate(build(5945.), initial_point(), accuracy_level=3)
        scene['gas_network']['averaged_pressure_iterations'] = 8193
        data, _ = load()
        with tempfile.TemporaryDirectory(prefix='fsri-bayes-test-') as temp, patch.object(controller, 'native') as solver:
            target = Path(temp)/'never-launched'
            with self.assertRaisesRegex(ValueError, 'native input contract'):
                controller.evaluate(scene, target, data, controller.DEFAULT_PROBE, 10.)
            solver.assert_not_called()
            self.assertFalse(target.exists())


class SearchObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = build(5945.)
        cls.data, cls.manifest = load()

    def test_unknown_observables_are_not_invented(self):
        audit = data_audit(self.data, self.manifest)
        self.assertFalse(audit['quantitative_validation_claim_allowed'])
        self.assertFalse(audit['sensor_mapping_confirmed'])
        run = synthetic_run(self.scene)
        summary = loss(self.scene, run, self.data)
        self.assertFalse(summary['validation_passed'])
        self.assertEqual(summary['cascade']['reacted_components'], 0)
        self.assertIsNone(summary['cascade']['observed_failure_count'])
        self.assertEqual(summary['features']['minimum_autonomous_propagation'], 32.)

    def test_observed_sensor_permutation_does_not_change_loss(self):
        run = synthetic_run(self.scene)
        permuted = dict(self.data)
        for i in range(1, 10):
            permuted[f'T{i}_k'] = self.data[f'T{10-i}_k']
        self.assertEqual(loss(self.scene, run, self.data), loss(self.scene, run, permuted))

    def test_peak_softening_nonzero_and_kelvin_scale(self):
        target = targets(self.data)
        self.assertTrue(np.any(target['burst']))
        self.assertTrue(np.all(target['q'] > 250.))
        score = loss(self.scene, synthetic_run(self.scene), self.data)
        self.assertGreater(score['peak_weight'], 0.)
        self.assertLess(score['peak_weight'], 1.)
        self.assertGreater(score['phase_loss']['cooling'], 0.)

    def test_prefix_is_lower_bound_not_finished_calibration(self):
        run = synthetic_run(self.scene)
        full = loss(self.scene, run, self.data, include_features=False)
        count = 401
        run['time_s'] = run['time_s'][:count]
        run['cell_mean_temperature_k'] = {k: v[:count] for k, v in run['cell_mean_temperature_k'].items()}
        run['partial'] = True
        prefix = loss(self.scene, run, self.data, include_features=False)
        self.assertLessEqual(prefix['objective'], full['objective'])
        self.assertFalse(prefix['complete'])
        self.assertNotIn('cascade', prefix)

    def test_cascade_is_component_conversion_not_event_message_count(self):
        t = np.arange(0., 5946., 5.)
        converted = np.zeros((len(t), 30))
        converted[:, 3] = .025*np.clip((t-2000.)/10., 0., 1.)
        converted[:, 8] = .025*np.clip((t-2100.)/10., 0., 1.)
        r = synthetic_run(self.scene, converted=converted)
        r['events'] = [dict(kind='runaway_secondary', time_s=5.)]*1000
        c = cascade(self.scene, r)
        self.assertTrue(c['autonomous_cascade'])
        self.assertEqual([e['component'] for e in c['events']], ['B03', 'B08'])
        self.assertAlmostEqual(c['events'][0]['time_s'], 2001.)
        self.assertAlmostEqual(c['inter_event_delays_s'][0], 100.)

    def test_numerical_comparison_rejects_incomplete_and_changed_energy(self):
        r = synthetic_run(self.scene)
        self.assertTrue(numerical_comparison(r, copy.deepcopy(r))['accepted'])
        partial = copy.deepcopy(r)
        partial['partial'] = True
        self.assertFalse(numerical_comparison(r, partial)['accepted'])
        changed = copy.deepcopy(r)
        changed['gas']['hood']['released_energy_j'][-1] = 1000.
        self.assertFalse(numerical_comparison(r, changed)['accepted'])


class BayesOptimizerTests(unittest.TestCase):
    def test_gp_interpolates_and_variance_is_not_negative(self):
        x = np.array([[0.], [.5], [1.]])
        gp = GaussianProcess(x, [1., 0., 1.])
        m, sd = gp.predict(x)
        np.testing.assert_allclose(m, [1., 0., 1.], atol=.002)
        self.assertTrue(np.all(sd > 0.))
        self.assertTrue(np.all(expected_improvement(m, sd, 0.) >= 0.))

    def test_resume_and_bounds(self):
        initial = np.array([.3, .7])
        rows = []
        for i in range(9):
            x, meta = propose(rows, 2, initial, initial_design=4, pool_size=128)
            self.assertTrue(np.all((x >= 0.)&(x <= 1.)))
            rows.append(dict(x=x.tolist(), objective=float(np.sum((x-.4)**2)),
                             complete=True, constraint=-1., wall_s=10.+i))
        resumed = json.loads(json.dumps(rows))
        a, ma = propose(rows, 2, initial, initial_design=4, pool_size=128)
        b, mb = propose(resumed, 2, initial, initial_design=4, pool_size=128)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ma, mb)
        self.assertIn('gp-', ma['method'])

    def test_pruned_runs_do_not_supply_a_fake_objective(self):
        rows = [dict(x=[.1], objective=10., complete=True, constraint=-1., wall_s=10.),
                dict(x=[.9], objective=None, complete=False, constraint=1., wall_s=2.)]
        a, meta = propose(rows, 1, [.5], initial_design=2, pool_size=128)
        rows[1]['objective'] = -1000000.
        b, altered = propose(rows, 1, [.5], initial_design=2, pool_size=128)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(meta, altered)
        self.assertTrue(0. < meta['feasible_probability'] <= 1.)

    def test_trust_region_shrinks_on_plateau_but_keeps_global_bounds(self):
        rows = [dict(x=[.5], objective=1., complete=True, constraint=-1.) for _ in range(5)]
        self.assertLess(trust_region(rows), .75)
        self.assertGreaterEqual(trust_region(rows*20), .08)


class SearchControllerTests(unittest.TestCase):
    """Exercise orchestration, restart and rejection with no CUDA invocation."""
    def settings(self, root):
        return SimpleNamespace(output=root/'search', base=controller.DEFAULT_BASE,
            probe=controller.DEFAULT_PROBE, seed=123, trials=2, initial_design=2,
            trial_timeout=60., budget_hours=1.)

    def test_complete_workflow_resume_and_no_false_validation(self):
        scene = build(5945.)
        run = synthetic_run(scene)
        data, _ = load()
        score = dict(loss(scene, run, data), constraint=-1., native_sha256='mock',
                     scene_hash='mock', wall_s=10., realtime_ratio=594.5)
        with tempfile.TemporaryDirectory(prefix='fsri-bayes-test-') as temp:
            args = self.settings(Path(temp))
            with patch.object(controller, 'evaluate', return_value=score), \
                 patch.object(controller, 'read_run', return_value=run), \
                 patch.object(controller, 'report_best'), redirect_stdout(StringIO()):
                self.assertEqual(controller.run_search(args), 0)
                history = (args.output/'observations.json').read_bytes()
                self.assertEqual(len(json.loads(history)), 2)
                self.assertEqual(controller.run_search(args), 0)
                self.assertEqual(history, (args.output/'observations.json').read_bytes())
            state = json.loads((args.output/'status.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertFalse(state['validation_passed'])
            self.assertFalse(state['autonomous_cascade'])
            self.assertTrue((args.output/'best-numerical-verification.json').exists())

    def test_bad_preflight_stops_before_any_optimization(self):
        with tempfile.TemporaryDirectory(prefix='fsri-bayes-test-') as temp:
            args = self.settings(Path(temp))
            with patch.object(controller, 'evaluate', return_value=dict(complete=False, constraint=1., error='mock fail')), \
                 patch.object(controller, 'propose') as proposer, redirect_stdout(StringIO()):
                self.assertEqual(controller.run_search(args), 2)
                proposer.assert_not_called()
            state = json.loads((args.output/'status.json').read_text())
            self.assertEqual(state['status'], 'needs-attention')
            self.assertFalse((args.output/'observations.json').exists())

    def test_incompatible_resume_does_not_reset_results(self):
        with tempfile.TemporaryDirectory(prefix='fsri-bayes-test-') as temp:
            args = self.settings(Path(temp))
            args.output.mkdir()
            (args.output/'contract.json').write_text('{}')
            (args.output/'observations.json').write_text('[{"preserved":true}]')
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                controller.run_search(args)
            self.assertEqual(json.loads((args.output/'observations.json').read_text()), [{'preserved': True}])

    def test_cache_lineage_uses_only_exact_complete_controls(self):
        scene = candidate(build(5945.), initial_point(), accuracy_level=2)
        expected = controller.canonical_hash(scene)
        core = controller.sha(controller.DEFAULT_PROBE)
        with tempfile.TemporaryDirectory(prefix='fsri-cache-test-') as temp:
            root = Path(temp)
            folder = root/'control-time2-space1'
            (folder/'replay').mkdir(parents=True)
            (folder/'scenario.json').write_text(json.dumps(scene))
            with gzip.open(folder/'replay/run.json.gz', 'wt') as f:
                json.dump({'time_s': [0., 5945.], 'advanced_to_s': 5945., 'partial': False}, f)
            result = dict(complete=True, constraint=-1., native_sha256=core, scene_hash=expected)
            (folder/'evaluation.json').write_text(json.dumps(result))
            self.assertEqual(controller.cached_control([root], 2, 1, scene, controller.DEFAULT_PROBE), folder)
            changed = copy.deepcopy(scene)
            changed['gas_network']['averaged_payload_iterations'] = 128
            self.assertIsNone(controller.cached_control([root], 2, 1, changed, controller.DEFAULT_PROBE))
            result['complete'] = False
            (folder/'evaluation.json').write_text(json.dumps(result))
            self.assertIsNone(controller.cached_control([root], 2, 1, scene, controller.DEFAULT_PROBE))

    def test_report_error_preserves_best_and_does_not_abort(self):
        scene = candidate(build(5945.), initial_point())
        with tempfile.TemporaryDirectory(prefix='fsri-report-test-') as temp:
            out = Path(temp)
            row = dict(trial='old-label', objective=10., calibrated=False, validation_passed=False)
            with patch.object(controller, 'report_best', side_effect=RuntimeError('plot failed')):
                controller.safe_report(out, out/'trial-1', scene, {}, {}, row)
            result = json.loads((out/'best.json').read_text())
            self.assertEqual(result['report_error'], 'plot failed')
            self.assertEqual(result['objective'], 10.)
            self.assertFalse(result['validation_passed'])

    def test_provisional_search_continues_after_failed_candidate(self):
        scene = build(5945.)
        run = synthetic_run(scene)
        data, _ = load()
        score = dict(loss(scene, run, data), constraint=-1., native_sha256='mock',
                     scene_hash='mock', wall_s=10., realtime_ratio=594.5)
        def controls_then_warning(evaluate_one, **unused):
            evaluate_one(0, 1)
            evaluate_one(1, 1)
            return dict(accepted=False, reason='unresolved R3 branch')
        attempted = []
        def evaluate(scene, folder, *unused, **options):
            if folder.name == 'trial-0001':
                attempted.append(1)
                return dict(complete=False, constraint=1., objective=None, error='rejected trial', wall_s=1.)
            if folder.name == 'trial-0002':
                attempted.append(2)
            return copy.deepcopy(score)
        with tempfile.TemporaryDirectory(prefix='fsri-continue-test-') as temp:
            args = self.settings(Path(temp))
            args.continue_unverified_search = True
            args.trials = 3
            with patch.object(controller, 'evaluate', side_effect=evaluate), \
                 patch.object(controller, 'select_numerics', side_effect=controls_then_warning), \
                 patch.object(controller, 'read_run', return_value=run), \
                 patch.object(controller, 'report_best'), redirect_stdout(StringIO()):
                self.assertEqual(controller.run_search(args), 0)
            self.assertEqual(attempted, [1, 2])
            history = json.loads((args.output/'observations.json').read_text())
            self.assertEqual(len(history), 3)
            self.assertFalse(history[1]['complete'])
            self.assertTrue(history[2]['complete'])
            state = json.loads((args.output/'status.json').read_text())
            self.assertTrue(state['numerical_policy']['provisional'])
            self.assertFalse(state['validation_passed'])


class NumericalPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = build(5945.)
        cls.data, _ = load()

    def control(self, level, layers, energy):
        run = synthetic_run(self.scene)
        run['gas']['hood']['released_energy_j'][-1] = energy
        return dict(run=run, score=dict(loss(self.scene, run, self.data), constraint=-1.),
                    folder=f'test-level{level}-layers{layers}')

    def test_failed_coarse_time_selects_checked_finer_time(self):
        seen = []
        def evaluate(level, layers):
            seen.append((level, layers))
            return self.control(level, layers, [100., 150., 154., 155.][level])
        result = select_numerics(evaluate)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['accuracy_level'], 1)
        self.assertEqual(result['layers'], 1)
        self.assertIn((2, 1), seen)
        self.assertFalse(result['comparisons'][0]['accepted'])
        self.assertTrue(result['comparisons'][1]['accepted'])

    def test_failed_time_ladder_blocks_optimization(self):
        result = select_numerics(lambda level, layers: self.control(level, layers, 100.*2**level))
        self.assertFalse(result['accepted'])
        self.assertIn('Time refinement', result['reason'])

    def test_spatial_refinement_rechecks_time_on_selected_space(self):
        seen = []
        def evaluate(level, layers):
            seen.append((level, layers))
            return self.control(level, layers, {1:100., 2:150., 4:152.}[layers])
        result = select_numerics(evaluate)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['accuracy_level'], 0)
        self.assertEqual(result['layers'], 2)
        self.assertIn((1, 2), seen)
        self.assertEqual(result['comparisons'][-1]['kind'], 'time-after-space')

    def test_comparisons_survive_later_control_failure(self):
        snapshots = []
        def evaluate(level, layers):
            if level == 2:
                raise RuntimeError('later control fails')
            return self.control(level, layers, 100.*2**level)
        with self.assertRaisesRegex(RuntimeError, 'later control fails'):
            select_numerics(evaluate, on_comparison=lambda history: snapshots.append(copy.deepcopy(history)))
        self.assertEqual(len(snapshots), 1)
        self.assertFalse(snapshots[0][0]['accepted'])

    def test_search_permission_is_not_convergence_acceptance(self):
        controls = {}
        for level, energy in ((0, 10000.), (1, 8000.)):
            c = self.control(level, 1, energy)
            c['run']['released_energy_j_history'][-1] = 40000.
            c['score']['wall_s'] = 10.+level
            controls[(level, 1)] = c
        strict = dict(accepted=False, reason='R3 branch failed')
        result = exploration_policy(controls, strict)
        self.assertTrue(result['search_allowed'])
        self.assertFalse(result['accepted'])
        self.assertFalse(result['numerically_verified'])
        self.assertLess(result['recorded_coupled_heat_l1_relative'], .05)
        self.assertGreater(result['supporting_comparison']['recorded_external_r3_relative_difference'], .05)
        controls[(1, 1)]['score']['constraint'] = 1.
        self.assertFalse(exploration_policy(controls, strict)['search_allowed'])

    def test_throughflow_split_can_conserve_but_miss_reaction(self):
        coarse = throughflow(.01)
        fine = throughflow(.0001)
        self.assertLess(abs(coarse['chemical_closure_j']), 1e-12)
        self.assertGreater(coarse['relative_heat_error'], .7)
        self.assertLess(fine['relative_heat_error'], coarse['relative_heat_error'])


if __name__ == '__main__':
    unittest.main()

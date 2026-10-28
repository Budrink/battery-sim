from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_fsri30_vent_retention import module_exits, restrict_exits, snapshot, BASE
from fsri30_vent_geometry import seal_module,sealed_cut_audit


def fixture():
    return dict(metadata=dict(heated_component='B10'), gas_history_groups=dict(outside=['air-']),
                gas_network=dict(compartments=[dict(id='body-07'), dict(id='body-22'),
                    dict(id='air-0', ambient_boundary=dict(renewal_flow_m3_s=.01))],
                    channels=[dict(id='vent-a', kind='passage', **{'from': 'body-07', 'to': 'air-0'}, area_m2=2.),
                              dict(id='vent-b', kind='passage', **{'from': 'body-22', 'to': 'air-0'}, area_m2=3.)]))


class VentRetentionTests(unittest.TestCase):
    def test_sealed_cut_is_explicit_and_audits_mass_and_counterexchange(self):
        base=fixture();closed=seal_module(base)
        self.assertEqual(module_exits(closed,allow_closed=True),['vent-a','vent-b'])
        self.assertEqual([e['area_m2'] for e in base['gas_network']['channels']],[2.,3.])
        with self.assertRaises(ValueError):module_exits(closed)
        fields=('transferred_mass_kg','transferred_chemical_j','transferred_oxidizer_kg',
                'transferred_vapor_kg','transferred_latent_j','mixed_mass_kg','delivered_gas_energy_j','effective_area_m2')
        run=dict(time_s=[0.,1.],channels={key:{f:[0.,0.] for f in fields} for key in ('vent-a','vent-b')})
        self.assertTrue(sealed_cut_audit(closed,run)['accepted'])
        for field in fields:
            leak=deepcopy(run);leak['channels']['vent-a'][field][1]=1e-14
            self.assertFalse(sealed_cut_audit(closed,leak)['accepted'])

    @unittest.skipUnless(BASE.exists(), 'Retained fixture not installed')
    def test_sealed_real_geometry_has_no_positive_area_path_to_exterior(self):
        base=json.loads(BASE.read_text());closed=seal_module(base)
        cut=set(module_exits(base));edges=closed['gas_network']['channels']
        graph={n['id']:set() for n in closed['gas_network']['compartments']}
        original={e['id']:e for e in base['gas_network']['channels']}
        for edge in edges:
            if edge['id'] in cut:self.assertEqual(edge['area_m2'],0.)
            else:self.assertEqual(edge,original[edge['id']])
            if edge.get('area_m2',1.)>0. or edge.get('kind')=='rupture':
                graph[edge['from']].add(edge['to']);graph[edge['to']].add(edge['from'])
        reached=set();pending=['body-10']
        while pending:
            node=pending.pop()
            if node in reached:continue
            reached.add(node);pending.extend(graph[node]-reached)
        self.assertFalse(any(name.startswith('air-') for name in reached))
        for key in ('cells','shells','interfaces','boundaries','heating_protocols'):
            self.assertEqual(base[key],closed[key])

    def test_only_two_access_areas_change_and_input_is_immutable(self):
        base = fixture()
        original = deepcopy(base)
        for scale in (.25, .1):
            candidate = restrict_exits(base, scale)
            for a, b in zip(base['gas_network']['channels'], candidate['gas_network']['channels']):
                self.assertEqual(b['area_m2'], a['area_m2']*scale)
                b['area_m2'] = a['area_m2']
            self.assertEqual(candidate, original)
        self.assertEqual(base, original)
        self.assertEqual(restrict_exits(base, 1.), base)

    def test_invalid_or_sealed_restrictions_rejected(self):
        for scale in (0., -1., 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                restrict_exits(fixture(), scale)

    def test_unknown_exits_and_internal_ambient_bypasses_rejected(self):
        for change in ('third', 'unknown', 'ambient', 'renamed'):
            scene = fixture()
            if change == 'third':
                edge = deepcopy(scene['gas_network']['channels'][0]); edge['id'] = 'vent-c'
                scene['gas_network']['channels'].append(edge)
            elif change == 'unknown':
                scene['gas_network']['channels'][0]['to'] = 'ambient'
            elif change == 'ambient':
                scene['gas_network']['compartments'][0]['ambient_boundary'] = {'area_m2': 1.}
            else:
                scene['gas_network']['channels'][0]['id'] = 'unqualified-bypass'
            with self.assertRaises(ValueError):
                module_exits(scene)

    @unittest.skipUnless(BASE.exists(), 'Retained fixture not installed')
    def test_real_fixture_preserves_chemistry_heater_mesh_and_ambient(self):
        scene = json.loads(BASE.read_text())
        for scale in (.25, .1):
            candidate = restrict_exits(scene, scale)
            by_id = {e['id']: e for e in scene['gas_network']['channels']}
            changed = []
            for edge in candidate['gas_network']['channels']:
                if edge != by_id[edge['id']]:
                    changed.append(edge['id'])
                    self.assertEqual(edge['area_m2'], by_id[edge['id']]['area_m2']*scale)
                    edge['area_m2'] = by_id[edge['id']]['area_m2']
            self.assertEqual(sorted(changed), module_exits(scene))
            self.assertEqual(candidate, scene)

    def test_budget_uses_global_minus_outside_and_never_extrapolates(self):
        fields = ('escaped_energy_j', 'escaped_chemical_j', 'escaped_mass_kg',
                  'gas_released_energy_j', 'gas_chemical_j', 'gas_oxidizer_kg',
                  'gas_sensible_energy_j', 'thermal_boundary_energy_out_j')
        totals = {key: [0., 10., 20.] for key in fields}
        channels = {key: {f: [0., 1., 3.] for f in
                    ('transferred_mass_kg', 'transferred_chemical_j',
                     'delivered_gas_energy_j', 'transferred_oxidizer_kg')}
                    for key in ('vent-a', 'vent-b')}
        run = dict(time_s=[0., 5., 10.], channels=channels, process_totals=totals,
                   gas_groups={'outside': dict(released_energy_j=[0., 4., 9.],
                       chemical_j=[0., 2., 3.], oxidizer_kg=[0., 1., 2.],
                       sensible_energy_j=[0., 3., 6.], maximum_pressure_pa=[1., 2., 3.])},
                   gas_film_trace={'film': dict(solid='B11:case:+x', heat_to_solid_j=[0., 2., 4.])},
                   cell_mean_temperature_k={'B10': [300., 400., 500.], 'B11': [300., 310., 320.]})
        result = snapshot(fixture(), run, 0., 9.)
        self.assertEqual(result['interval_s'], [0., 5.])
        self.assertEqual(result['r3_inside_including_battery_pores_j'], 6.)
        self.assertEqual(result['vent_net']['transferred_mass_kg'], 2.)
        self.assertEqual(result['film_to_other_cans_j'], 2.)
        for start, end in ((0., 11.), (-1., 5.), (6., 9.)):
            with self.assertRaises(ValueError):
                snapshot(fixture(), run, start, end)


if __name__ == '__main__':
    unittest.main()

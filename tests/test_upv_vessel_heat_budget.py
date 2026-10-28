import sys
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from upv_vessel_heat_budget import wall_inventory


class VesselBudget(TestCase):
    def test_stored_heat_is_not_mistaken_for_removed_heat(self):
        scene = dict(cells=[dict(id='chamber-wall', role='passive',
            thermal={'heat_capacity_j_per_k': 20.})])
        run = dict(time_s=[0., 10.], thermal_paths={'objects': {'chamber-wall': {
            'reference_capacity_weighted_temperature_k': [300., 310.]}}},
            gas_film_trace={'a': {'solid': 'chamber-wall', 'heat_to_solid_j': [10., 260.]},
                            'b': {'solid': 'C3', 'heat_to_solid_j': [0., 999.]}})
        last = wall_inventory(scene, run)['snapshots'][-1]
        self.assertEqual(last['net_heat_from_gas_j'], 250.)
        self.assertEqual(last['stored_energy_change_j'], 200.)
        self.assertEqual(last['inferred_heat_to_room_j'], 50.)

    def test_unknown_solid_heat_path_is_not_silently_ignored(self):
        scene = dict(cells=[dict(id='chamber-wall', thermal={'heat_capacity_j_per_k': 20.})],
            ports=[dict(id='wall', cell_id='chamber-wall'), dict(id='other', cell_id='C3')],
            interfaces=[dict(port_a='wall', port_b='other')])
        with self.assertRaisesRegex(ValueError, 'unaccounted solid interface'):
            wall_inventory(scene, {})

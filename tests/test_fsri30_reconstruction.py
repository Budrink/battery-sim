from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from generate_fsri30 import build, DIAMETER, HEIGHT, BOX
from fsri30_search_model import split_hood
from fsri30_reconstruction import (source_geometry, transfer_r2, partition,
                                   polygon_area, exterior_area, R2_KEYS)
from check_fsri30_source_reconstruction import cases, metrics


def scene():
    s = build(5945.); split_hood(s)
    return s


class ReconstructionTests(unittest.TestCase):
    def test_ablation_matrix_separates_geometry_and_kinetics(self):
        base=scene(); copies=cases(base)
        for name in ('reference','r2-only'):
            self.assertNotIn('geometry_reconstruction',copies[name]['metadata'])
        for name in ('reference','geometry-only'):
            self.assertNotIn('r2_transfer',copies[name]['metadata'])
        self.assertIn('r2_transfer',copies['combined']['metadata'])
        self.assertFalse(any('No scheduled ignition, measured-temperature forcing, time shifts or UPV fitted coefficients' == note
                             for note in copies['combined']['metadata']['assumptions']))
        self.assertEqual(copies['combined']['metadata']['near_field']['outlets'],2)
        self.assertEqual(copies['geometry-only']['gas_network'],copies['combined']['gas_network'])
        self.assertEqual(base['simulation']['minimum_dt_s'],.01)
        self.assertNotIn('fresh_from_zero',base['metadata'])

    def test_r1_inert_residue_does_not_count_as_r2_conversion(self):
        base=scene();ids=[c['id'] for c in base['cells'] if c['role']=='battery']
        run=dict(time_s=[0.,1.],started_at_s=0.,advanced_to_s=1.,wall_s=1.,
            cell_mean_temperature_k={k:[298.15,300.] for k in ids},
            chemistry={k:dict(residue_kg=[0.,.025],r2_generated_gas_kg=[0.,0.]) for k in ids},
            gas={'body-00':dict(pressure_pa=[101325.,101325.])},
            process_totals={k:[0.,0.] for k in ('gas_released_energy_j','gas_chemical_j','gas_oxidizer_kg')},
            solver_profile=dict(minimum_regular_step_s=.01))
        with patch('check_fsri30_source_reconstruction.loss',return_value={'complete':np.bool_(False)}):
            result=metrics(base,run,{})
        self.assertEqual(result['r2_converted_components'],0)
        import json
        json.dumps(result,allow_nan=False)

    def test_only_three_r2_parameters_transfer_from_actual_accepted_scene(self):
        base = scene(); result = transfer_r2(base)
        for a, b in zip(base['cells'], result['cells']):
            if a['role'] != 'battery':
                self.assertEqual(a, b); continue
            law = deepcopy(b['conversion'])
            self.assertAlmostEqual(law['reaction_rate_per_s'], .01999639111839507)
            self.assertEqual(law['reaction_activation_j_per_mol'], 100000.)
            law.update({k:a['conversion'][k] for k in R2_KEYS})
            self.assertEqual(law, a['conversion'])
        for key in ('gas_network', 'shells', 'interfaces', 'boundaries', 'simulation', 'primary_failures'):
            self.assertEqual(base[key], result[key])

    def test_partition_closes_box_and_rejects_cylinder_overlap(self):
        s = scene(); meta = s['metadata']; t = meta['geometry_budget']['abs_thickness_m']
        inner = np.array(BOX)-2*t; positions = list(meta['approximate_positions_m'].values())
        polygons, shared, walls = partition(positions, inner)
        self.assertAlmostEqual(sum(polygon_area(p) for p in polygons), inner[0]*inner[1], places=14)
        self.assertTrue(all(polygon_area(p) > math.pi*(DIAMETER/2)**2 for p in polygons))
        for face in ('-x','+x','-y','+y'):
            self.assertAlmostEqual(sum(w['length_m'] for w in walls if w['face'] == face),
                inner[1] if face[1] == 'x' else inner[0], places=13)
        self.assertGreater(len(shared), 30)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            partition([[0.,0.],[.001,0.]], [.1,.1,.1])

    def test_volumes_gas_inventory_chemistry_and_ruptures_preserved(self):
        base = scene(); s = source_geometry(base)
        for key in ('shells', 'interfaces', 'simulation', 'primary_failures'):
            self.assertEqual(base[key], s[key])
        self.assertEqual([c for c in base['cells'] if c['role'] == 'battery'],
                         [c for c in s['cells'] if c['role'] == 'battery'])
        for kind in ('rupture', 'internal'):
            self.assertEqual([e for e in base['gas_network']['channels'] if e['kind'] == kind],
                             [e for e in s['gas_network']['channels'] if e['kind'] == kind])
        for field in ('volume_m3','initial_mass_kg'):
            self.assertAlmostEqual(sum(n[field] for n in base['gas_network']['compartments']),
                                   sum(n[field] for n in s['gas_network']['compartments']), places=13)

    def test_vents_only_at_boundary_no_internal_teleport(self):
        base = scene(); s = source_geometry(base)
        geom = s['metadata']['geometry_reconstruction']; gas = s['gas_network']
        self.assertAlmostEqual(sum(v['area_m2'] for v in geom['central_front_back_apertures']), exterior_area(base))
        boundary = {w['component'] for w in geom['wall_faces'] if w['face'][1]=='y'}
        vents = [e for e in gas['channels'] if e['id'].startswith('vent-')]
        self.assertGreater(len(vents), 1)
        self.assertTrue(all(int(e['from'].split('-')[-1]) in boundary for e in vents))
        self.assertFalse(any(e['id'].startswith('exhaust-') for e in gas['channels']))
        graph = {}
        for e in gas['channels']:
            if e['kind'] == 'rupture': continue
            graph.setdefault(e['from'], set()).add(e['to']); graph.setdefault(e['to'], set()).add(e['from'])
        seen = {'hood'}; pending = ['hood']
        while pending:
            for key in graph.get(pending.pop(), ())-seen:
                seen.add(key); pending.append(key)
        self.assertTrue(all(f'{band}-{j:02d}' in seen for j in range(30) for band in ('top','body','base')))

    def test_wall_contacts_are_local_and_thermal_areas_close(self):
        s = source_geometry(scene()); t = s['metadata']['geometry_budget']['abs_thickness_m']
        inner = np.array(BOX)-2*t; geom = s['metadata']['geometry_reconstruction']
        for face in ('-x','+x','-y','+y','-z','+z'):
            inside = ('-' if face[0]=='+' else '+')+face[1]
            contacts = [c for c in s['gas_network']['thermal_contacts'] if c['port_id']==f'ABS:{face}:{inside}']
            area = sum((c['connectivity']['background_conductance_w_per_k']+
                        .5*c['connectivity']['active_conductance_w_per_k'])/8 for c in contacts)
            axis = 'xyz'.index(face[1]); expected = inner[(axis+1)%3]*inner[(axis+2)%3]
            if face[1]=='y': expected -= geom['unmeasured_total_aperture_area_m2']/2
            self.assertAlmostEqual(area, expected, places=13)
            if axis < 2:
                allowed = {w['component'] for w in geom['wall_faces'] if w['face']==face}
                self.assertTrue(all(int(c['compartment'].split('-')[-1]) in allowed for c in contacts))


if __name__ == '__main__': unittest.main()

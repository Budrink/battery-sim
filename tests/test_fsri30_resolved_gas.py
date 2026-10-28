import math
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from free_gas_mesh import exterior_grid
from fsri30_resolved_gas import build_resolved
from check_fsri30_resolved_gas import compare_mesh
from test_fsri30_reconstruction import scene


class ResolvedGasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=scene()
        cls.levels=[build_resolved(cls.base,h) for h in (.04,.02)]

    def test_grid_inventory_connected_and_no_inside_shortcuts(self):
        for h in (.04,.02,.01):
            nodes,faces=exterior_grid([.092,.172,.092],h)
            self.assertAlmostEqual(sum(n.volume for n in nodes.values()),1.,places=12)
            by_id={n.name:n for n in nodes.values()};graph={n:set() for n in by_id}
            for a,b,area,length,axis in faces:
                self.assertGreater(area,0.);self.assertGreater(length,0.)
                graph[a].add(b);graph[b].add(a)
                self.assertAlmostEqual(by_id[a].high[axis],by_id[b].low[axis])
            seen=set();pending=[next(iter(graph))]
            while pending:
                key=pending.pop()
                if key in seen:continue
                seen.add(key);pending.extend(graph[key]-seen)
            self.assertEqual(len(seen),len(nodes))

    def test_refinement_does_not_change_materials_chemistry_or_inventory(self):
        for s in self.levels:
            for key in ('initial_mass_kg','volume_m3'):
                self.assertAlmostEqual(sum(n[key] for n in s['gas_network']['compartments']),
                    sum(n[key] for n in self.base['gas_network']['compartments']),places=11)
            oxygen=lambda sc:sum(n['initial_mass_kg']*n.get('initial_oxidizer_mass_fraction',0.)
                                for n in sc['gas_network']['compartments'])
            self.assertAlmostEqual(oxygen(s),oxygen(self.base),places=12)
        for a,b in zip(self.levels[0]['cells'],self.levels[1]['cells']):
            self.assertEqual(a.get('conversion'),b.get('conversion'))
            self.assertEqual(a['thermal']['heat_capacity_j_per_k'],b['thermal']['heat_capacity_j_per_k'])
            self.assertEqual(a['thermal']['conductivity_w_per_m_k'],b['thermal']['conductivity_w_per_m_k'])

    def test_local_surface_ports_and_openings(self):
        for s in self.levels:
            ports={p['id']:p for p in s['ports']}
            for contact in s['gas_network']['thermal_contacts']:
                if not contact['port_id'].startswith('gasmesh:'):continue
                port=ports[contact['port_id']]
                self.assertIn('thermal_volume',contact)
                self.assertGreater(port['area_m2'],0.)
                self.assertIn('gas_surface_connectivity',contact)
            openings=sum(e['area_m2'] for e in s['gas_network']['channels'] if e['id'].startswith('vent-'))
            self.assertAlmostEqual(openings,s['metadata']['geometry_reconstruction']['unmeasured_total_aperture_area_m2'],places=13)
            self.assertFalse(any(b['id'].startswith('room-') for b in s['boundaries']))
            self.assertFalse(any(n['id']=='hood' for n in s['gas_network']['compartments']))

    def test_distributed_resistance_tracks_length_not_pixel_count(self):
        for s in self.levels:
            for e in s['gas_network']['channels']:
                if not e['id'].startswith('air-path-'):continue
                self.assertAlmostEqual(e['hydraulic_resistance_factor'],e['mixing_length_m']/.05)

    def test_stirring_physical_length_is_independent_of_grid(self):
        for s in self.levels:
            self.assertEqual(s['gas_network']['thermal_stirring_length_m'],.02)
            self.assertEqual(s['metadata']['thermal_stirring_prior']['physical_length_m'],.02)
            self.assertFalse(s['gas_network']['pressure_pattern_predictor'])
        trial=build_resolved(self.base,.02,pressure_patterns=True)
        self.assertTrue(trial['gas_network']['pressure_pattern_predictor'])

    def test_local_refinement_preserves_volume_faces_and_surface_area(self):
        nodes,edges=exterior_grid([.092,.172,.092],.02,vent_spacing=.005)
        self.assertAlmostEqual(sum(n.volume for n in nodes.values()),1.,places=12)
        by_id={n.name:n for n in nodes.values()};seen=set();graph={k:set() for k in by_id}
        for a,b,area,distance,axis in edges:
            v,w=by_id[a],by_id[b];self.assertAlmostEqual(v.high[axis],w.low[axis])
            self.assertGreater(area,0.);self.assertGreater(distance,0.)
            self.assertAlmostEqual(distance,w.center[axis]-v.center[axis])
            graph[a].add(b);graph[b].add(a)
        pending=[next(iter(graph))]
        while pending:
            key=pending.pop()
            if key in seen:continue
            seen.add(key);pending.extend(graph[key]-seen)
        self.assertEqual(len(seen),len(nodes))
        trial=build_resolved(self.base,.02,vent_spacing=.005,wall_spacing=.01)
        self.assertEqual(trial['metadata']['air_mesh']['abs_spacing_m'],.01)
        self.assertEqual(trial['metadata']['air_mesh']['external_nodes'],len(nodes))
        for a,b in zip(trial['cells'],self.levels[0]['cells']):
            self.assertEqual(a['thermal']['heat_capacity_j_per_k'],b['thermal']['heat_capacity_j_per_k'])

    def test_only_external_heater_is_powered_and_cutoff_is_autonomous(self):
        for s in self.levels:
            self.assertFalse(any(c.get('electrical',{}).get('enabled',False) for c in s['cells'] if c['role']=='battery'))
            p=s['heating_protocols'][0]
            self.assertEqual(p['ramp_k_per_s'],.1)
            self.assertEqual(p['cutoff_rate_k_per_s'],10.)
            self.assertEqual(p['cutoff_temperature_k'],773.15)
            self.assertNotIn('cutoff_time_s',p)

    def test_cold_agreement_cannot_qualify_reactive_mesh_convergence(self):
        run=dict(time_s=[0.,10.],cell_mean_temperature_k={'B00':[298.15,300.]},
            process_totals=dict(gas_released_energy_j=[0.,0.]),
            gas_groups=dict(outside=dict(maximum_temperature_k=[298.15,298.15])))
        info=dict(all_gas_r3_j=0.,r2_converted_components=0,model_conversion_events=[])
        comparison=compare_mesh(run,run,info,info)
        self.assertFalse(comparison['accepted'])
        self.assertFalse(comparison['reactive_window_exercised'])
        run['process_totals']['gas_released_energy_j'][-1]=100.
        info.update(all_gas_r3_j=100.,r2_converted_components=1,
                    model_conversion_events=[dict(component='B00',time_s=5.)])
        self.assertTrue(compare_mesh(run,run,info,info)['accepted'])
        rounded=deepcopy(run);rounded['time_s'][-1]-=1e-10
        self.assertTrue(compare_mesh(run,rounded,info,info)['accepted'])
        shifted=deepcopy(run);shifted['time_s'][-1]-=.001
        with self.assertRaisesRegex(ValueError,'do not align'):compare_mesh(run,shifted,info,info)


if __name__=='__main__':unittest.main()

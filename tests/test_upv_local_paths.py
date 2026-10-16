import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import calibrate_upv_area as area
import upv_local_mass_paths as local
from gas_void_geometry import VoidGrid, contains


class LocalPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scene,_=area.old.geometry.inputs('no-cooling')
        cls.before=area.build(scene,area.BASE,17)
        cls.after=local.attach(cls.before)

    def test_frozen_physics_and_inventory(self):
        a,b=self.before,self.after
        self.assertEqual(area.old.geometry.thermal_signature(a),area.old.geometry.thermal_signature(b))
        self.assertEqual([c.get('conversion') for c in a['cells']],[c.get('conversion') for c in b['cells']])
        cv=a['gas_network']['cv_j_per_kg_k'];tref=a['gas_network']['reference_temperature_k']
        def total(s,key):return sum(key(n) for n in s['gas_network']['compartments'])
        for f in (lambda n:n['volume_m3'],lambda n:n['initial_mass_kg'],lambda n:n['initial_mass_kg']*cv*(n['initial_temperature_k']-tref),
                  lambda n:n['initial_mass_kg']*n.get('initial_oxidizer_mass_fraction',0)):
            self.assertAlmostEqual(total(a,f),total(b,f),places=8)
        self.assertTrue(any(n['id'].startswith('near-') for n in a['gas_network']['compartments']))
        self.assertFalse(any(n['id'].startswith(('near-', 'gap-ISS2-inlet', 'gap-TIM2-inlet')) for n in b['gas_network']['compartments']))

    def test_local_rupture_receivers(self):
        gas=self.after['gas_network'];nodes={n['id']:n for n in gas['compartments']}
        rupture=[p for p in gas['channels'] if p['kind']=='rupture']
        self.assertEqual(len(rupture),54)
        self.assertGreater(len({p['from'] for p in rupture}),6)
        self.assertGreater(len({p['to'] for p in rupture}),6)
        self.assertEqual(len({(p['port_id'],p['shell_thermal_volume']) for p in rupture}),54)
        for p in rupture:
            self.assertNotEqual(p['to'],'chamber')
            self.assertIn('local_equilibrium',nodes[p['from']])
        for p in gas['channels']:
            if p['to']=='chamber':self.assertTrue(p['id'].startswith('void-face-'))
            self.assertNotEqual(p['to'],'outside_domain')

    def test_voids_outside_obstacles(self):
        meta=self.after['metadata']['local_mass_paths']
        for c in meta['void_cells']:
            self.assertFalse(any(contains(box,c['center']) for box in meta['obstacles']))

    def test_solid_separates_free_components(self):
        grid=VoidGrid(([0,0,0],[3,1,1]),[([1,0,0],[2,1,1])],[[],[],[]])
        self.assertEqual(len(grid.cells),2)
        self.assertTrue(all(b is None for a,b,*_ in grid.faces()))
        self.assertIsNone(grid.locate([1.5,.5,.5]))
        grid=VoidGrid(([0,0,0],[3,1,1]),[],[[1,2],[],[]])
        self.assertEqual(sum(b is not None for a,b,*_ in grid.faces()),2)

    def test_cooling_plate_is_an_obstacle(self):
        s,_=area.old.geometry.inputs('active-cooling');s=local.attach(area.build(s,area.BASE,17))
        self.assertFalse(any(p['kind']=='rupture' and ':case:-z:' in p['port_id'] for p in s['gas_network']['channels']))
        self.assertTrue(s['metadata']['local_mass_paths']['blocked_paths'])
        for c in s['metadata']['local_mass_paths']['void_cells']:
            self.assertFalse(any(contains(box,c['center']) for box in s['metadata']['local_mass_paths']['obstacles']))

    def test_explicit_holder_can_close_route(self):
        # A user-supplied solid closes the full +y sector; do not silently reroute
        # blocked local holes to a distant chamber or to another side.
        s=local.attach(self.before,obstacles=[([0,.0435,-.114],[.07,.064,.114])])
        self.assertFalse(any(p['kind']=='rupture' and ':case:+y:' in p['port_id'] for p in s['gas_network']['channels']))
        self.assertTrue(s['metadata']['local_mass_paths']['blocked_paths'])

    def test_contact_h_unchanged_and_local(self):
        before={c['port_id']:c for c in self.before['gas_network']['thermal_contacts']}
        for c in self.after['gas_network']['thermal_contacts']:
            self.assertEqual(c['h_w_per_m2_k'],before[c['port_id']]['h_w_per_m2_k'])
            if c['compartment'].startswith('void-'):self.assertIn('thermal_volume',c)


if __name__=='__main__':unittest.main()

import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import calibrate_upv_stochastic as model
import compare_upv_shell_mesh as mesh

class StochasticProtocol(unittest.TestCase):
    def test_arbitrary_mesh_remap(self):
        values=np.arange(27,dtype=float).tolist()
        for grid in [(1,2,3),(1,3,6),(4,5,7),(1,1,1)]:
            remapped=model.remap_uniform_field(values,(3,3,3),grid)
            self.assertEqual(len(remapped),np.prod(grid))
            self.assertAlmostEqual(np.mean(remapped),np.mean(values),places=12)
            self.assertGreaterEqual(min(remapped),min(values))
            self.assertLessEqual(max(remapped),max(values)+1e-12)
        # Distinguish correct axis/storage ordering from a mere mean check.
        self.assertTrue(np.allclose(model.remap_uniform_field([1,2,3],(1,3,1),(1,2,1)),[4/3,8/3]))
        scene,_=model.geometry.inputs('no-cooling')
        self.assertEqual(mesh.grid_for_size(scene,64.),(1,2,3))
        self.assertEqual(mesh.grid_for_size(scene,32.),(1,3,6))
        for size in mesh.SIZES_MM:
            grid=mesh.grid_for_size(scene,size);refined=model.refine_shells(scene,grid)
            self.assertEqual(model.physical_signature(scene),model.physical_signature(refined))
            for a,b in zip(scene['shells'],refined['shells']):
                for face,v in a['initial_temperature_fields_k'].items():
                    self.assertAlmostEqual(np.mean(v),np.mean(b['initial_temperature_fields_k'][face]),places=10)

    def test_refinement_is_conservative(self):
        scene,_=model.geometry.inputs('no-cooling');old=copy.deepcopy(scene)
        refined=model.refine_shells(scene,model.GRID)
        self.assertEqual(scene,old)
        self.assertEqual(model.physical_signature(scene),model.physical_signature(refined))
        for a,b in zip(scene['shells'],refined['shells']):
            for face,v in a['initial_temperature_fields_k'].items():
                self.assertAlmostEqual(np.mean(v),np.mean(b['initial_temperature_fields_k'][face]),places=10)
        self.assertEqual(refined['cells'],scene['cells'])

    def test_candidates_only_change_barrier(self):
        scene,_=model.geometry.inputs('no-cooling');a=model.build(scene,model.BASE,17)
        for _,p in model.CASES:
            b=model.build(scene,p,29)
            for key in ('cells','shells','interfaces','ports','boundaries','measurement_probes'):
                self.assertEqual(a[key],b[key])
            for key in ('compartments','thermal_contacts'):
                self.assertEqual(a['gas_network'][key],b['gas_network'][key])
            for x,y in zip(a['gas_network']['channels'],b['gas_network']['channels']):
                expected=copy.deepcopy(x)
                if x['kind']=='rupture':
                    expected['opening_pressure_pa']=p['pressure_pa']
                    expected['damage_opening'].update(seed=29,pixel_rate_per_s=p['rate_per_s'],pixel_pressure_exponent=p['pressure_exponent'])
                self.assertEqual(expected,y)

    def test_invalid_parameters(self):
        scene,_=model.geometry.inputs('no-cooling')
        for p in [dict(model.BASE,heat_capacity=1),dict(model.BASE,pressure_pa=float('nan')),dict(model.BASE,rate_per_s=-1)]:
            with self.assertRaises(ValueError):model.build(scene,p,17)

if __name__=='__main__':unittest.main()

import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from free_gas_mesh import exterior_grid


class NestedJets(unittest.TestCase):
    envelope=[.092,.172,.092]

    def planes(self,nodes,axis):
        return np.unique([x for n in nodes.values() for x in (n.low[axis],n.high[axis])])

    def test_coarse_planes_retained_and_fine_planes_aligned(self):
        coarse,_=exterior_grid(self.envelope,.02)
        fine,_=exterior_grid(self.envelope,.01)
        local,faces=exterior_grid(self.envelope,.02,vent_spacing=.01)
        for axis in range(3):
            c,f,l=(self.planes(nodes,axis) for nodes in (coarse,fine,local))
            for x in c:self.assertLess(np.min(abs(l-x)),1e-12)
            for x in l:self.assertLess(np.min(abs(f-x)),1e-12)
        self.assertAlmostEqual(sum(n.volume for n in local.values()),1.,places=12)
        self.assertGreater(len(local),len(coarse))
        self.assertLess(len(local),len(fine)//2)
        self.assertTrue(all(area>0. and distance>0. for _,_,area,distance,_ in faces))

    def test_same_spacing_is_exact_identity_not_reanchored_grid(self):
        coarse,faces=exterior_grid(self.envelope,.02)
        local,local_faces=exterior_grid(self.envelope,.02,vent_spacing=.02)
        self.assertEqual(set(local),set(coarse));self.assertEqual(local_faces,faces)
        for key in local:
            np.testing.assert_array_equal(local[key].low,coarse[key].low)
            np.testing.assert_array_equal(local[key].high,coarse[key].high)

    def test_local_cells_exactly_union_reference_children(self):
        local,_=exterior_grid(self.envelope,.02,vent_spacing=.01)
        fine,_=exterior_grid(self.envelope,.01)
        fl=np.array([n.low for n in fine.values()]);fh=np.array([n.high for n in fine.values()])
        for n in local.values():
            overlap=np.prod(np.maximum(0.,np.minimum(fh,n.high)-np.maximum(fl,n.low)),axis=1)
            self.assertAlmostEqual(float(overlap.sum()),n.volume,places=14)
            children=overlap>1e-18
            self.assertTrue(np.allclose(overlap[children],np.prod(fh[children]-fl[children],axis=1),rtol=1e-10,atol=1e-18))


if __name__=='__main__':unittest.main()

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fsri30_cylinder_cutcell import cylinder_box_intersection, cylinder_side_area, aligned_cylinder_gas_grid, aligned_gas_faces


class CylinderCutCellTest(unittest.TestCase):
    def test_quadrants_and_axial_slices_close(self):
        r, h = .009, .065
        pieces = []
        for x0, x1 in ((-r, 0.), (0., r)):
            for y0, y1 in ((-r, 0.), (0., r)):
                for z0, z1 in ((-h / 2, 0.), (0., h / 2)):
                    pieces.append(cylinder_box_intersection((x0, y0, z0), (x1, y1, z1), (0., 0., 0.), r, h))
        self.assertAlmostEqual(sum(p['displaced_volume_m3'] for p in pieces), math.pi*r*r*h, places=13)
        self.assertAlmostEqual(sum(p['lateral_contact_area_m2'] for p in pieces), 2*math.pi*r*h, places=12)
        self.assertAlmostEqual(sum(p['top_cap_area_m2'] for p in pieces), math.pi*r*r, places=12)
        self.assertAlmostEqual(sum(p['bottom_cap_area_m2'] for p in pieces), math.pi*r*r, places=12)
        self.assertTrue(all(p['free_volume_m3'] >= 0 for p in pieces))

    def test_disjoint_and_full(self):
        r, h = .009, .065
        inside = cylinder_box_intersection((-.02, -.02, -.04), (.02, .02, .04), (0., 0., 0.), r, h)
        self.assertAlmostEqual(inside['displaced_volume_m3'], math.pi*r*r*h, places=13)
        self.assertAlmostEqual(inside['lateral_contact_area_m2'], 2*math.pi*r*h, places=12)
        outside = cylinder_box_intersection((.02, .02, -.04), (.03, .03, .04), (0., 0., 0.), r, h)
        self.assertEqual(outside['displaced_volume_m3'], 0.)
        self.assertEqual(outside['lateral_contact_area_m2'], 0.)

    def test_325mm_mesh_aligns_cylinder_and_conserves(self):
        r, h = .009, .065
        box = (.092, .172, .092)
        axes, cells = aligned_cylinder_gas_grid(box,
            [{'id': 'B', 'center': (0.,0.,0.), 'radius': r, 'height': h}])
        for plane in (-.0325, 0., .0325):
            self.assertTrue(any(abs(x-plane) < 1e-12 for x in axes[2]))
        self.assertTrue(all(b-a <= .0325+1e-12 for axis in axes for a,b in zip(axis,axis[1:])))
        self.assertAlmostEqual(sum(c['free_volume_m3'] for c in cells), math.prod(box)-math.pi*r*r*h, places=12)
        contacts = [p for c in cells for p in c['contacts']]
        self.assertAlmostEqual(sum(p['lateral_contact_area_m2'] for p in contacts), 2*math.pi*r*h, places=11)
        for face in ('-x','+x','-y','+y'):
            self.assertAlmostEqual(sum(cylinder_side_area(c['low_m'], c['high_m'], (0.,0.,0.), r, h, face)
                                       for c in cells), math.pi*r*h/2, places=11)
        faces = aligned_gas_faces(cells, [{'id': 'B', 'center': (0.,0.,0.), 'radius': r, 'height': h}])
        self.assertTrue(faces)
        self.assertTrue(all(p['area_m2'] > 0 for p in faces))
        self.assertEqual(len({(p['from'], p['to']) for p in faces}), len(faces))


if __name__ == '__main__':
    unittest.main()

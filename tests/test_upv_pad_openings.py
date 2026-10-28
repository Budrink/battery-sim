import collections
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import upv_r123_model as model
from gas_void_geometry import cell_box


class PadOpenings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.closed,_=model.build('no-cooling',duration=1.)
        cls.opened,_=model.build('no-cooling',duration=1.,gas_geometry={'pad_edge_clearance_m':.002})

    def test_only_pad_footprint_and_derived_mass_change(self):
        a={c['id']:c for c in self.closed['cells']};b={c['id']:c for c in self.opened['cells']}
        for name,c in a.items():
            if c['material_id']not in ('ISS','TIM'):
                self.assertEqual(c,b[name]);continue
            before=c['geometry']['dimensions_m'];after=b[name]['geometry']['dimensions_m']
            self.assertEqual(before[0],after[0]);self.assertEqual(c['pose'],b[name]['pose'])
            for k in (1,2):self.assertAlmostEqual(before[k]-after[k],.004)
            ratio=math.prod(after)/math.prod(before)
            self.assertAlmostEqual(b[name]['thermal']['heat_capacity_j_per_k'],c['thermal']['heat_capacity_j_per_k']*ratio)
            for key in set(c['thermal'])-{'heat_capacity_j_per_k'}:self.assertEqual(c['thermal'][key],b[name]['thermal'][key])
        self.assertEqual(self.closed['shells'],self.opened['shells'])

    def test_no_added_air_or_oxygen_and_chamber_stays_sealed(self):
        inventories=[]
        for scene in (self.closed,self.opened):
            nodes=[n for n in scene['gas_network']['compartments']if not n['id'].startswith('inside-')]
            self.assertAlmostEqual(sum(n['volume_m3']for n in nodes),.12,places=12)
            inventories.append(sum(n['initial_mass_kg']*n['initial_oxidizer_mass_fraction']for n in nodes))
            self.assertTrue(all(e['to']!='outside_domain'for e in scene['gas_network']['channels']))
        self.assertAlmostEqual(*inventories,places=12)

    def test_exposed_case_area_is_the_open_band_not_a_full_face(self):
        scene=self.opened;objects={c['id']:c for c in scene['cells']};ports={p['id']:p for p in scene['ports']}
        contacts=scene['metadata']['pad_openings']['exposed_contacts']
        for pad,cell,face in [('ISS1','C3','-x'),('TIM1','C3','+x'),('ISS2','C2','-x'),('TIM2','C2','+x')]:
            area=sum(c['area_m2']for c in contacts if c['solid']==cell+':case:'+face)
            covered=math.prod(objects[pad]['geometry']['dimensions_m'][1:])
            self.assertAlmostEqual(area,ports[cell+':'+face]['area_m2']-covered,places=12)
            self.assertGreater(area,0.)
        configured={c['port_id']:c for c in scene['gas_network']['thermal_contacts']}
        for c in contacts:self.assertEqual(configured[c['port']]['connectivity']['kind'],'directed_geometry')

    def test_perimeter_voids_reachable_and_never_inside_a_pad(self):
        scene=self.opened;meta=scene['metadata'];boxes=meta['gas_geometry']['void_boxes']
        graph=collections.defaultdict(set)
        for e in scene['gas_network']['channels']:
            if e['kind']=='passage':graph[e['from']].add(e['to']);graph[e['to']].add(e['from'])
        start=next(iter(boxes));seen={start};queue=[start]
        for a in queue:
            for b in graph[a]-seen:seen.add(b);queue.append(b)
        external={n['id']for n in scene['gas_network']['compartments']if not n['id'].startswith('inside-')}
        self.assertEqual(seen,external)
        pads=[cell_box(c)for c in scene['cells']if c['material_id']in ('ISS','TIM')]
        for name in meta['pad_openings']['gallery_nodes']:
            c=[(a+b)/2 for a,b in zip(boxes[name]['lo'],boxes[name]['hi'])]
            self.assertFalse(any(all(a[k]<c[k]<b[k]for k in range(3))for a,b in pads))

    def test_cooling_plate_does_not_gain_a_fictitious_inlet(self):
        scene,_=model.build('active-cooling',duration=1.,gas_geometry={'pad_edge_clearance_m':.002})
        self.assertFalse(any(e['id'].startswith('rupture-')and'--z-'in e['id']for e in scene['gas_network']['channels']))

    def test_invalid_clearance_rejected(self):
        for value in (-.001,.01,float('nan')):
            with self.assertRaises(ValueError):model.build('no-cooling',duration=1.,gas_geometry={'pad_edge_clearance_m':value})


if __name__=='__main__':unittest.main()

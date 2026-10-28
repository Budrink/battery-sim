import json
import math
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fsri30_six_zones import six_zone_batteries
from fsri30_aligned_gas import align_inner_gas


class AlignedGasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'data/development/fsri30-packed-fine-2406-20261027-r1/cascade-0000/scenario.json'
        if not path.exists():raise unittest.SkipTest('Local FSRI scenario absent')
        cls.old=six_zone_batteries(json.loads(path.read_text()))
        cls.new=align_inner_gas(cls.old)

    def test_gas_inventory_and_connections(self):
        a,b=self.old['gas_network'],self.new['gas_network']
        for k in ('volume_m3','initial_mass_kg'):
            self.assertTrue(math.isclose(sum(n[k] for n in a['compartments']),sum(n[k] for n in b['compartments']),rel_tol=1e-11))
        ids={n['id'] for n in b['compartments']}
        self.assertEqual(len(ids),len(b['compartments']))
        self.assertTrue(all(e['from'] in ids and e['to'] in ids for e in b['channels']))
        self.assertTrue(all(f['compartment'] in ids for f in b['thermal_contacts']))
        self.assertEqual(sum(e['id'].startswith('rupture-') for e in b['channels']),90)

    def test_axial_grid_and_lid_opening(self):
        meta=self.new['metadata']['aligned_inner_gas']
        for x in (-.0325,0.,.0325): self.assertTrue(any(abs(y-x)<1e-12 for y in meta['axial_boundaries_m']))
        vents=[e for e in self.new['gas_network']['channels'] if e['id'].startswith('aligned-vent-')]
        self.assertAlmostEqual(sum(e['area_m2'] for e in vents),math.pi*(.0254/2)**2,places=12)
        self.assertFalse(any(e['id'].startswith('vent-') for e in self.new['gas_network']['channels']))

    def test_lateral_film_areas(self):
        # Undo port-patch weights to compare effective directed conductance.
        old=self.old['gas_network']['thermal_contacts'];new=self.new['gas_network']['thermal_contacts']
        cells={c['id']:c for c in self.old['cells'] if c.get('role')=='battery'}
        for name in cells:
            c=cells[name];dims=c['geometry']['dimensions_m']
            for face in ('-x','+x','-y','+y'):
                port=name+':case:'+face+':outer';axis='xyz'.index(face[1])
                patch_count=2 if axis==0 else 6
                former=sum(f['connectivity']['active_conductance_w_per_k'] for f in old if f['port_id']==port)
                actual=sum(f['connectivity']['active_conductance_w_per_k']/patch_count for f in new if f['port_id']==port)
                width=dims[1]+.00036 if axis==0 else dims[0]
                port_area=width*(dims[2]+.00036)
                expected=former*(math.pi*.009*.065/2)/port_area
                self.assertTrue(math.isclose(actual,expected,rel_tol=1e-9), (name,face,actual,expected))
            for face in ('-z','+z'):
                port=name+':case:'+face+':outer'
                former=sum(f['connectivity']['active_conductance_w_per_k'] for f in old if f['port_id']==port)
                actual=sum(f['connectivity']['active_conductance_w_per_k']/3 for f in new if f['port_id']==port)
                expected=former*math.pi*.009**2/(dims[0]*dims[1])
                self.assertTrue(math.isclose(actual,expected,rel_tol=1e-9), (name,face,actual,expected))


if __name__=='__main__': unittest.main()

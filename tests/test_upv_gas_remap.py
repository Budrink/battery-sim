import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upv_gas_remap import unvented_remap
import upv_r123_model as model
from gas_void_geometry import VoidGrid
from upv_r123_data import chamber_pressure_pa


class GasRemap(unittest.TestCase):
    def test_observer_does_not_read_corner_pressure(self):
        run=dict(gas={'chamber':dict(pressure_pa=[999.])},gas_observers={'chamber':dict(pressure_pa=[123.])})
        self.assertEqual(chamber_pressure_pa(run),[123.])
        del run['gas_observers'];self.assertEqual(chamber_pressure_pa(run),[999.])

    def test_grid_preserves_exact_envelope_and_rejects_outside_cuts(self):
        bounds=([-.12345678901234]*3,[.12345678901234]*3)
        grid=VoidGrid(bounds,[],[[0.]]*3)
        self.assertAlmostEqual(sum(c['volume']for c in grid.cells.values()),(.24691357802468)**3,places=15)
        for cuts in ([[.2],[],[]],[[-.2],[],[]],[[float('nan')],[],[]]):
            with self.assertRaises(ValueError):VoidGrid(bounds,[],cuts)

    def test_identity_and_split_inventory(self):
        old=dict(gas_network=dict(compartments=[dict(id='chamber',volume_m3=2.),dict(id='inside',volume_m3=1.)]),
            metadata=dict(gas_geometry=dict(void_boxes={})))
        new=copy.deepcopy(old);new['gas_network']['compartments']=[dict(id='a',volume_m3=.5),dict(id='b',volume_m3=1.5),old['gas_network']['compartments'][1]]
        new['metadata']['gas_geometry']['void_boxes']={
            'a':dict(lo=[0,0,0],hi=[.5,1,1]),'b':dict(lo=[.5,0,0],hi=[2,1,1])}
        self.assertEqual(unvented_remap(old,new),[dict(to=0,fraction=.25,**{'from':0}),
            dict(to=1,fraction=.75,**{'from':0}),dict(to=2,fraction=1.,**{'from':1})])
        self.assertEqual(unvented_remap(new,new),[dict(to=i,fraction=1.,**{'from':i})for i in range(3)])
        broken=copy.deepcopy(new);broken['gas_network']['compartments'][2]['volume_m3']=2.
        with self.assertRaises(ValueError):unvented_remap(old,broken)

    def test_real_geometry_inventory(self):
        path=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/restart-gas-baseline-20261019-r1/scenario.json'
        if not path.exists():self.skipTest('Optional local pre-rupture fixture')
        old=json.loads(path.read_text());new,_=model.build('no-cooling',duration=950.)
        mapping=unvented_remap(old,new)
        self.assertTrue(mapping)
        # A spatial rebase NEVER sends pore inventory into the chamber.
        a=old['gas_network']['compartments'];b=new['gas_network']['compartments']
        for e in mapping:
            if a[e['from']]['id'].startswith('inside-'):
                self.assertEqual(a[e['from']]['id'],b[e['to']]['id']);self.assertEqual(e['fraction'],1.)


if __name__=='__main__':unittest.main()

"""Scenario/ledger contract; CUDA analytic tests live in process_network.cpp."""
import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_upv_gas_energy as gas


class GasEnergyProtocol(unittest.TestCase):
    def test_no_duplicate_measured_environment(self):
        for name in ('no-cooling','active-cooling'):
            s,_,_,_=gas.prepare(name)
            self.assertTrue(all(b['id']=='wall-room' or b['id'].startswith('cooling-') for b in s['boundaries']))
            net=s['gas_network'];self.assertEqual(net['compartments'][1]['volume_m3'],.12)
            self.assertEqual(net['compartments'][1]['initial_oxidizer_mass_fraction'],.232)
            self.assertTrue(any(p['port_id']=='wall:inner' for p in net['thermal_contacts']))
            self.assertTrue(any(p['port_id'].startswith('C2:') for p in net['thermal_contacts']))
            cells={c['id']:c for c in s['cells']};ports={p['id']:p for p in s['ports']}
            self.assertGreater(cells['chamber-wall']['thermal']['heat_capacity_j_per_k'],0)
            self.assertTrue(all(ports[p['port_id']]['cell_id'] not in ('C1','C2','C3') for p in net['thermal_contacts']))

    def test_single_chemical_budget(self):
        s,_,_,_=gas.prepare('no-cooling');before=copy.deepcopy(s);p=gas.INITIAL
        baked=gas.bake(s,p);self.assertEqual(s,before)
        law=next(c['conversion'] for c in baked['cells'] if c['id']=='C2')
        total=law['mass_kg']*law['net_heat_j_per_kg']
        self.assertAlmostEqual(total,p['solid_energy_j']+p['gas_energy_j'])
        self.assertAlmostEqual(total*law['gas_chemical_fraction'],p['gas_energy_j'])
        self.assertEqual(law['gas_yield'],p['gas_yield'])
        for c,b in zip(s['cells'],baked['cells']):self.assertEqual(c['thermal'],b['thermal'])


if __name__=='__main__':unittest.main()

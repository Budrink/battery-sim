from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fsri30_abs_barrier import damage_abs_barrier
from fsri30_data import ROOT

BASE=ROOT/'data/development/fsri30-abs-seams-20261025-r2/scenario.json'
PARAMETERS=dict(pressure_pa=1000.,onset_k=450.,rate_per_s=.01,
    temperature_scale_k=40.,area_rate_per_s=1.,fracture_j_m2=100.)

class AbsBarrierTests(unittest.TestCase):
    def test_local_connections_no_inventory_or_heat_refit(self):
        old=json.loads(BASE.read_text());before=deepcopy(old);new=damage_abs_barrier(old,**PARAMETERS)
        self.assertEqual(old,before)
        for key in old:
            if key not in ('cells','ports','gas_network','metadata'):self.assertEqual(old[key],new[key],key)
        for a,b in zip(old['cells'],new['cells']):
            expected=deepcopy(b)
            if a['id'].startswith('ABS:'):expected.pop('degradation')
            self.assertEqual(a,expected)
        network=deepcopy(new['gas_network']);n=len(old['gas_network']['channels'])
        additions=network['channels'][n:];network['channels']=network['channels'][:n]
        self.assertEqual(network,old['gas_network'])
        meta=new['metadata']['abs_barrier_damage']
        self.assertGreater(len(additions),0);self.assertGreater(len(meta['omitted_ambiguous']),0)
        self.assertLess(meta['connected_area_m2'],meta['total_abs_face_area_m2'])
        contacts=old['gas_network']['thermal_contacts'];ports={p['id']:p for p in new['ports']}
        seen=set()
        for edge in additions:
            wall=ports[edge['port_id']]['cell_id'];local=edge['shell_thermal_volume']
            self.assertNotIn((wall,local),seen);seen.add((wall,local))
            self.assertNotIn('area_m2',edge)
            for endpoint in ('from','to'):
                adjacent=[c for c in contacts if c.get('thermal_volume')==local and
                    ports.get(c['port_id'],{}).get('cell_id')==wall and c['compartment']==edge[endpoint]]
                self.assertEqual(len(adjacent),1)
            self.assertEqual(edge['damage_opening']['max_damage_step'],.005)
            self.assertGreater(edge['damage_opening']['fracture_energy_j_per_m2'],0.)
        with self.assertRaisesRegex(ValueError,'already'):damage_abs_barrier(new,**PARAMETERS)

    def test_invalid_material_parameters(self):
        old=json.loads(BASE.read_text())
        for key in PARAMETERS:
            for value in (0.,-1.,float('nan'),float('inf')):
                p=dict(PARAMETERS);p[key]=value
                with self.assertRaises(ValueError):damage_abs_barrier(old,**p)

if __name__=='__main__':unittest.main()

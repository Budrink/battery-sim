import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_upv_module_scaling import replicate

SOURCE=Path(__file__).resolve().parents[1]/'data/calibration/upv-separated-chemistry-v6/five-channel-best-and-validation-20261022-r1/trial-000/scenario.json'


class ModuleScaling(unittest.TestCase):
    def test_connected_replicas_keep_routes_unique_and_source_frozen(self):
        if not SOURCE.exists():self.skipTest('Optional local UPV scene')
        scene=json.loads(SOURCE.read_text());before=json.dumps(scene,sort_keys=True)
        target=replicate(scene,4,10.)
        self.assertEqual(json.dumps(scene,sort_keys=True),before)
        self.assertEqual(sum(c['role']=='battery'for c in target['cells']),12)
        nodes={g['id']for g in target['gas_network']['compartments']}
        self.assertEqual(len(nodes),4*len(scene['gas_network']['compartments']))
        ids=[e['id']for e in target['gas_network']['channels']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(sum(e.startswith('connecting-tube-')for e in ids),3)
        self.assertTrue(all(e['kind']=='passage'for e in target['gas_network']['channels']if e['id'].startswith('connecting-tube-')))
        for edge in target['gas_network']['channels']:
            self.assertIn(edge['from'],nodes)
            self.assertTrue(edge['to']=='outside_domain'or edge['to']in nodes)
        self.assertEqual(target['measurement_probes'],[])
        self.assertTrue(target['metadata']['not_experimental_validation'])


if __name__=='__main__':unittest.main()

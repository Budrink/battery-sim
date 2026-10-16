import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import calibrate_upv_casing as trial


class PhysicalCasingProtocol(unittest.TestCase):
    def test_capacity_and_exterior_binding(self):
        original=trial.read_scene('no-cooling');before=copy.deepcopy(original)
        scene=trial.casing(original,trial.start_heat())
        self.assertEqual(original,before)
        cells={c['id']:c for c in scene['cells']}
        for identifier,inventory in scene['metadata']['casing_inventory'].items():
            self.assertAlmostEqual(cells[identifier]['thermal']['heat_capacity_j_per_k']+inventory['case_capacity_j_k'],412.87)
            self.assertEqual(cells[identifier]['thermal']['conductivity_w_per_m_k'],[.9,28.3,28.3])
        self.assertTrue(all(':case:' in p['cell_id'] for p in scene['measurement_probes']))
        self.assertFalse(any(p['cell_id'] in ('C1','C3') for p in scene['ports']))
        self.assertEqual([c['pose'] for c in scene['cells']],[c['pose'] for c in original['cells']])
        self.assertEqual(scene['primary_failures'],[])

    def test_native_capacity_rotation_and_surface_observer(self):
        probe=Path('/tmp/battery-sim-cuda-v1/battery_sim_casing_probe')
        if not probe.is_file():self.skipTest('Build casing probe first')
        scene=trial.casing(trial.read_scene('no-cooling'),trial.start_heat())
        scene['simulation'].update(duration_s=1.,output_interval_s=1.)
        with tempfile.TemporaryDirectory() as d:
            runner=trial.Runner(probe,Path(d))
            try:r=runner.run(scene)
            finally:runner.close()
        self.assertEqual(set(r['temperature_k']),{'101','102','107','108'})
        for c in r['applied_objects']:
            if c['shell_contents_id']:
                axis='xyz'.index(c['id'][-1]);k=c['conductivity_w_m_k']
                self.assertEqual(k[axis],.3)
                self.assertEqual(k[(axis+1)%3],50)


if __name__=='__main__':unittest.main()

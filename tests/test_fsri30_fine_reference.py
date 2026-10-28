from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from complete_fsri30_fine_reference import reference_request


class FineReferenceTests(unittest.TestCase):
    def scene(self):
        return dict(simulation=dict(minimum_dt_s=.01),gas_network=dict(pressure_pattern_predictor=False),
            failure_sample_times_s=[0.,2200.,5945.])

    def test_capture_does_not_stop_or_refine(self):
        r=reference_request(self.scene(),Path('/tmp/state.bin'),'pinned-core')
        self.assertFalse(r['stop_after_committed_restart'])
        self.assertNotIn('refinement',r)
        self.assertEqual(r['committed_restart_at_s'],2200.)

    def test_capture_cannot_add_step_boundary(self):
        with self.assertRaisesRegex(ValueError,'existing observation'):
            reference_request(self.scene(),Path('/tmp/state.bin'),'core',2200.1)

    def test_floor_unchanged(self):
        s=self.scene();s['simulation']['minimum_dt_s']=.005
        with self.assertRaisesRegex(ValueError,'floor changed'):
            reference_request(s,Path('/tmp/state.bin'),'core')

    def test_rejected_predictor_stays_off(self):
        s=self.scene();s['gas_network']['pressure_pattern_predictor']=True
        with self.assertRaisesRegex(ValueError,'rejected pressure predictor'):
            reference_request(s,Path('/tmp/state.bin'),'core')


if __name__=='__main__':unittest.main()

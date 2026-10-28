import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from report_upv_frozen_comparison import replay_timing,boundary_provenance


class ComparisonTiming(unittest.TestCase):
    def test_identified_boundary_is_never_reported_as_blind_frozen_validation(self):
        plain=boundary_provenance(dict(metadata=dict(frozen_validation=dict(parameters_refitted=False))))
        self.assertFalse(plain['validation_refitted'])
        scene=dict(metadata=dict(cooling_boundary_identification={},frozen_validation=dict(parameters_refitted=True)))
        report=boundary_provenance(scene)
        self.assertTrue(report['validation_refitted'])
        self.assertTrue(report['cooling_boundary_conditioned'])
        self.assertIn('NOT fully blind',report['validation_scope'])

    def timing(self,log,run):
        with tempfile.TemporaryDirectory()as name:
            folder=Path(name)
            (folder/'cuda.log').write_text(log)
            return replay_timing(folder,run,1000.)

    def test_rejected_log_does_not_extend_past_preserved_snapshot(self):
        run=dict(started_at_s=0.,advanced_to_s=90.,wall_s=30.,
                 time_s=[0.,85.],events=[],partial=True)
        report=self.timing('Replay: t=30 wall_s=10\n'
                           'Replay: t=60 wall_s=20\n'
                           'Replay: t=95 wall_s=30.3\n',run)
        self.assertEqual(report['whole_run_realtime_ratio'],3.)
        self.assertEqual(report['timing_scope'],'preserved committed prefix')
        self.assertAlmostEqual(report['worst_20s_window']['realtime_ratio'],3.)

    def test_completed_timing_preserves_native_progress(self):
        run=dict(started_at_s=0.,advanced_to_s=120.,wall_s=40.,
                 time_s=[0.,120.],events=[])
        report=self.timing('Replay: t=30 wall_s=10\n'
                           'Replay: t=60 wall_s=20\n'
                           'Replay: t=90 wall_s=30\n',run)
        self.assertEqual(report['whole_run_realtime_ratio'],3.)
        self.assertEqual(report['timing_scope'],'completed replay')
        self.assertAlmostEqual(report['worst_20s_window']['realtime_ratio'],3.)

    def test_phase_estimates_do_not_overwrite_whole_run_duration(self):
        run=dict(started_at_s=0.,advanced_to_s=1200.,wall_s=400.,
                 time_s=[0.,1200.],events=[])
        report=self.timing('Replay: t=300 wall_s=100\n'
                           'Replay: t=600 wall_s=200\n'
                           'Replay: t=900 wall_s=300\n',run)
        self.assertEqual(report['whole_run_realtime_ratio'],3.)
        self.assertEqual(set(report['phase_estimates']),
                         {'pre_burst','observed_event_and_relaxation','cooling_tail'})


if __name__=='__main__':unittest.main()

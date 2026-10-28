import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from realtime_guard import RealtimeFloor, completed_performance


class RealtimeGuardTests(unittest.TestCase):
    def test_exact_quarter_realtime_is_accepted(self):
        g = RealtimeFloor()
        for wall in range(5, 31, 5):
            result = g.observe(wall, wall*.25)
        self.assertTrue(result['accepted'])

    def test_previous_rejection_now_passes_rolling_but_not_average_floor(self):
        g=RealtimeFloor()
        g.observe(0.,0.)
        self.assertTrue(g.observe(20.,20.*.288966)['accepted'])
        self.assertFalse(completed_performance(20.*.288966,20.)['accepted'])
        self.assertTrue(completed_performance(925.,1850.)['accepted'])
        self.assertFalse(completed_performance(925.,1850.01)['accepted'])

    def test_bad_completed_timing_is_rejected(self):
        for simulated,wall in [(1.,0.),(1.,float('nan')),(-1.,2.),(float('inf'),2.)]:
            with self.subTest(simulated=simulated,wall=wall),self.assertRaises(ValueError):
                completed_performance(simulated,wall)

    def test_fast_warmup_does_not_hide_slow_event(self):
        g = RealtimeFloor()
        for wall in range(5, 41, 5):
            g.observe(wall, wall*10)
        for wall in range(45, 66, 5):
            result = g.observe(wall, 400+(wall-40)*.1)
        self.assertFalse(result['accepted'])
        self.assertAlmostEqual(result['realtime_ratio'], .1)

    def test_startup_and_sparse_reports(self):
        g = RealtimeFloor()
        self.assertIsNone(g.observe(100., 0.))
        self.assertIsNone(g.observe(105., 5.))
        result = g.observe(125., 25.)
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['realtime_ratio'], 1.)

    def test_parser_uses_native_wall_clock(self):
        g = RealtimeFloor()
        self.assertIsNone(g.inspect('Replay: t=2 / 925 s, wall_s=100.1, steps=2'))
        result = g.inspect('Replay: t=4 / 925 s, wall_s=120.1, steps=3')
        self.assertFalse(result['accepted'])
        self.assertIsNone(g.inspect('Replay: t=4 / 925 s, wall_s=120.1, steps=3'))

    def test_stuck_kernel_cannot_escape_floor_by_not_reporting(self):
        g=RealtimeFloor();line='Replay: t=4 / 925 s, wall_s=10, steps=3'
        self.assertIsNone(g.inspect(line,now=100.))
        self.assertIsNone(g.inspect(line,now=115.))
        result=g.inspect(line,now=121.)
        self.assertFalse(result['accepted']);self.assertTrue(result['stalled'])


if __name__ == '__main__':
    unittest.main()

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refine_fsri30_candidate import paired_events,worth_finer_check


def row(level,time):
    return dict(level_mm=level,status='pruned',physical_error=None,native_sha256='same-core',
                parameters=dict(rate=.049),run_sha256=f'history-{level}',
                assessment=dict(accepted=False,primary_offset_s=time-1763.,model=dict(
                    pulse_groups=[dict(time_s=time,channels=['B10'])])))


class RefinementTests(unittest.TestCase):
    def test_primary_probe_does_not_require_claiming_whole_cascade_accepted(self):
        r=row(80,1803.)
        self.assertTrue(worth_finer_check(r))
        self.assertFalse(r['assessment']['accepted'])
        r['physical_error']='ledger';self.assertFalse(worth_finer_check(r))

    def test_pair_measures_actual_shift_without_parameter_change(self):
        a,b=row(80,1803.),row(40,1730.)
        before=deepcopy((a,b));pair=paired_events(a,b)
        self.assertEqual(pair['coarse_minus_fine_s'],73.)
        self.assertEqual((a,b),before)
        b['parameters']['rate']=.03
        with self.assertRaises(ValueError):paired_events(a,b)

    def test_absent_or_different_primary_is_not_a_mesh_pair(self):
        a,b=row(80,1803.),row(40,1730.)
        b['assessment']['model']['pulse_groups'][0]['channels']=['B11']
        self.assertIsNone(paired_events(a,b))
        b['assessment']['model']['pulse_groups']=[]
        self.assertFalse(worth_finer_check(b));self.assertIsNone(paired_events(a,b))


if __name__=='__main__':unittest.main()

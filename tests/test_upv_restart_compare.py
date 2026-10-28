from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_upv_restart import compare


class RestartInterval(unittest.TestCase):
    def reference(self):
        return dict(time_s=[0.,835.,900.,925.,990.],started_at_s=0.,
                    temperature_k={'101':[300.,350.,700.,500.,400.]})

    def resumed(self):
        return dict(time_s=[835.,900.,925.],started_at_s=835.,
                    temperature_k={'101':[350.,700.,500.]})

    def test_explicit_known_interval_can_verify_a_pruned_reference(self):
        result=compare(self.reference(),self.resumed(),until_s=925.)
        self.assertTrue(result['bitwise_equal']);self.assertEqual(result['compared_until_s'],925.)

    def test_default_still_requires_the_entire_suffix(self):
        with self.assertRaises(ValueError):compare(self.reference(),self.resumed())

    def test_unknown_time_and_changed_samples_are_not_hidden(self):
        with self.assertRaises(ValueError):compare(self.reference(),self.resumed(),until_s=950.)
        r=self.resumed();r['temperature_k']['101'][1]+=1.
        result=compare(self.reference(),r,until_s=925.)
        self.assertFalse(result['bitwise_equal']);self.assertEqual(result['max_absolute_difference'],1.)


if __name__=='__main__':unittest.main()

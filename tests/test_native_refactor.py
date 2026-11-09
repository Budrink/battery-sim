"""Regression/performance guard itself must not hide physics changes or noisy timing."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "check_native_refactor", Path(__file__).resolve().parents[1] / "scripts/check_native_refactor.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class NativeRefactorTests(unittest.TestCase):
    def timing(self, a, b):
        return check.performance_summary([{"wall_s": x} for x in a],
                                         [{"wall_s": x} for x in b], .05)

    def test_repeatable_speed(self):
        self.assertEqual(self.timing([10, 10, 10], [10.1, 9.9, 10])["status"], "passed")

    def test_slowdown_fails(self):
        self.assertEqual(self.timing([10, 10, 10], [12, 12, 12])["status"], "regression")

    def test_noise_is_not_a_pass(self):
        self.assertEqual(self.timing([7, 10, 13], [7, 10, 13])["status"], "inconclusive_noise")

    def test_outlier_is_reported(self):
        self.assertEqual(self.timing([10] * 5, [10, 10, 10, 10, 20])["status"],
                         "inconclusive_noise")

    def test_invalid_timing(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.timing([10], [value])

    def test_only_timing_is_ignored(self):
        a = {"wall_s": 1, "temperature": [300], "solver_profile": {"prepare_s": 1, "steps": 5}}
        b = {"wall_s": 2, "temperature": [300], "solver_profile": {"prepare_s": 2, "steps": 5}}
        self.assertEqual(check.physical_result(a), check.physical_result(b))
        b["temperature"][0] = 301
        self.assertTrue(list(check.differences(check.physical_result(a), check.physical_result(b))))
        b["temperature"][0] = 300
        b["solver_profile"]["steps"] = 6
        self.assertTrue(list(check.differences(check.physical_result(a), check.physical_result(b))))


if __name__ == "__main__":
    unittest.main()

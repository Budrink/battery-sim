"""Short native contract -> scenario -> existing CUDA worker integration."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

COMPILER = WORKER = EXAMPLE = None


class ExperimentCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if COMPILER is None:
            raise unittest.SkipTest("Run via CTest with native executables")

    def test_compiled_snapshot_runs_through_unchanged_worker_contract(self):
        with tempfile.TemporaryDirectory(prefix="battery-experiment-cli-") as temporary:
            root = Path(temporary)
            source = json.loads(EXAMPLE.read_text())
            source["numerics"]["simulation"]["duration_s"] = .03
            experiment = root / "experiment.json"
            experiment.write_text(json.dumps(source))
            compiled = subprocess.run([str(COMPILER), str(experiment)], capture_output=True,
                                      text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            scenario = json.loads(compiled.stdout)
            self.assertEqual(scenario["metadata"]["experiment_contract"], source)
            (root / "scenario.json").write_text(compiled.stdout)
            request = root / "job.json"
            request.write_text(json.dumps(dict(schema_version=1, job_id="contract", attempt_id="one",
                                               scenario_path="scenario.json")))
            result = subprocess.run([str(WORKER), str(request), "--output", str(root / "result")],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            terminal = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(terminal["status"], "completed")
            self.assertEqual(json.loads((root / "result/scenario.json").read_text()), scenario)
            source["assembly"]["batteries"][0]["model_revision"] = "absent"
            experiment.write_text(json.dumps(source))
            invalid = subprocess.run([str(COMPILER), str(experiment)], capture_output=True,
                                     text=True, timeout=30)
            self.assertEqual(invalid.returncode, 2)
            self.assertEqual(invalid.stdout, "")

    def test_explicit_upgrade_preserves_executable_parameters(self):
        source = json.loads(EXAMPLE.read_text())
        migrated = subprocess.run([str(COMPILER), str(EXAMPLE), "--upgrade-v2"],
                                  capture_output=True, text=True, timeout=30)
        self.assertEqual(migrated.returncode, 0, migrated.stderr)
        upgraded = json.loads(migrated.stdout)
        self.assertEqual(upgraded["version"], 2)
        if source["version"] == 2:
            self.assertEqual(upgraded, source)
        with tempfile.TemporaryDirectory(prefix="battery-contract-upgrade-") as temporary:
            path = Path(temporary) / "experiment.json"
            path.write_text(migrated.stdout)
            before = subprocess.run([str(COMPILER), str(EXAMPLE)], capture_output=True, text=True, timeout=30)
            after = subprocess.run([str(COMPILER), str(path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(before.returncode, 0, before.stderr)
            self.assertEqual(after.returncode, 0, after.stderr)
            old, new = json.loads(before.stdout), json.loads(after.stdout)
            for document in (old, new):
                document.pop("source")
                document.pop("metadata")
            self.assertEqual(old, new)


if __name__ == "__main__":
    COMPILER = Path(sys.argv.pop(1)).resolve()
    WORKER = Path(sys.argv.pop(1)).resolve()
    EXAMPLE = Path(sys.argv.pop(1)).resolve()
    unittest.main()

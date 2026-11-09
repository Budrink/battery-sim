"""Process-level contract checks. Real CUDA, no calibration or long experiments."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

WORKER = None
SCENARIO = None


class WorkerCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if WORKER is None:
            raise unittest.SkipTest("Run via CTest with an explicit CUDA worker binary")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="battery-worker-cli-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        scenario = json.loads(SCENARIO.read_text())
        scenario["simulation"].update(duration_s=0.03, step_interval_s=0.01)
        (self.root / "scenario.json").write_text(json.dumps(scenario))
        self.job = dict(schema_version=1, job_id="cli", attempt_id="one", scenario_path="scenario.json")

    def command(self, name="one"):
        request = self.root / "job.json"
        request.write_text(json.dumps(self.job))
        return [str(WORKER), str(request), "--output", str(self.root / name)]

    def run_job(self, name="one", **kwargs):
        result = subprocess.run(self.command(name), capture_output=True, text=True, timeout=30, **kwargs)
        messages = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertTrue(messages)
        self.assertEqual([m["sequence"] for m in messages], list(range(len(messages))))
        self.assertEqual(messages[-1]["type"], "terminal")
        self.assertEqual(messages[-1]["exit_code"], result.returncode)
        return result, messages

    def test_complete_and_duplicate(self):
        result, messages = self.run_job()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(messages[-1]["status"], "completed")
        original = (self.root / "one/result.json").read_bytes()
        result, messages = self.run_job()
        self.assertEqual(result.returncode, 3)
        self.assertEqual((self.root / "one/result.json").read_bytes(), original)
        self.assertEqual(messages[-1]["reason"], "output_exists")

    def test_existing_file_not_overwritten(self):
        (self.root / "one").write_text("keep")
        result, _ = self.run_job()
        self.assertEqual(result.returncode, 3)
        self.assertEqual((self.root / "one").read_text(), "keep")

    def test_invalid_request(self):
        self.job["unknown"] = 123
        result, messages = self.run_job()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(messages[-1]["status"], "invalid_request")

    def test_gpu_initialization_failure_is_not_validation(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        result, messages = self.run_job(env=env)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(messages[-1]["status"], "failed")
        self.assertEqual(messages[-1]["stage"], "initialization")
        self.assertFalse((self.root / "one/checkpoint.json").exists())

    def test_sigterm_saves_resumable_state(self):
        scenario = json.loads((self.root / "scenario.json").read_text())
        scenario["simulation"]["duration_s"] = 2
        (self.root / "scenario.json").write_text(json.dumps(scenario))
        process = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            first = json.loads(process.stdout.readline())
            self.assertEqual(first["type"], "started")
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 130, stdout + stderr)
            self.assertEqual(json.loads(stdout.splitlines()[-1])["status"], "cancelled")
            self.assertTrue((self.root / "one/checkpoint.json").is_file())
            self.job.update(attempt_id="two", restart_from="one/checkpoint.json", limits={"max_steps": 1})
            result, messages = self.run_job("two")
            self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
            self.assertEqual(messages[-1]["status"], "limit_reached")
            prior = json.loads((self.root / "one/result.json").read_text())
            resumed = json.loads((self.root / "two/result.json").read_text())
            self.assertEqual(resumed["started_at_s"], prior["time_s"])
            self.assertGreater(resumed["time_s"], prior["time_s"])
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    WORKER = Path(sys.argv.pop(1)).resolve()
    SCENARIO = Path(sys.argv.pop(1)).resolve()
    unittest.main()

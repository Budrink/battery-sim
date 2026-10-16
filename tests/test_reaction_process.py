"""Parser/schema/diagnostic contract; dynamics are checked independently in CUDA."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from battery_sim.io import load_scenario

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(os.environ.get("BATTERY_SIM_REACTION_PROBE", "/tmp/battery-sim-cuda-v1/battery_sim_reaction_probe"))


class ReactionContract(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "examples/kinetic-process-demo.json").read_text())
        self.scene["simulation"].update(duration_s=2, output_interval_s=1)

    def run_native(self, scene):
        if not PROBE.is_file():
            self.skipTest("Build native reaction probe first")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(scene))
            return subprocess.run([str(PROBE), str(path)], text=True, capture_output=True, timeout=30)

    def test_optional_json_schema_validation(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("Optional jsonschema package is not installed; native validation is tested separately")
        schema = json.loads((ROOT / "schemas/scenario-v0.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(self.scene, schema)

    def test_native_contract(self):
        result = self.run_native(self.scene)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertLess(data["max_thermal_residual_j"], 1e-6)
        self.assertLess(data["max_chemical_residual_j"], 1e-6)
        initial = data["samples"][0]["objects"][0]
        self.assertAlmostEqual(initial["remaining_j"], 120000)
        self.assertEqual(initial["released_j"], 0)
        self.assertLess(initial["damage"], .01)  # only 1/27 activated, NOT all destroyed

    def test_historical_solver_fails_closed(self):
        with self.assertRaisesRegex(Exception, "native CUDA"):
            load_scenario(ROOT / "examples/kinetic-process-demo.json")

    def test_invalid_process_contracts(self):
        variants = []
        scene = copy.deepcopy(self.scene)
        scene["cells"][0]["runaway"]["duration_s"] = 1
        variants.append(scene)
        scene = copy.deepcopy(self.scene)
        scene["cells"][0]["runaway"]["energy_outlet"]["sink"] = "neighbor"
        variants.append(scene)
        scene = copy.deepcopy(self.scene)
        scene["cells"][0]["runaway"]["energy_outlet"]["port_id"] = "missing"
        variants.append(scene)
        scene = copy.deepcopy(self.scene)
        scene["ports"][0]["centroid_m"] = [0, 0, 0]
        variants.append(scene)
        scene = copy.deepcopy(self.scene)
        scene["cells"][0]["thermal"]["internal_heat_transfer"] = "isothermal"
        variants.append(scene)
        scene = copy.deepcopy(self.scene)
        scene["cells"][0]["runaway"]["kinetics"]["max_fraction_per_step"] = 1
        variants.append(scene)
        scene = copy.deepcopy(self.scene)
        scene["cells"][0]["runaway"]["energy_outlet"]["opening_end_conversion"] = 0
        variants.append(scene)
        for scene in variants:
            with self.subTest(scene=scene):
                result = self.run_native(scene)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

"""Native JSON contract and a small conservative barrier/receiver example."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(os.environ.get("BATTERY_SIM_REACTION_PROBE", "/tmp/battery-sim-cuda-v1/battery_sim_reaction_probe"))


class ApertureConnections(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "examples/aperture-surface-demo.json").read_text())
        self.scene["simulation"].update(duration_s=.02, output_interval_s=.01)

    def run_scene(self, scene):
        if not PROBE.is_file():
            self.skipTest("Build native reaction probe first")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(scene))
            return subprocess.run([str(PROBE), str(path)], capture_output=True, text=True, timeout=30)

    def test_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("Optional jsonschema not installed")
        jsonschema.validate(self.scene, json.loads((ROOT / "schemas/scenario-v0.schema.json").read_text()))

    def test_native_energy_and_visibility(self):
        result = self.run_scene(self.scene)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertLess(data["max_thermal_residual_j"], 1e-8)
        self.assertLess(data["max_mass_residual_kg"], 1e-13)
        last = data["samples"][-1]
        channel = last["channels"][0]
        self.assertGreater(channel["surface_energy_j"], 0)
        self.assertGreater(channel["fracture_work_j"], 0)
        self.assertAlmostEqual(channel["transferred_energy_j"], channel["surface_energy_j"] + channel["delivered_gas_energy_j"])
        objects = {obj["id"]: obj for obj in last["objects"]}
        self.assertGreater(objects["near"]["temperature_k"], 300)
        self.assertGreater(objects["screen"]["temperature_k"], 300)
        self.assertAlmostEqual(objects["far"]["temperature_k"], 300)

    def test_no_silent_invalid_settings(self):
        for link in [{}, {"h_w_per_m2_k": -1}, {"h_w_per_m2_k": 1, "extra": 2}]:
            with self.subTest(link=link):
                scene = copy.deepcopy(self.scene)
                scene["gas_network"]["channels"][0]["surface_link"] = link
                self.assertNotEqual(self.run_scene(scene).returncode, 0)
        scene = copy.deepcopy(self.scene)
        scene["gas_network"]["channels"][0]["damage_opening"]["fracture_energy_j_per_m2"] = -1
        self.assertNotEqual(self.run_scene(scene).returncode, 0)


if __name__ == "__main__":
    unittest.main()

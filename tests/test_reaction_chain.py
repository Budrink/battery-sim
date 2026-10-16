"""Schema/native entry points for R123; CUDA numerical checks live in CTest."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(os.environ.get("BATTERY_SIM_REACTION_PROBE", "/tmp/battery-sim-cuda-v1/battery_sim_reaction_probe"))


class ReactionChainContract(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "examples/reaction-chain-demo.json").read_text())

    def validator(self, definition=None):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("Optional jsonschema not installed")
        schema = json.loads((ROOT / "schemas/scenario-v0.schema.json").read_text())
        if definition:
            schema = {"$defs": schema["$defs"], "$ref": "#/$defs/" + definition}
        jsonschema.Draft202012Validator.check_schema(schema)
        return jsonschema.Draft202012Validator(schema)

    def test_new_schema_requires_no_lengths(self):
        self.validator().validate(self.scene)
        law = self.scene["cells"][0]["conversion"]
        self.assertFalse(any(k in law for k in ("l0_m", "lcrit_m", "energy_scale_j_per_kg")))
        for key in ("degradation_rate_per_s", "degradation_reference_k", "degradation_activation_j_per_mol"):
            missing = copy.deepcopy(law)
            del missing[key]
            self.assertTrue(list(self.validator("conversion").iter_errors(missing)))

    def test_obsolete_chemistry_is_rejected(self):
        law = self.scene["cells"][0]["conversion"]
        for field in ("l0_m", "lcrit_m", "energy_scale_j_per_kg", "activation_rate_per_s", "absorption_j_per_kg"):
            obsolete = dict(law, **{field: 1.})
            self.assertTrue(list(self.validator("conversion").iter_errors(obsolete)))

    def test_r1_heat_is_rejected_by_schema(self):
        self.scene["cells"][0]["conversion"]["absorption_j_per_kg"] = 1
        self.assertTrue(list(self.validator().iter_errors(self.scene)))

    def test_native_parser_and_precursor_initialization(self):
        if not PROBE.is_file():
            self.skipTest("Build CUDA reaction probe first")
        self.scene["simulation"].update(duration_s=.01, output_interval_s=.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r123.json"
            path.write_text(json.dumps(self.scene))
            run = subprocess.run([str(PROBE), str(path)], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        data = json.loads(run.stdout)
        initial = data["samples"][0]["objects"][0]
        self.assertAlmostEqual(initial["activated_kg"], .004*.4)
        self.assertNotIn("material_l_max_m", initial)
        self.assertLess(data["max_thermal_residual_j"], 1e-7)
        self.assertLess(data["max_chemical_residual_j"], 1e-7)

    def test_native_rejects_old_fields_without_schema_dependency(self):
        if not PROBE.is_file():
            self.skipTest("Build CUDA reaction probe first")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obsolete.json"
            for key in ("lcrit_m", "activation_rate_per_s", "absorption_j_per_kg"):
                scene = copy.deepcopy(self.scene)
                scene["cells"][0]["conversion"][key] = 1.
                path.write_text(json.dumps(scene))
                run = subprocess.run([str(PROBE), str(path)], capture_output=True, text=True, timeout=10)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn("Obsolete chemistry field", run.stderr)


if __name__ == "__main__":
    unittest.main()

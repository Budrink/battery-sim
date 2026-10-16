"""Public JSON contract for independent pressure transport and local mixtures."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(os.environ.get("BATTERY_SIM_REACTION_PROBE", "/tmp/battery-sim-cuda-v1/battery_sim_reaction_probe"))


class SplitTransport(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "examples/split-transport-demo.json").read_text())
        self.scene["simulation"].update(duration_s=.2, output_interval_s=.1)

    def native(self, scene):
        if not PROBE.is_file():
            self.skipTest("Build native reaction probe first")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(scene))
            return subprocess.run([str(PROBE), str(path)], text=True, capture_output=True, timeout=60)

    def test_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("Optional jsonschema not installed")
        schema = json.loads((ROOT / "schemas/scenario-v0.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(self.scene, schema)

    def test_rupture_mixture_and_chamber(self):
        result = self.native(self.scene)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertLess(data["max_mass_residual_kg"], 1e-11)
        self.assertLess(data["max_thermal_residual_j"], 1e-7)
        self.assertLess(data["max_chemical_residual_j"], 1e-7)
        final = data["samples"][-1]
        for sample in data["samples"]:
            for gas in sample["gas"][:2]:
                self.assertEqual(gas["temperature_k"], gas["material_temperature_k"])
        self.assertGreater(final["channels"][0]["transferred_kg"], 0)
        self.assertGreater(final["channels"][1]["transferred_kg"], 0)
        self.assertGreater(final["fracture_work_j"], 0)
        self.assertGreater(final["gas"][0]["free_volume_m3"], 1e-6)
        self.assertGreater(final["gas"][1]["temperature_k"], 301)
        self.assertGreater(final["gas"][2]["chemical_j"] + final["gas"][2]["released_energy_j"], 0)
        self.assertEqual(final["channels"][0]["surface_energy_j"], 0)

    def test_cpu_local_loader_and_balances(self):
        self.check_cpu_local_loader_and_balances('explicit')

    def test_cpu_implicit_loader_and_balances(self):
        self.check_cpu_local_loader_and_balances('implicit')

    def check_cpu_local_loader_and_balances(self, integrator):
        self.scene['gas_network'].update(mass_backend='cpu_local',local_exchange_interval_s=.0001)
        self.scene['gas_network']['local_integrator']=integrator
        # Long progressive-opening stiffness belongs to the replay audit;
        # fracture/flow/rollback are covered by hybrid_leading_aperture in CTest.
        self.scene['simulation'].update(duration_s=.002,output_interval_s=.001)
        result=self.native(self.scene)
        self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertLess(data['max_mass_residual_kg'],1e-11)
        self.assertLess(data['max_thermal_residual_j'],1e-7)
        for sample in data['samples']:
            for gas in sample['gas'][:2]:
                self.assertEqual(gas['temperature_k'],gas['material_temperature_k'])

    def test_rejects_ambiguous_or_double_counted_transport(self):
        variants = []
        s = copy.deepcopy(self.scene);s['gas_network']['mass_backend']='typo';variants.append(s)
        s = copy.deepcopy(self.scene);s['gas_network']['local_integrator']='typo';variants.append(s)
        s = copy.deepcopy(self.scene);s['gas_network']['local_integrator']='implicit';variants.append(s)
        for key in ('implicit_temperature_tolerance_k', 'implicit_tracer_relative_tolerance'):
            s = copy.deepcopy(self.scene);s['gas_network'].update(mass_backend='cpu_local',local_integrator='implicit')
            s['gas_network'][key]=0;variants.append(s)
        s = copy.deepcopy(self.scene);s['gas_network'].update(mass_backend='cpu_local',local_exchange_interval_s=0);variants.append(s)
        s = copy.deepcopy(self.scene);s['gas_network'].update(mass_backend='cpu_local',transport='legacy_coupled');variants.append(s)
        s = copy.deepcopy(self.scene);s["gas_network"]["transport"] = "typo";variants.append(s)
        s = copy.deepcopy(self.scene);s["gas_network"].pop("transport");variants.append(s)
        s = copy.deepcopy(self.scene);s["gas_network"]["compartments"][1]["local_equilibrium"]["thermal_volume"] = 1;variants.append(s)
        s = copy.deepcopy(self.scene);s["gas_network"]["compartments"][1]["local_equilibrium"]["unknown"] = 1;variants.append(s)
        s = copy.deepcopy(self.scene);s["gas_network"]["channels"][0]["surface_link"] = {"h_w_per_m2_k": 1};variants.append(s)
        for index, s in enumerate(variants):
            with self.subTest(index=index):
                self.assertNotEqual(self.native(s).returncode, 0)


if __name__ == "__main__":
    unittest.main()

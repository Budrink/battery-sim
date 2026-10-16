"""Public contract of the opt-in native casing/conversion/gas model."""
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


class ShellGasContract(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "examples/shell-gas-network-demo.json").read_text())
        self.scene["simulation"].update(duration_s=.2, output_interval_s=.1)

    def native(self, scene):
        if not PROBE.is_file():
            self.skipTest("Build native reaction probe first")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(scene))
            return subprocess.run([str(PROBE), str(path)], text=True, capture_output=True, timeout=30)

    def test_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("Optional jsonschema not installed")
        schema = json.loads((ROOT / "schemas/scenario-v0.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(self.scene, schema)

    def test_native_physical_path(self):
        result = self.native(self.scene)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertLess(data["max_mass_residual_kg"], 1e-11)
        self.assertLess(data["max_thermal_residual_j"], 1e-7)
        self.assertLess(data["max_chemical_residual_j"], 1e-7)
        initial, final = data["samples"][0], data["samples"][-1]
        panels = [o for o in initial["objects"] if ":case:" in o["id"]]
        self.assertEqual(len(panels), 6)
        self.assertTrue(all(abs(o["temperature_k"] - 300) < 1e-10 for o in panels))
        self.assertGreater(initial["objects"][0]["temperature_k"], 499)
        self.assertTrue(final["channels"][0]["open"])
        self.assertGreater(final["channels"][0]["transferred_kg"], 0)
        self.assertGreater(final["objects"][0]["reactant_kg"], .002)
        self.assertGreater(final["gas"][1]["chemical_j"], 0)

    def test_python_fails_closed(self):
        with self.assertRaisesRegex(Exception, "native CUDA"):
            load_scenario(ROOT / "examples/shell-gas-network-demo.json")

    def test_progressive_whole_shell(self):
        scene = json.loads((ROOT / "examples/progressive-shell-opening-demo.json").read_text())
        scene["simulation"].update(duration_s=.2, output_interval_s=.1)
        result = self.native(scene)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(len(data["samples"][-1]["channels"]), 6)
        self.assertLess(data["max_mass_residual_kg"], 1e-11)
        self.assertLess(data["max_thermal_residual_j"], 1e-7)
        core = scene['cells'][0]['geometry']['dimensions_m']
        thickness = scene['shells'][0]['thickness_m']
        for channel in data["samples"][-1]["channels"]:
            self.assertGreaterEqual(channel["max_seam_damage"], channel["max_bulk_damage"])
            self.assertGreaterEqual(channel["opening_fraction"], 0)
            self.assertLessEqual(channel["opening_fraction"], 1)
            axis = 'xyz'.index(channel['id'][-1])
            panel = [thickness if k == axis else core[k] + (2*thickness if k > axis else 0) for k in range(3)]
            area = panel[(axis+1)%3] * panel[(axis+2)%3]
            self.assertAlmostEqual(channel["effective_area_m2"], area * channel["opening_fraction"])

    def test_progressive_area_ceiling_rejected(self):
        scene = json.loads((ROOT / 'examples/progressive-shell-opening-demo.json').read_text())
        scene['gas_network']['channels'][0]['area_m2'] = 1e-7
        result = self.native(scene)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('obsolete area_m2 ceiling', result.stderr)

    def test_stochastic_pixels(self):
        scene = json.loads((ROOT / 'examples/stochastic-shell-pixels-demo.json').read_text())
        result = self.native(scene)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertLess(data['max_mass_residual_kg'], 1e-11)
        self.assertLess(data['max_thermal_residual_j'], 1e-7)
        counts = [576*c['opening_fraction'] for c in data['samples'][-1]['channels']]
        for n in counts:
            self.assertAlmostEqual(n, round(n), places=9)
        self.assertGreater(sum(counts), 0)
        self.assertLess(sum(counts), 6*576)
        again = self.native(scene)
        self.assertEqual(json.loads(again.stdout)['samples'], data['samples'])

    def test_invalid_stochastic_inputs(self):
        base = json.loads((ROOT / 'examples/stochastic-shell-pixels-demo.json').read_text())
        for change in [{'seed': -1}, {'seed': 1.5}, {'pixel_rate_per_s': -1},
                       {'pixel_pressure_exponent': .5}, {'initial_mechanical_damage': .5}, {'mode': 'unknown'}]:
            scene = copy.deepcopy(base)
            scene['gas_network']['channels'][0]['damage_opening'].update(change)
            self.assertNotEqual(self.native(scene).returncode, 0)
        for divisions in [[0,24,24], [-1,24,24], [24,24]]:
            scene = copy.deepcopy(base);scene['shells'][0]['surface_divisions'] = divisions
            self.assertNotEqual(self.native(scene).returncode, 0)

    def test_stochastic_area(self):
        scene=json.loads((ROOT/'examples/stochastic-shell-pixels-demo.json').read_text())
        scene['shells'][0]['surface_divisions']=[3,3,3]
        for channel in scene['gas_network']['channels']:
            channel['damage_opening'].update(mode='stochastic_area',fragment_mean_area_m2=1e-6)
        result=self.native(scene);self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertLess(data['max_thermal_residual_j'],1e-7)
        self.assertLess(data['max_mass_residual_kg'],1e-11)
        fractions=[c['opening_fraction'] for c in data['samples'][-1]['channels']]
        self.assertTrue(any(abs(9*f-round(9*f))>1e-6 for f in fractions))
        self.assertEqual(json.loads(self.native(scene).stdout)['samples'],data['samples'])
        for invalid in [0,-1]:
            bad=copy.deepcopy(scene);bad['gas_network']['channels'][0]['damage_opening']['fragment_mean_area_m2']=invalid
            self.assertNotEqual(self.native(bad).returncode,0)

    def test_invalid_progressive_laws(self):
        for law in [{"unknown": 1}, {"seam_strength_factor": 0}, {"seam_strength_factor": 1.1},
                    {"seam_width_m": -.001}, {"seam_width_m": .1}, {"exponent": .5},
                    {"damage_onset": 1}, {"pressure_damage_rate_per_s": -1}, {"max_damage_step": 0}]:
            with self.subTest(law=law):
                scene = copy.deepcopy(self.scene)
                scene["gas_network"]["channels"][0]["damage_opening"] = law
                self.assertNotEqual(self.native(scene).returncode, 0)

    def test_stochastic_fixed_fraction(self):
        scene=json.loads((ROOT/'examples/stochastic-shell-pixels-demo.json').read_text())
        scene['shells'][0]['surface_divisions']=[3,3,3]
        for c in scene['gas_network']['channels']:
            c['damage_opening'].update(mode='stochastic_fraction',fixed_area_fraction=.02)
        result=self.native(scene);self.assertEqual(result.returncode,0,result.stderr)
        data=json.loads(result.stdout)
        self.assertLess(data['max_thermal_residual_j'],1e-7)
        self.assertLess(data['max_mass_residual_kg'],1e-11)
        for c in data['samples'][-1]['channels']:
            count=c['opening_fraction']*9/.02
            self.assertAlmostEqual(count,round(count),places=8)
        for change in ({'fixed_area_fraction':0},{'fixed_area_fraction':-.1},{'fixed_area_fraction':1.1},
                       {'fragment_mean_area_m2':1e-5},{'mode':'stochastic_area'}):
            bad=copy.deepcopy(scene);bad['gas_network']['channels'][0]['damage_opening'].update(change)
            self.assertNotEqual(self.native(bad).returncode,0)

    def test_progressive_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("Optional jsonschema not installed")
        scene = json.loads((ROOT / "examples/progressive-shell-opening-demo.json").read_text())
        schema = json.loads((ROOT / "schemas/scenario-v0.schema.json").read_text())
        jsonschema.validate(scene, schema)

    def test_invalid_contracts(self):
        variants = []
        s = copy.deepcopy(self.scene)
        s["gas_network"]["compartments"][0].pop("contents_id")
        variants.append(s)
        s = copy.deepcopy(self.scene)
        s["shells"].append(copy.deepcopy(s["shells"][0]))
        variants.append(s)
        s = copy.deepcopy(self.scene)
        s["shells"][0]["initial_temperature_k"] = -2
        variants.append(s)
        s = copy.deepcopy(self.scene)
        s["cells"][0]["conversion"]["gas_yield"] = 2
        variants.append(s)
        s = copy.deepcopy(self.scene)
        s["cells"][0]["runaway"]["energy_j"] = 1
        variants.append(s)
        s = copy.deepcopy(self.scene)
        s["gas_network"]["channels"][0]["port_id"] = "missing"
        variants.append(s)
        s = copy.deepcopy(self.scene)
        s["gas_network"]["channels"][0]["port_id"] = "B:case:+z:contents"
        variants.append(s)
        s = copy.deepcopy(self.scene)
        s["boundaries"] = [{"id": "bypass", "port_id": "B:case:+z:contents", "temperature_k": 300, "conductance_w_per_k": 1}]
        variants.append(s)
        s = copy.deepcopy(self.scene)
        s["gas_network"]["thermal_contacts"].append({"compartment": "cavity", "port_id": "B:case:+z:contents", "h_w_per_m2_k": 10})
        variants.append(s)
        for index, scene in enumerate(variants):
            with self.subTest(index=index):
                self.assertNotEqual(self.native(scene).returncode, 0)


if __name__ == "__main__":
    unittest.main()

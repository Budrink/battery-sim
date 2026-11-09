import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LaboratoryModuleTests(unittest.TestCase):
    def test_gas_transport_not_silently_disabled(self):
        doc = json.loads((ROOT / "examples/contracts/module-4x4.json").read_text())
        seed = json.loads((ROOT / "examples/contracts/fsri30-gas-transport-preset.json").read_text())
        network = doc["assembly"]["gas_network"]
        for key, value in seed["network"].items():
            self.assertEqual(network[key], value, key)
        self.assertEqual(doc["assembly"]["gas_meshes"]["case"]["battery_surface_connectivity_per_m2"],
                         seed["battery_surface_connectivity_per_m2"])
        law = seed["battery_surface_connectivity_per_m2"]
        cold = law["background_conductance_w_per_k"] + law["active_conductance_w_per_k"] / (
            1 + math.exp(-law["geometry_reference_logit"]))
        self.assertAlmostEqual(cold, 20.)
        for key in ("mixing_diffusivity_m2_s", "mixing_flow_fraction", "thermal_stirring_coefficient"):
            self.assertGreater(network[key], 0)
        self.assertTrue(network["free_gas_response_only"])
        self.assertEqual(network["mixing_integrator"], "implicit_surfaces")
        self.assertIn("not universal", doc["metadata"]["gas_transport_preset"]["scope"])

    def test_reproducible_starter(self):
        spec = importlib.util.spec_from_file_location("module_generator", ROOT / "scripts/generate_laboratory_module.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        saved = json.loads((ROOT / "examples/contracts/module-4x4.json").read_text())
        self.assertEqual(saved, generator.make_module())
        self.assertEqual(len(saved["assembly"]["passive_cells"]), 31)
        self.assertEqual(len(saved["assembly"]["interfaces"]), 49)
        self.assertEqual(saved["conditions"]["heating_protocols"][0]["sensor_id"], "B05:case:+x")
        preset = json.loads((ROOT / "examples/contracts/fsri30-heater-preset.json").read_text())
        protocol = saved["conditions"]["heating_protocols"][0]
        for key, value in preset["protocol"].items():
            self.assertEqual(protocol[key], value)
        self.assertEqual(saved["conditions"]["boundaries"], [])
        self.assertEqual(saved["assembly"]["gas_meshes"]["case"]["openings"], [
            {"id": "top-opening", "face": "+z", "shape": "circle", "center_m": [0, 0], "size_m": [.02, .02]}])
        self.assertEqual(saved["metadata"]["editor_enclosures"]["case"]["properties"]["ambient_h_w_m2_k"], 0)
        self.assertEqual(saved["conditions"]["source_drives"][0]["current_a"], 0)
        self.assertEqual(saved["provenance"]["status"], "synthetic")

    def test_material_transfer_is_scaled_not_reinvented(self):
        doc = json.loads((ROOT / "examples/contracts/module-4x4.json").read_text())
        seed = json.loads((ROOT / "examples/contracts/fsri30-material-preset.json").read_text())
        model = doc["battery_models"][0]
        shell = model["shell"]
        self.assertEqual(shell["damage_onset_k"], 900.)
        for face in ("+z", "-z"):
            self.assertEqual(shell["face_materials"][face]["damage_onset_k"], 450.)
        self.assertEqual({v["port_suffix"] for v in model["vents"]},
                         {"case:+z:outer", "case:-z:outer"})
        for vent in model["vents"]:
            self.assertTrue(vent["channel"]["damage_opening"]["whole_plate"])
        for b in doc["assembly"]["batteries"]:
            self.assertEqual({v["vent_id"] for v in doc["assembly"]["vent_connections"]
                              if v["battery_id"] == b["id"]}, {"top", "bottom"})
        scale = math.prod(model["geometry"]["dimensions_m"]) / seed["reference_volume_m3"]
        actual = model["contents"]["conversion"]
        for key, value in seed["conversion"].items():
            if key not in ("mass_kg", "volatile_phase"):
                self.assertEqual(actual[key], value, key)
        self.assertAlmostEqual(actual["mass_kg"], seed["conversion"]["mass_kg"] * scale)
        self.assertAlmostEqual(actual["volatile_phase"]["initial_mass_kg"],
                               seed["conversion"]["volatile_phase"]["initial_mass_kg"] * scale)
        self.assertEqual(model["shell"]["contents_connectivity_per_m2"],
                         seed["shell"]["contents_connectivity_per_m2"])
        self.assertEqual(doc["conditions"]["source_drives"][0]["current_a"], 0)
        self.assertGreaterEqual(doc["numerics"]["simulation"]["duration_s"], 3000)
        gases = doc["assembly"]["gas_network"]["compartments"]
        for b in doc["assembly"]["batteries"]:
            pores = [g for g in gases if g.get("local_equilibrium", {}).get("cell_id") == b["id"]]
            self.assertEqual(len(pores), 27)
            self.assertEqual({g["local_equilibrium"]["thermal_volume"] for g in pores}, set(range(27)))
            self.assertEqual(b["initial_state"]["charged_fraction"], 1)
            for g in pores:
                self.assertEqual(g["initial_oxidizer_mass_fraction"], 0)
                for key, value in seed["r3"].items():
                    self.assertEqual(g[key], value)
                self.assertAlmostEqual(g["initial_mass_kg"]*287*298.15/g["volume_m3"], 101325)

from pathlib import Path
import json
import math
import runpy
import unittest
from battery_sim.io import load_scenario
from battery_sim.model import ModelError

ROOT = Path(__file__).resolve().parents[1]

class MultiscaleGeometryTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "examples/narada-52-multiscale.json"
        self.raw = json.loads(self.path.read_text())

    def test_generated_scenario_and_provenance(self):
        generate = runpy.run_path(str(ROOT / "scripts/generate_narada_multiscale.py"))["make_scenario"]
        self.assertEqual(generate(), self.raw)
        self.assertEqual(self.raw["metadata"]["calibration_status"], "numerical-reference-fit")
        self.assertEqual(self.raw["observations"], {})
        self.assertEqual(self.raw["simulation"]["step_interval_s"], 1)

    def test_reference_fit_does_not_force_secondary_failures(self):
        generate = runpy.run_path(str(ROOT / "scripts/generate_narada_multiscale.py"))["make_scenario"]
        prior = generate(calibrated=False)
        self.assertEqual(prior, json.loads((ROOT / "examples/narada-52-multiscale-prior.json").read_text()))
        self.assertEqual(prior["primary_failures"], self.raw["primary_failures"])
        self.assertEqual(len(self.raw["primary_failures"]), 1)
        for old, new in zip(prior["cells"][:52], self.raw["cells"][:52]):
            for key in ("thermal", "degradation", "runaway"):
                self.assertEqual(old[key], new[key])
        self.assertTrue(all(e["law"]["kind"] == "first_order" for e in self.raw["interfaces"]))
        self.assertTrue(all(e["law"]["damage_rate_at_onset_per_s"] > 0 for e in self.raw["interfaces"]))

    def test_zone_counts_and_no_parallel_bypass(self):
        cells = {c["id"]: c for c in self.raw["cells"]}
        ports = {p["id"]: p for p in self.raw["ports"]}
        batteries = [c for c in cells.values() if c["role"] == "battery"]
        pads = [c for c in cells.values() if c["role"] == "passive"]
        self.assertEqual(len(batteries), 52)
        self.assertEqual(len(pads), 87)
        self.assertEqual(sum(math.prod(c["thermal"]["divisions"]) for c in batteries), 416)
        self.assertEqual(sum(math.prod(c["thermal"]["divisions"]) for c in pads), 34416)
        neighbors = {c["id"]: set() for c in pads}
        for edge in self.raw["interfaces"]:
            a, b = (ports[edge[p]] for p in ("port_a", "port_b"))
            self.assertEqual(cells[a["cell_id"]]["role"], "battery")
            self.assertEqual(cells[b["cell_id"]]["role"], "passive")
            neighbors[b["cell_id"]].add(a["cell_id"])
            self.assertEqual(a["centroid_m"], b["centroid_m"])
            self.assertEqual(a["normal"], [-v for v in b["normal"]])
            self.assertEqual(a["area_m2"], b["area_m2"])
            for p in (a, b):
                cell = cells[p["cell_id"]]
                axis = "xyz".index(p["surface"][1])
                sign = 1 if p["surface"][0] == "+" else -1
                self.assertAlmostEqual(p["centroid_m"][axis]-cell["pose"]["position_m"][axis],
                                       sign*cell["geometry"]["dimensions_m"][axis]/2)
        self.assertTrue(all(len(n) == 2 for n in neighbors.values()))
        for c in pads:
            self.assertEqual(c["runaway"]["energy_j"], 0)
            self.assertAlmostEqual(c["thermal"]["heat_capacity_j_per_k"], math.prod(c["geometry"]["dimensions_m"])*1.2e6)

    def test_python_reference_does_not_silently_misinterpret_passive_material(self):
        with self.assertRaisesRegex(ModelError, "native CUDA"):
            load_scenario(self.path)

if __name__ == "__main__":
    unittest.main()

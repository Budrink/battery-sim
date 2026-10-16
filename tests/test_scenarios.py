from __future__ import annotations

import json
from pathlib import Path
import unittest

from battery_sim.io import load_scenario, scenario_from_dict
from battery_sim.model import ModelError
from battery_sim.solver import Simulator


ROOT = Path(__file__).resolve().parents[1]


class ScenarioTests(unittest.TestCase):
    def test_cascade_example(self) -> None:
        result = Simulator(load_scenario(ROOT / "examples/three-cell-cascade.json")).run()
        self.assertEqual(result.runaway_order, ("cell-a", "cell-b", "cell-c"))
        self.assertTrue(result.propagated)
        self.assertTrue(result.validation["pass"])
        self.assertLess(abs(result.energy_balance_residual_j), 1.0e-5)

    def test_contained_example(self) -> None:
        result = Simulator(load_scenario(ROOT / "examples/three-cell-contained.json")).run()
        self.assertEqual(result.runaway_order, ("cell-a",))
        self.assertFalse(result.propagated)
        self.assertTrue(result.validation["pass"])
        self.assertLess(abs(result.energy_balance_residual_j), 1.0e-5)

    def test_duplicate_cell_id_is_rejected(self) -> None:
        with (ROOT / "examples/three-cell-contained.json").open(encoding="utf-8") as stream:
            raw = json.load(stream)
        raw["cells"][1]["id"] = "cell-a"
        with self.assertRaisesRegex(ModelError, "cell ids must be unique"):
            scenario_from_dict(raw)

    def test_python_reference_rejects_measured_forcing(self) -> None:
        raw=json.loads((ROOT / "examples/three-cell-contained.json").read_text())
        raw["boundaries"][0]["temperature_profile"]=[[0,300],[1000,400]]
        with self.assertRaisesRegex(ModelError,"native CUDA"):
            scenario_from_dict(raw)

if __name__ == "__main__":
    unittest.main()

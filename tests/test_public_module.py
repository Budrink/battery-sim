from pathlib import Path
import json
import math
import runpy
import unittest

from battery_sim.io import load_scenario

ROOT = Path(__file__).resolve().parents[1]


class PublicModuleGeometryTests(unittest.TestCase):
    def test_generated_artifact_is_reproducible(self):
        generate = runpy.run_path(str(ROOT / "scripts/generate_narada_52.py"))["make_scenario"]
        raw = json.loads((ROOT / "examples/narada-52-schematic.json").read_text())
        self.assertEqual(generate(), raw)
        self.assertEqual(raw["metadata"]["calibration_status"], "uncalibrated")
        self.assertEqual(raw["observations"], {})  # no manufactured measurements

    def test_thermal_paths_stay_on_faces_and_module_fits(self):
        path = ROOT / "examples/narada-52-schematic.json"
        load_scenario(path)  # solver's normal validation path
        raw = json.loads(path.read_text())
        cells = {c["id"]: c for c in raw["cells"]}
        ports = {p["id"]: p for p in raw["ports"]}
        self.assertEqual(len(cells), 52)
        self.assertEqual(len(raw["interfaces"]), 87)
        graph = {cid: set() for cid in cells}
        pairs = set()
        for edge in raw["interfaces"]:
            pa, pb = ports[edge["port_a"]], ports[edge["port_b"]]
            pair = tuple(sorted([pa["cell_id"], pb["cell_id"]]))
            self.assertNotIn(pair, pairs)
            pairs.add(pair)
            graph[pair[0]].add(pair[1])
            graph[pair[1]].add(pair[0])
            axis = next(i for i, n in enumerate(pa["normal"]) if n)
            gap = sum((pb["centroid_m"][i]-pa["centroid_m"][i])*pa["normal"][i]
                      for i in range(3))
            self.assertAlmostEqual(gap, 0.010 if axis == 0 else 0.002)
            for p in (pa, pb):
                c = cells[p["cell_id"]]
                d = c["geometry"]["dimensions_m"]
                pos = c["pose"]["position_m"]
                self.assertAlmostEqual(abs(p["centroid_m"][axis]-pos[axis]), d[axis]/2)
                for i in range(3):
                    if i != axis:
                        self.assertLessEqual(abs(p["centroid_m"][i]-pos[i]) +
                                             math.sqrt(p["area_m2"])/2, d[i]/2 + 1e-10)
            self.assertEqual(pa["normal"], [-n for n in pb["normal"]])
        visited, todo = set(), [next(iter(cells))]
        while todo:
            node = todo.pop()
            if node not in visited:
                visited.add(node)
                todo.extend(graph[node]-visited)
        self.assertEqual(len(visited), 52)
        for axis, envelope in enumerate([0.760, 1.150, 0.237]):
            low = min(c["pose"]["position_m"][axis]-c["geometry"]["dimensions_m"][axis]/2
                      for c in cells.values())
            high = max(c["pose"]["position_m"][axis]+c["geometry"]["dimensions_m"][axis]/2
                       for c in cells.values())
            self.assertLessEqual(high-low, envelope)


if __name__ == "__main__":
    unittest.main()

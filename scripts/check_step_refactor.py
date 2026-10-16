#!/usr/bin/env python3
"""Compare native replays, not fitted parameters, before/after solver refactoring.

The old executable must be saved BEFORE rebuilding. Scenario bytes and numerical
settings are identical. This is regression evidence, not renewed validation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(executable, scenario):
    result = subprocess.run([str(executable), str(scenario)], input="{}\n", text=True,
                            capture_output=True, check=True)
    value = json.loads(result.stdout)
    if "error" in value:
        raise RuntimeError(value["error"])
    return value


def compare(a, b, path=""):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Different output keys: {path}")
        return [item for k in a if k != "wall_s"
                for item in compare(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError(f"Different observation counts: {path}")
        return [item for i, (x, y) in enumerate(zip(a, b))
                for item in compare(x, y, f"{path}/{i}")]
    if isinstance(a, (int, float)):
        if not math.isfinite(a) or not math.isfinite(b):
            raise AssertionError(f"Nonfinite numerical output: {path}")
        delta = abs(a-b)
        # Much tighter than the accepted 0.5% calibration tolerance.
        allowed = 1e-8 + 1e-10 * max(abs(a), abs(b))
        return [(path, delta, delta/allowed)]
    if a != b:
        raise AssertionError(f"Different categorical/event output: {path}: {a} != {b}")
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        "heat-no-cooling": "upv-physical-casing-v5/heat/no-cooling-fitted-scenario.json",
        "heat-active-cooling": "upv-physical-casing-v5/heat/active-cooling-fitted-scenario.json",
        "failure-no-cooling": "upv-gas-energy-v6/no-cooling-fit-scenario.json",
        "failure-active-cooling": "upv-gas-energy-v6/active-cooling-validate-scenario.json",
    }
    report = {"before_sha256": hashlib.sha256(args.before.read_bytes()).hexdigest(),
              "after_sha256": hashlib.sha256(args.after.read_bytes()).hexdigest(), "cases": {}}
    for name, relative in cases.items():
        scenario = ROOT / "data/calibration" / relative
        before, after = run(args.before, scenario), run(args.after, scenario)
        differences = compare(before, after)
        worst = max(differences, key=lambda x: x[2])
        temperatures = [x[1] for x in differences if "temperature_k" in x[0] or "final_fields_k" in x[0]]
        pressures = [x[1] for x in differences if "pressure_pa" in x[0]]
        item = {"scenario_sha256": hashlib.sha256(scenario.read_bytes()).hexdigest(),
                "steps_before": before["steps"], "steps_after": after["steps"],
                "wall_s_before": before["wall_s"], "wall_s_after": after["wall_s"],
                "max_temperature_delta_k": max(temperatures, default=0),
                "max_pressure_delta_pa": max(pressures, default=0),
                "worst_normalized_difference": worst,
                "passed": worst[2] <= 1 and before["steps"] == after["steps"]}
        report["cases"][name] = item
        print(name, json.dumps(item), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    if not all(x["passed"] for x in report["cases"].values()):
        raise SystemExit("Refactor regression failed; inspect report")


if __name__ == "__main__":
    main()

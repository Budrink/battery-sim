#!/usr/bin/env python3
"""Replay frozen heating inputs without fitting or modifying old artifacts."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument("--probe", type=Path, default=Path("/tmp/battery-sim-cuda-v1/battery_sim_casing_probe"))
    cli.add_argument("--reference", type=Path, default=ROOT / "data/calibration/upv-training-refit-v13")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    report = {"protocol": "Frozen heating replay; no fitting, changed inputs or time interpolation", "cases": []}
    for name, folder in [("no-cooling", args.reference), ("active-cooling", args.reference / "cooling-check/nominal")]:
        scene = folder / "warmup-scenario.json"
        old = json.loads((folder / "warmup.json").read_text())
        process = subprocess.run([str(args.probe.resolve()), str(scene.resolve())], input="{}\n",
                                 text=True, capture_output=True, check=True, timeout=300)
        current = json.loads(process.stdout)
        if "error" in current:
            raise RuntimeError(current["error"])
        assert current["time_s"] == old["time_s"], "Changed observation times"
        assert current["applied_objects"] == old["applied_objects"], "Changed frozen material/geometry"
        errors = {}
        for key, before in old["temperature_k"].items():
            after = current["temperature_k"][key]
            assert len(before) == len(after)
            errors[key] = max(abs(a-b) for a, b in zip(before, after))
        assert max(errors.values()) <= 1e-10, errors
        assert current["final_fields_k"] == old["final_fields_k"], "Changed final field"
        assert current["steps"] == old["steps"], "Changed thermal timestep selection"
        report["cases"].append(dict(name=name, max_absolute_temperature_change_k=errors,
                                    identical_final_field=True, steps=current["steps"],
                                    max_thermal_residual_j=current["max_thermal_residual_j"]))
    text = json.dumps(report, indent=2)
    if args.output:
        # Never overwrite a previous regression result.
        with args.output.open("x") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

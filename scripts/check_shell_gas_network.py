#!/usr/bin/env python3
"""Run three temporal resolutions; report conservation and convergence, not a fit."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--probe", type=Path, default=Path("/tmp/battery-sim-cuda-v1/battery_sim_reaction_probe"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for level in range(3):
        factor = 2 ** level
        command = [str(args.probe), str(ROOT / "examples/shell-gas-network-demo.json"),
                   "--max-dt", str(.002 / factor), "--fraction", str(.005 / factor),
                   "--temperature-step", str(.25 / factor)]
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=300)
        data = json.loads(result.stdout)
        (args.output_dir / f"level-{level}.json").write_text(json.dumps(data, indent=2) + "\n")
        last = data["samples"][-1]
        report = {key: data[key] for key in ("steps", "wall_s", "max_mass_residual_kg", "max_thermal_residual_j", "max_chemical_residual_j")}
        report.update(level=level, temperatures_k={o["id"]: o["temperature_k"] for o in last["objects"]},
                      gas_pressures_pa={g["id"]: g["pressure_pa"] for g in last["gas"]},
                      rupture_s=last["channels"][0]["opened_s"], transferred_kg=last["channels"][0]["transferred_kg"])
        results.append(report)
        print(json.dumps(report), flush=True)
    deltas = []
    for a, b in zip(results, results[1:]):
        deltas.append({"levels": [a["level"], b["level"]],
                       "max_solid_temperature_difference_k": max(abs(a["temperatures_k"][key]-b["temperatures_k"][key]) for key in a["temperatures_k"]),
                       "max_gas_pressure_difference_pa": max(abs(a["gas_pressures_pa"][key]-b["gas_pressures_pa"][key]) for key in a["gas_pressures_pa"]),
                       "rupture_time_difference_s": abs(a["rupture_s"]-b["rupture_s"])})
    summary = {"status": "synthetic numerical verification, not experimental validation", "runs": results, "successive_differences": deltas}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(deltas), flush=True)
    for r in results:
        assert r["max_mass_residual_kg"] < 1e-10
        assert r["max_thermal_residual_j"] < 1e-6
        assert r["max_chemical_residual_j"] < 1e-6
    assert deltas[-1]["max_solid_temperature_difference_k"] < .1


if __name__ == "__main__":
    main()

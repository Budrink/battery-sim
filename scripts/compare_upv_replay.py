#!/usr/bin/env python3
"""Compare an unfitted UPV replay with a frozen report and its observations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = {"101": "C1 / ISS", "102": "C1 / TIM", "107": "C3 / TIM", "108": "C3 / ISS"}


def compare(reference, current):
    old = reference["result"]
    if current["applied_objects"] != old["applied_objects"]:
        raise ValueError("Frozen material properties or geometry changed")
    t = np.asarray(current["time_s"])
    before_t = np.asarray(old["time_s"])
    if len(t) != len(before_t) or not np.allclose(t, before_t, atol=1e-8, rtol=0):
        raise ValueError("Observation schedules differ")
    conservation = {key: current.get(key, 0.) for key in (
        "max_thermal_residual_j", "max_chemical_residual_j", "max_mass_residual_kg", "max_oxidizer_residual_kg")}
    if (conservation["max_thermal_residual_j"] > 1e-4 or conservation["max_chemical_residual_j"] > 1e-4
            or conservation["max_mass_residual_kg"] > 1e-10 or conservation["max_oxidizer_residual_kg"] > 1e-10):
        raise ValueError(f"Conservation failed: {conservation}")
    report = dict(no_calibration=True, identical_materials_and_geometry=True, conservation=conservation,
                  old_steps=old["steps"], new_steps=current["steps"], old_wall_s=old["wall_s"],
                  new_wall_s=current["wall_s"], simulated_s=float(t[-1]),
                  simulated_s_per_wall_s=float(t[-1]/current["wall_s"]), channels={})
    for key in CHANNELS:
        before = np.asarray(old["temperature_k"][key])
        after = np.asarray(current["temperature_k"][key])
        observed = np.asarray(reference["observed_temperature_k"][key], dtype=float)
        phases = {}
        for lo, hi in [(3, 30), (30, 300), (300, 1800)]:
            mask = (t > lo) & (t <= hi) & np.isfinite(observed)
            phases[f"{lo}-{hi}"] = {name: float(np.sqrt(np.mean((value[mask]-observed[mask])**2)))
                                    for name, value in [("old_rmse_k", before), ("new_rmse_k", after)]}
        report["channels"][key] = dict(max_change_k=float(np.max(np.abs(after-before))), phases=phases)
    for name, run in [("old", old), ("new", current)]:
        report[name+"_pressure_peak_bar"] = max(run["gas"]["chamber"]["pressure_pa"])/1e5
        report[name+"_chamber_released_j"] = run["gas"]["chamber"]["released_energy_j"][-1]
        report[name+"_near_released_j"] = sum(v["released_energy_j"][-1] for k, v in run["gas"].items() if k.startswith("near-"))
    return report


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument("reference", type=Path, help="Frozen report with result and experimental observations")
    cli.add_argument("candidate", type=Path, help="Native replay output, without fitting")
    cli.add_argument("output", type=Path)
    cli.add_argument("--plot-start", type=float, default=3.)
    cli.add_argument("--plot-end", type=float, default=300.)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["comparison.json", "temperature.png", "pressure.png"]:
        if (args.output/name).exists():
            raise ValueError("Do not overwrite an existing comparison")
    reference = json.loads(args.reference.read_text())
    current = json.loads(args.candidate.read_text())
    if "error" in current:
        raise ValueError(current["error"])
    report = compare(reference, current)
    old = reference["result"]
    t = np.asarray(current["time_s"])
    mask = (t >= args.plot_start) & (t <= args.plot_end)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout="constrained")
    for ax, (key, name) in zip(axes.flat, CHANNELS.items()):
        for values, color, style, width, label in [
            (reference["observed_temperature_k"][key], "#333333", "-", 1.5, "Experiment"),
            (old["temperature_k"][key], "#e7772c", "-", 2.5, "Frozen reference"),
            (current["temperature_k"][key], "#146c94", "--", 1.4, "Continuous gas reaction")]:
            ax.plot(t[mask], np.asarray(values, dtype=float)[mask]-273.15, color=color, ls=style, lw=width, label=label)
        ax.set(title=name, xlabel="Time after event, s", ylabel="Temperature, °C")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("UPV replay — continuous gas reaction, frozen thermal model")
    fig.savefig(args.output/"temperature.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    mask = t <= 30
    ax.plot(t[mask], np.asarray(reference["observed_pressure_bar"])[mask], color="#333333", label="Experiment")
    for run, color, style, label in [(old, "#e7772c", "-", "Frozen reference"), (current, "#146c94", "--", "Continuous gas reaction")]:
        ax.plot(t[mask], np.asarray(run["gas"]["chamber"]["pressure_pa"])[mask]/1e5, color=color, ls=style, label=label)
    ax.set(xlabel="Time after event, s", ylabel="Chamber pressure, bar")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(args.output/"pressure.png", dpi=150)
    plt.close(fig)
    with (args.output/"comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two casing probes on identical inputs; never modify a calibrated seed.

Only wall-clock profiling is excluded. Temperatures, composition, pressure,
events, ledgers and accepted step counts must be exactly equal, not merely close.
This is a regression check, not experimental validation or recalibration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_result(result):
    result = dict(result)
    result.pop("wall_s", None)
    # Retain integer iteration/retry counters; omit measured CPU/GPU timing only.
    profile = dict(result.get("solver_profile", {}))
    for key in ("prepare_s", "cpu_gas_s", "cuda_gas_s", "download_s", "upload_s"):
        profile.pop(key, None)
    result["solver_profile"] = profile
    return result


def differences(a, b, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            yield path + ": keys differ"
        for key in a.keys() & b.keys():
            yield from differences(a[key], b[key], path + "." + key)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield path + ": lengths differ"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, f"{path}[{i}]")
    elif a != b:
        yield f"{path}: {a!r} != {b!r}"


def run(probe, scene, request, out):
    process = subprocess.run([str(Path(probe).resolve()), str(scene)],
                             input=json.dumps(request) + "\n", text=True,
                             capture_output=True, check=True, timeout=180)
    result = json.loads(process.stdout)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def performance_summary(before, after, max_slowdown=None, max_noise=.05):
    """Use alternating pairs; a noisy measurement must not certify a refactor."""
    def describe(samples):
        values = [float(s["wall_s"]) for s in samples]
        if not values or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Invalid wall-clock samples")
        median = statistics.median(values)
        mad = statistics.median(abs(v - median) for v in values) / median
        return dict(samples_s=values, median_s=median, relative_mad=mad)
    a, b = describe(before), describe(after)
    ratio = statistics.median(y["wall_s"] / x["wall_s"] for x, y in zip(before, after))
    # MAD catches broad noise; spread also catches a minority of thermally throttled runs.
    spread = max((max(s["samples_s"]) - min(s["samples_s"])) / s["median_s"] for s in (a, b))
    noisy = max(a["relative_mad"], b["relative_mad"]) > max_noise or spread > 4 * max_noise
    status = ("not_enforced" if max_slowdown is None else
              "inconclusive_noise" if noisy else
              "regression" if ratio > 1 + max_slowdown else "passed")
    return dict(before=a, after=b, paired_median_ratio=ratio,
                max_slowdown=max_slowdown, max_relative_mad=max_noise,
                relative_spread=spread, status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, default=ROOT / "examples/reaction-chain-demo.json")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--contract", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--max-slowdown", type=float,
                        help="Allowed relative slowdown, e.g. 0.05; needs >=3 pairs and >=1 warmup")
    parser.add_argument("--max-timing-noise", type=float, default=.05)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("Invalid repeat/warmup count")
    if not math.isfinite(args.max_timing_noise) or args.max_timing_noise <= 0:
        parser.error("Timing noise must be finite and positive")
    if args.max_slowdown is not None and (
            not math.isfinite(args.max_slowdown) or args.max_slowdown < 0 or
            args.repeats < 3 or args.warmups < 1):
        parser.error("Performance guard needs a finite nonnegative limit, >=3 repeats and >=1 warmup")
    args.out.mkdir(parents=True, exist_ok=True)
    scene = json.loads(args.scenario.read_text())
    scene.setdefault("measurement_probes", [])
    scene.pop("failure_sample_times_s", None)
    if args.duration is not None:
        scene["simulation"]["duration_s"] = args.duration
    scene_path = (args.out / "scenario.json").resolve()
    if scene_path == args.scenario.resolve():
        raise ValueError("Output directory must not overwrite the source scenario")
    scene_path.write_text(json.dumps(scene) + "\n")
    request = {"stop_after_first_rupture": False}
    if args.restart:
        request.update(restart_from=str(args.restart.resolve()), restart_contract=args.contract)
    samples = {"before": [], "after": []}
    delta = []
    reference = None
    for iteration in range(args.warmups + args.repeats):
        warm = iteration < args.warmups
        index = iteration - args.warmups
        order = ("before", "after") if iteration % 2 == 0 else ("after", "before")
        for label in order:
            suffix = f"-warmup-{iteration}" if warm else ("" if index == 0 else f"-{index}")
            result = run(getattr(args, label), scene_path, request, args.out / f"{label}{suffix}.json")
            physical = physical_result(result)
            if reference is None:
                reference = physical
            delta.extend(differences(reference, physical, f"{label}{suffix}"))
            if result["advanced_to_s"] + 1e-6 < scene["simulation"]["duration_s"]:
                delta.append(f"{label}{suffix}: incomplete segment")
            if not warm:
                samples[label].append(result)
    before, after = samples["before"][0], samples["after"][0]
    performance = performance_summary(samples["before"], samples["after"],
                                      args.max_slowdown, args.max_timing_noise)
    performance_ok = performance["status"] in ("passed", "not_enforced")
    summary = dict(passed=not delta and performance_ok, physics_passed=not delta,
                   performance=performance, source_scenario=str(args.scenario.resolve()),
                   source_sha256=digest(args.scenario), before_sha256=digest(args.before),
                   after_sha256=digest(args.after), steps=after["steps"],
                   advanced_steps=after["steps"] - after["steps_at_start"],
                   started_at_s=after["started_at_s"], advanced_to_s=after["advanced_to_s"],
                   differences=delta[:30], difference_count=len(delta))
    (args.out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if delta:
        raise SystemExit(1)
    if not performance_ok:
        raise SystemExit(2 if performance["status"] == "inconclusive_noise" else 1)


if __name__ == "__main__":
    main()

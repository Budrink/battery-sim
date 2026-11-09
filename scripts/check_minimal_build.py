#!/usr/bin/env python3
"""Check an actually compiled core/server preset, not only its requested options (Linux).

Inspect compile commands for UI/tool source leakage and ldd for graphics dependencies.
This does not start CUDA or a window. Run after building the selected preset.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def check(build, mode):
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        match = re.match(r"([^:#/][^:]*):[^=]+=(.*)", line)
        if match:
            cache[match[1]] = match[2]
    expected = {"BATTERY_SIM_BUILD_VIEWER": "OFF", "BATTERY_SIM_BUILD_TOOLS": "OFF",
                "BUILD_TESTING": "OFF",
                "BATTERY_SIM_BUILD_RUNTIME": "OFF" if mode == "core" else "ON"}
    for name, value in expected.items():
        if cache.get(name) != value:
            raise RuntimeError(f"{name} must be {value}, got {cache.get(name)}")
    commands = json.loads((build / "compile_commands.json").read_text())
    if not commands:
        raise RuntimeError("Empty compilation database")
    allowed = ["native/core/"]
    if mode == "server":
        allowed += ["native/run/", "native/job/", "native/apps/headless/",
                    "native/apps/worker/", "native/tools/contracts/"]
    for command in commands:
        path = Path(command["file"])
        if not path.is_absolute():
            path = Path(command["directory"]) / path
        source = path.resolve().relative_to(ROOT).as_posix()
        if not any(source.startswith(prefix) for prefix in allowed):
            raise RuntimeError(f"Unexpected source in {mode}: {source}")
        flags = command.get("command", " ".join(command.get("arguments", [])))
        for forbidden in ("native/ui/", "native/visualization/", "imgui-src", "glfw-src", "implot-src", "glm-src"):
            if forbidden in flags:
                raise RuntimeError(f"Graphics include leakage: {source}: {forbidden}")
    if not (build / "libbattery_native_core.a").is_file():
        raise RuntimeError("Core library has not been built")
    dependencies = {}
    if mode == "server":
        for name in ("battery_sim_headless", "battery_sim_worker", "battery_sim_compile_experiment"):
            binary = build / name
            result = subprocess.run(["ldd", str(binary)], text=True, capture_output=True, check=True)
            if re.search(r"lib(?:GL|EGL|glfw|X11|wayland|SDL|Qt)", result.stdout):
                raise RuntimeError(f"Graphics dependency in {name}: {result.stdout}")
            if "not found" in result.stdout:
                raise RuntimeError(f"Missing dependency in {name}: {result.stdout}")
            dependencies[name] = result.stdout.splitlines()
    return dict(passed=True, mode=mode, build=str(build), compilation_units=len(commands),
                allowed_sources=allowed, dynamic_dependencies=dependencies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--mode", choices=("core", "server"), required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.build.resolve(), args.mode), indent=2))

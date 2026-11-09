#!/usr/bin/env python3
"""Import the archived FSRI30 run without regenerating its fitted gas graph.

This is a fixture-specific, checked conversion, not a generic scenario importer.
Native per-tile vent routes stay in the explicit gas graph: collapsing their
different internal donors into a model-level single vent would change physics.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/development/fsri30-wall-delay-full-20261027-r1/scenario.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shell-mm", type=float, default=0.2)
    parser.add_argument("--wrap-heater", action="store_true", help="Explicitly replace the legacy B10 plate by a wrap")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        raise ValueError("Refusing to overwrite an existing file")
    if not 0 < args.shell_mm < 1:
        raise ValueError("Shell thickness must be between 0 and 1 mm")
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    assert source["primary_failures"] == [], "Do not import an imposed ignition"
    thickness = args.shell_mm / 1000
    scene = copy.deepcopy(source)
    # Contract object IDs disallow ':'. Rename only the six passive case IDs and
    # every exact/prefixed reference; thermal/gas topology otherwise stays intact.
    id_map = {f"ABS:{sign}{axis}": f"ABS-{'plus' if sign == '+' else 'minus'}{axis}"
              for axis in "xyz" for sign in "-+"}
    def rename(value):
        if isinstance(value, str):
            for old, new in id_map.items():
                if value == old or value.startswith(old + ":"):
                    return new + value[len(old):]
            return value
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {rename(key): rename(item) for key, item in value.items()}
        return value
    scene = rename(scene)
    previous = {}
    for shell in scene["shells"]:
        previous[shell["contents_id"]] = shell["thickness_m"]
        shell["thickness_m"] = thickness
        for face in shell.get("face_materials", {}).values():
            if "thickness_m" in face:
                face["thickness_m"] = thickness
    # Explicit shell passage lengths represent the same wall thickness.
    for channel in scene["gas_network"]["channels"]:
        if channel["kind"] != "rupture":
            continue
        battery = channel.get("port_id", "").split(":")[0]
        if battery in previous and "mixing_length_m" in channel:
            assert math.isclose(channel["mixing_length_m"], previous[battery])
            channel["mixing_length_m"] = thickness
    provenance = dict(status="mixed", notes=(
        "Archived fitted FSRI30 candidate, not independently validated. "
        f"Only shell thickness and corresponding vent path length changed to {args.shell_mm:g} mm. "
        "Fitted heater, R1/R2/R3, inventory, contacts and explicit gas routes retained."),
        sources=[str(SOURCE.relative_to(ROOT)), source["source"]])
    experiment = dict(format="battery-sim.experiment", version=1,
        id="fsri30-shell-020mm", name="FSRI30 / 30 cylinders / shell 0.20 mm",
        provenance=provenance, battery_models=[],
        assembly=dict(batteries=[], passive_cells=[], ports=scene["ports"],
                      interfaces=scene["interfaces"], gas_network=scene["gas_network"], vent_connections=[]),
        conditions=dict(boundaries=scene["boundaries"], heating_protocols=scene["heating_protocols"],
                        electrical_drives=[]),
        numerics=dict(simulation=scene["simulation"], batteries={}), measurements=[],
        metadata=copy.deepcopy(scene["metadata"]))
    models = {}
    shells = {s["contents_id"]: s for s in scene["shells"]}
    for cell in scene["cells"]:
        if cell["role"] != "battery":
            experiment["assembly"]["passive_cells"].append(cell)
            continue
        bid = cell["id"]
        thermal = copy.deepcopy(cell["thermal"])
        conversion = copy.deepcopy(cell["conversion"])
        shell = copy.deepcopy(shells[bid])
        state = dict(temperature_k=thermal.pop("initial_temperature_k"), damage_fraction=0,
                     aging_fraction=0, precursor_fraction=conversion.pop("initial_precursor_fraction", 0),
                     charged_fraction=conversion.pop("initial_charged_fraction", 0),
                     shell_temperature_k=shell.pop("initial_temperature_k"))
        grid = dict(divisions=thermal.pop("divisions"),
                    internal_heat_transfer=thermal.pop("internal_heat_transfer", "finite_volume"))
        grid["conversion"] = {k: conversion.pop(k) for k in
                              ("max_fraction_per_step", "max_temperature_step_k") if k in conversion}
        gas_id = conversion.pop("gas_compartment")
        shell.pop("contents_id")
        dx, dy, dz = cell["geometry"]["dimensions_m"]
        geometry = dict(kind="cylinder", dimensions_m=[math.sqrt(dx * dy / math.pi), dz])
        grid["geometry_approximation"] = dict(geometry=cell["geometry"], reason=
            "Archived equal-volume prismatic thermal mesh; cylinder rendering. Gas/contact routes retained.")
        model = dict(format="battery-sim.battery", version=1, revision="1", provenance=provenance,
                     geometry=geometry, contents=dict(material_id=cell["material_id"], thermal=thermal,
                     conversion=conversion), shell=shell, vents=[])
        signature = json.dumps(model, sort_keys=True)
        if signature not in models:
            model["id"] = f"fsri30-cylinder-{len(models) + 1}"
            models[signature] = model["id"]
            experiment["battery_models"].append(model)
        pose = copy.deepcopy(cell["pose"])
        pose.setdefault("orientation_xyzw", [0, 0, 0, 1])
        experiment["assembly"]["batteries"].append(dict(id=bid, model_id=models[signature],
            model_revision="1", pose=pose, initial_state=state, gas_compartment=gas_id))
        experiment["numerics"]["batteries"][bid] = grid
    assert len(experiment["assembly"]["batteries"]) == 30
    experiment["metadata"]["editor_import"] = dict(
        source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        shell_thickness_before_m=previous, shell_thickness_after_m=thickness,
        case_id_mapping=id_map,
        scope="Geometry variant; historical trajectory is not a validation of this thickness/current core.",
        explicit_routes="Per-tile vents preserved in gas_network, not collapsed into single-donor model vents.",
        archived_observers={k: v for k, v in scene.items() if k not in {
            "name", "source", "metadata", "simulation", "cells", "shells", "ports", "interfaces",
            "gas_network", "boundaries", "heating_protocols", "primary_failures"}})
    compiler = ROOT / "build/dev/battery_sim_compile_experiment"
    with tempfile.TemporaryDirectory(prefix="fsri30-contract-") as temporary:
        candidate = Path(temporary) / "experiment.json"
        candidate.write_text(json.dumps(experiment))
        result = subprocess.run([str(compiler), str(candidate)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        upgraded = subprocess.run([str(compiler), str(candidate), "--upgrade-v2"], capture_output=True,
                                  text=True, check=True)
        if args.wrap_heater:
            wrapped = subprocess.run([str(compiler), str(candidate), "--mount-heater", "heater", "B10", "wrap"],
                                     capture_output=True, text=True, check=True)
    compiled = json.loads(result.stdout)
    # Assert exact retained physics (apart from explicit default fields introduced by the contract).
    for key in ("gas_network", "ports", "interfaces", "boundaries", "heating_protocols", "simulation", "shells"):
        assert compiled[key] == scene[key], f"Unexpected change in {key}"
    for before, after in zip(scene["cells"], compiled["cells"]):
        for key, value in before.items():
            if key == "thermal":
                assert all(after[key][k] == v for k, v in value.items())
            elif key == "pose":
                assert after[key]["position_m"] == value["position_m"]
            else:
                assert after[key] == value, (before["id"], key)
    # Create only after validation; never overwrite source, working 4x4 or an existing variant.
    with target.open("x") as stream:
        stream.write(wrapped.stdout if args.wrap_heater else upgraded.stdout)
    print(json.dumps(dict(file=str(target), batteries=30, models=len(models),
        shell_mm=args.shell_mm, gas_volumes=len(scene["gas_network"]["compartments"]),
        channels=len(scene["gas_network"]["channels"]), retained_physics_check="passed"), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deterministic report-inspired module. Thermal priors are NOT calibrated."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_scenario() -> dict:
    evidence = json.loads((ROOT / "data/manifests/narada-52-public-case.json").read_text())
    spec = evidence["published"]["specification"]
    nx, ny = 4, 13
    dims = [v / 1000 for v in spec["cell_dimensions_mm"]]
    gaps = [0.010, 0.002]
    pitch = [dims[i] + gaps[i] for i in range(2)]
    cells, ports, edges, boundaries = [], [], [], []

    def cid(x, y):
        return f"r{x+1}-c{y+1:02d}"

    def pos(x, y):
        return [round((x-(nx-1)/2)*pitch[0], 9),
                round((y-(ny-1)/2)*pitch[1], 9), dims[2]/2]

    for x in range(nx):
        for y in range(ny):
            name, p = cid(x, y), pos(x, y)
            cells.append({
                "id": name, "geometry": {"kind": "prismatic", "dimensions_m": dims},
                "pose": {"position_m": p},
                "thermal": {"heat_capacity_j_per_k": 5500.0, "initial_temperature_k": 298.15,
                    "divisions": [8, 4, 8], "conductivity_w_per_m_k": [1.0, 1.0, 1.0]},
                "degradation": {"aging_fraction": 0.0, "initial_damage": 0.0,
                    "onset_k": 465.0, "rate_at_onset_per_s": 0.025,
                    "temperature_scale_k": 30.0, "aging_damage_multiplier": 1.0},
                "runaway": {"energy_j": 4000000.0, "duration_s": 60.0}})
            ports.append({
                "id": f"{name}-ambient", "cell_id": name, "kind": "cooling",
                "surface": "+z", "area_m2": dims[0]*dims[1],
                "centroid_m": [p[0], p[1], dims[2]], "normal": [0, 0, 1]})
            boundaries.append({"id": f"{name}-sink", "port_id": f"{name}-ambient",
                "temperature_k": 298.15, "conductance_w_per_k": 0.8})

    base_law = {
        "kind": "first_order", "l0_m": 0.00045, "l_low_m": 0.00010,
        "l_high_m": 0.001, "transition_temperature_k": 425.0,
        "transition_width_k": 15.0, "hysteresis_k": 15.0, "relaxation_time_s": 2.0,
        "activation_width_m": 0.00004,
        "damage_onset_k": 680.0, "damage_rate_at_onset_per_s": 0.0002,
        "damage_temperature_scale_k": 45.0, "damage_l_shift_m": -0.00015,
        "damage_conductance_factor_at_one": 0.6, "destruction_damage": 0.9,
        "destroyed_l0_m": 0.0009, "destroyed_l_m": 0.00005,
        "destroyed_active_conductance_w_per_k": 0.5}
    for x in range(nx):
        for y in range(ny):
            for axis, dx, dy in [(0, 1, 0), (1, 0, 1)]:
                if x+dx >= nx or y+dy >= ny:
                    continue
                a, b = cid(x, y), cid(x+dx, y+dy)
                pa, pb = pos(x, y), pos(x+dx, y+dy)
                pa[axis] += dims[axis]/2
                pb[axis] -= dims[axis]/2
                broad = axis == 1
                # Square patches stay within both facing surfaces.
                area = 0.0196 if broad else 0.0025
                for name, point, sign in [(a, pa, 1), (b, pb, -1)]:
                    normal = [0, 0, 0]
                    normal[axis] = sign
                    ports.append({
                        "id": f"{a}-{b}-{name}", "cell_id": name, "kind": "contact",
                        "surface": ("+" if sign > 0 else "-") + "xyz"[axis],
                        "area_m2": area, "centroid_m": [round(v, 9) for v in point],
                        "normal": normal})
                law = copy.deepcopy(base_law)
                law.update({
                    "background_conductance_w_per_k": 4.0 if broad else 0.5,
                    "active_conductance_w_per_k": 24.0 if broad else 7.0,
                    "destroyed_background_conductance_w_per_k": 0.4 if broad else 0.1})
                edges.append({
                    "id": f"{a}--{b}", "port_a": f"{a}-{b}-{a}", "port_b": f"{a}-{b}-{b}",
                    "material_a": "cell-case-unspecified", "material_b": "cell-case-unspecified",
                    "interface_material": "assumed-face-pad" if broad else "assumed-row-bridge",
                    "initial_l_m": law["l_low_m"], "law": law})
    return {
        "$schema": "../schemas/scenario-v0.schema.json", "name": "narada-52-schematic",
        "source": evidence["dataset_id"],
        "metadata": {
            "experiment_id": evidence["experiment_id"], "adapter": evidence["adapter"],
            "split": evidence["split"], "calibration_status": "uncalibrated",
            "geometry_status": "assumed 4x13 layout",
            "source_manifest": "data/manifests/narada-52-public-case.json",
            "sources": evidence["sources"], "reported": evidence["published"],
            "applicability_note": evidence["applicability_note"], "soc": None, "soh": None,
            "assumptions": {
                "layout": "4x13; axes x=174mm width, y=72mm thickness, z=207mm height",
                "row_gap_m": gaps[0], "in_row_gap_m": gaps[1],
                "trigger": "r2-c07 at 5s; experimental position/time unknown; no heater ramp",
                "initial_temperature_k": 298.15, "specific_heat_j_per_kg_k": 1000.0,
                "heat_capacity": "Published family mass 5.5kg x assumed 1000 J/kg/K",
                "spatial_model": "CUDA finite volumes: 8x4x8 per battery; mesh size is NOT l0",
                "bulk_conductivity": "Isotropic 1 W/m/K placeholder, NOT a measured material property",
                "runaway": "4MJ into cell over 60s: independent prior, NOT electrical energy 896Wh",
                "cell_damage": "465K onset, all damage kinetics provisional",
                "interfaces": "87 assumed nearest-face paths, including 39 cross-row bridges",
                "connectivity": "All l/l0, transition and damage constants are provisional",
                "l0": "Order-parameter threshold, NOT the geometric gap",
                "aging": "Zero initial aging/damage for exploration; experimental SOH unknown",
                "cooling": "0.8W/K per cell to 298.15K; effective boundary, no coolant model",
                "omitted": "Enclosure, busbars, other modules, gas, radiation, electrical network"},
            "unavailable": evidence["unavailable"],
            "reported_outcome": {"primary_cells": 1, "secondary_cells": 51,
                                 "no_inter_module_propagation": True},
            "comparison_note": "No validation verdict: schematic geometry/priors are uncalibrated."},
        "simulation": {"duration_s": 2400.0, "max_dt_s": 0.2,
                       "output_interval_s": 2.0, "stability_factor": 0.2},
        "cells": cells, "ports": ports, "interfaces": edges, "boundaries": boundaries,
        "primary_failures": [{"cell_id": "r2-c07", "time_s": 5.0}],
        "observations": {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/narada-52-schematic.json")
    args = parser.parse_args()
    raw = make_scenario()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(raw['cells'])} cells, {len(raw['interfaces'])} interfaces")


if __name__ == "__main__":
    main()

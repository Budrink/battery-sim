#!/usr/bin/env python3
"""Eight-zone batteries and resolved pads; numerical-reference fit, NOT experimental validation."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]

def make_scenario(calibrated: bool = True) -> dict:
    original = runpy.run_path(str(ROOT / "scripts/generate_narada_52.py"))["make_scenario"]()
    raw = copy.deepcopy(original)
    raw["name"] = "narada-52-multiscale"
    raw["simulation"].update(duration_s=6000.0, step_interval_s=1.0)
    old_ports = {p["id"]: p for p in original["ports"]}
    raw["ports"] = [p for p in raw["ports"] if p["kind"] == "cooling"]
    raw["interfaces"] = []
    for cell in raw["cells"]:
        cell.update(role="battery", material_id="battery-fast-zones-prior")
        cell["thermal"].update(divisions=[2, 2, 2], conductivity_w_per_m_k=[1000.0]*3)
    for old in original["interfaces"]:
        pa, pb = old_ports[old["port_a"]], old_ports[old["port_b"]]
        axis = next(i for i, n in enumerate(pa["normal"]) if n)
        center = [(a+b)/2 for a, b in zip(pa["centroid_m"], pb["centroid_m"])]
        gap = abs(pb["centroid_m"][axis]-pa["centroid_m"][axis])
        side = math.sqrt(pa["area_m2"])
        dims = [side]*3
        dims[axis] = round(gap, 9)
        divisions = [14, 2, 14] if axis == 1 else [4, 10, 10]
        pad_id = "pad-" + old["id"]
        # Each pad is a heat-storing solid, not an extra parallel edge.
        # These priors are illustrative model inputs, NOT a claimed material specification.
        bulk = copy.deepcopy(old["law"])
        for key in list(bulk):
            if "conductance_w_per_k" in key:
                del bulk[key]
        bulk.update(background_factor=1.0, active_factor=9.0,
                    destroyed_background_factor=12.0, destroyed_active_factor=0.0,
                    damage_conductance_factor_at_one=1.0, endpoint_damage_weight=1.0)
        raw["cells"].append({
            "id": pad_id, "role": "passive", "material_id": "interlayer-broad-prior" if axis == 1 else "interlayer-bridge-prior",
            "geometry": {"kind": "prismatic", "dimensions_m": dims},
            "pose": {"position_m": [round(x, 9) for x in center]},
            "thermal": {"heat_capacity_j_per_k": math.prod(dims)*1.2e6,
                        "initial_temperature_k": 298.15, "divisions": divisions,
                        "conductivity_w_per_m_k": [0.2]*3, "bulk_connectivity": bulk},
            "degradation": {"onset_k": 680.0, "rate_at_onset_per_s": 0.0002,
                            "temperature_scale_k": 45.0, "aging_damage_multiplier": 1.0},
            "runaway": {"energy_j": 0.0, "duration_s": 1.0}})
        for face, battery_port in enumerate((pa, pb)):
            port = copy.deepcopy(battery_port)
            port["id"] = f"{pad_id}-battery-{face}"
            pad_port = copy.deepcopy(port)
            pad_port.update(id=f"{pad_id}-face-{face}", cell_id=pad_id,
                            surface=("-" if face == 0 else "+") + "xyz"[axis],
                            normal=[-v for v in port["normal"]])
            raw["ports"].extend((port, pad_port))
            law = copy.deepcopy(old["law"])
            # Split the old EFFECTIVE contact prior across two contacts in series;
            # the new material resistance is additional, not fitted away.
            for key in list(law):
                if "conductance_w_per_k" in key:
                    law[key] *= 2
            law["endpoint_damage_weight"] = 1.0
            raw["interfaces"].append({
                "id": f"{pad_id}-contact-{face}", "port_a": port["id"], "port_b": pad_port["id"],
                "material_a": "battery-fast-zones-prior", "material_b": raw["cells"][-1]["material_id"],
                "interface_material": "battery-interlayer-contact-prior", "law": law,
                "initial_l_m": law["l_low_m"]})
    assumptions = raw["metadata"]["assumptions"]
    assumptions.update(
        spatial_model="52 batteries x 8 zones; 87 separately meshed passive interlayers; no direct battery-battery bypass",
        bulk_conductivity="Battery zones: 1000 W/m/K fast-equilibration PRIOR, not measured conductivity; pad baseline: 0.2 W/m/K",
        interlayer_capacity="Volumetric heat capacity 1.2 MJ/m3/K, provisional; effective solid filling each assumed contact gap",
        interlayer_damage="Phase-dependent bulk factors 1/10, destroyed factor 12; contact loss can reduce transfer independently",
        interfaces="174 battery-to-pad contacts replacing all 87 direct paths; old contact resistance split in two, pad resistance added",
        endpoint_damage="Contact/bulk damage is at least the weighted mean adjacent-volume damage, irreversible; weight=1 is a prior",
        time_integration="1 s outer intervals; adaptive global substeps capped by max_dt, conservative CFL, scheduled events, output and source exhaustion",
        lateral_pad_surfaces="Adiabatic; no implicit ambient sink or heat transfer through empty space")
    raw["metadata"]["geometry_status"] = "assumed 4x13; 87 explicit schematic interlayers"
    raw["metadata"]["comparison_note"] = "Not calibrated or validated against experiment; old effective parameters are not spatial material measurements."
    if calibrated:
        apply_reference_fit(raw)
    return raw

def apply_reference_fit(raw: dict) -> None:
    profile_path = ROOT / "data/calibration/narada-reference-fit.json"
    p = json.loads(profile_path.read_text())["parameters"]
    raw["simulation"]["duration_s"] = 2400.0
    for cell in raw["cells"]:
        cell["material_id"] = cell["material_id"].replace("-prior", "-reference-fit")
        if cell["role"] == "battery":
            continue  # C, energy, kinetics, eight zones and finite internal k are unchanged.
        broad = "broad" in cell["material_id"]
        k = [0.2]*3
        k[1 if broad else 0] = p["kb" if broad else "kx"]
        cell["thermal"]["conductivity_w_per_m_k"] = k
        cell["thermal"]["heat_capacity_j_per_k"] *= p["capacity"]
        # The old lumped model does not identify an additional bulk phase law.
        # Keep the nonlinear l/l0 response at both actual surface contacts.
        cell["thermal"].pop("bulk_connectivity")
    for edge in raw["interfaces"]:
        broad = "broad" in edge["material_b"]
        for key in ("material_a", "material_b", "interface_material"):
            edge[key] = edge[key].replace("-prior", "-reference-fit")
        law = edge["law"]
        law["background_conductance_w_per_k"] *= p["coldb" if broad else "coldx"]
        law["active_conductance_w_per_k"] *= p["gb" if broad else "gx"]
        law["transition_temperature_k"] = p["tb" if broad else "tx"]
        law["damage_onset_k"] = p["damageb" if broad else "damagex"]
        law["endpoint_damage_weight"] = p["endpoint"]
    raw["metadata"].update(
        calibration_status="numerical-reference-fit",
        calibration_profile="data/calibration/narada-reference-fit.json",
        comparison_note="Fitted to historical lumped-model ignition times, NOT measured experimental histories.")
    raw["metadata"]["assumptions"].update(
        bulk_conductivity="Battery k=1000; pad normal k=5 (broad) / 50 (bridge), tangential k=0.2 W/m/K; numerical surrogate parameters, not measured materials",
        interlayer_damage="Passive material damage accumulates; bulk conductivity is held constant in this fit. Contact l/l0/phase/damage remain dynamic.",
        interfaces="174 surface contacts; separate background/active conductances and local transition/damage thresholds fitted to old lumped timings",
        endpoint_damage="Weight=0 for reference reproduction: old model did not couple forced battery destruction directly into contact damage")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/narada-52-multiscale.json")
    parser.add_argument("--prior", action="store_true", help="Generate the original uncalibrated multiscale priors")
    args = parser.parse_args()
    raw = make_scenario(calibrated=not args.prior)
    args.output.write_text(json.dumps(raw, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"{args.output}: 52 batteries, 87 passive pads, {len(raw['interfaces'])} contacts")

if __name__ == "__main__":
    main()

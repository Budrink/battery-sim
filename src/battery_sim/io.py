from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import (
    AmbientBoundary,
    CellGeometry,
    CellParameters,
    ConnectivityLaw,
    EdgePhase,
    FailureEvent,
    Interface,
    ModelError,
    Observation,
    Port,
    PortKind,
    Pose,
    Scenario,
    ShapeKind,
    SimulationSettings,
    TransitionKind,
    Vec3,
)


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelError(f"{where} must be an object")
    return value


def _items(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelError(f"{where} must be an array")
    return value


def _enum(enum_type: type, value: Any, where: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ModelError(f"{where} must be one of: {choices}") from exc


def _number(data: dict[str, Any], key: str, where: str, default: float | None = None) -> float:
    if key not in data:
        if default is None:
            raise ModelError(f"{where}.{key} is required")
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{where}.{key} must be a number")
    return float(value)


def _string(data: dict[str, Any], key: str, where: str, default: str | None = None) -> str:
    if key not in data:
        if default is None:
            raise ModelError(f"{where}.{key} is required")
        return default
    value = data[key]
    if not isinstance(value, str):
        raise ModelError(f"{where}.{key} must be a string")
    return value


def _vec(data: dict[str, Any], key: str, where: str, default: list[float] | None = None) -> Vec3:
    value = data.get(key, default)
    if not isinstance(value, (list, tuple)):
        raise ModelError(f"{where}.{key} must be a three-element array")
    return Vec3.from_value(value)


def _parse_cell(raw: Any, index: int) -> CellParameters:
    where = f"cells[{index}]"
    data = _mapping(raw, where)
    if data.get("role", "battery") != "battery":
        raise ModelError("The historical lumped Python solver does not support passive multiscale solids; use the native CUDA solver")
    geometry_data = _mapping(data.get("geometry"), f"{where}.geometry")
    pose_data = _mapping(data.get("pose", {}), f"{where}.pose")
    thermal = _mapping(data.get("thermal"), f"{where}.thermal")
    if "heat_capacity_law" in thermal or "bulk_response" in thermal:
        raise ModelError("State-dependent node capacity and internal connectivity require the native CUDA solver")
    degradation = _mapping(data.get("degradation", {}), f"{where}.degradation")
    runaway = _mapping(data.get("runaway"), f"{where}.runaway")
    if runaway.get("profile", "constant_power") != "constant_power" or "fast_fraction" in runaway or "tail_time_s" in runaway or "kinetics" in runaway or "energy_outlet" in runaway or "reaction_structure" in thermal or thermal.get("bulk_destruction_on_boundary") or "bulk_failure_transient" in thermal:
        raise ModelError("Runaway pulse and destroyed bulk boundary state require the native CUDA solver")

    kind = _enum(ShapeKind, geometry_data.get("kind"), f"{where}.geometry.kind")
    dimensions = geometry_data.get("dimensions_m")
    if not isinstance(dimensions, list):
        raise ModelError(f"{where}.geometry.dimensions_m must be an array")
    orientation = pose_data.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0])
    if not isinstance(orientation, list) or len(orientation) != 4:
        raise ModelError(f"{where}.pose.orientation_xyzw must contain four numbers")

    return CellParameters(
        cell_id=_string(data, "id", where),
        geometry=CellGeometry(kind=kind, dimensions_m=tuple(float(v) for v in dimensions)),
        pose=Pose(
            position_m=_vec(pose_data, "position_m", f"{where}.pose", [0.0, 0.0, 0.0]),
            orientation_xyzw=tuple(float(v) for v in orientation),
        ),
        heat_capacity_j_per_k=_number(thermal, "heat_capacity_j_per_k", f"{where}.thermal"),
        initial_temperature_k=_number(
            thermal, "initial_temperature_k", f"{where}.thermal", 298.15
        ),
        aging_fraction=_number(degradation, "aging_fraction", f"{where}.degradation", 0.0),
        initial_damage=_number(degradation, "initial_damage", f"{where}.degradation", 0.0),
        damage_onset_k=_number(degradation, "onset_k", f"{where}.degradation", 420.0),
        damage_rate_at_onset_per_s=_number(
            degradation, "rate_at_onset_per_s", f"{where}.degradation", 0.0
        ),
        damage_temperature_scale_k=_number(
            degradation, "temperature_scale_k", f"{where}.degradation", 25.0
        ),
        aging_damage_multiplier=_number(
            degradation, "aging_damage_multiplier", f"{where}.degradation", 0.0
        ),
        runaway_energy_j=_number(runaway, "energy_j", f"{where}.runaway"),
        runaway_duration_s=_number(runaway, "duration_s", f"{where}.runaway"),
    )


def _parse_port(raw: Any, index: int) -> Port:
    where = f"ports[{index}]"
    data = _mapping(raw, where)
    return Port(
        port_id=_string(data, "id", where),
        cell_id=_string(data, "cell_id", where),
        kind=_enum(PortKind, data.get("kind", "contact"), f"{where}.kind"),
        surface=_string(data, "surface", where, "unspecified"),
        area_m2=_number(data, "area_m2", where),
        centroid_m=_vec(data, "centroid_m", where),
        normal=_vec(data, "normal", where),
    )


def _parse_law(raw: Any, where: str) -> ConnectivityLaw:
    data = _mapping(raw, where)
    background = _number(data, "background_conductance_w_per_k", where, 0.0)
    active = _number(data, "active_conductance_w_per_k", where, 0.0)
    l0 = _number(data, "l0_m", where)
    low = _number(data, "l_low_m", where, l0)
    return ConnectivityLaw(
        kind=_enum(TransitionKind, data.get("kind", "constant"), f"{where}.kind"),
        l0_m=l0,
        l_low_m=low,
        l_high_m=_number(data, "l_high_m", where, low),
        transition_temperature_k=_number(data, "transition_temperature_k", where, 400.0),
        transition_width_k=_number(data, "transition_width_k", where, 10.0),
        hysteresis_k=_number(data, "hysteresis_k", where, 0.0),
        relaxation_time_s=_number(data, "relaxation_time_s", where, 0.0),
        background_conductance_w_per_k=background,
        active_conductance_w_per_k=active,
        activation_width_m=_number(data, "activation_width_m", where, 1.0e-5),
        damage_onset_k=_number(data, "damage_onset_k", where, 450.0),
        damage_rate_at_onset_per_s=_number(data, "damage_rate_at_onset_per_s", where, 0.0),
        damage_temperature_scale_k=_number(data, "damage_temperature_scale_k", where, 25.0),
        damage_l_shift_m=_number(data, "damage_l_shift_m", where, 0.0),
        damage_conductance_factor_at_one=_number(
            data, "damage_conductance_factor_at_one", where, 1.0
        ),
        destruction_damage=_number(data, "destruction_damage", where, 1.0),
        destroyed_l0_m=_number(data, "destroyed_l0_m", where, l0),
        destroyed_l_m=_number(data, "destroyed_l_m", where, low),
        destroyed_background_conductance_w_per_k=_number(
            data, "destroyed_background_conductance_w_per_k", where, background
        ),
        destroyed_active_conductance_w_per_k=_number(
            data, "destroyed_active_conductance_w_per_k", where, active
        ),
    )


def _parse_interface(raw: Any, index: int) -> Interface:
    where = f"interfaces[{index}]"
    data = _mapping(raw, where)
    law = _parse_law(data.get("law"), f"{where}.law")
    return Interface(
        interface_id=_string(data, "id", where),
        port_a=_string(data, "port_a", where),
        port_b=_string(data, "port_b", where),
        material_a=_string(data, "material_a", where, "unspecified"),
        material_b=_string(data, "material_b", where, "unspecified"),
        interface_material=_string(data, "interface_material", where, "unspecified"),
        law=law,
        initial_l_m=_number(data, "initial_l_m", where, law.l_low_m),
        initial_damage=_number(data, "initial_damage", where, 0.0),
        initial_phase=_enum(EdgePhase, data.get("initial_phase", "low"), f"{where}.initial_phase"),
    )


def scenario_from_dict(raw: Any) -> Scenario:
    data = _mapping(raw, "scenario")
    if "shells" in data or "gas_network" in data or any("conversion" in c for c in data.get("cells", [])):
        raise ModelError("Physical casing, conversion and gas networks require the native CUDA solver")
    simulation = _mapping(data.get("simulation"), "simulation")
    observation_data = _mapping(data.get("observations", {}), "observations")
    times = observation_data.get("runaway_times_s", {})
    if not isinstance(times, dict):
        raise ModelError("observations.runaway_times_s must be an object")
    order = observation_data.get("runaway_order", [])
    if not isinstance(order, list) or any(not isinstance(value, str) for value in order):
        raise ModelError("observations.runaway_order must be an array of cell ids")
    propagated = observation_data.get("propagated")
    if propagated is not None and not isinstance(propagated, bool):
        raise ModelError("observations.propagated must be boolean or null")

    boundaries: list[AmbientBoundary] = []
    for index, raw_boundary in enumerate(_items(data.get("boundaries", []), "boundaries")):
        where = f"boundaries[{index}]"
        boundary = _mapping(raw_boundary, where)
        if boundary.get("temperature_profile"):
            raise ModelError("Measured temperature profiles require the native CUDA solver; the historical Python reference cannot replay them")
        boundaries.append(
            AmbientBoundary(
                boundary_id=_string(boundary, "id", where),
                port_id=_string(boundary, "port_id", where),
                temperature_k=_number(boundary, "temperature_k", where),
                conductance_w_per_k=_number(boundary, "conductance_w_per_k", where),
            )
        )

    failures: list[FailureEvent] = []
    for index, raw_failure in enumerate(
        _items(data.get("primary_failures", []), "primary_failures")
    ):
        where = f"primary_failures[{index}]"
        failure = _mapping(raw_failure, where)
        failures.append(
            FailureEvent(
                cell_id=_string(failure, "cell_id", where),
                time_s=_number(failure, "time_s", where),
            )
        )

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ModelError("metadata must be an object")
    scenario = Scenario(
        name=_string(data, "name", "scenario"),
        source=_string(data, "source", "scenario", "unspecified"),
        settings=SimulationSettings(
            duration_s=_number(simulation, "duration_s", "simulation"),
            max_dt_s=_number(simulation, "max_dt_s", "simulation"),
            output_interval_s=_number(simulation, "output_interval_s", "simulation"),
            stability_factor=_number(simulation, "stability_factor", "simulation", 0.2),
        ),
        cells=tuple(
            _parse_cell(item, index)
            for index, item in enumerate(_items(data.get("cells"), "cells"))
        ),
        ports=tuple(
            _parse_port(item, index)
            for index, item in enumerate(_items(data.get("ports"), "ports"))
        ),
        interfaces=tuple(
            _parse_interface(item, index)
            for index, item in enumerate(_items(data.get("interfaces", []), "interfaces"))
        ),
        boundaries=tuple(boundaries),
        primary_failures=tuple(failures),
        observation=Observation(
            propagated=propagated,
            runaway_order=tuple(order),
            runaway_times_s={str(key): float(value) for key, value in times.items()},
        ),
        metadata=dict(metadata),
    )
    scenario.validate()
    return scenario


def load_scenario(path: str | Path) -> Scenario:
    scenario_path = Path(path)
    try:
        with scenario_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ModelError(f"invalid JSON in {scenario_path}: {exc}") from exc
    return scenario_from_dict(raw)

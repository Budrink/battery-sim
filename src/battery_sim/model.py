from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


class ModelError(ValueError):
    """Raised when a scenario violates the canonical model contract."""


class ShapeKind(str, Enum):
    CYLINDER = "cylinder"
    PRISMATIC = "prismatic"
    POUCH = "pouch"


class PortKind(str, Enum):
    CONTACT = "contact"
    COOLING = "cooling"
    RADIATION = "radiation"
    VENT = "vent"


class CellMode(str, Enum):
    HEALTHY = "healthy"
    RUNAWAY = "runaway"
    SPENT = "spent"


class TransitionKind(str, Enum):
    CONSTANT = "constant"
    SMOOTH = "smooth"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"


class EdgePhase(str, Enum):
    LOW = "low"
    HIGH = "high"
    DESTROYED = "destroyed"


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    @classmethod
    def from_value(cls, value: list[float] | tuple[float, float, float]) -> "Vec3":
        if len(value) != 3:
            raise ModelError("a 3D vector must contain exactly three values")
        return cls(*(float(v) for v in value))

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        n = self.norm()
        if n <= 0.0:
            raise ModelError("a port normal cannot be the zero vector")
        return Vec3(self.x / n, self.y / n, self.z / n)


@dataclass(frozen=True)
class Pose:
    position_m: Vec3
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.orientation_xyzw) != 4:
            raise ModelError("orientation_xyzw must contain four values")
        n = math.sqrt(sum(v * v for v in self.orientation_xyzw))
        if not math.isfinite(n) or n <= 0.0:
            raise ModelError("orientation quaternion must be finite and non-zero")


@dataclass(frozen=True)
class CellGeometry:
    kind: ShapeKind
    dimensions_m: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = 2 if self.kind is ShapeKind.CYLINDER else 3
        if len(self.dimensions_m) != expected:
            label = "[radius, height]" if expected == 2 else "[x, y, z]"
            raise ModelError(f"{self.kind.value} dimensions must be {label}")
        if any((not math.isfinite(v)) or v <= 0.0 for v in self.dimensions_m):
            raise ModelError("all cell dimensions must be finite and positive")

    @property
    def volume_m3(self) -> float:
        if self.kind is ShapeKind.CYLINDER:
            radius, height = self.dimensions_m
            return math.pi * radius * radius * height
        x, y, z = self.dimensions_m
        return x * y * z


@dataclass(frozen=True)
class CellParameters:
    cell_id: str
    geometry: CellGeometry
    pose: Pose
    heat_capacity_j_per_k: float
    initial_temperature_k: float
    aging_fraction: float
    initial_damage: float
    damage_onset_k: float
    damage_rate_at_onset_per_s: float
    damage_temperature_scale_k: float
    aging_damage_multiplier: float
    runaway_energy_j: float
    runaway_duration_s: float

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ModelError("cell id cannot be empty")
        if self.heat_capacity_j_per_k <= 0.0:
            raise ModelError(f"cell {self.cell_id}: heat capacity must be positive")
        if self.initial_temperature_k <= 0.0:
            raise ModelError(f"cell {self.cell_id}: temperature must be above 0 K")
        if not 0.0 <= self.aging_fraction <= 1.0:
            raise ModelError(f"cell {self.cell_id}: aging_fraction must be in [0, 1]")
        if not 0.0 <= self.initial_damage <= 1.0:
            raise ModelError(f"cell {self.cell_id}: initial_damage must be in [0, 1]")
        if self.damage_rate_at_onset_per_s < 0.0 or self.damage_temperature_scale_k <= 0.0:
            raise ModelError(f"cell {self.cell_id}: invalid damage law")
        if self.aging_damage_multiplier < 0.0:
            raise ModelError(f"cell {self.cell_id}: aging multiplier cannot be negative")
        if self.runaway_energy_j < 0.0 or self.runaway_duration_s <= 0.0:
            raise ModelError(f"cell {self.cell_id}: invalid runaway source")


@dataclass(frozen=True)
class Port:
    port_id: str
    cell_id: str
    kind: PortKind
    surface: str
    area_m2: float
    centroid_m: Vec3
    normal: Vec3

    def __post_init__(self) -> None:
        if not self.port_id or not self.cell_id:
            raise ModelError("port and parent cell ids cannot be empty")
        if self.area_m2 <= 0.0:
            raise ModelError(f"port {self.port_id}: area must be positive")
        object.__setattr__(self, "normal", self.normal.normalized())


@dataclass(frozen=True)
class ConnectivityLaw:
    kind: TransitionKind
    l0_m: float
    l_low_m: float
    l_high_m: float
    transition_temperature_k: float
    transition_width_k: float
    hysteresis_k: float
    relaxation_time_s: float
    background_conductance_w_per_k: float
    active_conductance_w_per_k: float
    activation_width_m: float
    damage_onset_k: float
    damage_rate_at_onset_per_s: float
    damage_temperature_scale_k: float
    damage_l_shift_m: float
    damage_conductance_factor_at_one: float
    destruction_damage: float
    destroyed_l0_m: float
    destroyed_l_m: float
    destroyed_background_conductance_w_per_k: float
    destroyed_active_conductance_w_per_k: float

    def __post_init__(self) -> None:
        lengths = (
            self.l0_m,
            self.l_low_m,
            self.l_high_m,
            self.activation_width_m,
            self.destroyed_l0_m,
            self.destroyed_l_m,
        )
        if any((not math.isfinite(v)) or v < 0.0 for v in lengths):
            raise ModelError("connectivity lengths must be finite and non-negative")
        if self.activation_width_m <= 0.0:
            raise ModelError("activation_width_m must be positive")
        if self.transition_width_k <= 0.0 or self.damage_temperature_scale_k <= 0.0:
            raise ModelError("temperature scales must be positive")
        if self.relaxation_time_s < 0.0:
            raise ModelError("relaxation time cannot be negative")
        conductances = (
            self.background_conductance_w_per_k,
            self.active_conductance_w_per_k,
            self.destroyed_background_conductance_w_per_k,
            self.destroyed_active_conductance_w_per_k,
        )
        if any((not math.isfinite(v)) or v < 0.0 for v in conductances):
            raise ModelError("conductances must be finite and non-negative")
        if self.damage_rate_at_onset_per_s < 0.0:
            raise ModelError("edge damage rate cannot be negative")
        if self.damage_conductance_factor_at_one <= 0.0:
            raise ModelError("damage conductance factor must be positive")
        if not 0.0 <= self.destruction_damage <= 1.0:
            raise ModelError("destruction_damage must be in [0, 1]")


@dataclass(frozen=True)
class Interface:
    interface_id: str
    port_a: str
    port_b: str
    material_a: str
    material_b: str
    interface_material: str
    law: ConnectivityLaw
    initial_l_m: float
    initial_damage: float = 0.0
    initial_phase: EdgePhase = EdgePhase.LOW

    def __post_init__(self) -> None:
        if not self.interface_id or not self.port_a or not self.port_b:
            raise ModelError("interface and endpoint ids cannot be empty")
        if self.port_a == self.port_b:
            raise ModelError(f"interface {self.interface_id}: endpoints must differ")
        if self.initial_l_m < 0.0:
            raise ModelError(f"interface {self.interface_id}: initial l cannot be negative")
        if not 0.0 <= self.initial_damage <= 1.0:
            raise ModelError(f"interface {self.interface_id}: damage must be in [0, 1]")


@dataclass(frozen=True)
class AmbientBoundary:
    boundary_id: str
    port_id: str
    temperature_k: float
    conductance_w_per_k: float

    def __post_init__(self) -> None:
        if self.temperature_k <= 0.0 or self.conductance_w_per_k < 0.0:
            raise ModelError(f"boundary {self.boundary_id}: invalid thermal parameters")


@dataclass(frozen=True)
class FailureEvent:
    cell_id: str
    time_s: float

    def __post_init__(self) -> None:
        if self.time_s < 0.0:
            raise ModelError("failure event time cannot be negative")


@dataclass(frozen=True)
class SimulationSettings:
    duration_s: float
    max_dt_s: float
    output_interval_s: float
    stability_factor: float = 0.2

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0 or self.max_dt_s <= 0.0 or self.output_interval_s <= 0.0:
            raise ModelError("simulation times must be positive")
        if not 0.0 < self.stability_factor <= 0.5:
            raise ModelError("stability_factor must be in (0, 0.5]")


@dataclass(frozen=True)
class Observation:
    propagated: bool | None = None
    runaway_order: tuple[str, ...] = ()
    runaway_times_s: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    source: str
    settings: SimulationSettings
    cells: tuple[CellParameters, ...]
    ports: tuple[Port, ...]
    interfaces: tuple[Interface, ...]
    boundaries: tuple[AmbientBoundary, ...]
    primary_failures: tuple[FailureEvent, ...]
    observation: Observation = Observation()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ModelError("scenario name cannot be empty")
        cell_ids = [c.cell_id for c in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ModelError("cell ids must be unique")
        port_ids = [p.port_id for p in self.ports]
        if len(port_ids) != len(set(port_ids)):
            raise ModelError("port ids must be unique")
        interface_ids = [e.interface_id for e in self.interfaces]
        if len(interface_ids) != len(set(interface_ids)):
            raise ModelError("interface ids must be unique")

        cell_set = set(cell_ids)
        port_map = {p.port_id: p for p in self.ports}
        for port in self.ports:
            if port.cell_id not in cell_set:
                raise ModelError(f"port {port.port_id}: unknown cell {port.cell_id}")
        for interface in self.interfaces:
            if interface.port_a not in port_map or interface.port_b not in port_map:
                raise ModelError(f"interface {interface.interface_id}: unknown port")
            a = port_map[interface.port_a].cell_id
            b = port_map[interface.port_b].cell_id
            if a == b:
                raise ModelError(f"interface {interface.interface_id}: endpoints belong to one cell")
        for boundary in self.boundaries:
            if boundary.port_id not in port_map:
                raise ModelError(f"boundary {boundary.boundary_id}: unknown port")
        for event in self.primary_failures:
            if event.cell_id not in cell_set:
                raise ModelError(f"failure event: unknown cell {event.cell_id}")
            if event.time_s > self.settings.duration_s:
                raise ModelError(f"failure event for {event.cell_id} lies beyond the simulation")


@dataclass
class CellState:
    temperature_k: float
    damage: float
    mode: CellMode = CellMode.HEALTHY
    runaway_started_s: float | None = None
    released_energy_j: float = 0.0


@dataclass
class InterfaceState:
    l0_m: float
    l_m: float
    damage: float
    phase: EdgePhase
    conductance_w_per_k: float = 0.0


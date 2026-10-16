from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .laws import advance_interface_state, conductance, thermal_damage_rate
from .model import CellMode, CellState, EdgePhase, InterfaceState, Scenario


_EPS = 1.0e-10


@dataclass(frozen=True)
class CellSnapshot:
    temperature_k: float
    damage: float
    mode: str
    released_energy_j: float


@dataclass(frozen=True)
class InterfaceSnapshot:
    l0_m: float
    l_m: float
    damage: float
    phase: str
    conductance_w_per_k: float


@dataclass(frozen=True)
class Snapshot:
    time_s: float
    cells: dict[str, CellSnapshot]
    interfaces: dict[str, InterfaceSnapshot]


@dataclass(frozen=True)
class EventRecord:
    time_s: float
    kind: str
    entity_id: str
    detail: str = ""


@dataclass(frozen=True)
class SimulationResult:
    scenario_name: str
    source: str
    snapshots: tuple[Snapshot, ...]
    events: tuple[EventRecord, ...]
    propagated: bool
    runaway_order: tuple[str, ...]
    runaway_times_s: dict[str, float]
    initial_stored_energy_j: float
    final_stored_energy_j: float
    released_energy_j: float
    boundary_energy_out_j: float
    energy_balance_residual_j: float
    validation: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        delta = self.final_stored_energy_j - self.initial_stored_energy_j
        scale = max(abs(delta), abs(self.released_energy_j), 1.0)
        return {
            "scenario": self.scenario_name,
            "source": self.source,
            "propagated": self.propagated,
            "runaway_order": list(self.runaway_order),
            "runaway_times_s": self.runaway_times_s,
            "energy": {
                "initial_stored_j": self.initial_stored_energy_j,
                "final_stored_j": self.final_stored_energy_j,
                "delta_stored_j": delta,
                "runaway_released_j": self.released_energy_j,
                "boundary_energy_out_j": self.boundary_energy_out_j,
                "balance_residual_j": self.energy_balance_residual_j,
                "relative_balance_residual": self.energy_balance_residual_j / scale,
            },
            "validation": self.validation,
        }


class Simulator:
    """Deterministic lumped-cell solver with stateful thermal interfaces."""

    def __init__(self, scenario: Scenario):
        scenario.validate()
        self.scenario = scenario
        self._cells = {cell.cell_id: cell for cell in scenario.cells}
        self._ports = {port.port_id: port for port in scenario.ports}
        self._interfaces = {edge.interface_id: edge for edge in scenario.interfaces}
        self._edge_cells = {
            edge.interface_id: (
                self._ports[edge.port_a].cell_id,
                self._ports[edge.port_b].cell_id,
            )
            for edge in scenario.interfaces
        }
        self._boundary_cells = {
            boundary.boundary_id: self._ports[boundary.port_id].cell_id
            for boundary in scenario.boundaries
        }

    def run(self) -> SimulationResult:
        cell_states = {
            cell.cell_id: CellState(
                temperature_k=cell.initial_temperature_k,
                damage=cell.initial_damage,
            )
            for cell in self.scenario.cells
        }
        edge_states: dict[str, InterfaceState] = {}
        for edge in self.scenario.interfaces:
            initial_l0 = (
                edge.law.destroyed_l0_m
                if edge.initial_phase is EdgePhase.DESTROYED
                else edge.law.l0_m
            )
            state = InterfaceState(
                l0_m=initial_l0,
                l_m=edge.initial_l_m,
                damage=edge.initial_damage,
                phase=edge.initial_phase,
            )
            state.conductance_w_per_k = conductance(edge.law, state)
            edge_states[edge.interface_id] = state

        events: list[EventRecord] = []
        snapshots: list[Snapshot] = []
        runaway_order: list[str] = []
        runaway_times: dict[str, float] = {}
        scheduled = sorted(self.scenario.primary_failures, key=lambda event: event.time_s)
        next_scheduled = 0
        primary_cells = {event.cell_id for event in scheduled}
        boundary_energy_out_j = 0.0

        def trigger(cell_id: str, time_s: float, origin: str) -> None:
            state = cell_states[cell_id]
            if state.mode is not CellMode.HEALTHY:
                return
            state.mode = CellMode.RUNAWAY
            state.runaway_started_s = time_s
            runaway_order.append(cell_id)
            runaway_times[cell_id] = time_s
            events.append(EventRecord(time_s, f"runaway_{origin}", cell_id))

        def apply_scheduled(time_s: float) -> None:
            nonlocal next_scheduled
            while next_scheduled < len(scheduled) and scheduled[next_scheduled].time_s <= time_s + _EPS:
                event = scheduled[next_scheduled]
                trigger(event.cell_id, event.time_s, "primary")
                next_scheduled += 1

        def finish_sources(time_s: float) -> None:
            for cell_id, state in cell_states.items():
                if state.mode is not CellMode.RUNAWAY or state.runaway_started_s is None:
                    continue
                duration = self._cells[cell_id].runaway_duration_s
                if time_s >= state.runaway_started_s + duration - _EPS:
                    state.mode = CellMode.SPENT
                    events.append(EventRecord(time_s, "runaway_complete", cell_id))

        def snapshot(time_s: float) -> None:
            snapshots.append(
                Snapshot(
                    time_s=time_s,
                    cells={
                        cell_id: CellSnapshot(
                            temperature_k=state.temperature_k,
                            damage=state.damage,
                            mode=state.mode.value,
                            released_energy_j=state.released_energy_j,
                        )
                        for cell_id, state in cell_states.items()
                    },
                    interfaces={
                        edge_id: InterfaceSnapshot(
                            l0_m=state.l0_m,
                            l_m=state.l_m,
                            damage=state.damage,
                            phase=state.phase.value,
                            conductance_w_per_k=state.conductance_w_per_k,
                        )
                        for edge_id, state in edge_states.items()
                    },
                )
            )

        apply_scheduled(0.0)
        for cell_id, state in cell_states.items():
            if state.damage >= 1.0:
                trigger(cell_id, 0.0, "secondary")
        snapshot(0.0)

        initial_energy = sum(
            self._cells[cell_id].heat_capacity_j_per_k * state.temperature_k
            for cell_id, state in cell_states.items()
        )
        time_s = 0.0
        next_output_s = self.scenario.settings.output_interval_s

        while time_s < self.scenario.settings.duration_s - _EPS:
            apply_scheduled(time_s)
            finish_sources(time_s)

            dt_s = min(
                self.scenario.settings.max_dt_s,
                self.scenario.settings.duration_s - time_s,
                max(next_output_s - time_s, 0.0),
            )
            if next_scheduled < len(scheduled):
                scheduled_dt = scheduled[next_scheduled].time_s - time_s
                if scheduled_dt > _EPS:
                    dt_s = min(dt_s, scheduled_dt)
            for cell_id, state in cell_states.items():
                if state.mode is CellMode.RUNAWAY and state.runaway_started_s is not None:
                    source_end = state.runaway_started_s + self._cells[cell_id].runaway_duration_s
                    if source_end > time_s + _EPS:
                        dt_s = min(dt_s, source_end - time_s)

            if dt_s <= _EPS:
                if next_output_s <= time_s + _EPS:
                    snapshot(time_s)
                    next_output_s += self.scenario.settings.output_interval_s
                    continue
                raise RuntimeError("time step collapsed; inspect event and output schedules")

            # The interface update can increase conductance. Re-evaluate the
            # explicit stability limit from the same pre-step state until the
            # proposed edge state and dt agree.
            proposed_edges: dict[str, InterfaceState] = {}
            proposed_edge_events: dict[str, str] = {}
            for _ in range(12):
                proposed_edges.clear()
                proposed_edge_events.clear()
                for edge_id, edge in self._interfaces.items():
                    cell_a, cell_b = self._edge_cells[edge_id]
                    edge_temperature = 0.5 * (
                        cell_states[cell_a].temperature_k + cell_states[cell_b].temperature_k
                    )
                    candidate, edge_event = advance_interface_state(
                        edge.law, edge_states[edge_id], edge_temperature, dt_s
                    )
                    proposed_edges[edge_id] = candidate
                    if edge_event is not None:
                        proposed_edge_events[edge_id] = edge_event

                conductance_sums = {cell_id: 0.0 for cell_id in self._cells}
                for edge_id, state in proposed_edges.items():
                    cell_a, cell_b = self._edge_cells[edge_id]
                    conductance_sums[cell_a] += state.conductance_w_per_k
                    conductance_sums[cell_b] += state.conductance_w_per_k
                for boundary in self.scenario.boundaries:
                    conductance_sums[self._boundary_cells[boundary.boundary_id]] += (
                        boundary.conductance_w_per_k
                    )

                stable_dt = dt_s
                for cell_id, total_g in conductance_sums.items():
                    if total_g > 0.0:
                        stable_dt = min(
                            stable_dt,
                            self.scenario.settings.stability_factor
                            * self._cells[cell_id].heat_capacity_j_per_k
                            / total_g,
                        )
                if stable_dt >= dt_s - _EPS:
                    break
                dt_s = stable_dt
            else:
                raise RuntimeError("interface/stability iteration did not converge")

            edge_states = dict(proposed_edges)
            energy_delta = {cell_id: 0.0 for cell_id in self._cells}

            for edge_id, state in edge_states.items():
                cell_a, cell_b = self._edge_cells[edge_id]
                heat_rate_a = state.conductance_w_per_k * (
                    cell_states[cell_b].temperature_k - cell_states[cell_a].temperature_k
                )
                transferred = heat_rate_a * dt_s
                energy_delta[cell_a] += transferred
                energy_delta[cell_b] -= transferred

            for boundary in self.scenario.boundaries:
                cell_id = self._boundary_cells[boundary.boundary_id]
                outward_rate = boundary.conductance_w_per_k * (
                    cell_states[cell_id].temperature_k - boundary.temperature_k
                )
                outward_energy = outward_rate * dt_s
                energy_delta[cell_id] -= outward_energy
                boundary_energy_out_j += outward_energy

            for cell_id, state in cell_states.items():
                if state.mode is CellMode.RUNAWAY:
                    cell = self._cells[cell_id]
                    remaining = max(cell.runaway_energy_j - state.released_energy_j, 0.0)
                    released = min(cell.runaway_energy_j / cell.runaway_duration_s * dt_s, remaining)
                    state.released_energy_j += released
                    energy_delta[cell_id] += released

            old_temperatures = {
                cell_id: state.temperature_k for cell_id, state in cell_states.items()
            }
            for cell_id, state in cell_states.items():
                cell = self._cells[cell_id]
                state.temperature_k += energy_delta[cell_id] / cell.heat_capacity_j_per_k
                if state.mode is CellMode.HEALTHY:
                    average_temperature = 0.5 * (
                        old_temperatures[cell_id] + state.temperature_k
                    )
                    rate = thermal_damage_rate(
                        average_temperature,
                        cell.damage_onset_k,
                        cell.damage_rate_at_onset_per_s,
                        cell.damage_temperature_scale_k,
                    )
                    rate *= 1.0 + cell.aging_damage_multiplier * cell.aging_fraction
                    state.damage = min(1.0, state.damage + rate * dt_s)

            time_s += dt_s
            for edge_id, edge_event in proposed_edge_events.items():
                events.append(EventRecord(time_s, edge_event, edge_id))
            finish_sources(time_s)
            apply_scheduled(time_s)
            for cell_id, state in cell_states.items():
                if state.mode is CellMode.HEALTHY and state.damage >= 1.0:
                    trigger(cell_id, time_s, "secondary")

            if next_output_s <= time_s + _EPS:
                # Store the requested decimal output time rather than its
                # accumulated binary floating-point representation.
                snapshot(min(next_output_s, self.scenario.settings.duration_s))
                next_output_s += self.scenario.settings.output_interval_s

        if not snapshots or snapshots[-1].time_s < self.scenario.settings.duration_s - _EPS:
            snapshot(self.scenario.settings.duration_s)

        final_energy = sum(
            self._cells[cell_id].heat_capacity_j_per_k * state.temperature_k
            for cell_id, state in cell_states.items()
        )
        released_energy = sum(state.released_energy_j for state in cell_states.values())
        energy_residual = (
            final_energy
            - initial_energy
            - (released_energy - boundary_energy_out_j)
        )
        propagated = any(cell_id not in primary_cells for cell_id in runaway_order)

        # Avoid a runtime import cycle: validation only needs the frozen result.
        preliminary = SimulationResult(
            scenario_name=self.scenario.name,
            source=self.scenario.source,
            snapshots=tuple(snapshots),
            events=tuple(events),
            propagated=propagated,
            runaway_order=tuple(runaway_order),
            runaway_times_s=dict(runaway_times),
            initial_stored_energy_j=initial_energy,
            final_stored_energy_j=final_energy,
            released_energy_j=released_energy,
            boundary_energy_out_j=boundary_energy_out_j,
            energy_balance_residual_j=energy_residual,
            validation={},
        )
        from .validation import compare_with_observation

        validation = compare_with_observation(self.scenario, preliminary)
        return SimulationResult(**{**preliminary.__dict__, "validation": validation})

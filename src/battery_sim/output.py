from __future__ import annotations

import csv
import json
from pathlib import Path

from .solver import SimulationResult


def write_result(result: SimulationResult, output_directory: str | Path) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    cell_ids = list(result.snapshots[0].cells) if result.snapshots else []
    edge_ids = list(result.snapshots[0].interfaces) if result.snapshots else []
    fields = ["time_s"]
    for cell_id in cell_ids:
        fields.extend(
            [
                f"cell.{cell_id}.temperature_k",
                f"cell.{cell_id}.damage",
                f"cell.{cell_id}.mode",
                f"cell.{cell_id}.released_energy_j",
            ]
        )
    for edge_id in edge_ids:
        fields.extend(
            [
                f"interface.{edge_id}.l0_m",
                f"interface.{edge_id}.l_m",
                f"interface.{edge_id}.damage",
                f"interface.{edge_id}.phase",
                f"interface.{edge_id}.conductance_w_per_k",
            ]
        )

    with (output / "timeseries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for snapshot in result.snapshots:
            row: dict[str, str | float] = {"time_s": snapshot.time_s}
            for cell_id, state in snapshot.cells.items():
                row[f"cell.{cell_id}.temperature_k"] = state.temperature_k
                row[f"cell.{cell_id}.damage"] = state.damage
                row[f"cell.{cell_id}.mode"] = state.mode
                row[f"cell.{cell_id}.released_energy_j"] = state.released_energy_j
            for edge_id, state in snapshot.interfaces.items():
                row[f"interface.{edge_id}.l0_m"] = state.l0_m
                row[f"interface.{edge_id}.l_m"] = state.l_m
                row[f"interface.{edge_id}.damage"] = state.damage
                row[f"interface.{edge_id}.phase"] = state.phase
                row[f"interface.{edge_id}.conductance_w_per_k"] = state.conductance_w_per_k
            writer.writerow(row)

    with (output / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "kind", "entity_id", "detail"])
        for event in result.events:
            writer.writerow([event.time_s, event.kind, event.entity_id, event.detail])

    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result.summary(), stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")

    return output

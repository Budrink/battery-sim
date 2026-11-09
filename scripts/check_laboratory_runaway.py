#!/usr/bin/env python3
"""Check a completed default-module job; this is a demo acceptance, not validation."""
import argparse
import json
from pathlib import Path


def summarize(path):
    result = json.loads((path / "result.json").read_text())
    if result["status"] != "completed":
        raise ValueError("Run is not completed: " + result["status"])
    document = json.loads((path / "scenario.json").read_text())
    if document.get("format") != "battery-sim.experiment":
        raise ValueError("Expected the unmodified experiment contract snapshot")
    events = [json.loads(line) for line in (path / "events.ndjson").read_text().splitlines()]
    reactions = [e for e in events if e["kind"] in ("runaway_primary", "runaway_secondary")]
    completed = [e for e in events if e["kind"] == "runaway_complete"]
    openings = [e for e in events if e["kind"] == "shell_rupture"]
    maximum = {}
    peak_time = {}
    first = last = None
    for line in (path / "observations.ndjson").read_text().splitlines():
        row = json.loads(line)
        if first is None:
            first = row
        last = row
        for c in row["cells"]:
            if len(c["id"]) == 3 and c["id"].startswith("B"):
                if c["temperature_k"] > maximum.get(c["id"], 0):
                    maximum[c["id"]] = c["temperature_k"]
                    peak_time[c["id"]] = row["time_s"]
    assert first["time_s"] == 0, "Demo must be checked from a cold start"
    assert all(c["temperature_k"] < 300 for c in first["cells"]), "Hot start is not acceptance"
    assert any(e["entity_id"] == "B05" for e in reactions), "Heated battery did not run away"
    assert any("B05" in e["entity_id"] for e in openings), "Heated battery did not rupture"
    assert maximum["B05"] > 700, "Missing hot thermal excursion"
    assert any(e["entity_id"] == "B05" for e in completed), "Primary reaction did not finish"
    for key in ("energy_balance_residual_j", "chemical_balance_residual_j"):
        assert abs(result["validation"][key]) < 1e-4, key
    return dict(status="demo_passed", experimental_validation=False,
                completed_s=last["time_s"], wall_s=result["wall_s"],
                reaction_onsets=reactions, completed_reactions=completed,
                first_opening=next(e for e in openings if "B05" in e["entity_id"]),
                maximum_temperature_c={k: v-273.15 for k, v in maximum.items()},
                peak_time_s=peak_time, balances=result["validation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, help="Save the acceptance summary")
    args = parser.parse_args()
    payload = json.dumps(summarize(args.run), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")

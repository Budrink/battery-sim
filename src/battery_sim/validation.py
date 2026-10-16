from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .model import Scenario

if TYPE_CHECKING:
    from .solver import SimulationResult


def compare_with_observation(scenario: Scenario, result: "SimulationResult") -> dict[str, Any]:
    """Compare model outputs with optional canonical observations.

    Dataset-specific parsers should stop at ``Scenario``. This function then
    applies the same comparison semantics to public experiments and local tests.
    """
    expected = scenario.observation
    checks: dict[str, Any] = {}
    passed = True

    if expected.propagated is not None:
        match = expected.propagated == result.propagated
        checks["propagated"] = {
            "expected": expected.propagated,
            "actual": result.propagated,
            "pass": match,
        }
        passed = passed and match

    if expected.runaway_order:
        actual = list(result.runaway_order)
        wanted = list(expected.runaway_order)
        match = actual == wanted
        checks["runaway_order"] = {"expected": wanted, "actual": actual, "pass": match}
        passed = passed and match

    if expected.runaway_times_s:
        residuals: dict[str, float | None] = {}
        for cell_id, expected_time in expected.runaway_times_s.items():
            actual_time = result.runaway_times_s.get(cell_id)
            residuals[cell_id] = None if actual_time is None else actual_time - expected_time
            passed = passed and actual_time is not None
        checks["runaway_time_residuals_s"] = residuals

    return {"available": bool(checks), "pass": passed if checks else None, "checks": checks}

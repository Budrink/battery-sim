from __future__ import annotations

import math

from .model import (
    ConnectivityLaw,
    EdgePhase,
    InterfaceState,
    TransitionKind,
)


def _bounded_exp(value: float) -> float:
    return math.exp(max(-80.0, min(80.0, value)))


def logistic(value: float) -> float:
    if value >= 0.0:
        e = math.exp(-min(value, 80.0))
        return 1.0 / (1.0 + e)
    e = math.exp(max(value, -80.0))
    return e / (1.0 + e)


def thermal_damage_rate(
    temperature_k: float,
    onset_k: float,
    rate_at_onset_per_s: float,
    scale_k: float,
) -> float:
    """Empirical thermal-dose rate used until a calibrated kinetics law replaces it."""
    if temperature_k < onset_k or rate_at_onset_per_s <= 0.0:
        return 0.0
    return rate_at_onset_per_s * _bounded_exp((temperature_k - onset_k) / scale_k)


def _target_l(
    law: ConnectivityLaw,
    state: InterfaceState,
    temperature_k: float,
) -> tuple[float, EdgePhase]:
    if state.phase is EdgePhase.DESTROYED:
        return law.destroyed_l_m, EdgePhase.DESTROYED

    if law.kind is TransitionKind.CONSTANT:
        return law.l_low_m, state.phase

    if law.kind is TransitionKind.SMOOTH:
        x = (temperature_k - law.transition_temperature_k) / law.transition_width_k
        fraction = logistic(x)
        return law.l_low_m + (law.l_high_m - law.l_low_m) * fraction, state.phase

    if law.kind is TransitionKind.FIRST_ORDER:
        phase = state.phase
        upper = law.transition_temperature_k + 0.5 * law.hysteresis_k
        lower = law.transition_temperature_k - 0.5 * law.hysteresis_k
        if phase is EdgePhase.LOW and temperature_k >= upper:
            phase = EdgePhase.HIGH
        elif phase is EdgePhase.HIGH and temperature_k <= lower:
            phase = EdgePhase.LOW
        return (law.l_high_m if phase is EdgePhase.HIGH else law.l_low_m), phase

    # Mean-field continuous transition. l is continuous at T_c and has a
    # square-root critical branch above it; l_high_m is a safety cap.
    reduced = max(
        (temperature_k - law.transition_temperature_k) / law.transition_width_k,
        0.0,
    )
    amplitude = law.l_high_m - law.l_low_m
    target = law.l_low_m + amplitude * min(math.sqrt(reduced), 1.0)
    phase = EdgePhase.HIGH if reduced > 0.0 else EdgePhase.LOW
    return target, phase


def conductance(law: ConnectivityLaw, state: InterfaceState) -> float:
    if state.phase is EdgePhase.DESTROYED:
        background = law.destroyed_background_conductance_w_per_k
        active = law.destroyed_active_conductance_w_per_k
    else:
        background = law.background_conductance_w_per_k
        active = law.active_conductance_w_per_k

    activation = logistic((state.l_m - state.l0_m) / law.activation_width_m)
    damage_factor = law.damage_conductance_factor_at_one ** state.damage
    return max((background + active * activation) * damage_factor, 0.0)


def advance_interface_state(
    law: ConnectivityLaw,
    state: InterfaceState,
    temperature_k: float,
    dt_s: float,
) -> tuple[InterfaceState, str | None]:
    """Advance edge order parameter, damage, and phase without storing heat."""
    damage = min(
        1.0,
        state.damage
        + thermal_damage_rate(
            temperature_k,
            law.damage_onset_k,
            law.damage_rate_at_onset_per_s,
            law.damage_temperature_scale_k,
        )
        * dt_s,
    )

    event: str | None = None
    phase = state.phase
    l0_m = state.l0_m
    if phase is not EdgePhase.DESTROYED and damage >= law.destruction_damage:
        phase = EdgePhase.DESTROYED
        l0_m = law.destroyed_l0_m
        event = "interface_destroyed"

    intermediate = InterfaceState(
        l0_m=l0_m,
        l_m=state.l_m,
        damage=damage,
        phase=phase,
    )
    target, new_phase = _target_l(law, intermediate, temperature_k)
    if new_phase is not phase and event is None:
        event = f"phase_{phase.value}_to_{new_phase.value}"
    phase = new_phase

    if phase is not EdgePhase.DESTROYED:
        target += law.damage_l_shift_m * damage
    target = max(target, 0.0)
    if law.relaxation_time_s <= 0.0:
        l_m = target
    else:
        fraction = 1.0 - math.exp(-dt_s / law.relaxation_time_s)
        l_m = state.l_m + (target - state.l_m) * fraction

    updated = InterfaceState(
        l0_m=l0_m,
        l_m=l_m,
        damage=damage,
        phase=phase,
    )
    updated.conductance_w_per_k = conductance(law, updated)
    return updated, event

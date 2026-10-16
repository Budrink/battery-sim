from __future__ import annotations

import unittest

from battery_sim.laws import advance_interface_state
from battery_sim.model import ConnectivityLaw, EdgePhase, InterfaceState, TransitionKind


def law(**overrides) -> ConnectivityLaw:
    values = {
        "kind": TransitionKind.FIRST_ORDER,
        "l0_m": 0.5,
        "l_low_m": 0.1,
        "l_high_m": 1.0,
        "transition_temperature_k": 400.0,
        "transition_width_k": 20.0,
        "hysteresis_k": 20.0,
        "relaxation_time_s": 0.0,
        "background_conductance_w_per_k": 0.1,
        "active_conductance_w_per_k": 10.0,
        "activation_width_m": 0.05,
        "damage_onset_k": 450.0,
        "damage_rate_at_onset_per_s": 0.0,
        "damage_temperature_scale_k": 20.0,
        "damage_l_shift_m": 0.0,
        "damage_conductance_factor_at_one": 1.0,
        "destruction_damage": 1.0,
        "destroyed_l0_m": 0.8,
        "destroyed_l_m": 0.05,
        "destroyed_background_conductance_w_per_k": 0.02,
        "destroyed_active_conductance_w_per_k": 0.1,
    }
    values.update(overrides)
    return ConnectivityLaw(**values)


class ConnectivityLawTests(unittest.TestCase):
    def test_first_order_transition_has_hysteresis(self) -> None:
        state = InterfaceState(0.5, 0.1, 0.0, EdgePhase.LOW)
        heated, event = advance_interface_state(law(), state, 411.0, 1.0)
        self.assertEqual(heated.phase, EdgePhase.HIGH)
        self.assertEqual(event, "phase_low_to_high")
        self.assertEqual(heated.l_m, 1.0)

        still_high, event = advance_interface_state(law(), heated, 395.0, 1.0)
        self.assertEqual(still_high.phase, EdgePhase.HIGH)
        self.assertIsNone(event)

        cooled, event = advance_interface_state(law(), still_high, 389.0, 1.0)
        self.assertEqual(cooled.phase, EdgePhase.LOW)
        self.assertEqual(event, "phase_high_to_low")

    def test_destruction_can_replace_l0(self) -> None:
        destructive = law(
            damage_onset_k=300.0,
            damage_rate_at_onset_per_s=0.6,
            destruction_damage=0.5,
        )
        state = InterfaceState(0.5, 0.1, 0.0, EdgePhase.LOW)
        destroyed, event = advance_interface_state(destructive, state, 300.0, 1.0)
        self.assertEqual(event, "interface_destroyed")
        self.assertEqual(destroyed.phase, EdgePhase.DESTROYED)
        self.assertEqual(destroyed.l0_m, 0.8)
        self.assertEqual(destroyed.l_m, 0.05)


if __name__ == "__main__":
    unittest.main()

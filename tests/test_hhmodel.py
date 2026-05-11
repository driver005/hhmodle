import unittest

from hhmodel import (
    HHParameters,
    HHState,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    euler_step,
    simulate,
    steady_state,
)


class HHModelTests(unittest.TestCase):
    def test_steady_state_at_rest(self):
        m, h, n = steady_state(-65.0)
        self.assertAlmostEqual(m, 0.0529, places=3)
        self.assertAlmostEqual(h, 0.5961, places=3)
        self.assertAlmostEqual(n, 0.3177, places=3)

    def test_simulation_initializes_gates_from_resting_steady_state(self):
        _, states = simulate(duration_ms=0.5, dt_ms=0.01, input_current=0.0, method="rk4")
        expected = steady_state(-65.0)
        self.assertAlmostEqual(states[0].gate_fast_on, expected[0], places=8)
        self.assertAlmostEqual(states[0].gate_fast_off, expected[1], places=8)
        self.assertAlmostEqual(states[0].gate_slow_on, expected[2], places=8)

    def test_euler_updates_gates_before_voltage(self):
        params = HHParameters()
        state = HHState(-65.0, 0.1, 0.7, 0.2)
        dt = 0.01
        next_state = euler_step(0.0, dt, state, params, 0.0)

        m1 = state.gate_fast_on + dt * (
            alpha_m(state.potential) * (1.0 - state.gate_fast_on)
            - beta_m(state.potential) * state.gate_fast_on
        )
        h1 = state.gate_fast_off + dt * (
            alpha_h(state.potential) * (1.0 - state.gate_fast_off)
            - beta_h(state.potential) * state.gate_fast_off
        )
        n1 = state.gate_slow_on + dt * (
            alpha_n(state.potential) * (1.0 - state.gate_slow_on)
            - beta_n(state.potential) * state.gate_slow_on
        )
        ionic = (
            params.g_na * (m1**3) * h1 * (state.potential - params.e_na)
            + params.g_k * (n1**4) * (state.potential - params.e_k)
            + params.g_l * (state.potential - params.e_l)
        )
        expected_v = state.potential + dt * (0.0 - ionic) / params.capacity
        self.assertAlmostEqual(next_state.potential, expected_v, places=8)

    def test_rk4_spikes_with_standard_current(self):
        _, states = simulate(duration_ms=50.0, dt_ms=0.01, input_current=10.0, method="rk4")
        self.assertTrue(any(s.potential > 0.0 for s in states))


if __name__ == "__main__":
    unittest.main()

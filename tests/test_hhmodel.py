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
    ionic_currents,
    simulate_trace,
    simulate,
    step_current,
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

    def test_simulate_trace_contains_currents(self):
        trace = simulate_trace(duration_ms=1.0, dt_ms=0.1, input_current=0.0, method="rk4")
        self.assertEqual(len(trace.times), len(trace.states))
        self.assertEqual(len(trace.times), len(trace.currents))
        self.assertAlmostEqual(trace.currents[0].ext, 0.0)

    def test_step_current_protocol(self):
        current = step_current(amplitude=8.0, start_ms=1.0, stop_ms=2.0)
        self.assertEqual(current(0.9), 0.0)
        self.assertEqual(current(1.0), 8.0)
        self.assertEqual(current(1.5), 8.0)
        self.assertEqual(current(2.1), 0.0)

    def test_ionic_currents_sum_matches_total(self):
        params = HHParameters()
        state = HHState(-60.0, 0.1, 0.6, 0.3)
        currents = ionic_currents(0.0, state, params, 3.0)
        self.assertAlmostEqual(
            currents.ionic_total,
            currents.na + currents.k + currents.leak,
            places=12,
        )

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            simulate_trace(
                duration_ms=1.0,
                dt_ms=0.01,
                params=HHParameters(capacity=0.0),
            )

    def test_invalid_step_current_raises(self):
        with self.assertRaises(ValueError):
            step_current(amplitude=5.0, start_ms=2.0, stop_ms=1.0)


if __name__ == "__main__":
    unittest.main()

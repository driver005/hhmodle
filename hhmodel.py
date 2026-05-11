"""Hodgkin-Huxley neuron model with Euler and RK4 integration."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Callable, List, Tuple, Union


CurrentInput = Union[float, Callable[[float], float]]


@dataclass(frozen=True)
class HHParameters:
    capacity: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.4


@dataclass(frozen=True)
class HHState:
    potential: float
    gate_fast_on: float
    gate_fast_off: float
    gate_slow_on: float


@dataclass(frozen=True)
class HHCurrents:
    ext: float
    na: float
    k: float
    leak: float

    @property
    def ionic_total(self) -> float:
        return self.na + self.k + self.leak


@dataclass(frozen=True)
class HHTrace:
    times: List[float]
    states: List[HHState]
    currents: List[HHCurrents]


def _vtrap(x: float, y: float) -> float:
    if abs(x / y) < 1e-6:
        return y * (1 - (x / y) / 2.0)
    return x / (exp(x / y) - 1.0)


def alpha_m(v: float) -> float:
    return 0.1 * _vtrap(-(v + 40.0), 10.0)


def beta_m(v: float) -> float:
    return 4.0 * exp(-(v + 65.0) / 18.0)


def alpha_h(v: float) -> float:
    return 0.07 * exp(-(v + 65.0) / 20.0)


def beta_h(v: float) -> float:
    return 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))


def alpha_n(v: float) -> float:
    return 0.01 * _vtrap(-(v + 55.0), 10.0)


def beta_n(v: float) -> float:
    return 0.125 * exp(-(v + 65.0) / 80.0)


def steady_state(v: float) -> Tuple[float, float, float]:
    m = alpha_m(v) / (alpha_m(v) + beta_m(v))
    h = alpha_h(v) / (alpha_h(v) + beta_h(v))
    n = alpha_n(v) / (alpha_n(v) + beta_n(v))
    return m, h, n


def _input_current(input_current: CurrentInput, t: float) -> float:
    if callable(input_current):
        return input_current(t)
    return input_current


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _validate_parameters(params: HHParameters) -> None:
    if params.capacity <= 0.0:
        raise ValueError("capacity must be positive")
    if params.g_na < 0.0 or params.g_k < 0.0 or params.g_l < 0.0:
        raise ValueError("conductances must be non-negative")


def ionic_currents(
    t: float, state: HHState, params: HHParameters, input_current: CurrentInput
) -> HHCurrents:
    v = state.potential
    ext = _input_current(input_current, t)
    na = params.g_na * (state.gate_fast_on**3) * state.gate_fast_off * (v - params.e_na)
    k = params.g_k * (state.gate_slow_on**4) * (v - params.e_k)
    leak = params.g_l * (v - params.e_l)
    return HHCurrents(ext=ext, na=na, k=k, leak=leak)


def derivatives(
    t: float, state: HHState, params: HHParameters, input_current: CurrentInput
) -> HHState:
    v = state.potential
    m = state.gate_fast_on
    h = state.gate_fast_off
    n = state.gate_slow_on

    currents = ionic_currents(t, state, params, input_current)
    dv_dt = (currents.ext - currents.ionic_total) / params.capacity
    dm_dt = alpha_m(v) * (1.0 - m) - beta_m(v) * m
    dh_dt = alpha_h(v) * (1.0 - h) - beta_h(v) * h
    dn_dt = alpha_n(v) * (1.0 - n) - beta_n(v) * n
    return HHState(dv_dt, dm_dt, dh_dt, dn_dt)


def euler_step(
    t: float, dt: float, state: HHState, params: HHParameters, input_current: CurrentInput
) -> HHState:
    v = state.potential
    m = state.gate_fast_on
    h = state.gate_fast_off
    n = state.gate_slow_on

    m_next = _clip01(m + dt * (alpha_m(v) * (1.0 - m) - beta_m(v) * m))
    h_next = _clip01(h + dt * (alpha_h(v) * (1.0 - h) - beta_h(v) * h))
    n_next = _clip01(n + dt * (alpha_n(v) * (1.0 - n) - beta_n(v) * n))

    state_with_new_gates = HHState(v, m_next, h_next, n_next)
    currents = ionic_currents(t, state_with_new_gates, params, input_current)
    v_next = v + dt * (currents.ext - currents.ionic_total) / params.capacity
    return HHState(v_next, m_next, h_next, n_next)


def rk4_step(
    t: float, dt: float, state: HHState, params: HHParameters, input_current: CurrentInput
) -> HHState:
    k1 = derivatives(t, state, params, input_current)

    s2 = HHState(
        state.potential + 0.5 * dt * k1.potential,
        state.gate_fast_on + 0.5 * dt * k1.gate_fast_on,
        state.gate_fast_off + 0.5 * dt * k1.gate_fast_off,
        state.gate_slow_on + 0.5 * dt * k1.gate_slow_on,
    )
    k2 = derivatives(t + 0.5 * dt, s2, params, input_current)

    s3 = HHState(
        state.potential + 0.5 * dt * k2.potential,
        state.gate_fast_on + 0.5 * dt * k2.gate_fast_on,
        state.gate_fast_off + 0.5 * dt * k2.gate_fast_off,
        state.gate_slow_on + 0.5 * dt * k2.gate_slow_on,
    )
    k3 = derivatives(t + 0.5 * dt, s3, params, input_current)

    s4 = HHState(
        state.potential + dt * k3.potential,
        state.gate_fast_on + dt * k3.gate_fast_on,
        state.gate_fast_off + dt * k3.gate_fast_off,
        state.gate_slow_on + dt * k3.gate_slow_on,
    )
    k4 = derivatives(t + dt, s4, params, input_current)

    return HHState(
        state.potential
        + (dt / 6.0)
        * (k1.potential + 2.0 * k2.potential + 2.0 * k3.potential + k4.potential),
        _clip01(
            state.gate_fast_on
            + (dt / 6.0)
            * (
                k1.gate_fast_on
                + 2.0 * k2.gate_fast_on
                + 2.0 * k3.gate_fast_on
                + k4.gate_fast_on
            )
        ),
        _clip01(
            state.gate_fast_off
            + (dt / 6.0)
            * (
                k1.gate_fast_off
                + 2.0 * k2.gate_fast_off
                + 2.0 * k3.gate_fast_off
                + k4.gate_fast_off
            )
        ),
        _clip01(
            state.gate_slow_on
            + (dt / 6.0)
            * (
                k1.gate_slow_on
                + 2.0 * k2.gate_slow_on
                + 2.0 * k3.gate_slow_on
                + k4.gate_slow_on
            )
        ),
    )


def step_current(amplitude: float, start_ms: float, stop_ms: float) -> Callable[[float], float]:
    if stop_ms < start_ms:
        raise ValueError("stop_ms must be greater than or equal to start_ms")

    def current(t: float) -> float:
        if start_ms <= t <= stop_ms:
            return amplitude
        return 0.0

    return current


def simulate_trace(
    duration_ms: float,
    dt_ms: float,
    input_current: CurrentInput = 10.0,
    method: str = "rk4",
    initial_potential: float = -65.0,
    params: HHParameters = HHParameters(),
) -> HHTrace:
    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be positive")
    if duration_ms < 0.0:
        raise ValueError("duration_ms must be non-negative")
    if method not in {"euler", "rk4"}:
        raise ValueError("method must be 'euler' or 'rk4'")
    _validate_parameters(params)

    m0, h0, n0 = steady_state(initial_potential)
    state = HHState(initial_potential, m0, h0, n0)

    steps = int(round(duration_ms / dt_ms))
    times: List[float] = [0.0]
    states: List[HHState] = [state]
    currents: List[HHCurrents] = [ionic_currents(0.0, state, params, input_current)]
    for i in range(steps):
        t = i * dt_ms
        if method == "rk4":
            state = rk4_step(t, dt_ms, state, params, input_current)
        else:
            state = euler_step(t, dt_ms, state, params, input_current)
        next_t = (i + 1) * dt_ms
        times.append(next_t)
        states.append(state)
        currents.append(ionic_currents(next_t, state, params, input_current))
    return HHTrace(times=times, states=states, currents=currents)


def simulate(
    duration_ms: float,
    dt_ms: float,
    input_current: CurrentInput = 10.0,
    method: str = "rk4",
    initial_potential: float = -65.0,
    params: HHParameters = HHParameters(),
) -> Tuple[List[float], List[HHState]]:
    trace = simulate_trace(
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        input_current=input_current,
        method=method,
        initial_potential=initial_potential,
        params=params,
    )
    return trace.times, trace.states

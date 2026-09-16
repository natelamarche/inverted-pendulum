import numpy as np

from pendulum.dynamics.parameters import PendulumParameters
from pendulum.simulation.integrators import rk4_step
from pendulum.simulation.simulator import Simulator


def test_reset_defaults_to_zero_state_and_returns_a_copy(
    pendulum_params: PendulumParameters,
) -> None:
    simulator = Simulator(pendulum_params, dt=0.01)

    returned_state = simulator.reset()
    returned_state[0] = 1.0

    np.testing.assert_array_equal(simulator.get_state(), np.zeros(4))


def test_reset_copies_supplied_state(pendulum_params: PendulumParameters) -> None:
    simulator = Simulator(pendulum_params, dt=0.01)
    initial_state = np.array([0.1, -0.2, 0.3, -0.4])

    simulator.reset(initial_state)
    initial_state[0] = 99.0

    np.testing.assert_allclose(simulator.get_state(), [0.1, -0.2, 0.3, -0.4])


def test_step_matches_one_rk4_step_and_returns_a_copy(
    pendulum_params: PendulumParameters,
) -> None:
    dt = 0.01
    torque = 0.1
    initial_state = np.array([0.05, -0.1, 0.2, -0.3])
    simulator = Simulator(pendulum_params, dt=dt)
    simulator.reset(initial_state)
    expected_state = rk4_step(initial_state, torque, dt, pendulum_params)

    returned_state = simulator.step(torque)

    np.testing.assert_allclose(returned_state, expected_state)
    returned_state[0] = 99.0
    np.testing.assert_allclose(simulator.get_state(), expected_state)


def test_zero_input_keeps_equilibrium_stationary(
    pendulum_params: PendulumParameters,
) -> None:
    simulator = Simulator(pendulum_params, dt=0.01)

    state = simulator.step(torque=0.0)

    np.testing.assert_allclose(state, np.zeros(4), atol=1e-12)


def test_held_acceleration_follows_analytic_arm_motion(pendulum_params):
    for acceleration in (-100.0, 0.0, 100.0):
        simulator = Simulator(pendulum_params, dt=0.001)
        simulator.reset(np.array([0.2, 0.4, -0.3, 0.5]))
        for _ in range(100):
            state = simulator.step_acceleration(acceleration)
        elapsed = 0.1
        np.testing.assert_allclose(
            state[[0, 2]],
            [
                0.2 - 0.3 * elapsed + 0.5 * acceleration * elapsed**2,
                -0.3 + acceleration * elapsed,
            ],
            atol=1e-12,
        )
        state[0] = 99.0
        assert simulator.get_state()[0] != 99.0

import numpy as np

from pendulum.dynamics.model import accelerations, state_derivative
from pendulum.dynamics.parameters import PendulumParameters


def test_stationary_downward_state_is_an_equilibrium(
    pendulum_params: PendulumParameters,
) -> None:
    derivative = state_derivative(np.zeros(4), torque=0.0, params=pendulum_params)

    np.testing.assert_allclose(derivative, np.zeros(4), atol=1e-12)


def test_derivative_starts_with_current_angular_velocities(
    pendulum_params: PendulumParameters,
) -> None:
    state = np.array([0.3, -0.2, 1.25, -0.75])

    derivative = state_derivative(state, torque=0.0, params=pendulum_params)

    np.testing.assert_allclose(derivative[:2], state[2:])


def test_arm_torque_accelerates_arm_and_couples_into_pendulum(
    pendulum_params: PendulumParameters,
) -> None:
    phi_ddot, theta_ddot = accelerations(
        np.zeros(4), torque=0.1, params=pendulum_params
    )

    assert phi_ddot > 0.0
    assert theta_ddot < 0.0


def test_gravity_accelerates_displaced_pendulum_toward_downward_position(
    pendulum_params: PendulumParameters,
) -> None:
    state = np.array([0.0, np.pi / 2, 0.0, 0.0])

    _, theta_ddot = accelerations(state, torque=0.0, params=pendulum_params)

    assert theta_ddot < 0.0

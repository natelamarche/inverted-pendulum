import numpy as np

from pendulum.dynamics.model import (
    accelerations,
    state_derivative,
    state_derivative_acceleration,
)
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


def test_pendulum_acceleration_changes_sign_with_theta(
    pendulum_params: PendulumParameters,
) -> None:
    positive_theta = np.array([0.0, 0.4, 0.0, 0.0])
    negative_theta = np.array([0.0, -0.4, 0.0, 0.0])

    _, positive_theta_ddot = accelerations(
        positive_theta, torque=0.0, params=pendulum_params
    )
    _, negative_theta_ddot = accelerations(
        negative_theta, torque=0.0, params=pendulum_params
    )

    assert positive_theta_ddot < 0.0
    assert negative_theta_ddot > 0.0
    np.testing.assert_allclose(positive_theta_ddot, -negative_theta_ddot)


def test_opposite_torques_produce_opposite_arm_accelerations(
    pendulum_params: PendulumParameters,
) -> None:
    positive_phi_ddot, _ = accelerations(
        np.zeros(4), torque=0.1, params=pendulum_params
    )
    negative_phi_ddot, _ = accelerations(
        np.zeros(4), torque=-0.1, params=pendulum_params
    )

    assert positive_phi_ddot > 0.0
    assert negative_phi_ddot < 0.0
    np.testing.assert_allclose(positive_phi_ddot, -negative_phi_ddot)


def test_gravity_accelerates_displaced_pendulum_toward_downward_position(
    pendulum_params: PendulumParameters,
) -> None:
    state = np.array([0.0, np.pi / 2, 0.0, 0.0])

    _, theta_ddot = accelerations(state, torque=0.0, params=pendulum_params)

    assert theta_ddot < 0.0


def test_derivative_remains_finite_over_reasonable_states(
    pendulum_params: PendulumParameters,
) -> None:
    states = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([np.pi, -np.pi / 2, 2.0, -2.0]),
        np.array([-np.pi, np.pi / 2, -5.0, 5.0]),
        np.array([2 * np.pi, np.pi, 10.0, -10.0]),
    ]
    torques = [
        -pendulum_params.motor_torque_limit,
        0.0,
        pendulum_params.motor_torque_limit,
    ]

    for state in states:
        for torque in torques:
            derivative = state_derivative(state, torque, pendulum_params)

            assert np.all(np.isfinite(derivative))


def test_acceleration_input_matches_instantaneous_torque_dynamics(pendulum_params):
    rng = np.random.default_rng(42)
    for _ in range(50):
        state = rng.uniform(-3.0, 3.0, size=4)
        torque = rng.uniform(-0.5, 0.5)
        expected = state_derivative(state, torque, pendulum_params)
        actual = state_derivative_acceleration(state, expected[2], pendulum_params)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

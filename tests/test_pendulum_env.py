from unittest.mock import patch

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pendulum.dynamics.parameters import PendulumParameters
from pendulum.envs.pendulum_env import PendulumEnv


@pytest.fixture
def env(pendulum_params: PendulumParameters) -> PendulumEnv:
    return PendulumEnv(
        params=pendulum_params,
        dt=0.01,
        max_episode_steps=3,
    )


def test_env_passes_gymnasium_checker(env: PendulumEnv) -> None:
    check_env(env, skip_render_check=True)


def test_reset_is_seeded_and_resets_simulator(env: PendulumEnv) -> None:
    env.sample_distribution_factor = np.array([0.1, 0.2, 0.3, 0.4])

    first_observation, first_info = env.reset(seed=123)
    env.step(np.array([0.5], dtype=np.float32))
    second_observation, second_info = env.reset(seed=123)

    np.testing.assert_array_equal(second_observation, first_observation)
    assert env.steps == 0
    assert first_info == second_info == {}


def test_reset_without_noise_starts_at_equilibrium(env: PendulumEnv) -> None:
    observation, _ = env.reset()

    np.testing.assert_allclose(env.simulator.get_state(), env.state_e)
    np.testing.assert_allclose(
        observation,
        np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-7,
    )


def test_sampled_theta_error_is_limited_by_cutoff(env: PendulumEnv) -> None:
    env.sample_distribution_factor = np.array([0.0, 100.0, 0.0, 0.0])

    env.reset(seed=1)

    theta_error = env.simulator.get_state()[1] - env.state_e[1]
    assert abs(theta_error) <= env.theta_cutoff_error


def test_observation_represents_simulator_state(env: PendulumEnv) -> None:
    state = np.array([0.25, -0.75, 1.5, -2.0])
    env.simulator.reset(state)

    observation = env._get_observation()

    expected = np.array(
        [state[0], np.sin(state[1]), np.cos(state[1]), state[2], state[3]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(observation, expected)
    assert env.observation_space.contains(observation)


def test_action_is_scaled_to_motor_torque_limit(env: PendulumEnv) -> None:
    action = np.array([0.4], dtype=np.float32)

    with patch.object(env.simulator, "step") as simulator_step:
        env.step(action)

    simulator_step.assert_called_once_with(
        pytest.approx(0.4 * env.params.motor_torque_limit)
    )


def test_episode_truncates_at_exact_step_limit(env: PendulumEnv) -> None:
    env.reset()

    for expected_steps in range(1, env.max_episode_steps + 1):
        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))

        assert env.steps == expected_steps
        assert not terminated
        assert truncated is (expected_steps == env.max_episode_steps)


@pytest.mark.parametrize(
    ("theta_error", "expected_terminated"),
    [
        (0.5, False),
        (1.0, False),
        (1.01, True),
        (-1.01, True),
    ],
)
def test_episode_termination_uses_theta_cutoff(
    pendulum_params: PendulumParameters,
    theta_error: float,
    expected_terminated: bool,
) -> None:
    env = PendulumEnv(
        params=pendulum_params,
        dt=0.01,
        max_episode_steps=3,
        theta_cutoff_error=1.0,
    )
    env.simulator.reset(env.state_e + np.array([0.0, theta_error, 0.0, 0.0]))

    with patch.object(env.simulator, "step"):
        _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))

    assert bool(terminated) is expected_terminated


def test_termination_is_periodic_in_pendulum_angle(env: PendulumEnv) -> None:
    env.simulator.reset(np.array([0.0, -np.pi, 0.0, 0.0]))

    with patch.object(env.simulator, "step"):
        _, _, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))

    assert not terminated


def test_reward_is_periodic_in_pendulum_angle(env: PendulumEnv) -> None:
    rewards = []
    for theta in (-np.pi, np.pi, 3 * np.pi):
        env.simulator.reset(np.array([0.0, theta, 0.0, 0.0]))
        rewards.append(env._get_reward(torque=0.0))

    assert rewards == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_reward_applies_state_and_torque_costs(env: PendulumEnv) -> None:
    state = env.state_e + np.array([1.0, 0.2, 2.0, -3.0])
    torque = 0.1
    env.simulator.reset(state)

    reward = env._get_reward(torque)

    expected_cost = 1.0 + 20.0 * 0.2**2 + 0.1 * 2.0**2 + 0.1 * 3.0**2
    expected_cost += 4.0 * torque**2
    assert reward == pytest.approx(-expected_cost)

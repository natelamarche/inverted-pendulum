from unittest.mock import call, patch

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
        sample_range_lower=np.zeros(4),
        sample_range_upper=np.zeros(4),
    )


def test_env_passes_gymnasium_checker(env: PendulumEnv) -> None:
    env.sample_range_upper = np.array([0.1, 0.2, 0.3, 0.4])
    check_env(env, skip_render_check=True)


def test_reset_is_seeded_and_resets_simulator(env: PendulumEnv) -> None:
    env.sample_range_lower = np.array([0.01, 0.02, 0.03, 0.04])
    env.sample_range_upper = np.array([0.1, 0.2, 0.3, 0.4])

    for seed in range(10):
        first_observation, first_info = env.reset(seed=seed)
        first_state = env.simulator.get_state().copy()
        env.step(np.array([0.5], dtype=np.float32))
        second_observation, second_info = env.reset(seed=seed)

        np.testing.assert_array_equal(second_observation, first_observation)
        np.testing.assert_array_equal(env.simulator.get_state(), first_state)
        assert env.steps == 0
        assert first_info == second_info == {}


def test_explicit_reset_starts_at_equilibrium(env: PendulumEnv) -> None:
    observation, _ = env.reset(options={"initial_state": env.state_e.copy()})

    np.testing.assert_allclose(env.simulator.get_state(), env.state_e)
    np.testing.assert_allclose(
        observation,
        np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-7,
    )


def test_sampling_mixes_balance_and_swing_up_without_clipping(env: PendulumEnv):
    env.theta_cutoff_error = np.pi / 4
    env.sample_range_lower = np.array([0.02, np.pi / 2, 0.03, 0.04])
    env.sample_range_upper = np.array([0.1, np.pi, 0.1, 0.2])
    errors = []
    for seed in range(256):
        env.reset(seed=seed)
        errors.append(env.simulator.get_state() - env.state_e)
    errors = np.array(errors)
    magnitudes = np.abs(errors)
    balance = magnitudes[:, 1] < np.pi / 16
    swing = magnitudes[:, 1] >= np.pi / 2
    assert np.all(balance | swing)
    assert 80 < balance.sum() < 176
    assert np.all(magnitudes[balance] <= np.array([0.1, np.pi / 16, 0.1, 0.2]))
    assert np.all(magnitudes[swing] >= env.sample_range_lower - 1e-12)
    assert np.all(magnitudes[swing] <= env.sample_range_upper + 1e-12)
    for samples in (errors[balance], errors[swing]):
        assert np.all(np.any(samples > 0, axis=0))
        assert np.all(np.any(samples < 0, axis=0))


def test_observation_represents_simulator_state(env: PendulumEnv) -> None:
    state = np.array([0.25, -0.75, 1.5, -2.0])
    env.simulator.reset(state)

    observation = env._get_observation()

    expected = np.array(
        [
            state[0] / (2 * np.pi),
            np.sin(state[1]),
            np.cos(state[1]),
            state[2] / 10.0,
            state[3] / 15.0,
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(observation, expected)
    assert env.observation_space.contains(observation)


def test_action_is_scaled_and_held_for_one_control_step(env: PendulumEnv) -> None:
    action = np.array([0.4], dtype=np.float32)

    with patch.object(env.simulator, "step") as simulator_step:
        env.step(action)

    assert (
        simulator_step.call_args_list
        == [call(pytest.approx(0.4 * env.params.motor_torque_limit))]
        * env.controller_stride
    )
    assert simulator_step.call_count * env.simulator.dt == pytest.approx(env.dt)
    assert env.steps == 1


def test_episode_truncates_at_exact_step_limit(env: PendulumEnv) -> None:
    env.reset(options={"initial_state": env.state_e.copy()})

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
        sample_range_lower=np.zeros(4),
        sample_range_upper=np.zeros(4),
        theta_cutoff_error=1.0,
    )
    env.reset(options={"initial_state": env.state_e.copy()})

    with patch.object(env.simulator, "step"):
        env.step(np.array([0.0], dtype=np.float32))
        assert env.flag_captured
        env.simulator.reset(env.state_e + np.array([0.0, theta_error, 0.0, 0.0]))
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

    expected_cost = 10.0 + 20.0 * 0.2**2 + 1.0 * 2.0**2 + 0.1 * 3.0**2
    expected_cost += 10.0 * torque**2
    assert reward == pytest.approx(-expected_cost)


def test_torque_costs_observation_history_and_reset(env: PendulumEnv):
    env.max_episode_steps = 10
    env.reset(options={"initial_state": env.state_e.copy()})
    limit = env.params.motor_torque_limit
    with patch.object(env.simulator, "step"):
        for action, coefficient in [(1.0, 15.0), (1.0, 10.0), (-1.0, 30.0)]:
            obs, reward, terminated, truncated, _ = env.step(np.array([action]))
            assert not terminated and not truncated
            assert reward == pytest.approx(-coefficient * limit**2)
            assert obs[-1] == pytest.approx(action)
            np.testing.assert_array_equal(obs, env._get_observation())
        obs, _ = env.reset(options={"initial_state": env.state_e.copy()})
        assert obs[-1] == 0.0
        assert env.previous_torque == 0.0
        assert env.step(np.array([0.0]))[1] == pytest.approx(0.0)


def test_capture_latches_and_reset_allows_another_swing_up(env: PendulumEnv):
    env.max_episode_steps = 10
    with patch.object(env.simulator, "step"):
        env.reset(options={"initial_state": [0.0, 0.0, 0.0, 0.0]})
        assert not env.step(np.array([0.0]))[2]
        assert not env.flag_captured
        env.simulator.reset(env.state_e.copy())
        assert not env.step(np.array([0.0]))[2]
        assert env.flag_captured
        env.simulator.reset(np.zeros(4))
        _, reward, terminated, _, _ = env.step(np.array([0.0]))
        assert terminated
        assert reward == pytest.approx(-20 * np.pi**2 - 1_000_000)
        env.reset(options={"initial_state": [0.0, 0.0, 0.0, 0.0]})
        assert not env.flag_captured
        assert not env.step(np.array([0.0]))[2]


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("phi", [-2 * np.pi - 0.01, 2 * np.pi + 0.01])
def test_arm_limit_applies_before_and_after_capture(env, captured, phi):
    with patch.object(env.simulator, "step"):
        env.reset(options={"initial_state": env.state_e.copy()})
        if captured:
            env.step(np.array([0.0]))
        env.simulator.reset(np.array([phi, np.pi if captured else 0.0, 0.0, 0.0]))
        assert env.step(np.array([0.0]))[2]

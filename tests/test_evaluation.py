from unittest.mock import patch

import numpy as np
import pytest
import torch

from pendulum.envs.pendulum_env import PendulumEnv
from pendulum.rl.evaluation import evaluate_policy
from pendulum.rl.policy import ActorCritic


@pytest.mark.parametrize("terminated", [False, True])
def test_evaluation_metrics_and_terminal_step(pendulum_params, terminated):
    env = PendulumEnv(
        params=pendulum_params,
        dt=0.01,
        max_episode_steps=2,
        sample_range_lower=np.zeros(4),
        sample_range_upper=np.zeros(4),
    )
    policy = ActorCritic(5, 1)
    with torch.no_grad():
        for parameter in policy.actor.parameters():
            parameter.zero_()
        policy.actor[-1].bias.fill_(np.arctanh(0.4))

    def step(action):
        np.testing.assert_allclose(action, [0.4], atol=1e-6)
        env.steps += 1
        # Include a full revolution to check angle wrapping.
        env.simulator.state[1] = np.pi + 2 * np.pi + 0.2
        done = env.steps == 2
        return env._get_observation(), -3.0, terminated and done, done, {}

    with patch.object(env, "step", side_effect=step):
        metrics = evaluate_policy(policy, env, seeds=[1, 2])

    assert metrics == pytest.approx(
        {
            "mean_return": -6.0,
            "mean_episode_steps": 2.0,
            "mean_episode_seconds": 0.02,
            "survival_rate": 0.0 if terminated else 1.0,
            "angle_rmse_rad": 0.2,
            "torque_rms_nm": 0.4 * pendulum_params.motor_torque_limit,
        }
    )
    assert policy.training


def test_evaluation_is_repeatable_without_consuming_torch_rng(pendulum_params):
    env = PendulumEnv(
        params=pendulum_params,
        dt=0.01,
        max_episode_steps=3,
        sample_range_lower=np.zeros(4),
        sample_range_upper=np.array([0.02, 0.05, 0.05, 0.1]),
    )
    policy = ActorCritic(5, 1)
    policy.eval()
    rng_state = torch.get_rng_state().clone()
    first = evaluate_policy(policy, env, seeds=[10, 11])
    second = evaluate_policy(policy, env, seeds=[10, 11])
    assert first == second
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert not policy.training


def test_evaluation_rejects_empty_seeds():
    with pytest.raises(ValueError, match="at least one seed"):
        evaluate_policy(ActorCritic(5, 1), None, seeds=[])

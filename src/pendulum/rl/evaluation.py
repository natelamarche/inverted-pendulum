from collections.abc import Sequence

import numpy as np
import torch

from pendulum.envs.pendulum_env import PendulumEnv
from pendulum.rl.policy import ActorCritic


def evaluate_policy(
    policy: ActorCritic,
    env: PendulumEnv,
    seeds: Sequence[int] = tuple(range(10_000, 10_020)),
) -> dict[str, float]:
    if len(seeds) == 0:
        raise ValueError("Evaluation requires at least one seed")

    returns = []
    lengths = []
    survivors = 0
    angle_squared_sum = 0.0
    torque_squared_sum = 0.0
    was_training = policy.training
    parameter = next(policy.parameters())
    policy.eval()
    try:
        with torch.no_grad():
            for seed in seeds:
                observation, _ = env.reset(seed=seed)
                episode_return = 0.0
                length = 0
                while True:
                    obs = torch.as_tensor(
                        observation, dtype=parameter.dtype, device=parameter.device
                    )
                    action = torch.tanh(policy.actor(obs)).cpu().numpy()
                    observation, reward, terminated, truncated, _ = env.step(action)
                    episode_return += reward
                    length += 1

                    theta_error = env.simulator.get_state()[1] - env.state_e[1]
                    theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
                    torque = env.previous_torque
                    angle_squared_sum += float(theta_error**2)
                    torque_squared_sum += torque**2

                    if terminated or truncated:
                        survivors += int(truncated and not terminated)
                        returns.append(episode_return)
                        lengths.append(length)
                        break
    finally:
        policy.train(was_training)

    steps = sum(lengths)
    return {
        "mean_return": float(np.mean(returns)),
        "mean_episode_steps": float(np.mean(lengths)),
        "mean_episode_seconds": float(np.mean(lengths) * env.dt),
        "survival_rate": survivors / len(seeds),
        "angle_rmse_rad": float(np.sqrt(angle_squared_sum / steps)),
        "torque_rms_nm": float(np.sqrt(torque_squared_sum / steps)),
    }

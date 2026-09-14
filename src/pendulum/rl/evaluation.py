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
    up_survivors = 0
    up_total = 0
    angle_squared_sum = 0.0
    torque_squared_sum = 0.0
    was_training = policy.training
    parameter = next(policy.parameters())
    policy.eval()
    try:
        with torch.no_grad():
            for seed in seeds:
                observation, info = env.reset(seed=seed)
                upright = info['upright']
                up_total += 1 if upright else 0
                
                episode_return = 0.0
                length = 0
                while True:
                    obs = torch.as_tensor(
                        observation, dtype=parameter.dtype, device=parameter.device
                    )
                    action = torch.tanh(policy.get_action_mean(obs)).cpu().numpy()
                    observation, reward, terminated, truncated, _ = env.step(action)
                    episode_return += reward
                    length += 1

                    theta_error = env.simulator.get_state()[1] - env.state_e[1]
                    theta_error = np.arctan2(np.sin(theta_error), np.cos(theta_error))
                    torque = env.previous_torque
                    angle_squared_sum += float(theta_error**2)
                    torque_squared_sum += torque**2

                    if terminated or truncated:
                        survived = truncated and not terminated
                        survivors += int(survived)
                        up_survivors += int(survived and upright)
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
        "action_std": float(torch.exp(policy.log_std)),
        "upright_survival_rate": float(up_survivors/up_total),
    }

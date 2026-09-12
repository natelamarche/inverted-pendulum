import csv
import argparse

from pathlib import Path

from pendulum.envs.pendulum_env import PendulumEnv
from pendulum.dynamics.parameters import PendulumParameters
from pendulum.rl.ppo import PPO
from pendulum.rl.policy import ActorCritic
from pendulum.rl.evaluation import evaluate_policy

from stable_baselines3.common.env_checker import check_env

import numpy as np


def make_env() -> PendulumEnv:
    # based on measured components
    params = PendulumParameters(
        arm_length=0.123,
        arm_inertia=4.904e-5,
        arm_damping=1.0e-4,
        pendulum_mass=0.0228,
        pendulum_com_length=0.1106,
        pendulum_com_inertia=2.261e-4,
        pendulum_damping=1.5e-4,
        motor_torque_limit=0.1,
    )

    env = PendulumEnv(
        params=params,
        dt=0.01,
        max_episode_steps=1000,
        theta_cutoff_error=np.pi,
        sample_range_lower=np.array([0.0, 0.0, 0.0, 0.0]),
        sample_range_upper=np.array([0.1, np.pi / 2, 0.1, 0.2]),
    )

    check_env(env)

    return env


def main():
    parser = argparse.ArgumentParser(description="RL Controller trainer")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}.")

    env = make_env()
    eval_env = make_env()
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = ActorCritic(5, 1)

    ppo = PPO(
        env=env,
        policy=policy,
        learning_rate=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=1.0,
        entropy_coef=0.001,
        rollout_steps=8192,
        batch_size=64,
        epochs=5,
    )

    if args.checkpoint is not None:
        ppo.load(args.checkpoint)

    total_timesteps = 1_000_000
    timesteps = 0
    iterations = 0
    eval_every_iterations = 4
    best_score = None

    def record_evaluation(writer, metrics_file):
        nonlocal best_score
        metrics = evaluate_policy(policy, eval_env)
        writer.writerow({"timesteps": timesteps, **metrics})
        metrics_file.flush()
        print(
            f"Evaluation at {timesteps:,} steps: "
            f"return={metrics['mean_return']:.2f}, "
            f"length={metrics['mean_episode_steps']:.1f} steps "
            f"({metrics['mean_episode_seconds']:.2f}s), "
            f"survival={metrics['survival_rate']:.0%}, "
            f"angle RMSE={metrics['angle_rmse_rad']:.4f} rad, "
            f"torque RMS={metrics['torque_rms_nm']:.4f} Nm"
        )

        score = (
            metrics["survival_rate"],
            metrics["mean_episode_steps"],
            -metrics["angle_rmse_rad"],
        )
        if best_score is None or score > best_score:
            best_score = score
            ppo.save(output_dir / "rl_controller_best.pth")

    try:
        with (output_dir / "evaluation.csv").open("w", newline="") as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=[
                    "timesteps",
                    "mean_return",
                    "mean_episode_steps",
                    "mean_episode_seconds",
                    "survival_rate",
                    "angle_rmse_rad",
                    "torque_rms_nm",
                ],
            )
            writer.writeheader()
            record_evaluation(writer, metrics_file)
            while timesteps < total_timesteps:
                ppo.train_iteration()
                timesteps += ppo.rollout_steps
                iterations += 1

                print(f"Iteration: {iterations}: {timesteps:,} steps")

                if (
                    iterations % eval_every_iterations == 0
                    or timesteps >= total_timesteps
                ):
                    record_evaluation(writer, metrics_file)
                if iterations % 25 == 0:
                    ppo.save(output_dir / "rl_controller_latest.pth")
        ppo.save(output_dir / "rl_controller.pth")
    finally:
        env.close()
        eval_env.close()

    print("Done:")
    print(f"Iterations: {iterations}: {timesteps:,} steps")


if __name__ == "__main__":
    main()

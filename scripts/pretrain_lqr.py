import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from pendulum.controllers.lqr import LQRController
from pendulum.envs.pendulum_env import PendulumEnv
from pendulum.rl.policy import ActorCritic
from train_rl import make_env


def make_teacher(env: PendulumEnv) -> LQRController:
    return LQRController(
        env.params,
        Q=env.state_error_reward_matrix,
        R=np.array([[env.torque_error_reward]]),
        dt=env.dt,
    )


def make_dataset(env, teacher, count, rng, bounds):
    percent_upright = 0.7
    count_upright = round(count * percent_upright)
    
    errors = rng.uniform(-1.0, 1.0, (count_upright, 4)) * bounds
    errors *= rng.choice([0.05, 0.2, 1.0], size=(count_upright, 1))
    states = env.state_e + errors
    observations = []
    targets = []
    for state in states:
        observation, _ = env.reset(options={"initial_state": state})
        observations.append(observation)
        targets.append(
            np.clip(
                teacher.get_discrete_action(state) / env.params.motor_torque_limit,
                -1.0,
                1.0,
            )
        )
    
    count_relaxed = count - count_upright
    
    relaxed_errors = rng.uniform(-1.0, 1.0, (count_relaxed, 4)) * np.array([np.pi/4, np.pi/4, 1.0, 1.0])
    
    relaxed_state_e = env.state_e + np.array([0, np.pi, 0, 0])
    relaxed_state_e[1] = np.arctan2(np.sin(relaxed_state_e[1]), np.cos(relaxed_state_e[1]))
    
    relaxed_states = relaxed_state_e + relaxed_errors
    
    for state in relaxed_states:
        observation, _ = env.reset(options={"initial_state": state})
        observations.append(observation)
        targets.append(0.0)
    
    return (
        torch.from_numpy(np.asarray(observations, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.float32)).unsqueeze(-1),
    )


def fit_actor(policy, training, validation, epochs, batch_size, learning_rate):
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=learning_rate)
    observations, targets = training
    validation_obs, validation_targets = validation
    best_weights = deepcopy(policy.actor.state_dict())
    with torch.no_grad():
        initial_mse = torch.mean(
            (torch.tanh(policy.get_action_mean(validation_obs)) - validation_targets)
            ** 2
        ).item()
    best_mse = initial_mse
    print(f"Initial held-out action MSE: {initial_mse:.8f}", flush=True)
    for epoch in range(1, epochs + 1):
        policy.train()
        for indices in torch.randperm(len(observations)).split(batch_size):
            prediction = torch.tanh(policy.get_action_mean(observations[indices]))
            loss = torch.mean((prediction - targets[indices]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite imitation loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 0.5)
            optimizer.step()
        policy.eval()
        with torch.no_grad():
            mse = torch.mean(
                (
                    torch.tanh(policy.get_action_mean(validation_obs))
                    - validation_targets
                )
                ** 2
            ).item()
        if mse < best_mse:
            best_mse = mse
            best_weights = deepcopy(policy.actor.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}: held-out action MSE={mse:.8f}", flush=True)
    policy.actor.load_state_dict(best_weights)
    return initial_mse, best_mse


def evaluate_local(env, action_fn, starts):
    """Require a full episode and the last two seconds within 5 deg / 0.5 rad/s."""
    successes = 0
    angle_squared = []
    torque_squared = []
    change_squared = []
    for state in starts:
        obs, _ = env.reset(options={"initial_state": state})
        held = []
        previous = 0.0
        for _ in range(env.max_episode_steps):
            obs, _, terminated, truncated, _ = env.step(
                np.array([action_fn(obs)], dtype=np.float32)
            )
            physical = env.simulator.get_state()
            error = np.arctan2(np.sin(physical[1] - np.pi), np.cos(physical[1] - np.pi))
            torque = env.previous_torque
            angle_squared.append(error**2)
            torque_squared.append(torque**2)
            change_squared.append((torque - previous) ** 2)
            previous = torque
            held.append(abs(error) < np.deg2rad(5) and abs(physical[3]) < 0.5)
            if terminated or truncated:
                break
        window = min(len(held), max(1, round(2.0 / env.dt)))
        successes += int(truncated and not terminated and all(held[-window:]))
    return {
        "success_rate": successes / len(starts),
        "angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(angle_squared)))),
        "torque_rms_nm": float(np.sqrt(np.mean(torque_squared))),
        "torque_change_rms_nm": float(np.sqrt(np.mean(change_squared))),
    }


def save_checkpoint(policy, output, ppo_learning_rate, metadata):
    # PPO.load expects one optimizer group containing actor, critic and log_std.
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_learning_rate)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pretraining": metadata,
        },
        output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("models/lqr_pretrained.pth")
    )
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--validation-samples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--initial-std",
        type=float,
        default=0.5,
        help="Pre-tanh exploration std for subsequent PPO training",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--tilt-deg", type=float, default=10.0)
    parser.add_argument("--arm-deg", type=float, default=5.0)
    parser.add_argument("--arm-speed", type=float, default=0.2)
    parser.add_argument("--pendulum-speed", type=float, default=0.5)
    args = parser.parse_args()
    for name in (
        "samples",
        "validation_samples",
        "epochs",
        "batch_size",
        "learning_rate",
        "ppo_learning_rate",
        "initial_std",
        "eval_episodes",
        "tilt_deg",
        "arm_deg",
        "arm_speed",
        "pendulum_speed",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.tilt_deg >= 45:
        parser.error("--tilt-deg must be below 45 for this local balancing task")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = make_env()
    try:
        teacher = make_teacher(env)
        policy = ActorCritic(6, 1, initial_std=args.initial_std)
        bounds = np.array(
            [
                np.deg2rad(args.arm_deg),
                np.deg2rad(args.tilt_deg),
                args.arm_speed,
                args.pendulum_speed,
            ]
        )
        training = make_dataset(env, teacher, args.samples, rng, bounds)
        validation = make_dataset(env, teacher, args.validation_samples, rng, bounds)
        initial_mse, best_mse = fit_actor(
            policy,
            training,
            validation,
            args.epochs,
            args.batch_size,
            args.learning_rate,
        )
        starts = env.state_e + rng.uniform(-1, 1, (args.eval_episodes, 4)) * bounds
        teacher_metrics = evaluate_local(
            env,
            lambda obs: np.clip(
                teacher.get_discrete_action(env.simulator.get_state())
                / env.params.motor_torque_limit,
                -1,
                1,
            ),
            starts,
        )
        with torch.inference_mode():
            actor_metrics = evaluate_local(
                env,
                lambda obs: torch.tanh(
                    policy.get_action_mean(torch.from_numpy(obs))
                ).item(),
                starts,
            )
        print(f"LQR local evaluation: {teacher_metrics}", flush=True)
        print(f"Actor local evaluation: {actor_metrics}", flush=True)
        save_checkpoint(
            policy,
            args.output,
            args.ppo_learning_rate,
            {
                "source": "local LQR imitation; actor only",
                "seed": args.seed,
                "bounds": bounds.tolist(),
                "initial_validation_mse": initial_mse,
                "best_validation_mse": best_mse,
                "teacher_metrics": teacher_metrics,
                "actor_metrics": actor_metrics,
            },
        )
        print(f"Saved {args.output}. Critic is untrained; PPO optimizer is fresh.")
        if actor_metrics["success_rate"] < 1.0:
            print(
                "Some local trials failed: inspect this checkpoint before PPO training."
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()

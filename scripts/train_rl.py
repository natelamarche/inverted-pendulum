from pendulum.envs.pendulum_env import PendulumEnv
from pendulum.dynamics.parameters import PendulumParameters
from pendulum.rl.ppo import PPO
from pendulum.rl.policy import ActorCritic

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
        max_episode_steps=500, 
        theta_cutoff_error=np.pi / 2,
        sample_distribution_factor=np.array([0.02, 0.05, 0.05, 0.10])
    )

    check_env(env)

    return env


def main():
    env: PendulumEnv = make_env()

    policy = ActorCritic(5, 1)
    
    ppo = PPO(
        env=env,
        policy=policy,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=1.0,
        entropy_coef=0.001,
        rollout_steps=8192,
        batch_size=64,
        epochs=5
    )
    
    total_timesteps = 1_000_000
    timesteps = 0
    iterations = 0
    
    try:
        while timesteps < total_timesteps:
            ppo.train_iteration()
            timesteps += ppo.rollout_steps
            iterations += 1
            
            print(f"Iteration: {iteration}: {timesteps:,} steps")
            
            if iteration % 25 == 0:
                ppo.save("models/rl_controller_latest.ph")
        ppo.save("models/rl_controller.ph")
    finally:
        env.close()    
    
    print(f"Done:") 
    print(f"Iterations: {iterations}: {timesteps:,} steps")

if __name__ == "__main__":
    main()

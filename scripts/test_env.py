from pendulum.envs.pendulum_env import PendulumEnv
from pendulum.dynamics.parameters import PendulumParameters
from pendulum.simulation.visualtization import plot_state_history

import numpy as np


def main():
    params = PendulumParameters(
        arm_length=0.2,
        arm_inertia=0.01,
        arm_damping=0.002,
        pendulum_mass=0.1,
        pendulum_com_length=0.15,
        pendulum_com_inertia=0.001,
        pendulum_damping=0.001,
        motor_torque_limit=0.5,
    )

    dt = 0.01
    env = PendulumEnv(params=params, dt=dt, max_episode_steps=1000)

    obs, info = env.reset()

    timestamps = 1000

    state_history = np.empty((timestamps, 4))
    time_history = np.empty(timestamps)

    for i in range(timestamps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert np.all(np.isfinite(obs))

        state_history[i, :] = env.simulator.get_state()
        time_history[i] = i * dt

        if terminated or truncated:
            print(f"{'Terminated' if terminated else 'Truncated'} at i={i}")
            obs, info = env.reset()

    plot_state_history(time_history, state_history)


if __name__ == "__main__":
    main()

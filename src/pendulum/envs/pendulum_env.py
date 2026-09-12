import gymnasium as gym
import numpy as np

from pendulum.dynamics.parameters import PendulumParameters
from pendulum.simulation.simulator import Simulator


class PendulumEnv(gym.Env):
    def __init__(
        self,
        params: PendulumParameters,
        dt: float,
        max_episode_steps: int,
        sample_range_lower: np.ndarray,
        sample_range_upper: np.ndarray,
        theta_cutoff_error: float = np.pi / 2,
    ):
        self.params: PendulumParameters = params

        # [phi, sin(theta), cos(theta), phi_dot, theta_dot]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -1.0, -1.0, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, 1.0, 1.0, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.state_e = np.array([0.0, np.pi, 0.0, 0.0])
        self.torque_e = 0.0

        self.steps = 0
        self.max_episode_steps = max_episode_steps
        self.theta_cutoff_error = theta_cutoff_error

        # [phi, theta, phi_dot, theta_dot]
        self.sample_range_lower = sample_range_lower
        self.sample_range_upper = sample_range_upper
        initial_state: np.ndarray = self._sample_initial_state()

        self.simulator = Simulator(self.params, dt)
        self.simulator.reset(initial_state)

        self.state_error_reward_matrix = np.diag([1.0, 20.0, 0.1, 0.1])
        self.torque_error_reward = 4.0

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if options is not None and "initial_state" in options:
            initial_state = np.asarray(options["initial_state"], dtype=float)
            if initial_state.shape != (4,) or not np.all(np.isfinite(initial_state)):
                raise ValueError("initial_state must contain four finite values")
        else:
            initial_state = self._sample_initial_state()

        self.simulator.reset(initial_state)

        observation = self._get_observation()

        self.steps = 0

        return observation, {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        torque = action[0] * self.params.motor_torque_limit

        self.simulator.step(torque)
        self.steps += 1

        observation = self._get_observation()

        reward = self._get_reward(torque)

        theta_error = self.simulator.get_state()[1] - self.state_e[1]
        wrapped_theta_error = np.atan2(np.sin(theta_error), np.cos(theta_error))

        terminated = bool(abs(wrapped_theta_error) > self.theta_cutoff_error)

        truncated = self.steps >= self.max_episode_steps

        return observation, reward, terminated, truncated, {}

    def _get_observation(self) -> np.ndarray:
        phi, theta, phi_dot, theta_dot = self.simulator.get_state()

        return np.array(
            [phi, np.sin(theta), np.cos(theta), phi_dot, theta_dot], dtype=np.float32
        )

    def _get_reward(self, torque: float) -> float:
        state = self.simulator.get_state()

        state_error = state - self.state_e
        state_error[1] = np.arctan2(np.sin(state_error[1]), np.cos(state_error[1]))

        state_reward = state_error @ self.state_error_reward_matrix @ state_error

        torque_error = torque - self.torque_e
        torque_reward = self.torque_error_reward * torque_error**2

        return -(state_reward + torque_reward)

    def _sample_initial_state(self) -> np.ndarray:
        sample = (
            self.np_random.random(size=(4,))
            * (self.sample_range_upper - self.sample_range_lower)
            + self.sample_range_lower
        ) * self.np_random.choice([-1, 1], size=4)
        sample[1] = np.clip(
            sample[1], -self.theta_cutoff_error, self.theta_cutoff_error
        )

        return self.state_e + sample

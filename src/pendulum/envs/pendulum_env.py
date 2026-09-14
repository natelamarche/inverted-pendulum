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
            low=np.array(
                [-np.inf, -1.0, -1.0, -np.inf, -np.inf, -np.inf], dtype=np.float32
            ),
            high=np.array([np.inf, 1.0, 1.0, np.inf, np.inf, np.inf], dtype=np.float32),
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
        self.flag_captured = False

        # [phi, theta, phi_dot, theta_dot]
        self.sample_range_lower = sample_range_lower
        self.sample_range_upper = sample_range_upper
        initial_state, _ = self._sample_initial_state()

        self.controller_stride = 1
        self.dt = dt
        self.simulator = Simulator(self.params, self.dt / self.controller_stride)
        self.simulator.reset(initial_state)

        self.state_error_reward_matrix = np.diag([10.0, 20.0, 1.0, 0.1])
        self.torque_error_reward = 10.0
        self.torque_change_reward_flagged = 20.0
        self.torque_change_reward = 20.0
        self.previous_torque = 0.0

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        up = None
        
        if options is not None and "initial_state" in options:
            initial_state = np.asarray(options["initial_state"], dtype=float)
            if initial_state.shape != (4,) or not np.all(np.isfinite(initial_state)):
                raise ValueError("initial_state must contain four finite values")
        else:
            initial_state, up = self._sample_initial_state()

        self.simulator.reset(initial_state)

        self.steps = 0
        self.flag_captured = False
        self.previous_torque = 0.0

        observation = self._get_observation()

        info = {
            "upright": up
        }
        
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        requested_torque = float(action[0]) * self.params.motor_torque_limit
        torque = float(
            np.clip(
                requested_torque,
                -self.params.motor_torque_limit,
                self.params.motor_torque_limit,
            )
        )

        for _ in range(self.controller_stride):
            self.simulator.step(torque)

        self.steps += 1

        state = self.simulator.get_state()
        assert np.all(np.isfinite(state))

        theta_error = state[1] - self.state_e[1]
        theta_error = np.atan2(np.sin(theta_error), np.cos(theta_error))

        if abs(theta_error) < np.pi / 32:
            self.flag_captured = True

        terminated = bool(
            (abs(theta_error) > self.theta_cutoff_error and self.flag_captured)
            or abs(state[0]) > 2 * np.pi
        )

        truncated = self.steps >= self.max_episode_steps

        change_cost = (
            self.torque_change_reward_flagged
            if self.flag_captured
            else self.torque_change_reward
        ) * (torque - self.previous_torque) ** 2

        reward = (
            self._get_reward(torque) - change_cost - (1_000_000 if terminated else 0)
        )

        self.previous_torque = float(torque)
        observation = self._get_observation()

        return observation, reward, terminated, truncated, {}

    def _get_observation(self) -> np.ndarray:
        phi, theta, phi_dot, theta_dot = self.simulator.get_state()

        return np.array(
            [
                phi / (2 * np.pi),
                np.sin(theta),
                np.cos(theta),
                phi_dot / 10.0,
                theta_dot / 15.0,
                self.previous_torque / self.params.motor_torque_limit,
            ],
            dtype=np.float32,
        )

    def _get_reward(self, torque: float) -> float:
        state = self.simulator.get_state()

        state_error = state - self.state_e
        state_error[1] = np.arctan2(np.sin(state_error[1]), np.cos(state_error[1]))

        theta_error = state_error[1]
        theta_dot = state[3]

        near_upright = np.exp(-((theta_error / np.deg2rad(30.0)) ** 2))

        arrival_cost = 0.5 * near_upright * theta_dot**2

        state_reward = state_error @ self.state_error_reward_matrix @ state_error

        torque_error = torque - self.torque_e
        torque_reward = self.torque_error_reward * torque_error**2

        return -(state_reward + torque_reward + arrival_cost)

    def _sample_initial_state(self) -> np.ndarray:
        upright = self.np_random.choice([True, False])
        
        if not upright:
            sample = (
                self.np_random.random(size=(4,))
                * (self.sample_range_upper - self.sample_range_lower)
                + self.sample_range_lower
            ) * self.np_random.choice([-1, 1], size=4)
        else:
            sample = (
                self.np_random.random(size=(4,)) * np.array([0.1, np.pi / 16, 0.1, 0.2])
            ) * self.np_random.choice([-1, 1], size=4)

        return (self.state_e + sample, upright)

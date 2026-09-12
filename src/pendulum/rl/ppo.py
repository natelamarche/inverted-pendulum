from .buffer import RolloutBuffer
from .policy import ActorCritic
from ..envs.pendulum_env import PendulumEnv

import torch
import numpy as np


class PPO:
    def __init__(
        self,
        env: PendulumEnv,
        policy: ActorCritic,
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        value_coef: float,
        entropy_coef: float,
        rollout_steps: int,
        batch_size: int,
        epochs: int,
    ):
        self.env: PendulumEnv = env
        self.env.reset()

        self.rollout_buffer = RolloutBuffer(
            buffer_size=rollout_steps,
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        self.policy: ActorCritic = policy

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.rollout_steps = rollout_steps
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.epochs = epochs

    def select_action(
        self, observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            observation = torch.from_numpy(observation.copy())
            action, raw_action, log_prob, value = (
                t.numpy() for t in self.policy.act(observation)
            )
        return action, raw_action, log_prob, value

    def get_value(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            observation = torch.from_numpy(observation.copy())
            value = self.policy.get_value(observation).numpy()
        return value

    def simulate(self):
        self.rollout_buffer.reset()
        obs = self.env._get_observation()

        for _ in range(self.rollout_steps):
            action, raw_action, log_prob, value = self.select_action(obs)

            observation, reward, terminated, truncated, _ = self.env.step(action)

            next_value = self.get_value(observation)

            self.rollout_buffer.add(
                observation=obs,
                action=action,
                raw_action=raw_action,
                reward=reward,
                value=value,
                next_value=next_value,
                log_prob=log_prob,
                terminated=terminated,
                truncated=truncated,
            )

            if truncated or terminated:
                obs, _ = self.env.reset()
            else:
                obs = observation

        self.rollout_buffer.compute_returns_and_advantages()

    def _L_clip(self, ratio: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        return torch.min(
            ratio * advantage,
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantage,
        ).mean()

    def _L_value(self, returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return ((values - returns) ** 2).mean()

    def compute_loss(
        self,
        old_log_probs: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        entropies: torch.Tensor,
    ) -> torch.Tensor:
        ratios = (log_probs - old_log_probs).exp()
        return (
            -self._L_clip(ratios, advantages)
            + self.value_coef * self._L_value(returns, values)
            - self.entropy_coef * (entropies).mean()
        )

    def train_iteration(self):
        self.simulate()

        for _epoch in range(self.epochs):
            for (
                observations,
                raw_actions,
                old_log_probs,
                advantages,
                returns,
            ) in self.rollout_buffer.get_batches(self.batch_size):
                observations = torch.from_numpy(observations)
                raw_actions = torch.from_numpy(raw_actions)
                old_log_probs = torch.from_numpy(old_log_probs)
                advantages = torch.from_numpy(advantages)
                returns = torch.from_numpy(returns)

                log_probs, entropies, values = self.policy.evaluate_action(
                    observations, raw_actions
                )

                loss = self.compute_loss(
                    old_log_probs, log_probs, advantages, returns, values, entropies
                )

                self.optimizer.zero_grad()
                loss.backward()

                actor_parameters = [
                    *self.policy.actor.parameters(),
                    self.policy.log_std,
                ]
                torch.nn.utils.clip_grad_norm_(actor_parameters, max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), max_norm=0.5
                )

                self.optimizer.step()

    def save(self, path: str):
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=True)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

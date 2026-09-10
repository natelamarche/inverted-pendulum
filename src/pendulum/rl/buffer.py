import numpy as np


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        gamma: float,
        gae_lambda: float,
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros((buffer_size, observation_dim), dtype=np.float32)

        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)

        self.raw_actions = np.zeros((buffer_size, action_dim), dtype=np.float32)

        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.next_values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)

        self.terminates = np.zeros(buffer_size, dtype=np.bool_)
        self.truncates = np.zeros(buffer_size, dtype=np.bool_)

        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        raw_action: np.ndarray,
        reward: float,
        value: float,
        next_value: float,
        log_prob: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        assert self.pos != self.buffer_size

        self.observations[self.pos] = observation

        self.actions[self.pos] = action

        self.raw_actions[self.pos] = raw_action

        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.next_values[self.pos] = next_value
        self.log_probs[self.pos] = log_prob

        self.terminates[self.pos] = terminated
        self.truncates[self.pos] = truncated

        self.pos += 1

    def reset(self) -> None:
        self.observations.fill(0)

        self.actions.fill(0)

        self.raw_actions.fill(0)

        self.rewards.fill(0)
        self.values.fill(0)
        self.next_values.fill(0)
        self.log_probs.fill(0)

        self.terminates.fill(0)
        self.truncates.fill(0)

        self.advantages.fill(0)
        self.returns.fill(0)

        self.pos = 0

    def compute_returns_and_advantages(self) -> None:
        last_gae = 0.0

        for i in range(self.pos - 1, -1, -1):
            bootstrap = 1.0 - self.terminates[i]
            continue_gae = 1.0 - (self.terminates[i] or self.truncates[i])

            td_error = (
                self.rewards[i]
                + self.gamma * bootstrap * self.next_values[i]
                - self.values[i]
            )

            last_gae = td_error + self.gamma * self.gae_lambda * continue_gae * last_gae

            self.advantages[i] = last_gae

        self.returns = self.advantages + self.values

        self.advantages[: self.pos] = (
            self.advantages[: self.pos] - self.advantages[: self.pos].mean()
        ) / (self.advantages[: self.pos].std() + 1e-8)

    def get_batches(
        self,
        batch_size: int,
    ):
        indices = np.random.permutation(self.pos)

        for start in range(0, self.pos, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield (
                self.observations[batch_indices],
                self.raw_actions[batch_indices],
                self.log_probs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
            )

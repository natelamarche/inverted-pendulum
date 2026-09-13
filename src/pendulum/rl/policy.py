import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, initial_std: float = 0.5):
        super().__init__()
        # The final observation is previous torque, available only to the critic.
        self.actor = nn.Sequential(
            nn.Linear(observation_dim - 1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        self.log_std = nn.Parameter(
            torch.full((action_dim,), math.log(initial_std), dtype=torch.float32)
        )

        self.critic = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs[..., :-1])

    def get_action_distribution(self, obs: torch.Tensor) -> Normal:
        mu = self.get_action_mean(obs)
        std = torch.exp(self.log_std)

        return Normal(mu, std)

    def get_log_probability(
        self, dist: Normal, raw_action: torch.Tensor
    ) -> torch.Tensor:
        log_prob = dist.log_prob(raw_action)

        # Change of variables correction for tanh
        log_prob -= 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))

        return log_prob.sum(dim=-1)

    def act(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_action_distribution(obs)

        raw_action = dist.sample()
        action = torch.tanh(raw_action)

        log_prob = self.get_log_probability(dist, raw_action)

        value = self.critic(obs).squeeze(-1)

        return action, raw_action, log_prob, value

    def evaluate_action(
        self, obs: torch.Tensor, raw_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_action_distribution(obs)

        log_prob = self.get_log_probability(dist, raw_action)

        # fresh sample from CURRENT dist for entropy estimate
        entropy_raw_action = dist.rsample()
        entropy_log_prob = self.get_log_probability(dist, entropy_raw_action)
        entropy = -entropy_log_prob

        value = self.critic(obs).squeeze(-1)

        return log_prob, entropy, value

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

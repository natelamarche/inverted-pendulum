from pathlib import Path

import numpy as np
import torch

from pendulum.controllers.controller import Controller
from pendulum.rl.policy import ActorCritic


class RLController(Controller):
    def __init__(self, policy: ActorCritic):
        self.policy = policy
        self.policy.eval()
        super().__init__(self._predict)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "RLController":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        policy = ActorCritic(observation_dim=6, action_dim=1)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        return cls(policy)

    def _predict(self, observation: np.ndarray) -> float:
        parameter = next(self.policy.parameters())
        with torch.inference_mode():
            obs = torch.as_tensor(
                observation, dtype=parameter.dtype, device=parameter.device
            )
            return float(torch.tanh(self.policy.actor(obs)).item())

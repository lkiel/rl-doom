import typing as t

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticCnnPolicy


class RandomizedActorCritic(ActorCriticCnnPolicy):
    def __init__(self, exploration_factor: float, **kwargs):
        super().__init__(**kwargs)
        self.exploration_factor = exploration_factor

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> t.Tuple[th.Tensor, th.Tensor, th.Tensor]:
        actions, values, log_prob = super().forward(obs, deterministic)

        if np.random.rand() < self.exploration_factor:
            actions = th.from_numpy(np.random.randint(self.action_space.n, size=len(actions)))

        return actions, values, log_prob

import gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, **kwargs):
        super().__init__(observation_space, features_dim)

        channels, height, width = observation_space.shape

        self.cnn = nn.Sequential(
            nn.LayerNorm([channels, height, width]),
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0, bias=False),
            nn.LayerNorm([32, 24, 39]),  # TODO: find automatically the weights of the layer norm
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LayerNorm([64, 11, 18]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LayerNorm([64, 9, 16]),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim, bias=False),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

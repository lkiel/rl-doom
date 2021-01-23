import gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from models.hooks import Hooks


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, norm: str = None, **kwargs):
        super().__init__(observation_space, features_dim)

        channels, height, width = observation_space.shape

        layers = []

        # Initialize first layer
        if norm == 'layer': layers.append(nn.LayerNorm([channels, height, width]))
        if norm == 'batch': layers.append(nn.BatchNorm2d(channels))

        layers = self.add_conv_layer(layers, channels, 32, ks=8, s=4, p=0, norm=norm, **kwargs)
        layers = self.add_conv_layer(layers, 32, 64, ks=4, s=2, p=0, norm=norm, **kwargs)
        layers = self.add_conv_layer(layers, 64, 64, ks=3, s=1, p=0, norm=norm, **kwargs)
        self.cnn = nn.Sequential(*layers, nn.Flatten())

        n_flatten = self.compute_shape(self.cnn)[1]

        self.linear = nn.Sequential(*self.linear_layer(n_flatten, features_dim, norm=norm))

        self.hooks = Hooks(self.cnn, self.linear)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

    # TODO: improve organisation
    def add_conv_layer(self, previous_layers, n_in, n_out, ks, s, p, norm=None, **kwargs):
        layers = previous_layers + [nn.Conv2d(n_in, n_out, ks, s, p, bias=norm is None)]

        if norm == 'layer':
            layers.append(nn.LayerNorm(self.compute_shape(layers)[1:]))

        layers.append(nn.LeakyReLU(**kwargs))

        if norm == 'batch':
            layers.append(nn.BatchNorm2d(self.compute_shape(layers)[1]))

        return layers

    def linear_layer(self, n_in, n_out, norm=None, **kwargs):
        layers = [nn.Linear(n_in, n_out, bias=norm is None),]

        if norm == 'layer':
            layers.append(nn.LayerNorm(n_out))

        layers.append(nn.LeakyReLU(**kwargs))

        if norm == 'batch':
            layers.append(nn.BatchNorm1d(n_out))

        return layers

    def compute_shape(self, layers):
        with th.no_grad():
            return nn.Sequential(*layers)(th.as_tensor(self._observation_space.sample()[None]).float()).shape


if __name__ == '__main__':
    from gym import spaces
    import numpy as np
    print(CNNFeatureExtractor(spaces.Box(0, 1, (3, 100, 160), dtype=np.float32), norm='layer'))

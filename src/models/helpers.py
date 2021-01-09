from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from config import ModelConfig


def init_net(net: nn.Module, model_config: ModelConfig):
    for m in net:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    a=model_config.policy_kwargs['features_extractor_kwargs']['negative_slope'],
                                    mode='fan_in',
                                    nonlinearity='leaky_relu')


def init_weights(model: ActorCriticPolicy, model_config: ModelConfig):
    # Initialize feature extractor
    init_net(model.policy.features_extractor.cnn, model_config)
    init_net(model.policy.features_extractor.linear, model_config)

    # Initialize shared net
    init_net(model.policy.mlp_extractor.shared_net, model_config)

    # Initialize policy net
    init_net(model.policy.mlp_extractor.policy_net, model_config)

    # Initialize value net
    init_net(model.policy.mlp_extractor.value_net, model_config)

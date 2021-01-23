import json
import typing as t
from pathlib import Path

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

import constants
import paths
from environments import constants as environment_constants
from helpers.learning_schedules import combine_scheds, sched_cos


def load(config_path: str):
    """Loads a configuration from a given path and scenario name.

    :param config_path: The path to the JSON file containing the model and scenarios configurations.
    :param scenario: The name of the scenario to be retrieved from the config.
    :return: A configuration object with information about the environment and the model hyperparameters.
    """
    with open(config_path) as f:
        params = json.load(f)

    return TrainingConfig(params)


class EnvironmentConfig:
    """Class holding environment configuration information"""
    def __init__(self, params: t.Dict):
        self.scenario = params['scenario']

        self.n_envs = params['n_parallel']
        self.frame_stack = params['frame_stack']
        self.frame_skip = params['frame_skip']

        # Action space parameters
        self.action_combination = params['action_combination']
        self.action_noop = params['action_noop']

        # Observation space parameters
        self.raw_channels = params['obs_channels']
        self.raw_width = params['obs_width']
        self.raw_height = params['obs_height']
        self.crop = np.array(params['obs_crop'])
        self.resize = np.array(params['obs_resize'])

        self.game_mode = environment_constants.MODE_TO_VZD[params['vizdoom_mode']]
        self.screen_mode = environment_constants.CHANNELS_TO_VZD[self.raw_channels]
        self.screen_resolution = environment_constants.RESOLUTION_TO_VZD[(self.raw_width, self.raw_height)]

    def get_log_name(self):
        return 'scenario={}/nenvs={}_stack={}_skip={}'.format(
            self.scenario,
            self.n_envs,
            self.frame_stack,
            self.frame_skip,
        )

    def get_input_shape(self):
        width = self.resize[0] * (self.raw_width - sum(self.crop[[1, 3]]))
        height = self.resize[1] * (self.raw_height - sum(self.crop[[0, 2]]))

        return int(height), int(width), self.raw_channels


class LearningConfig:
    """Class holding learning configuration information"""
    def __init__(self, params: t.Dict):
        self.lr_start = params['lr_start']
        self.lr_mid = params['lr_mid']
        self.lr_end = params['lr_end']
        self.lr_pcts = params['lr_pcts']

    def get_learning_schedule(self):
        """Returns an annealed learning schedule according to the current configuration."""
        return combine_scheds(self.lr_pcts,
                              [sched_cos(self.lr_start, self.lr_mid),
                               sched_cos(self.lr_mid, self.lr_end)])


class ModelConfig:
    def __init__(self, params):
        self.model = params['name']
        self.model_class = constants.MODELS[self.model]
        self.model_params = params['parameters']
        self.model_params['tensorboard_log'] = paths.TENSORBOARD_LOGS
        self.policy_params = params['policy']
        self.policy_class = constants.POLICIES[self.model]
        self.policy_kwargs = {
            'features_extractor_class': constants.NETS[self.policy_params['feature_extractor']],
            'features_extractor_kwargs': {
                'negative_slope': self.policy_params['relu_slope']
            },
            'net_arch': params['policy']['net_arch'],
            'ortho_init': False,
            'activation_fn': lambda: nn.LeakyReLU(negative_slope=self.policy_params['relu_slope'])
        }

    def get_log_name(self):
        return 'algo={}_pi={}_vf={}'.format(
            self.model,
            '-'.join(map(str, self.policy_params['net_arch'][-1]['pi'])),
            '-'.join(map(str, self.policy_params['net_arch'][-1]['vf'])),
        )


class TrainingConfig:
    """Class holding configuration for a training session."""
    def __init__(self, params: t.Dict):
        self.params = params
        self.model_config = ModelConfig(params['model'])
        self.environment_config = EnvironmentConfig(params['env'])
        self.learning_config = LearningConfig(params['schedules'])

    def get_agent(self, env: VecEnv = None, load_from: str = None) -> BaseAlgorithm:
        """Creates a RL agent from the current config parameters.

        Args:
            env: A vectorized environment for the agent.
            load_from: An optional path pointing to an existing model.

        Returns:
            A RL agent according to the current config parameters if load_from is None. Otherwise, loads an existing
            model and overwrites the configuration keys found on the 'model.parameters' and 'schedules' config entries.
        """
        if load_from is None:
            return self.model_config.model_class(env=env,
                                                 policy=self.model_config.policy_class,
                                                 policy_kwargs=self.model_config.policy_kwargs,
                                                 learning_rate=self.learning_config.get_learning_schedule(),
                                                 **self.model_config.model_params)

        else:
            print(f'Loading model from {load_from}')
            return self.model_config.model_class.load(f'{paths.MODEL_LOGS}/{load_from}',
                                                      env=env,
                                                      learning_rate=self.learning_config.get_learning_schedule(),
                                                      **self.model_config.model_params)

    def get_log_name(self, input_shape: t.Tuple, output_length: int):
        """Returns a string characterizing the used configuration.

            Example:

                scenario=basic_algo=PPO_nenvs=1_input=100x160x3-RGB24_stack=1_skip=4_output=3

        """
        return '{}/{}_in={}_out={}'.format(
            self.environment_config.get_log_name(),
            self.model_config.get_log_name(),
            "x".join([str(dimension) for dimension in input_shape]),
            output_length,
        )

    def persist_model_params(self, path: str):
        """Saves the model parameters used for training at the provided location."""
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(f'{path}/model_parameters.json', "w") as out:
            json.dump(self.params, out)

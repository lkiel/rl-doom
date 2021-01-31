import typing as t

from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack, SubprocVecEnv, DummyVecEnv
from vizdoom.vizdoom import DoomGame
import paths
from config import EnvironmentConfig
from environments.doom_env import DoomEnv
from helpers import controls
from helpers.controls import ActionList


def create_game(environment_config: EnvironmentConfig) -> t.Tuple[DoomGame, ActionList]:
    """Creates an instance of VizDoom.

    Args:
        scenario: The name of the scenario to play.
        environment_config: An environment configuration instance.

    Returns:
        A Doom game instance that respects OpenAI's gym interface.
    """
    game = DoomGame()

    # Game configuration
    game.load_config(f'{paths.SCENARIOS}/{environment_config.scenario}.cfg')
    game.set_mode(environment_config.game_mode)
    game.set_screen_format(environment_config.screen_mode)
    game.init()

    possible_actions = controls.get_available_actions(
        game.get_available_buttons(),
        environment_config.action_combination,
        environment_config.action_noop
    )

    return game, possible_actions


def create_env(environment_config: EnvironmentConfig) -> DoomEnv:
    """Creates a Doom environment."""
    game, possible_actions = create_game(environment_config)
    env = DoomEnv(game, possible_actions, environment_config)
    return env


def create_vectorized_environment(n_envs: int, frame_stack: int, env_creation_func: t.Callable) -> VecTransposeImage:
    """Creates a vectorized environment for image-based models.

    :param n_envs: The number of parallel environment to run.
    :param frame_stack: The number of frame to stack in each environment.
    :param env_creation_func: A callable returning a Gym environment.
    :return: A vectorized environment with frame stacking and image transposition.
    """
    return VecTransposeImage(VecFrameStack(DummyVecEnv([env_creation_func] * n_envs), frame_stack))


def get_training_env(environment_config: EnvironmentConfig) -> VecTransposeImage:
    """Returns a vectorized environment suitable for training."""
    return create_vectorized_environment(environment_config.n_envs, environment_config.frame_stack,
                                         get_env_generating_function(environment_config))


def get_evaluation_env(environment_config: EnvironmentConfig) -> VecTransposeImage:
    """Returns a vectorized environment suitable for performance evaluation."""
    return create_vectorized_environment(1, environment_config.frame_stack,
                                         get_env_generating_function(environment_config))


def get_envs(environment_config: EnvironmentConfig) -> t.Tuple[VecTransposeImage, VecTransposeImage]:
    """Returns a pair of vectorized environments for training and evaluation respectively."""
    return get_training_env(environment_config), get_evaluation_env(environment_config)


def get_env_generating_function(environment_config: EnvironmentConfig) -> t.Callable:
    """Returns a function that generates a Doom environment wrapper."""

    return lambda: create_env(environment_config)

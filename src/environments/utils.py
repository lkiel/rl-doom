import typing as t

from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack, DummyVecEnv
from vizdoom.vizdoom import DoomGame

import paths
from config import EnvironmentConfig
from environments.doom_env import DoomEnv
from environments.doom_env_with_bots import DoomWithBots
from environments.doom_env_with_bots_curriculum import DoomWithCurriculum
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

    possible_actions = controls.get_available_actions(
        game.get_available_buttons(),
        environment_config.action_combination,
        environment_config.action_noop
    )

    return game, possible_actions


def create_env_with_bots(environment_config: EnvironmentConfig, eval: bool) -> DoomEnv:
    """Creates a Doom environment."""
    game, possible_actions = create_game(environment_config)

    game.set_doom_map(environment_config.env_args['map'])
    game.add_game_args('-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name AGENT +colorset 0' +
                       '+sv_forcerespawn 1 +sv_respawnprotect 1 +sv_nocrouch 1 +sv_noexit 1')
    game.init()

    if environment_config.env_args['curriculum'] and not eval:
        return DoomWithCurriculum(game, possible_actions, environment_config)
    else:
        return DoomWithBots(game, possible_actions, environment_config)


def create_env(environment_config: EnvironmentConfig, eval: bool) -> DoomEnv:
    """Creates a Doom environment."""
    if environment_config.env_type == 'multiplayer':
        return create_env_with_bots(environment_config, eval)

    game, possible_actions = create_game(environment_config)
    game.init()

    return DoomEnv(game, possible_actions, environment_config)


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
                                         get_env_generating_function(environment_config, eval=False))


def get_evaluation_env(environment_config: EnvironmentConfig) -> VecTransposeImage:
    """Returns a vectorized environment suitable for performance evaluation."""
    return create_vectorized_environment(1, environment_config.frame_stack,
                                         get_env_generating_function(environment_config, eval=True))


def get_envs(environment_config: EnvironmentConfig) -> t.Tuple[VecTransposeImage, VecTransposeImage]:
    """Returns a pair of vectorized environments for training and evaluation respectively."""
    return get_training_env(environment_config), get_evaluation_env(environment_config)


def get_env_generating_function(environment_config: EnvironmentConfig, eval: bool) -> t.Callable:
    """Returns a function that generates a Doom environment wrapper."""

    return lambda: create_env(environment_config, eval)

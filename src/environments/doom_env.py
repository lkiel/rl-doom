import typing as t

import gym
import numpy as np
from gym import spaces
from vizdoom.vizdoom import DoomGame

from config import EnvironmentConfig
from helpers.controls import ActionList
from helpers.frameutils import FramePreprocessor


class DoomEnv(gym.Env):
    """Wrapper environment following OpenAI's gym interface for a Doom game instance."""

    metadata = {'video.frames_per_second': 35}

    def __init__(self, game: DoomGame, possible_actions: ActionList, environment_config: EnvironmentConfig):
        super().__init__()

        self.action_space = spaces.Discrete(len(possible_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=environment_config.get_input_shape(), dtype=np.uint8)

        self.game = game
        self.possible_actions = possible_actions
        self.frame_skip = environment_config.frame_skip
        self.frame_preprocessor = FramePreprocessor(scale=environment_config.resize, crop=environment_config.crop)
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame

    def step(self, action: int) -> t.Tuple[np.array, int, bool, t.Dict[str, int]]:
        """Apply an action to the environment.

        Args:
            action:

        Returns:
            A tuple containing:
                - A numpy ndarray containing the current environment state.
                - The reward obtained by applying the provided action.
                - A boolean flag indicating whether the episode has ended.
                - An empty dict.
        """
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        done = self.game.is_episode_finished()

        self.state = self._game_frame(done)

        return self.state, reward, done, {}

    def reset(self):
        """Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        self.game.new_episode()
        self.state = self._game_frame()

        return self.state

    def render(self, **kwargs):
        return self._game_frame()

    def close(self):
        self.game.close()

    def _game_frame(self, done: bool = False):
        return self.frame_preprocessor(self.game.get_state().screen_buffer) if not done else self.empty_frame

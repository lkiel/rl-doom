import typing as t

import numpy as np
import vizdoom
from gym import Env
from gym import spaces
from stable_baselines3.common import vec_env

Frame = np.ndarray


class DoomEnv(Env):
    """Wrapper environment following OpenAI's gym interface for a VizDoom game instance."""

    def __init__(self,
                 game: vizdoom.DoomGame,
                 frame_processor: t.Callable,
                 frame_skip: int = 4):
        super().__init__()

        # Determine action space
        self.action_space = spaces.Discrete(game.get_available_buttons_size())

        # Determine observation space
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        new_h, new_w, new_c = frame_processor(np.zeros((h, w, c))).shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8)

        # Assign other variables
        self.game = game
        self.possible_actions = np.eye(self.action_space.n).tolist()  # VizDoom needs a list of buttons states.
        self.frame_skip = frame_skip
        self.frame_processor = frame_processor

        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame

    def step(self, action: int) -> t.Tuple[Frame, int, bool, t.Dict]:
        """Apply an action to the environment.

        Args:
            action:

        Returns:
            A tuple containing:
                - A numpy ndarray containing the current environment state.
                - The reward obtained by applying the provided action.
                - A boolean flag indicating whether the episode has ended.
                - An empty info dict.
        """
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        return self.state, reward, done, {}

    def reset(self) -> Frame:
        """Resets the environment.

        Returns:
            The initial state of the new environment.
        """
        self.game.new_episode()
        self.state = self._get_frame()

        return self.state

    def close(self) -> None:
        self.game.close()

    def render(self, mode='human'):
        pass

    def _get_frame(self, done: bool = False) -> Frame:
        return self.frame_processor(
            self.game.get_state().screen_buffer) if not done else self.empty_frame


class DoomWithBots(DoomEnv):

    def __init__(self, game, frame_processor, frame_skip, n_bots):
        super().__init__(game, frame_processor, frame_skip)
        self.n_bots = n_bots
        self.last_frags = 0
        self._reset_bots()

        # Redefine the action space using combinations.
        self.possible_actions = get_available_actions(np.array(game.get_available_buttons()))
        self.action_space = spaces.Discrete(len(possible_actions))

    def step(self, action):
        self.game.make_action(self.possible_actions[action], self.frame_skip)

        # Compute rewards.
        frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        reward = frags - self.last_frags
        self.last_frags = frags

        # Check for episode end.
        self._respawn_if_dead()
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        return self.state, reward, done, {}

    def reset(self):
        self._reset_bots()
        self.last_frags = 0

        return super().reset()

    def _respawn_if_dead(self):
        if not self.game.is_episode_finished():
            if self.game.is_player_dead():
                self.game.respawn_player()

    def _reset_bots(self):
        # Make sure you have the bots.cfg file next to the program entry point.
        self.game.send_game_command('removebots')
        for i in range(self.n_bots):
            self.game.send_game_command('addbot')


def create_env(scenario: str, **kwargs) -> DoomEnv:
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/{scenario}.cfg')
    game.init()

    # Wrap the game with the Gym adapter.
    return DoomEnv(game, **kwargs)


def create_env_with_bots(scenario, **kwargs) -> DoomEnv:
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/{scenario}.cfg')
    game.add_game_args('-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name AGENT +colorset 0' +
                       '+sv_forcerespawn 1 +sv_respawnprotect 1 +sv_nocrouch 1 +sv_noexit 1')
    game.init()

    return DoomWithBots(game, **kwargs)


def create_vec_env(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: create_env(**kwargs)] * n_envs))


def vec_env_with_bots(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: create_env_with_bots(**kwargs)] * n_envs))


def create_eval_vec_env(**kwargs) -> vec_env.VecTransposeImage:
    return create_vec_env(n_envs=1, **kwargs)

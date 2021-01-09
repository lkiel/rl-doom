import typing as t

import cv2
import numpy as np
import vizdoom
from gym import Env
from gym import spaces
from stable_baselines3 import ppo
from stable_baselines3.common import callbacks
from stable_baselines3.common import policies
from stable_baselines3.common import vec_env

Frame = np.ndarray


class DoomEnv(Env):
    """Wrapper environment following OpenAI's gym interface for a VizDoom game instance."""
    def __init__(self,
                 game: vizdoom.DoomGame,
                 action_space: spaces.Discrete,
                 observation_space: spaces.Box,
                 frame_skip: int = 4,
                 input_scaling_factor: float = .5):
        super().__init__()

        self.action_space = action_space
        self.observation_space = observation_space

        self.game = game
        self.possible_actions = np.eye(action_space.n).tolist()  # VizDoom needs a list of buttons states.
        self.frame_skip = frame_skip
        self.input_scaling_factor = input_scaling_factor

        self.empty_frame = np.zeros(observation_space.shape, dtype=np.uint8)
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
        return self._preprocess(self.game.get_state().screen_buffer) if not done else self.empty_frame

    def _preprocess(self, frame: np.ndarray) -> Frame:
        return cv2.resize(frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)


def create_env() -> DoomEnv:
    # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config('../scenarios/basic.cfg')
    game.init()

    # Wrap the environment with the Gym adapter.
    height, width, channels = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
    action_space = spaces.Discrete(game.get_available_buttons_size())
    observation_space = spaces.Box(low=0, high=255, shape=(height // 2, width // 2, channels), dtype=np.uint8)

    return DoomEnv(game, action_space, observation_space)


def create_vec_env() -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([create_env]))


if __name__ == '__main__':
    # Create training and evaluation environments.
    training_env, eval_env = create_vec_env(), create_vec_env()

    # Create an agent.
    agent = ppo.PPO(policies.ActorCriticCnnPolicy,
                    training_env,
                    learning_rate=1e-4,
                    tensorboard_log='logs/tensorboard')

    evaluation_callback = callbacks.EvalCallback(eval_env,
                                                 n_eval_episodes=10,
                                                 eval_freq=2500,
                                                 best_model_save_path='logs/models/basic')

    # Play!
    agent.learn(total_timesteps=25000, tb_log_name='ppo_basic', callback=evaluation_callback)

    training_env.close()
    eval_env.close()

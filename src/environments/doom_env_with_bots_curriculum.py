from collections import deque

import numpy as np

from config import EnvironmentConfig
from environments.doom_env_with_bots import DoomWithBots


class DoomWithCurriculum(DoomWithBots):

    def __init__(self, doom_game, possible_actions, environment_config: EnvironmentConfig,
                 initial_level=0, max_level=5, rolling_mean_length=10):
        super().__init__(doom_game, possible_actions, environment_config)
        doom_game.send_game_command('pukename change_difficulty 0')
        self.level = initial_level
        self.max_level = max_level
        self.rolling_mean_length = rolling_mean_length
        self.last_rewards = deque(maxlen=rolling_mean_length)
        self.reward_thresholds = [15, 20, 25, 30, 35, 40]

    def step(self, action, array=False):
        state, reward, done, infos = super().step(action, array)

        if done:
            self.last_rewards.append(self.total_rew)
            run_mean = np.mean(self.last_rewards)
            print(
                f'Avg. last 10 runs of {self.name}: {run_mean} (of length {len(self.last_rewards)}) current difficulty: {self.level}')
            if run_mean > self.reward_thresholds[self.level] and len(self.last_rewards) >= self.rolling_mean_length:
                self._change_difficulty()

        return state, reward, done, infos

    def reset(self):
        state = super().reset()
        self.game.send_game_command(f'pukename change_difficulty {self.level}')

        return state

    def _change_difficulty(self):
        if self.level < self.max_level:
            self.level += 1
            print(f'Changing difficulty for {self.name} to {self.level}')
            self.game.send_game_command(f'pukename change_difficulty {self.level}')
            self.last_rewards = deque(maxlen=self.rolling_mean_length)
        else:
            print(f'{self.name} at max level!')

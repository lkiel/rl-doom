import sys
import time

import cv2
import numpy as np
import torch as th
import vizdoom
from stable_baselines3 import ppo
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from common.envs import DoomWithBots
from common.utils import get_available_actions

sys.path.insert(0, '../src')  # To make pretrained models work (module structure different than notebooks)

print("P2")
game = vizdoom.DoomGame()
game.load_config(f'scenarios/bots_deathmatch_multimaps.cfg')

game.add_game_args("-join 127.0.0.1:12345 +name AI +cl_run 1")
game.set_doom_map('M')
game.set_mode(vizdoom.Mode.PLAYER)
game.init()

print("Joined game")
possible_actions = get_available_actions(np.array(game.get_available_buttons()))

print(possible_actions)

env = VecTransposeImage(
    VecFrameStack(
        DummyVecEnv([lambda: DoomWithBots(game, lambda frame: cv2.resize(frame[40:, 4:-4], None, fx=.5, fy=.5,
                                                                         interpolation=cv2.INTER_AREA),
                                          frame_skip=1, n_bots=0)]),
        4))

# Load agent
agent = ppo.PPO.load('../trained_agents/deathmatch_512_256-256_stack=4/best_model.zip')
agent.policy.to('cpu')

obs = None

t2 = time.time_ns()
t1 = t2

frame_counter = 0

while not game.is_episode_finished():
    if game.is_player_dead():
        game.respawn_player()

    t2 = time.time_ns()
    if t2 - t1 > 1e9:
        t1 = t2
        # print(frame_counter)
        frame_counter = 0

    if obs is not None:
        obs_tensor = th.as_tensor(obs)
        action, values, log_probs = agent.policy.forward(obs_tensor)
        obs, _, _, _ = env.step(action.cpu().numpy())
        frame_counter += 1
    else:
        obs, _, _, _ = env.step([0])

game.close()

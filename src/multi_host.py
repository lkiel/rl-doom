from __future__ import print_function

import time
from random import choice

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack, DummyVecEnv
from vizdoom.vizdoom import DoomGame, Mode, GameVariable

from environments import doom_env
from environments.doom_env import DoomEnv
from helpers import controls

episodes = 10
import torch as th
def player1():
    game = DoomGame()
    game.load_config('../scenarii/custom_bots_v3.cfg')
    game.set_doom_scenario_path('../scenarii/custom_bots_v3.wad')
    game.set_doom_map('flatmap3')
    game.set_mode(Mode.SPECTATOR)
    game.add_game_args("-host 2 -deathmatch +cl_run 1")
    game.add_game_args("+name Player1 +colorset 0")
    #game.set_ticrate(45)
    game.init()

    # Play until the game (episode) is over.
    while not game.is_episode_finished():

        game.advance_action()

        # Check if player is dead
        if game.is_player_dead():
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

    game.close()

def player2():
    game = DoomGame()
    game.load_config('../scenarii/custom_bots_v3.cfg')
    game.set_doom_scenario_path('../scenarii/custom_bots_v3.wad')
    game.set_doom_map('flatmap3')
    game.set_mode(Mode.PLAYER)
    game.add_game_args("-join 127.0.0.1 +name Leo +cl_run 1")

    game.init()

    possible_actions = controls.get_available_actions(game.get_available_buttons(),
                                                      add_combinations=True,
                                                      add_noop=False)
    env = VecTransposeImage(VecFrameStack(DummyVecEnv([lambda: DoomEnv(game, possible_actions)]), n_stack=4))

    agent = PPO.load(r'D:\Leandro\ML\doom_rl\trained_agents\flatmap_3_out=18_stack=4\agent.zip')

    obs=None

    t1 = time.time_ns()
    t2 = time.time_ns()
    print_freq = 100
    i = 0
    for i in range(episodes):
        while not game.is_episode_finished():
            i+=1
            if game.is_player_dead():
                game.respawn_player()
            t1 = t2
            t2 = time.time_ns()
            if i % print_freq == 0:
                print(f'{(t2 - t1) // 1e6}')
            if obs is not None:
                obs_tensor = th.as_tensor(obs).to('cuda')
                action, values, log_probs = agent.policy.forward(obs_tensor)
                obs, _, _, _ = env.step(action.cpu().numpy())
            else:
                obs, _, _, _  = env.step([0])

            #game.make_action(choice(actions))

        print("Player2 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
        game.new_episode()

    game.close()

from multiprocessing import Process
if __name__ == '__main__':
    p1 = Process(target=player1)
    p1.start()
    player2()

    print("Done")



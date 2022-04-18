import time
from multiprocessing import Process

import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from vizdoom.vizdoom import DoomGame, Mode, GameVariable, Button

import config
from environments import utils
from environments.doom_env import DoomEnv
from helpers import cli

episodes = 10


def player1(n_bots):
    game = DoomGame()
    game.load_config('../scenarios/bots_deathmatch_multimaps.cfg')
    game.set_doom_map('M')
    game.set_mode(Mode.SPECTATOR)
    game.add_game_args("-host 2 -deathmatch +cl_run 1")
    game.add_game_args("+name EoD +colorset 0")

    game.set_ticrate(35)

    game.init()

    for i in range(n_bots):
        game.send_game_command("addbot")

    while not game.is_episode_finished():
        game.advance_action()

        if game.is_player_dead():
            game.respawn_player()

    time.sleep(2)
    game.close()


def player2(args):
    load_from = args.load
    config_path = args.config

    # Load config for session
    conf = config.load(config_path)

    game, possible_actions = utils.create_game(conf.environment_config)
    game.add_game_args("-join 127.0.0.1 +name HAL9001 +cl_run 1")
    game.set_doom_map('M')
    game.set_mode(Mode.PLAYER)
    game.init()

    env = VecTransposeImage(
        VecFrameStack(
            DummyVecEnv([lambda: DoomEnv(game, possible_actions, conf.environment_config)]),
            conf.environment_config.frame_stack))

    # Load agent
    agent = conf.get_agent(env=env, load_from=load_from)
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
            #print(frame_counter)
            frame_counter = 0

        if obs is not None:
            obs_tensor = th.as_tensor(obs)
            action, values, log_probs = agent.policy.forward(obs_tensor)
            obs, _, _, _ = env.step(action.cpu().numpy())
            frame_counter += 1
        else:
            obs, _, _, _ = env.step([0])

        # game.make_action(choice(actions))

    game.close()


if __name__ == '__main__':
    parser = cli.get_parser()
    parser.add_argument('-b', '--bots', type=int)
    args = parser.parse_args()

    p1 = Process(target=player1, args=(args.bots,))
    p1.start()
    player2(args)

    print("Done")

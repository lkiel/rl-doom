import time
from multiprocessing import Pool

import cv2
from stable_baselines3 import ppo
from stable_baselines3.common import callbacks
from stable_baselines3.common import policies

from callbacks.LayerMonitoring import LayerActivationMonitoring
from environments import utils as env_utils
from helpers import cli
from models import helpers

import config
from common import envs


def evaluate_training(n: int, config_path):
    print(n, config_path)
    time.sleep((n % 3) * 10)

    # Load config for session
    conf = config.load(config_path)

    # Create environments.
    env, eval_env = env_utils.get_envs(conf.environment_config)

    # Build the agent
    agent = conf.get_agent(env=env)
    helpers.init_weights(agent, conf.model_config)

    print(agent.policy)

    # Callbacks
    layer_monitoring = LayerActivationMonitoring()
    evaluation_callback = callbacks.EvalCallback(eval_env,
                                                 n_eval_episodes=10,
                                                 eval_freq=8192,
                                                 log_path=f'logs/evaluations/defend_the_center_gerelu_{n}',
                                                 best_model_save_path=f'logs/models/defend_the_center_gerelu_{n}')

    # Play!
    agent.learn(total_timesteps=500000, tb_log_name='ppo_defend_the_center_gerelu', callback=evaluation_callback)

    env.close()
    eval_env.close()
    print(f'Finished {n}')
    return n


if __name__ == '__main__':
    # Extract command line arguments
    parser = cli.get_parser()
    args = parser.parse_args()

    load_from = args.load
    features_only = args.features_only
    config_path = args.config

    with Pool(3) as pool:
        pool.starmap(evaluate_training, zip(range(6), [config_path] * 6))

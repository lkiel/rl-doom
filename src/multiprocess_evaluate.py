import time
from multiprocessing import Pool

import cv2
from stable_baselines3 import ppo
from stable_baselines3.common import callbacks
from stable_baselines3.common import policies
from callbacks.FragEvalCallback import FragEvalCallback
from callbacks.LayerMonitoring import LayerActivationMonitoring
from environments import utils as env_utils
from helpers import cli
from models import helpers

import config

NAME = 'dm_shaping_curriculum'


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
    evaluation_callback = FragEvalCallback(eval_env,
                                           n_eval_episodes=10,
                                           eval_freq=16384,
                                           log_path=f'logs/evaluations/{NAME}_{n}',
                                           best_model_save_path=f'logs/models/{NAME}_{n}',
                                           deterministic=True)

    # Play!
    agent.learn(total_timesteps=3000000, tb_log_name=NAME, callback=[evaluation_callback, layer_monitoring])

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

from stable_baselines3.common import evaluation

import config
from environments import utils as env_utils

from helpers import cli

if __name__ == '__main__':
    # Extract command line arguments
    parser = cli.get_parser()
    args = parser.parse_args()

    scenario = args.scenario
    load_from = args.load
    config_path = args.config

    # Load config for session
    conf = config.load(config_path)

    # Create environments.
    eval_env = env_utils.get_evaluation_env(conf.environment_config)

    # Load agent
    agent = conf.get_agent(env=eval_env, load_from=load_from)

    mean_reward, std_reward = evaluation.evaluate_policy(agent, eval_env, n_eval_episodes=100)
    print(f'Mean reward {mean_reward} +/- {std_reward}')

    eval_env.close()

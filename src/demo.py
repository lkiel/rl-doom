import time

from stable_baselines3.common import evaluation

import config
from environments import utils as env_utils

from helpers import cli

ORANGE = '\033[1m\x1B[31m'
NOCOLOR = '\033[0m'


def colored_button(button_str, value):
    if value > 0:
        return f'{ORANGE}{button_str}: {value}{NOCOLOR}'
    else:
        return f'{button_str}: {value}'


def print_action(buttons, possible_actions, action_index):
    print(' | '.join([colored_button(b, a) for b, a in zip(buttons, possible_actions[action_index])]), end=' \r', flush=True)


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
    env = env_utils.get_evaluation_env(conf.environment_config)

    # Load agent
    agent = conf.get_agent(env=env, load_from=load_from)

    # Extract button acronyms
    buttons = env.venv.envs[0].game.get_available_buttons()
    buttons = [str(b).split('.')[1] for b in buttons]

    # Extract possible actions
    possible_actions = env.venv.envs[0].possible_actions

    for i in range(10):
        obs = env.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=False)
            print_action(buttons, possible_actions, action[0])
            obs, reward, done, _ = env.step(action)
            time.sleep(1/45.0)

    env.close()

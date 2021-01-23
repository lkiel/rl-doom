from stable_baselines3.common.callbacks import EvalCallback

import config
from environments import constants as env_constants
from environments import utils as env_utils
from helpers import cli
from models import helpers
import paths

if __name__ == '__main__':
    # Extract command line arguments
    parser = cli.get_parser()
    args = parser.parse_args()

    load_from = args.load
    features_only = args.features_only
    config_path = args.config

    # Load config for session
    conf = config.load(config_path)

    # Create environments.
    env, eval_env = env_utils.get_envs(conf.environment_config)

    # Set-up the logging folder
    name_suffix = conf.get_log_name(env_constants.OBS_SHAPE, env.action_space.n)
    model_folder = f'{paths.MODEL_LOGS}/{name_suffix}/'

    # Save the parameters used next to the best model
    conf.persist_model_params(model_folder)

    # Build the agent
    agent = conf.get_agent(env=env, load_from=load_from)

    if not load_from:
        helpers.init_weights(agent, conf.model_config)

    # Start the training process.
    evaluation_callback = EvalCallback(eval_env,
                                       n_eval_episodes=10,
                                       eval_freq=15000,
                                       best_model_save_path=model_folder,
                                       deterministic=True)

    agent.learn(total_timesteps=5000000, tb_log_name=name_suffix.replace('/', '_'), callback=[evaluation_callback])

    env.close()
    eval_env.close()

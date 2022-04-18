import time
from multiprocessing import Pool

import cv2
from stable_baselines3 import ppo
from stable_baselines3.common import callbacks
from stable_baselines3.common import policies

from common import envs


def evaluate_training(n: int):
    print(n)
    time.sleep((n % 3) * 10)

    def frame_processor(frame):
        return cv2.resize(
            frame[40:-41, :], None,
            fx=0.5, fy=0.5,
            interpolation=cv2.INTER_AREA)

    # Create training and evaluation environments.
    training_env = envs.create_vec_env(frame_skip=4, frame_processor=frame_processor)
    eval_env = envs.create_vec_env(frame_skip=4, frame_processor=frame_processor)

    # Create an agent.
    agent = ppo.PPO(policy=policies.ActorCriticCnnPolicy,
                    env=training_env,
                    n_steps=4096,
                    batch_size=32,
                    learning_rate=1e-4,
                    tensorboard_log='logs/tensorboard',
                    policy_kwargs={'features_extractor_kwargs': {'features_dim': 64}})

    print(agent.policy)

    evaluation_callback = callbacks.EvalCallback(eval_env,
                                                 n_eval_episodes=10,
                                                 eval_freq=5000,
                                                 log_path=f'logs/evaluations/basic_run_fs=4_small_{n}',
                                                 best_model_save_path=f'logs/models/basic_run_fs=4_small_{n}')

    # Play!
    agent.learn(total_timesteps=50000, tb_log_name='ppo_basic', callback=evaluation_callback)

    training_env.close()
    eval_env.close()
    print(f'Finished {n}')
    return n


if __name__ == '__main__':
    with Pool(3) as pool:
        pool.map(evaluate_training, range(6))
